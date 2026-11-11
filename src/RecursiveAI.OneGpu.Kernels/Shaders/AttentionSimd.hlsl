// Fused scaled dot-product attention on Apple GPUs' simdgroup matrices:  O = softmax(Scale * Q K^T + Mask) V  for float32, float16 and bfloat16
// tensors, with float32 accumulation and softmax. The semantics, push constants and bindings are those of Attention.hlsl.
//
// The scores are computed transposed, S^T = K Q^T, so that the key fragments are read along their rows, the contiguous direction of the tensor
// (a fragment of K^T would gather two elements a row apart per lane).
//
// A simdgroup (32 lanes) owns 16 query rows and walks over the keys in blocks of 16, once:
// the online normaliser of Milakov and Gimelshein (2018) in the tiled form of Dao et al. (FlashAttention, 2022), with one change that suits
// matrix registers. The layout of the elements of a simdgroup matrix is not specified, so an accumulator cannot be scaled per row where it
// lives; scaling one costs a round trip through workgroup memory. The kernel therefore keeps a reference maximum per row that it raises
// only when a block's maximum exceeds it by more than REF_SLACK (8, so a probability is at most e^8, harmless in float32 accumulators and
// representable in float16 and bfloat16). The result is the same softmax whatever the reference: numerator and denominator use the same
// reference. Most rows set it on their first block, when the accumulator is still zero and needs no scaling; a later raise rescales the
// accumulators of the whole workgroup (all its simdgroups take part together because workgroup barriers must be reached by every thread).
//
// Keys and values are read straight from the buffers into matrix fragments whenever the block lies inside the key range and the head
// dimension is a whole number of fragments (the cache serves the rows neighbouring fragments share); only edge blocks and padded head
// dimensions are staged through workgroup memory. Scores and probabilities pass between the matrix units and the lanes through
// workgroup memory, where lane l of a simdgroup owns row l & 15 and every other column (l >> 4) + 2 i of each 16-wide block; the two lanes of a
// row meet in shuffles that maximise or add two numbers, which do not depend on operand order. Every other order is fixed by the code:
// blocks in increasing key order, the matrix products are the hardware's and depend only on their operands, there are no atomics and
// no subgroup arithmetic; whether a rescale happens depends only on the data. Two runs on one device give identical bits.
//
// The probabilities are rounded to the element type before the product with V, and the normaliser sums the rounded values.

#include "Common.hlsl"

#if ELEM_F32
typedef float Elem;
#elif ELEM_BF16
typedef bfloat16_t Elem;
#else
typedef float16_t Elem;
#endif

#if HD_D32
#define DT 2
#elif HD_D64
#define DT 4
#else
#define DT 8
#endif

// Key blocks of 16 per step. Two or four per step ran no faster (measured on an Apple M2) and the workgroup memory (about 13 KB with float32 and one
// block) must stay under half of Metal's limit, which shader validation counts twice.
#define NB 1

#define BC (NB * 16)
#define DIM (DT * 16)
// With QD the Q fragments are read from the tensor at every step instead of being held in registers (a float32 head of 64 would take 32 of them), which
// lets a workgroup of four simdgroups keep enough work in flight; the host picks it when the head dimension is a whole number of fragments and there are
// at least 16 query rows. Without it a workgroup is one simdgroup, which keeps short queries (decoder steps) cheap.
#if QD
#define SGS 4
#else
#define SGS 1
#endif
#define THREADS (SGS * 32)
#define ROWS (SGS * 16)
// Padded row of the staged tile, so fragment loads spread over the banks.
#define TS (DIM + 8)
#define REF_SLACK 8.0

#define FLAG_MASK 1u
#define FLAG_CAUSAL 2u
#define FLAG_LENS 4u
#define FLAG_ZERO 8u

struct AttnParams
{
    uint Heads;
    uint Group;       // query heads per key/value head
    uint QLen;
    uint KLen;
    uint D;
    float Scale;
    int CausalOff;
    uint Flags;
    uint QBatch; uint QHead; uint QRow;
    uint KBatch; uint KHead; uint KRow;
    uint VBatch; uint VHead; uint VRow;
    uint OBatch; uint OHead; uint ORow;
    uint QOff; uint KOff; uint VOff; uint OOff;
    uint MOff; uint MBatch; uint MHead; uint MRow;
    uint LensOff;
    uint NSplit;
    uint KeysPerSplit;
    uint GroupBase;   // first workgroup of this dispatch in the whole problem's numbering (long problems are cut into several dispatches)
};

[[vk::binding(0, 0)]] StructuredBuffer<Elem> AQ;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> AK;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> AV;
[[vk::binding(3, 0)]] StructuredBuffer<float> AMask;
[[vk::binding(4, 0)]] StructuredBuffer<int> ALens;
[[vk::binding(5, 0)]] RWStructuredBuffer<Elem> AOut;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> APart;

// Declared after the buffers so that Metal numbers the buffers first, the way the host binds them.
[[vk::push_constant]] ConstantBuffer<AttnParams> AP;

groupshared Elem Tile[BC * TS];
groupshared float Sst[SGS * NB * 256];
groupshared Elem Pst[SGS * 16 * BC];
groupshared uint Rescale[SGS];

typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

bool IsNegInf(float x)
{
    return asuint(x) == 0xFF800000u;
}

// True for NaN and +infinity: a score like that makes the whole row NaN.
bool Poisoned(float x)
{
    uint t = asuint(x);
    return (t - INF_BITS) < 0x00800000u || t > 0xFF800000u;
}

// Stages BC rows of 16-bit tensor data (row = key) into Tile, zero beyond the key range and beyond the head dimension.
#define STAGE(BUF, BASE, ROWSTRIDE, KB, KEND)                                                       \
    for (uint e = tid; e < BC * DIM; e += THREADS)                                                     \
    {                                                                                               \
        uint row = e / DIM;                                                                         \
        uint c = e % DIM;                                                                           \
        uint key = (KB) + row;                                                                      \
        Elem v = (Elem)0.0f;                                                                        \
        if (c < AP.D && key < (KEND))                                                               \
        {                                                                                           \
            v = BUF[(BASE) + key * (ROWSTRIDE) + c];                                                \
        }                                                                                           \
        Tile[row * TS + c] = v;                                                                     \
    }

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void AttnSimd(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint lin = AP.GroupBase + gid.x;

    uint nQB = (AP.QLen + ROWS - 1u) / ROWS;
    uint split = lin % AP.NSplit;
    uint t = lin / AP.NSplit;
    uint qb = t % nQB;
    uint bh = t / nQB;
    uint b = bh / AP.Heads;
    uint h = bh % AP.Heads;
    uint kvh = h / AP.Group;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint r = lane & 15u;
    uint hh = lane >> 4;

    // Rows past the end recompute the last row and skip the writes, so every barrier is reached by every thread.
    uint q0 = qb * ROWS + sg * 16u;
#if QD
    // The block of 16 rows is slid back to end at the last row when it would run past it; the rows before q0 are another workgroup's and are not written.
    uint qs = min(q0, AP.QLen - 16u);
#else
    uint qs = q0;
#endif
    uint qRow = qs + r;
    bool live = qRow >= q0 && qRow < AP.QLen;
    uint qr = min(qRow, AP.QLen - 1u);

    uint kBegin = split * AP.KeysPerSplit;
    uint kEnd = min(AP.KLen, kBegin + AP.KeysPerSplit);
    if ((AP.Flags & FLAG_LENS) != 0u)
    {
        kEnd = min(kEnd, (uint)max(ALens[AP.LensOff + b], 0));
    }

    if ((AP.Flags & FLAG_CAUSAL) != 0u)
    {
        int lastQ = (int)min(AP.QLen - 1u, qb * ROWS + ROWS - 1u);
        int visible = lastQ + AP.CausalOff + 1;
        kEnd = min(kEnd, (uint)max(visible, 0));
    }

    // The same for every thread of the group, so the loops below and their barriers are uniform.
    uint nTiles = kEnd > kBegin ? (kEnd - kBegin + BC - 1u) / BC : 0u;
    bool wholeFragments = (AP.D == DIM);

    uint kBase = AP.KOff + b * AP.KBatch + kvh * AP.KHead;
    uint vBase = AP.VOff + b * AP.VBatch + kvh * AP.VHead;
    uint mBase = AP.MOff + b * AP.MBatch + h * AP.MHead + qr * AP.MRow;

    // Q fragments, kept in registers. The tile holds BC rows, so BC / 16 simdgroups stage their rows per round.
#if QD
    uint qLoad = AP.QOff + b * AP.QBatch + h * AP.QHead + qs * AP.QRow;
#else
    MatB qf[DT];
    for (uint rd = 0u; rd < SGS / NB; rd++)
    {
        for (uint e = tid; e < BC * DIM; e += THREADS)
        {
            uint row = e / DIM;
            uint c = e % DIM;
            uint srow = min(qb * ROWS + (rd * NB + row / 16u) * 16u + (row % 16u), AP.QLen - 1u);
            Elem v = (Elem)0.0f;
            if (c < AP.D)
            {
                v = AQ[AP.QOff + b * AP.QBatch + h * AP.QHead + srow * AP.QRow + c];
            }

            Tile[row * TS + c] = v;
        }

        GroupMemoryBarrierWithGroupSync();
        if (sg / NB == rd)
        {
            [unroll] for (uint kt = 0; kt < DT; kt++)
            {
                qf[kt].Load(Tile, (sg % NB) * 16 * TS + kt * 16, TS, MatrixLayout::ColumnMajor);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
#endif

    float scale = AP.Scale;
    uint flags = AP.Flags;
    int causalLimit = (int)qr + AP.CausalOff;

    MatC acc[DT];
    [unroll] for (uint z = 0; z < DT; z++)
    {
        acc[z] = MatC::Zero();
    }

    float m = NegInf();     // the row's reference maximum (both lanes of a row hold the same value)
    float l = 0.0;          // this lane's share of the normaliser
    uint bad = 0u;

    for (uint tile = 0u; tile < nTiles; tile++)
    {
        uint kb = kBegin + tile * BC;
        bool full = (wholeFragments && (kb + BC <= kEnd));

        if (!full)
        {
            // The previous step's value tile is no longer read once every simdgroup has passed the barrier before this one.
            GroupMemoryBarrierWithGroupSync();
            STAGE(AK, kBase, AP.KRow, kb, kEnd)
            GroupMemoryBarrierWithGroupSync();
        }

        [unroll] for (uint j = 0; j < NB; j++)
        {
            MatC s = MatC::Zero();
            [unroll] for (uint kt = 0; kt < DT; kt++)
            {
                MatA kf;
                if (full)
                {
                    kf.Load(AK, kBase + (kb + j * 16u) * AP.KRow + kt * 16u, AP.KRow, MatrixLayout::RowMajor);
                }
                else
                {
                    kf.Load(Tile, j * 16 * TS + kt * 16, TS, MatrixLayout::RowMajor);
                }

#if QD
                MatB qfr;
                qfr.Load(AQ, qLoad + kt * 16u, AP.QRow, MatrixLayout::ColumnMajor);
                s = MatC::MulAdd(kf, qfr, s);
#else
                s = MatC::MulAdd(kf, qf[kt], s);
#endif
            }

            s.Store(Sst, (sg * NB + j) * 256, 16, MatrixLayout::RowMajor);
        }

        GroupMemoryBarrierWithGroupSync();

        // The key tile is no longer read, so the value tile may take its place; it becomes visible at the barriers below.
        if (!full)
        {
            STAGE(AV, vBase, AP.VRow, kb, kEnd)
        }

        float sv[NB * 8];
        float tm = NegInf();
        [unroll] for (uint j2 = 0; j2 < NB; j2++)
        {
            [unroll] for (uint i = 0; i < 8; i++)
            {
                uint col = hh + 2u * i;
                uint key = kb + j2 * 16u + col;
                bool vis = key < kEnd;
                if ((flags & FLAG_CAUSAL) != 0u)
                {
                    vis = vis && (int)key <= causalLimit;
                }

                float sc = NegInf();
                if (vis)
                {
                    sc = Sst[(sg * NB + j2) * 256 + col * 16 + r] * scale;
                    if ((flags & FLAG_MASK) != 0u)
                    {
                        sc += AMask[mBase + key];
                    }

                    if (Poisoned(sc))
                    {
                        bad = 1u;
                    }
                }

                sv[j2 * 8 + i] = sc;
                if (sc > tm)
                {
                    tm = sc;
                }
            }
        }

        float tmo = WaveReadLaneAt(tm, lane ^ 16u);
        tm = (tmo > tm) ? tmo : tm;

        // A row that has seen no finite score takes the block's maximum as its reference (its accumulator is still zero, so nothing
        // needs scaling); a row whose block maximum is far above its reference raises the reference and scales what it has so far.
        bool fresh = IsNegInf(m);
        bool raise = !fresh && (tm > m + REF_SLACK);
        float alpha = 1.0;
        if (raise)
        {
            alpha = exp(m - tm);
        }

        if (fresh || raise)
        {
            m = tm;
        }

        l = l * alpha;
        bool anyRaise = WaveActiveAnyTrue(raise);
        if (lane == 0u)
        {
            Rescale[sg] = anyRaise ? 1u : 0u;
        }

        GroupMemoryBarrierWithGroupSync();

        uint anyRescale = 0u;
        [unroll] for (uint q = 0; q < SGS; q++)
        {
            anyRescale |= Rescale[q];
        }

        if (anyRescale != 0u)
        {
            // Rare: some row of the workgroup raised its reference. Every simdgroup takes the round trip together; rows that did not raise
            // multiply by 1.
            [unroll] for (uint dt = 0; dt < DT; dt++)
            {
                acc[dt].Store(Sst, sg * NB * 256, 16, MatrixLayout::ColumnMajor);
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (uint i = 0; i < 8; i++)
                {
                    uint col = hh + 2u * i;
                    Sst[sg * NB * 256 + col * 16 + r] *= alpha;
                }

                GroupMemoryBarrierWithGroupSync();
                acc[dt].Load(Sst, sg * NB * 256, 16, MatrixLayout::ColumnMajor);
                GroupMemoryBarrierWithGroupSync();
            }
        }

        [unroll] for (uint j3 = 0; j3 < NB; j3++)
        {
            [unroll] for (uint i = 0; i < 8; i++)
            {
                uint col = hh + 2u * i;
                float sc = sv[j3 * 8 + i];
                Elem p = (Elem)0.0f;
                if (!IsNegInf(sc))
                {
                    p = (Elem)exp(sc - m);
                }

                l += (float)p;
                Pst[sg * 16 * BC + j3 * 256 + col * 16 + r] = p;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint j4 = 0; j4 < NB; j4++)
        {
            MatA pa;
            pa.Load(Pst, sg * 16 * BC + j4 * 256, 16, MatrixLayout::ColumnMajor);
            [unroll] for (uint dt2 = 0; dt2 < DT; dt2++)
            {
                MatB vf;
                if (full)
                {
                    vf.Load(AV, vBase + (kb + j4 * 16u) * AP.VRow + dt2 * 16u, AP.VRow, MatrixLayout::RowMajor);
                }
                else
                {
                    vf.Load(Tile, j4 * 16 * TS + dt2 * 16, TS, MatrixLayout::RowMajor);
                }

                acc[dt2] = MatC::MulAdd(pa, vf, acc[dt2]);
            }
        }
    }

    {
        float mo = WaveReadLaneAt(m, lane ^ 16u);
        uint bo = WaveReadLaneAt(bad, lane ^ 16u);
        m = (mo > m) ? mo : m;
        bad = bad | bo;
    }

    l = l + WaveReadLaneAt(l, lane ^ 16u);

    bool empty = (l == 0.0);
    bool special = (bad != 0u) || empty;
    float fill = 0.0;
    if ((bad != 0u) || (empty && (flags & FLAG_ZERO) == 0u))
    {
        fill = QuietNaN();
    }

    float inv = empty ? 0.0 : 1.0 / l;
    uint oBase = AP.OOff + b * AP.OBatch + h * AP.OHead + qRow * AP.ORow;
    uint rec = ((bh * AP.QLen + qRow) * AP.NSplit + split) * (AP.D + 2u);
    if (AP.NSplit != 1u && live && hh == 0u)
    {
        APart[rec] = (bad != 0u) ? QuietNaN() : m;
        APart[rec + 1u] = l;
    }

    // The accumulators leave through workgroup memory, one matrix at a time.
    [unroll] for (uint dt3 = 0; dt3 < DT; dt3++)
    {
        GroupMemoryBarrierWithGroupSync();
        acc[dt3].Store(Sst, sg * NB * 256, 16, MatrixLayout::ColumnMajor);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint i = 0; i < 8; i++)
        {
            uint col = hh + 2u * i;
            uint d = dt3 * 16u + col;
            if (live && d < AP.D)
            {
                float v = Sst[sg * NB * 256 + col * 16 + r];
                if (AP.NSplit == 1u)
                {
                    AOut[oBase + d] = (Elem)(special ? fill : v * inv);
                }
                else
                {
                    APart[rec + 2u + d] = v;
                }
            }
        }
    }
}
