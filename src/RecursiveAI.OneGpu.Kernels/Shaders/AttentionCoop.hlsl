// Fused scaled dot-product attention on matrix hardware:  O = softmax(Scale * Q K^T + Mask) V  with 16-bit Q, K, V and O, float32 accumulation
// and float32 softmax. The semantics, push constants and bindings are those of Attention.hlsl; only the arithmetic differs: Q K^T and P V run as
// cooperative matrices (VK_KHR_cooperative_matrix on Vulkan, simdgroup_matrix on Metal), 16 x 16 x 16 at a time.
//
// A workgroup owns 64 query rows as four 32-wide subgroups of 16 rows each, and walks over the keys in tiles of BC keys. The softmax uses the
// exact row maximum, found in a first pass over the keys that computes only the scores (Milakov and Gimelshein 2018 normaliser in two passes
// instead of one); a second pass recomputes the scores, forms p = exp(s - max) with the final maximum, and accumulates p V. Because the maximum
// is final no accumulator is ever rescaled, so the accumulators stay in matrix registers: the layout of the elements of a cooperative matrix
// is not specified, so per-row scaling of one would have to go through workgroup memory on every tile. The price is a second Q K^T (1.5x the
// matrix work), which the matrix units absorb. Dao et al. (FlashAttention, 2022) give the tiling.
//
// Scores and probabilities leave and enter the matrix units through workgroup memory, where each lane of a subgroup handles one query row and
// every other column of its 16-wide block (lane l: row l & 15, columns (l >> 4) + 2 i). The two lanes of a row meet once per pass through one
// shuffle that adds (or maximises) two numbers, which does not depend on operand order. Every other order is fixed by the code: tiles in
// increasing key order, the matrix multiplications are the hardware's and depend only on their operands, there are no atomics and no subgroup
// arithmetic. Two runs on one device give identical bits.
//
// The probabilities are rounded to the 16-bit element type before the product with V, and the normaliser sums the rounded values, so the
// numerator and the denominator see the same p.
//
// HD selects the head dimension rounded up to 32, 64 or 128 (the matrices are 16 wide, padded with zeros). The host picks this kernel only
// where the device offers 16 x 16 x 16 matrices of the element type with a float32 accumulator and runs 32-wide subgroups, with full subgroups.

#include "Common.hlsl"

#if BF16
typedef bfloat16_t Elem;
#else
typedef float16_t Elem;
#endif

#if HD_D32
#define DT 2
#define BC 32
#elif HD_D64
#define DT 4
#define BC 32
#else
#define DT 8
#define BC 16
#endif

#define DIM (DT * 16)
#define NB (BC / 16)
#define SGS 4
// Padded row of the key/value tile, so fragment loads stay 16-byte aligned and spread over the banks.
#define TS (DIM + 8)

#define FLAG_MASK 1u
#define FLAG_CAUSAL 2u
#define FLAG_LENS 4u
#define FLAG_ZERO 8u

struct AttnParams
{
    uint HeadsGroup;  // heads in the low 16 bits, query heads per key/value head in the high 16 bits
    uint QLen;
    uint KLen;
    uint DFlags;      // head dimension in the low 16 bits, FLAG_* in the high 16 bits
    float Scale;
    int CausalOff;
    uint QBatch; uint QHead; uint QRow;
    uint KBatch; uint KHead; uint KRow;
    uint VBatch; uint VHead; uint VRow;
    uint OBatch; uint OHead; uint ORow;
    uint QOff; uint KOff; uint VOff; uint OOff;
    uint MOff; uint MBatch; uint MHead; uint MRow;
    uint LensOff;
    uint NSplit;
    uint KeysPerSplit;
    uint GridX;
    uint GroupBase;   // first workgroup of this dispatch in the whole problem's numbering (long problems are cut into several dispatches)
    uint GroupLimit;  // workgroups in the whole problem; the grid may overshoot
};

// Push constants are limited to 128 bytes on Vulkan, so two pairs of small values share a word.
#define P_HEADS (AP.HeadsGroup & 0xFFFFu)
#define P_GROUP (AP.HeadsGroup >> 16)
#define P_D (AP.DFlags & 0xFFFFu)
#define P_FLAGS (AP.DFlags >> 16)

[[vk::push_constant]] ConstantBuffer<AttnParams> AP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> AQ;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> AK;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> AV;
[[vk::binding(3, 0)]] StructuredBuffer<float> AMask;
[[vk::binding(4, 0)]] StructuredBuffer<int> ALens;
[[vk::binding(5, 0)]] RWStructuredBuffer<Elem> AOut;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> APart;

groupshared Elem Tile[BC * TS];
groupshared float Sst[SGS * 256];
groupshared Elem Pst[SGS * 16 * BC];

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

[shader("compute")]
[numthreads(128, 1, 1)]
void AttnCoop(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint lin = AP.GroupBase + gid.y * AP.GridX + gid.x;
    if (lin >= AP.GroupLimit)
    {
        return;
    }

    uint nQB = (AP.QLen + 63u) / 64u;
    uint split = lin % AP.NSplit;
    uint t = lin / AP.NSplit;
    uint qb = t % nQB;
    uint bh = t / nQB;
    uint b = bh / P_HEADS;
    uint h = bh % P_HEADS;
    uint kvh = h / P_GROUP;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint r = lane & 15u;
    uint hh = lane >> 4;

    // Rows past the end recompute the last row and skip the writes, so every barrier is reached by every thread.
    uint qRow = qb * 64u + sg * 16u + r;
    bool live = qRow < AP.QLen;
    uint qr = live ? qRow : (AP.QLen - 1u);

    uint kBegin = split * AP.KeysPerSplit;
    uint kEnd = min(AP.KLen, kBegin + AP.KeysPerSplit);
    if ((P_FLAGS & FLAG_LENS) != 0u)
    {
        kEnd = min(kEnd, (uint)max(ALens[AP.LensOff + b], 0));
    }

    if ((P_FLAGS & FLAG_CAUSAL) != 0u)
    {
        int lastQ = (int)min(AP.QLen - 1u, qb * 64u + 63u);
        int visible = lastQ + AP.CausalOff + 1;
        kEnd = min(kEnd, (uint)max(visible, 0));
    }

    // The same for every thread of the group, so the loops below and their barriers are uniform.
    uint nTiles = kEnd > kBegin ? (kEnd - kBegin + BC - 1u) / BC : 0u;

    uint kBase = AP.KOff + b * AP.KBatch + kvh * AP.KHead;
    uint vBase = AP.VOff + b * AP.VBatch + kvh * AP.VHead;
    uint mBase = AP.MOff + b * AP.MBatch + h * AP.MHead + qr * AP.MRow;

    // Q fragments, kept in registers for both passes. The tile holds BC rows, so BC / 16 subgroups stage their rows per round.
    MatA qf[DT];
    for (uint rd = 0u; rd < SGS / NB; rd++)
    {
        for (uint e = tid; e < BC * DIM; e += 128u)
        {
            uint row = e / DIM;
            uint c = e % DIM;
            uint srow = min(qb * 64u + (rd * NB + row / 16u) * 16u + (row % 16u), AP.QLen - 1u);
            Elem v = (Elem)0.0f;
            if (c < P_D)
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
                qf[kt].Load(Tile, (sg % NB) * 16 * TS + kt * 16, TS, MatrixLayout::RowMajor);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float scale = AP.Scale;
    uint flags = P_FLAGS;
    int causalLimit = (int)qr + AP.CausalOff;

    // Pass 1: the exact maximum of the visible scores of every row, and whether any score is NaN or +infinity.
    float m = NegInf();
    uint bad = 0u;
    for (uint tile = 0u; tile < nTiles; tile++)
    {
        uint kb = kBegin + tile * BC;
        GroupMemoryBarrierWithGroupSync();
        for (uint e = tid; e < BC * DIM; e += 128u)
        {
            uint row = e / DIM;
            uint c = e % DIM;
            uint key = kb + row;
            Elem v = (Elem)0.0f;
            if (c < P_D && key < kEnd)
            {
                v = AK[kBase + key * AP.KRow + c];
            }

            Tile[row * TS + c] = v;
        }

        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint j = 0; j < NB; j++)
        {
            MatC s = MatC::Zero();
            [unroll] for (uint kt = 0; kt < DT; kt++)
            {
                MatB kf;
                kf.Load(Tile, j * 16 * TS + kt * 16, TS, MatrixLayout::ColumnMajor);
                s = MatC::MulAdd(qf[kt], kf, s);
            }

            s.Store(Sst, sg * 256, 16, MatrixLayout::ColumnMajor);
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint i = 0; i < 8; i++)
            {
                uint col = hh + 2u * i;
                uint key = kb + j * 16u + col;
                bool vis = key < kEnd;
                if ((flags & FLAG_CAUSAL) != 0u)
                {
                    vis = vis && (int)key <= causalLimit;
                }

                if (vis)
                {
                    float sc = Sst[sg * 256 + col * 16 + r] * scale;
                    if ((flags & FLAG_MASK) != 0u)
                    {
                        sc += AMask[mBase + key];
                    }

                    if (Poisoned(sc))
                    {
                        bad = 1u;
                    }

                    if (sc > m)
                    {
                        m = sc;
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    // The two lanes of a row hold the maximum and the flag of half of its columns each.
    {
        float mo = WaveReadLaneAt(m, lane ^ 16u);
        uint bo = WaveReadLaneAt(bad, lane ^ 16u);
        m = (mo > m) ? mo : m;
        bad = bad | bo;
    }

    // Pass 2: p = exp(score - max) in the element type, the normaliser from the rounded p, and the accumulators of p V.
    MatC acc[DT];
    [unroll] for (uint z = 0; z < DT; z++)
    {
        acc[z] = MatC::Zero();
    }

    float l = 0.0;
    for (uint tile2 = 0u; tile2 < nTiles; tile2++)
    {
        uint kb = kBegin + tile2 * BC;
        GroupMemoryBarrierWithGroupSync();
        for (uint e = tid; e < BC * DIM; e += 128u)
        {
            uint row = e / DIM;
            uint c = e % DIM;
            uint key = kb + row;
            Elem v = (Elem)0.0f;
            if (c < P_D && key < kEnd)
            {
                v = AK[kBase + key * AP.KRow + c];
            }

            Tile[row * TS + c] = v;
        }

        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint j = 0; j < NB; j++)
        {
            MatC s = MatC::Zero();
            [unroll] for (uint kt = 0; kt < DT; kt++)
            {
                MatB kf;
                kf.Load(Tile, j * 16 * TS + kt * 16, TS, MatrixLayout::ColumnMajor);
                s = MatC::MulAdd(qf[kt], kf, s);
            }

            s.Store(Sst, sg * 256, 16, MatrixLayout::ColumnMajor);
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint i = 0; i < 8; i++)
            {
                uint col = hh + 2u * i;
                uint key = kb + j * 16u + col;
                bool vis = key < kEnd;
                if ((flags & FLAG_CAUSAL) != 0u)
                {
                    vis = vis && (int)key <= causalLimit;
                }

                Elem p16 = (Elem)0.0f;
                if (vis)
                {
                    float sc = Sst[sg * 256 + col * 16 + r] * scale;
                    if ((flags & FLAG_MASK) != 0u)
                    {
                        sc += AMask[mBase + key];
                    }

                    if (!IsNegInf(sc))
                    {
                        p16 = (Elem)exp(sc - m);
                    }
                }

                l += (float)p16;
                Pst[sg * 16 * BC + j * 256 + col * 16 + r] = p16;
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // The key tile is no longer read (every subgroup passed the barrier above), so the value tile takes its place.
        for (uint f = tid; f < BC * DIM; f += 128u)
        {
            uint row = f / DIM;
            uint c = f % DIM;
            uint key = kb + row;
            Elem v = (Elem)0.0f;
            if (c < P_D && key < kEnd)
            {
                v = AV[vBase + key * AP.VRow + c];
            }

            Tile[row * TS + c] = v;
        }

        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint j2 = 0; j2 < NB; j2++)
        {
            MatA pa;
            pa.Load(Pst, sg * 16 * BC + j2 * 256, 16, MatrixLayout::ColumnMajor);
            [unroll] for (uint dt = 0; dt < DT; dt++)
            {
                MatB vf;
                vf.Load(Tile, j2 * 16 * TS + dt * 16, TS, MatrixLayout::RowMajor);
                acc[dt] = MatC::MulAdd(pa, vf, acc[dt]);
            }
        }
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
    uint rec = ((bh * AP.QLen + qRow) * AP.NSplit + split) * (P_D + 2u);
    if (AP.NSplit != 1u && live && hh == 0u)
    {
        APart[rec] = (bad != 0u) ? QuietNaN() : m;
        APart[rec + 1u] = l;
    }

    // The accumulators leave through workgroup memory, one matrix at a time.
    [unroll] for (uint dt2 = 0; dt2 < DT; dt2++)
    {
        acc[dt2].Store(Sst, sg * 256, 16, MatrixLayout::ColumnMajor);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint i = 0; i < 8; i++)
        {
            uint col = hh + 2u * i;
            uint d = dt2 * 16u + col;
            if (live && d < P_D)
            {
                float v = Sst[sg * 256 + col * 16 + r];
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

        GroupMemoryBarrierWithGroupSync();
    }
}
