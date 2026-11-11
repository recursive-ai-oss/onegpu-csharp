// Fused scaled dot-product attention:  O = softmax(Scale * Q K^T + Mask) V,  without writing the scores to memory.
//
// The tiling follows Dao et al. ("FlashAttention", 2022): a workgroup owns a block of query rows and walks over the keys one tile at a time;
// the softmax statistics are the one-pass online normaliser of Milakov and Gimelshein (2018): a running maximum m, a running sum l of
// exp(s - m) and an accumulator acc = sum exp(s - m) v, rescaled by exp(m_old - m_new) whenever a key raises the maximum.
//
// A thread owns a query row (without WAVE) or half of one (with WAVE: two neighbouring threads share a row, each holding half of the head
// dimension, which halves the registers per thread). Keys are folded in one at a time in increasing index order, so every sum (the dot
// products over the head dimension, the normaliser, the P V accumulation over the keys) has an order the code fixes. There are no atomics.
// With WAVE the two halves of a dot product are added after one shuffle between the two threads of a pair; a sum of two numbers does not
// depend on which operand is first, so the bits are the same whatever the subgroup size or schedule.
// When the work has too few workgroups to fill a GPU the keys are cut into NSplit contiguous ranges (the cut depends only on the shape),
// each range gives a partial (m, l, acc), and AttentionMerge (AttentionCombine.hlsl) merges them in index order.
//
// Semantics, shared with the host class:
//   * score s[i][j] = Scale * dot(Q[i], K[j]) + Mask[i][j]   (the mask is optional, additive, float32; -infinity hides a key)
//   * a key is hidden when it is at or beyond the batch entry's key length, or, with Causal, when j > i + CausalOff
//   * a row with no visible key gives NaN in every output (Zero flag: 0), as does a row whose score is NaN or +infinity
//   * F16 builds read and write halves (packed in 32-bit words, see Common.hlsl) and accumulate in float32; they need even offsets and strides.
//
// HD selects the register footprint: the head dimension is rounded up to 32, 64, 128 or 256 and padded with zeros. The key and value tiles
// share 16 KB of workgroup memory in every build (BC keys of DIM elements each).

#include "Common.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

#if HD_D32
#define DIM 32u
#define BC 32u
#elif HD_D64
#define DIM 64u
#define BC 32u
#elif HD_D128
#define DIM 128u
#define BC 16u
#else
#define DIM 256u
#define BC 8u
#endif

#if WAVE
#define LANES 2u
#define PD (DIM / 2u)
#else
#define LANES 1u
#define PD DIM
#endif

#define BR (64u / LANES)

// The shader compiler keeps loops as loops, and a private array indexed by a loop variable lives in slow per-thread memory on some GPUs
// (Apple's in particular). Every access to q and acc therefore goes through these macros, which write one statement per element with
// a literal index so that the elements stay in registers. EACH_D covers the PD elements one thread holds.
#define U4(F, b) F((b)) F((b) + 1u) F((b) + 2u) F((b) + 3u)
#define U16(F, b) U4(F, (b)) U4(F, (b) + 4u) U4(F, (b) + 8u) U4(F, (b) + 12u)
#define U64(F, b) U16(F, (b)) U16(F, (b) + 16u) U16(F, (b) + 32u) U16(F, (b) + 48u)
#if HD_D32
#if WAVE
#define EACH_D(F) U16(F, 0u)
#else
#define EACH_D(F) U16(F, 0u) U16(F, 16u)
#endif
#elif HD_D64
#if WAVE
#define EACH_D(F) U16(F, 0u) U16(F, 16u)
#else
#define EACH_D(F) U64(F, 0u)
#endif
#elif HD_D128
#if WAVE
#define EACH_D(F) U64(F, 0u)
#else
#define EACH_D(F) U64(F, 0u) U64(F, 64u)
#endif
#else
#if WAVE
#define EACH_D(F) U64(F, 0u) U64(F, 64u)
#else
#define EACH_D(F) U64(F, 0u) U64(F, 64u) U64(F, 128u) U64(F, 192u)
#endif
#endif

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
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<AttnParams> AP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> AQ;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> AK;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> AV;
[[vk::binding(3, 0)]] StructuredBuffer<float> AMask;
[[vk::binding(4, 0)]] StructuredBuffer<int> ALens;
[[vk::binding(5, 0)]] RWStructuredBuffer<Elem> AOut;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> APart;

groupshared float gsK[BC * DIM];
groupshared float gsV[BC * DIM];

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

float LoadE(StructuredBuffer<Elem> buf, uint i)
{
#if F16
    return LoadH(buf, i);
#else
    return buf[i];
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void AttentionTile(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint lin = gid.y * AP.GridX + gid.x;
    uint nQB = (AP.QLen + BR - 1u) / BR;
    uint split = lin % AP.NSplit;
    uint t = lin / AP.NSplit;
    uint qb = t % nQB;
    uint bh = t / nQB;
    uint b = bh / AP.Heads;
    uint h = bh % AP.Heads;
    uint kvh = h / AP.Group;

    // Surplus threads of the last block recompute the last row and skip the writes, so every barrier below is reached by all threads.
    uint qRow = qb * BR + tid / LANES;
    bool live = qRow < AP.QLen;
    uint qr = live ? qRow : (AP.QLen - 1u);
    // The first head-dimension element this thread holds.
    uint dOff = (tid % LANES) * PD;

    // Keys of this workgroup's range that any of its rows can see. The bound is the same for every thread of the group.
    uint kBegin = split * AP.KeysPerSplit;
    uint kEnd = min(AP.KLen, kBegin + AP.KeysPerSplit);
    if ((AP.Flags & FLAG_LENS) != 0u)
    {
        kEnd = min(kEnd, (uint)max(ALens[AP.LensOff + b], 0));
    }

    if ((AP.Flags & FLAG_CAUSAL) != 0u)
    {
        int lastQ = (int)min(AP.QLen - 1u, qb * BR + BR - 1u);
        int visible = lastQ + AP.CausalOff + 1;
        kEnd = min(kEnd, (uint)max(visible, 0));
    }

    uint qBase = AP.QOff + b * AP.QBatch + h * AP.QHead + qr * AP.QRow;
    uint kBase = AP.KOff + b * AP.KBatch + kvh * AP.KHead;
    uint vBase = AP.VOff + b * AP.VBatch + kvh * AP.VHead;
    uint mBase = AP.MOff + b * AP.MBatch + h * AP.MHead + qr * AP.MRow;

    float q[PD];
    float acc[PD];
#define INIT_STEP(d) q[d] = 0.0; acc[d] = 0.0; if (dOff + (d) < AP.D) { q[d] = LoadE(AQ, qBase + dOff + (d)); }
    EACH_D(INIT_STEP)

    float m = NegInf();
    float l = 0.0;
    bool bad = false;

    // The trip count comes from push constants only, so every barrier in the loop is in uniform control flow, which WGSL requires (the workgroup
    // index is not known to be uniform there). Tiles past this workgroup's last visible key do nothing but meet at the barriers.
    uint tiles = (AP.KeysPerSplit + BC - 1u) / BC;
    for (uint tile = 0u; tile < tiles; tile++)
    {
        uint kb = kBegin + tile * BC;
        bool active = kb < kEnd;
        GroupMemoryBarrierWithGroupSync();
        if (active)
        {
            for (uint e = tid; e < BC * DIM; e += 64u)
            {
                uint r = e / DIM;
                uint c = e % DIM;
                uint key = kb + r;
                float kv = 0.0;
                float vv = 0.0;
                if (c < AP.D && key < kEnd)
                {
                    kv = LoadE(AK, kBase + key * AP.KRow + c);
                    vv = LoadE(AV, vBase + key * AP.VRow + c);
                }

                gsK[e] = kv;
                gsV[e] = vv;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Keys are folded in one at a time: the state per thread is only q, acc, m and l, which keeps the code small and the registers few.
        uint count = active ? min(BC, kEnd - kb) : 0u;
        for (uint j = 0u; j < count; j++)
        {
            uint key = kb + j;
            uint row = j * DIM;
            float dot = 0.0;
#define DOT_STEP(d) dot = mad(q[d], gsK[row + dOff + (d)], dot);
            EACH_D(DOT_STEP)
#if WAVE
            // The shuffle sits outside every divergent branch so that both threads of the pair always take part.
            dot = dot + WaveReadLaneAt(dot, WaveGetLaneIndex() ^ 1u);
#endif
            bool visible = true;
            if ((AP.Flags & FLAG_CAUSAL) != 0u)
            {
                visible = (int)key <= (int)qr + AP.CausalOff;
            }

            if (visible)
            {
                float sc = dot * AP.Scale;
                if ((AP.Flags & FLAG_MASK) != 0u)
                {
                    sc += AMask[mBase + key];
                }

                bad = bad || Poisoned(sc);
                if (!IsNegInf(sc))
                {
                    float p = 1.0;
                    if (sc > m)
                    {
                        float alpha = IsNegInf(m) ? 0.0 : exp(m - sc);
                        l *= alpha;
#define SCALE_STEP(d) acc[d] *= alpha;
                        EACH_D(SCALE_STEP)

                        m = sc;
                    }
                    else
                    {
                        p = exp(sc - m);
                    }

                    l += p;
#define FMA_STEP(d) acc[d] = mad(p, gsV[row + dOff + (d)], acc[d]);
                    EACH_D(FMA_STEP)
                }
            }
        }
    }

    if (!live)
    {
        return;
    }

    if (AP.NSplit == 1u)
    {
        float fill = 0.0;
        bool empty = (l == 0.0);
        if (bad || (empty && (AP.Flags & FLAG_ZERO) == 0u))
        {
            fill = QuietNaN();
        }

        float inv = empty ? 0.0 : 1.0 / l;
        uint oBase = AP.OOff + b * AP.OBatch + h * AP.OHead + qRow * AP.ORow + dOff;
        bool special = bad || empty;
#if F16
#define OUT_STEP(d) if (((d) & 1u) == 0u && dOff + (d) < AP.D) { AOut[(oBase + (d)) >> 1] = F2H(special ? fill : acc[d] * inv) | (F2H(special ? fill : acc[(d) | 1u] * inv) << 16); }
#else
#define OUT_STEP(d) if (dOff + (d) < AP.D) { AOut[oBase + (d)] = special ? fill : acc[d] * inv; }
#endif
        EACH_D(OUT_STEP)
    }
    else
    {
        uint rec = ((bh * AP.QLen + qRow) * AP.NSplit + split) * (AP.D + 2u);
        if (dOff == 0u)
        {
            APart[rec] = bad ? QuietNaN() : m;
            APart[rec + 1u] = l;
        }

#define PART_STEP(d) if (dOff + (d) < AP.D) { APart[rec + 2u + dOff + (d)] = acc[d]; }
        EACH_D(PART_STEP)
    }
}
