// Fused scaled dot-product attention:  O = softmax(Scale * Q K^T + Mask) V,  without writing the scores to memory.
//
// The tiling follows Dao et al. ("FlashAttention", 2022): a workgroup owns a block of query rows and walks over the keys one tile at a time;
// the softmax statistics are the one-pass online normaliser of Milakov and Gimelshein (2018): a running maximum m, a running sum l of
// exp(s - m) and an accumulator acc = sum exp(s - m) v, rescaled by exp(m_old - m_new) whenever a tile raises the maximum.
//
// One thread owns one query row, so every sum (the dot products over the head dimension, the normaliser, the P V accumulation over the keys)
// is a fixed sequential loop in increasing index order. There are no atomics and no subgroup operations, so the bits do not depend on the
// device's subgroup size, on occupancy or on scheduling.
// When the work has too few workgroups to fill a GPU the keys are cut into NSplit contiguous ranges (the cut depends only on the shape),
// each range gives a partial (m, l, acc), and AttentionCombine merges them in index order.
//
// Semantics, shared with the host class:
//   * score s[i][j] = Scale * dot(Q[i], K[j]) + Mask[i][j]   (the mask is optional, additive, float32; -infinity hides a key)
//   * a key is hidden when it is at or beyond the batch entry's key length, or, with Causal, when j > i + CausalOff
//   * a row with no visible key gives NaN in every output (Zero flag: 0), as does a row whose score is NaN or +infinity
//   * F16 builds read and write halves (packed in 32-bit words, see Common.hlsl) and accumulate in float32; they need even offsets and strides.
//
// DIM selects the register footprint: the head dimension is rounded up to 32, 64, 128 or 256 and padded with zeros. The key and value tiles
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

#define BR 64u

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
    uint qRow = qb * BR + tid;
    bool live = qRow < AP.QLen;
    uint qr = live ? qRow : (AP.QLen - 1u);

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

    float q[DIM];
    float acc[DIM];
    [unroll]
    for (uint d = 0u; d < DIM; d++)
    {
        q[d] = (d < AP.D) ? LoadE(AQ, qBase + d) : 0.0;
        acc[d] = 0.0;
    }

    float m = NegInf();
    float l = 0.0;
    bool bad = false;

    for (uint kb = kBegin; kb < kEnd; kb += BC)
    {
        GroupMemoryBarrierWithGroupSync();
        for (uint e = tid; e < BC * DIM; e += BR)
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

        GroupMemoryBarrierWithGroupSync();

        float s[BC];
        float tileMax = NegInf();
        [unroll]
        for (uint j = 0u; j < BC; j++)
        {
            float dot = 0.0;
            [unroll]
            for (uint d = 0u; d < DIM; d++)
            {
                dot = mad(q[d], gsK[j * DIM + d], dot);
            }

            uint key = kb + j;
            float sc = NegInf();
            bool visible = key < kEnd;
            if ((AP.Flags & FLAG_CAUSAL) != 0u)
            {
                visible = visible && ((int)key <= (int)qr + AP.CausalOff);
            }

            if (visible)
            {
                sc = dot * AP.Scale;
                if ((AP.Flags & FLAG_MASK) != 0u)
                {
                    sc += AMask[mBase + key];
                }

                bad = bad || Poisoned(sc);
            }

            s[j] = sc;
            tileMax = (sc > tileMax) ? sc : tileMax;
        }

        float mNew = (tileMax > m) ? tileMax : m;
        float shift = IsNegInf(mNew) ? 0.0 : mNew;
        float alpha = IsNegInf(m) ? 0.0 : exp(m - shift);
        float p[BC];
        float ls = 0.0;
        [unroll]
        for (uint j = 0u; j < BC; j++)
        {
            float pj = IsNegInf(s[j]) ? 0.0 : exp(s[j] - shift);
            p[j] = pj;
            ls += pj;
        }

        [unroll]
        for (uint d = 0u; d < DIM; d++)
        {
            float a = acc[d] * alpha;
            [unroll]
            for (uint j = 0u; j < BC; j++)
            {
                a = mad(p[j], gsV[j * DIM + d], a);
            }

            acc[d] = a;
        }

        l = l * alpha + ls;
        m = mNew;
    }

    if (!live)
    {
        return;
    }

    if (AP.NSplit == 1u)
    {
        float inv = 0.0;
        float fill = 0.0;
        bool empty = (l == 0.0);
        if (bad || (empty && (AP.Flags & FLAG_ZERO) == 0u))
        {
            fill = QuietNaN();
        }

        inv = empty ? 0.0 : 1.0 / l;
        uint oBase = AP.OOff + b * AP.OBatch + h * AP.OHead + qRow * AP.ORow;
#if F16
        for (uint d = 0u; d < AP.D; d += 2u)
        {
            float x0 = (bad || empty) ? fill : acc[d] * inv;
            float x1 = (bad || empty) ? fill : acc[d + 1u] * inv;
            AOut[(oBase + d) >> 1] = F2H(x0) | (F2H(x1) << 16);
        }
#else
        for (uint d = 0u; d < AP.D; d++)
        {
            AOut[oBase + d] = (bad || empty) ? fill : acc[d] * inv;
        }
#endif
    }
    else
    {
        uint rec = ((bh * AP.QLen + qRow) * AP.NSplit + split) * (AP.D + 2u);
        APart[rec] = bad ? QuietNaN() : m;
        APart[rec + 1u] = l;
        for (uint d = 0u; d < AP.D; d++)
        {
            APart[rec + 2u + d] = acc[d];
        }
    }
}
