// Index of the largest or smallest element along one axis (ArgMax and ArgMin). Ties give the lowest index; NaN counts as the extreme
// value, the first NaN winning. The result is a 64-bit integer per output element: low word, then high word (always 0).

#include "Block.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct ArgParams
{
    uint NumOut;
    uint R;
    uint InOff;
    uint OutOff;
    uint Op;     // 0 max, 1 min
    uint GridX;
    uint KRank;
    uint RStride;
    uint K0, K1, K2, K3;
    uint KS0, KS1, KS2, KS3;
};

[[vk::push_constant]] ConstantBuffer<ArgParams> AP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> AIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> AOut;

groupshared float gsV[256];
groupshared uint gsI[256];

#define ASTEP(R, D, S) \
    if (AP.KRank == R + 1u) { off += rem * AP.S; } \
    else if (AP.KRank > R) { uint c = rem % AP.D; rem /= AP.D; off += c * AP.S; }

uint ArgKeptOffset(uint o)
{
    uint rem = o;
    uint off = AP.InOff;
    ASTEP(0u, K0, KS0)
    ASTEP(1u, K1, KS1)
    ASTEP(2u, K2, KS2)
    ASTEP(3u, K3, KS3)
    return off;
}

bool Beats(float av, uint ai, float bv, uint bi)
{
    bool an = IsNaN(av);
    bool bn = IsNaN(bv);
    if (an || bn)
    {
        return (an && bn) ? (ai < bi) : an;
    }

    if (av == bv)
    {
        return ai < bi;
    }

    return (AP.Op == 0u) ? (av > bv) : (av < bv);
}

float ArgLoad(uint i)
{
#if F16
    return LoadH(AIn, i);
#else
    return AIn[i];
#endif
}

// One level of the halving tree, written out so that every barrier is visibly in uniform control flow (WGSL requires it).
#define ARG_STEP(S) \
    if (tid < (S) && Beats(gsV[tid + (S)], gsI[tid + (S)], gsV[tid], gsI[tid])) \
    { \
        gsV[tid] = gsV[tid + (S)]; \
        gsI[tid] = gsI[tid + (S)]; \
    } \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(256, 1, 1)]
void ArgBlock(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // No early return before the barriers (see ReduceBlock).
    uint o = gid.y * AP.GridX + gid.x;
    bool live = o < AP.NumOut;
    if (!live)
    {
        o = 0u;
    }

    uint baseOff = ArgKeptOffset(o);
    float bv = (AP.Op == 0u) ? NegInf() : PosInf();
    uint bi = 0xFFFFFFFFu;
    for (uint r = tid; r < AP.R; r += WG)
    {
        float v = ArgLoad(baseOff + r * AP.RStride);
        if (Beats(v, r, bv, bi))
        {
            bv = v;
            bi = r;
        }
    }

    gsV[tid] = bv;
    gsI[tid] = bi;
    GroupMemoryBarrierWithGroupSync();
    ARG_STEP(128u)
    ARG_STEP(64u)
    ARG_STEP(32u)
    ARG_STEP(16u)
    ARG_STEP(8u)
    ARG_STEP(4u)
    ARG_STEP(2u)
    ARG_STEP(1u)

    if (tid == 0u && live)
    {
        uint idx = (gsI[0] == 0xFFFFFFFFu) ? 0u : gsI[0];
        AOut[2u * (AP.OutOff + o)] = idx;
        AOut[2u * (AP.OutOff + o) + 1u] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ArgThread(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint o = (gid.y * AP.GridX + gid.x) * WG + tid;
    if (o >= AP.NumOut)
    {
        return;
    }

    uint baseOff = ArgKeptOffset(o);
    float bv = (AP.Op == 0u) ? NegInf() : PosInf();
    uint bi = 0xFFFFFFFFu;
    for (uint r = 0u; r < AP.R; r++)
    {
        float v = ArgLoad(baseOff + r * AP.RStride);
        if (Beats(v, r, bv, bi))
        {
            bv = v;
            bi = r;
        }
    }

    AOut[2u * (AP.OutOff + o)] = (bi == 0xFFFFFFFFu) ? 0u : bi;
    AOut[2u * (AP.OutOff + o) + 1u] = 0u;
}
