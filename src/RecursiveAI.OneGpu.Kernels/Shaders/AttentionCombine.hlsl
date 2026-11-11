// Second pass of Attention when the keys were cut into ranges: merges the partial (m, l, acc) of every query row in range order
// (Milakov and Gimelshein's merge rule), so the result depends only on the shape. Rows with no visible key or a NaN score give NaN
// (0 with the Zero flag); see Attention.hlsl.

#include "Common.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

#define FLAG_ZERO 8u

bool IsNegInf(float x)
{
    return asuint(x) == 0xFF800000u;
}

// Merges the partial results of the key ranges of every query row, in range order. One thread per output element (two for halves).
struct CombineParams
{
    uint Rows;        // batch * heads * qLen
    uint Heads;
    uint QLen;
    uint D;
    uint NSplit;
    uint Flags;
    uint OBatch; uint OHead; uint ORow;
    uint OOff;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<CombineParams> CP;

[[vk::binding(0, 0)]] StructuredBuffer<float> CPart;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> COut;

float MergeOne(uint rec0, uint stride, uint d, uint zeroFlag)
{
    float mx = NegInf();
    bool bad = false;
    for (uint s = 0u; s < CP.NSplit; s++)
    {
        float ms = CPart[rec0 + s * stride];
        bad = bad || IsNaN(ms);
        mx = (ms > mx) ? ms : mx;
    }

    float shift = IsNegInf(mx) ? 0.0 : mx;
    float lsum = 0.0;
    float asum = 0.0;
    for (uint s = 0u; s < CP.NSplit; s++)
    {
        uint rec = rec0 + s * stride;
        float ms = CPart[rec];
        float w = IsNegInf(ms) ? 0.0 : exp(ms - shift);
        lsum = mad(CPart[rec + 1u], w, lsum);
        asum = mad(CPart[rec + 2u + d], w, asum);
    }

    if (bad)
    {
        return QuietNaN();
    }

    if (lsum == 0.0)
    {
        return zeroFlag != 0u ? 0.0 : QuietNaN();
    }

    return asum / lsum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void AttentionMerge(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint idx = (gid.y * CP.GridX + gid.x) * 64u + tid;
#if F16
    uint per = CP.D >> 1;
#else
    uint per = CP.D;
#endif
    if (idx >= CP.Rows * per)
    {
        return;
    }

    uint row = idx / per;
    uint c = idx % per;
    uint bh = row / CP.QLen;
    uint q = row % CP.QLen;
    uint b = bh / CP.Heads;
    uint h = bh % CP.Heads;
    uint stride = CP.D + 2u;
    uint rec0 = row * CP.NSplit * stride;
    uint oBase = CP.OOff + b * CP.OBatch + h * CP.OHead + q * CP.ORow;
    uint zero = CP.Flags & FLAG_ZERO;
#if F16
    float x0 = MergeOne(rec0, stride, 2u * c, zero);
    float x1 = MergeOne(rec0, stride, 2u * c + 1u, zero);
    COut[(oBase + 2u * c) >> 1] = F2H(x0) | (F2H(x1) << 16);
#else
    COut[oBase + c] = MergeOne(rec0, stride, c, zero);
#endif
}
