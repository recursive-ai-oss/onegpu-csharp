// Convolution of groups that each see one input channel (depthwise convolution, with an optional channel multiplier): output channel co reads
// input channel co / Multiplier. A matrix product has nothing to work with here (K is only KH * KW), so every thread computes one output
// element directly, summing the window taps in increasing (kh, kw) order from zero in float32. The bias, residual and activation follow in the
// same thread, in the order the fused operator defines: sum, bias, residual, activation.
//
// F16 reads halves and writes halves (one thread per 32-bit word of output); the sum is still float32.

#include "Functions.hlsl"

struct DepthwiseParams
{
    uint Count;
    uint OffX;
    uint OffW;
    uint OffBias;
    uint OffRes;
    uint OffY;
    uint GridX;
    uint Channels;      // input channels = groups
    uint Multiplier;    // output channels per input channel
    uint H, W;
    uint OH, OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;
    uint HasBias;
    uint HasRes;
    uint Act;
    float P0;
    float P1;
};

[[vk::push_constant]] ConstantBuffer<DepthwiseParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> DwX;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> DwW;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> DwBias;
[[vk::binding(3, 0)]] StructuredBuffer<Elem> DwRes;
[[vk::binding(4, 0)]] RWStructuredBuffer<Elem> DwY;

float LoadX(uint i)
{
#if F16
    return LoadH(DwX, i);
#else
    return DwX[i];
#endif
}

float LoadW(uint i)
{
#if F16
    return LoadH(DwW, i);
#else
    return DwW[i];
#endif
}

float DepthwiseValue(uint e)
{
    uint ow = e % CP.OW;
    uint r = e / CP.OW;
    uint oh = r % CP.OH;
    uint plane = r / CP.OH;                       // n * (Channels * Multiplier) + co
    uint couts = CP.Channels * CP.Multiplier;
    uint co = plane % couts;
    uint n = plane / couts;
    uint c = co / CP.Multiplier;
    uint xBase = CP.OffX + (n * CP.Channels + c) * CP.H * CP.W;
    uint wBase = CP.OffW + co * CP.KH * CP.KW;
    float acc = 0.0;
    for (uint kh = 0u; kh < CP.KH; kh++)
    {
        int ih = (int)(oh * CP.SH + kh * CP.DH) - (int)CP.PH;
        if ((uint)ih >= CP.H)
        {
            continue;
        }

        for (uint kw = 0u; kw < CP.KW; kw++)
        {
            int iw = (int)(ow * CP.SW + kw * CP.DW) - (int)CP.PW;
            if ((uint)iw < CP.W)
            {
                acc += LoadW(wBase + kh * CP.KW + kw) * LoadX(xBase + (uint)ih * CP.W + (uint)iw);
            }
        }
    }

    float v = acc;
    if (CP.HasBias != 0u)
    {
#if F16
        v += LoadH(DwBias, CP.OffBias + co);
#else
        v += DwBias[CP.OffBias + co];
#endif
    }

    if (CP.HasRes != 0u)
    {
#if F16
        v += LoadH(DwRes, CP.OffRes + e);
#else
        v += DwRes[CP.OffRes + e];
#endif
    }

    if (CP.Act != 0u)
    {
        v = Unary(CP.Act - 1u, v, CP.P0, CP.P1);
    }

    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DepthwiseConvOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    if (t >= HalfWords(CP.OffY, CP.Count))
    {
        return;
    }

    uint word = (CP.OffY >> 1) + t;
    uint res = DwY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(DepthwiseValue(h - CP.OffY)) << (k * 16u));
        }
    }

    DwY[word] = res;
#else
    if (t < CP.Count)
    {
        DwY[CP.OffY + t] = DepthwiseValue(t);
    }
#endif
}
