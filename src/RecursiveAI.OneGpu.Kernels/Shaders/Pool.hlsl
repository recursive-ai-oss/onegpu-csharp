// Max and average pooling over the two trailing axes of an NCHW tensor, one thread per output element (one per 32-bit word for halves).
// The window taps sit at o * stride - pad + k * dilation, visited in (kh, kw) order, as the ONNX operators define them.
//
// Max follows IEEE maxNum: NaN taps are skipped unless every tap inside the input is NaN, in which case the first of them is the result;
// a window with no tap inside the input gives the lowest finite value of the type (-3.4028235e38, or -65504 for halves). NaN is tested on the bit pattern, because Metal compiles with fast math.
// Average sums the taps inside the input from zero and divides by their count, or, with IncludePad, by the count of taps inside the padded
// input (a tap beyond the padding, which ceil mode can produce, is never counted). An empty window gives NaN.

#include "Common.hlsl"

struct PoolParams
{
    uint Count;
    uint OffX;
    uint OffY;
    uint GridX;
    uint H, W;
    uint OH, OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;        // padding before
    uint PEH, PEW;      // padding after
    uint Average;       // 0 max, 1 average
    uint IncludePad;
};

[[vk::push_constant]] ConstantBuffer<PoolParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> PoolX;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> PoolY;

float LoadPool(uint i)
{
#if F16
    return LoadH(PoolX, i);
#else
    return PoolX[i];
#endif
}

float PoolValue(uint e)
{
    uint ox = e % CP.OW;
    uint r = e / CP.OW;
    uint oy = r % CP.OH;
    uint plane = r / CP.OH;
    uint xBase = CP.OffX + plane * CP.H * CP.W;
#if F16
    float best = -65504.0;
#else
    // The lowest finite float, from its bits with a term that is zero at run time (Average is 0 or 1): the shader compiler prints a folded constant as 3.4028235e38, which WGSL rejects as
    // not representable.
    float best = asfloat(0xff7fffffu | (CP.Average >> 8));
#endif
    float nanValue = 0.0;
    float sum = 0.0;
    bool any = false;
    bool sawNaN = false;
    uint count = 0u;
    for (uint kh = 0u; kh < CP.KH; kh++)
    {
        int ih = (int)(oy * CP.SH + kh * CP.DH) - (int)CP.PH;
        bool hIn = (uint)ih < CP.H;
        bool hPad = ih >= -(int)CP.PH && ih < (int)(CP.H + CP.PEH);
        if (!hIn && !(CP.Average != 0u && CP.IncludePad != 0u && hPad))
        {
            continue;
        }

        for (uint kw = 0u; kw < CP.KW; kw++)
        {
            int iw = (int)(ox * CP.SW + kw * CP.DW) - (int)CP.PW;
            bool wIn = (uint)iw < CP.W;
            if (hIn && wIn)
            {
                float v = LoadPool(xBase + (uint)ih * CP.W + (uint)iw);
                if (CP.Average != 0u)
                {
                    sum += v;
                    count++;
                }
                else if (IsNaN(v))
                {
                    if (!sawNaN)
                    {
                        sawNaN = true;
                        nanValue = v;
                    }
                }
                else if (!any || v > best)
                {
                    best = v;
                    any = true;
                }
            }
            else if (CP.Average != 0u && CP.IncludePad != 0u && iw >= -(int)CP.PW && iw < (int)(CP.W + CP.PEW))
            {
                count++;
            }
        }
    }

    if (CP.Average != 0u)
    {
        return count == 0u ? QuietNaN() : sum / (float)count;
    }

    return (!any && sawNaN) ? nanValue : best;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void PoolOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    if (t >= HalfWords(CP.OffY, CP.Count))
    {
        return;
    }

    uint word = (CP.OffY >> 1) + t;
    uint res = PoolY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(PoolValue(h - CP.OffY)) << (k * 16u));
        }
    }

    PoolY[word] = res;
#else
    if (t < CP.Count)
    {
        PoolY[CP.OffY + t] = PoolValue(t);
    }
#endif
}
