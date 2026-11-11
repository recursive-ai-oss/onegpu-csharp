// Separable resampling of the two trailing axes of an NCHW tensor (nearest and linear). The host computes, per output row and column, the
// source positions and weights (ONNX Resize coordinate transformation and rounding, in float32) and uploads them as a table of
// (index, weight bits) pairs: Taps entries per output position, the rows' table first and the columns' after it. Each output element then
// sums its column taps per row tap, in increasing tap order, and the row sums likewise. A tap whose weight is exactly zero is skipped, so an
// infinity or NaN beside an exactly hit sample cannot turn it into NaN (0 * inf). With one tap of weight one per axis (nearest) the output is a bit
// copy of the selected element.

#include "Common.hlsl"

struct ResizeParams
{
    uint Count;
    uint OffX;
    uint OffY;
    uint GridX;
    uint H, W;
    uint OH, OW;
    uint TapsY, TapsX;
    uint TabY, TabX;    // word offsets of the two tables
    uint OffTab;
};

[[vk::push_constant]] ConstantBuffer<ResizeParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> RsX;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> RsY;
[[vk::binding(2, 0)]] StructuredBuffer<uint> RsTab;

float LoadRs(uint i)
{
#if F16
    return LoadH(RsX, i);
#else
    return RsX[i];
#endif
}

float ResizeValue(uint e)
{
    uint ox = e % CP.OW;
    uint r = e / CP.OW;
    uint oy = r % CP.OH;
    uint plane = r / CP.OH;
    uint xBase = CP.OffX + plane * CP.H * CP.W;
    float outer = 0.0;
    bool firstY = true;
    for (uint ty = 0u; ty < CP.TapsY; ty++)
    {
        uint ey = CP.OffTab + CP.TabY + (oy * CP.TapsY + ty) * 2u;
        float wy = asfloat(RsTab[ey + 1u]);
        if (wy == 0.0 && !(firstY && ty == CP.TapsY - 1u))
        {
            continue;
        }

        uint iy = RsTab[ey];
        float inner = 0.0;
        bool firstX = true;
        for (uint tx = 0u; tx < CP.TapsX; tx++)
        {
            uint ex = CP.OffTab + CP.TabX + (ox * CP.TapsX + tx) * 2u;
            float wx = asfloat(RsTab[ex + 1u]);
            if (wx == 0.0 && !(firstX && tx == CP.TapsX - 1u))
            {
                continue;
            }

            float v = wx * LoadRs(xBase + iy * CP.W + RsTab[ex]);
            inner = firstX ? v : inner + v;
            firstX = false;
        }

        float term = wy * inner;
        outer = firstY ? term : outer + term;
        firstY = false;
    }

    return outer;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ResizeOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    if (t >= HalfWords(CP.OffY, CP.Count))
    {
        return;
    }

    uint word = (CP.OffY >> 1) + t;
    uint res = RsY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(ResizeValue(h - CP.OffY)) << (k * 16u));
        }
    }

    RsY[word] = res;
#else
    if (t < CP.Count)
    {
        RsY[CP.OffY + t] = ResizeValue(t);
    }
#endif
}
