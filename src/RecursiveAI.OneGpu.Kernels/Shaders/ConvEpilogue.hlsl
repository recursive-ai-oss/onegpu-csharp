// The end of a convolution: y = activation(accumulator + bias[channel] + residual), element-wise over an NCHW tensor.
//
// F32 works in place: the matrix product wrote the accumulators into y and this pass rewrites them. F16 reads float32 accumulators from a
// scratch buffer and writes halves (the bias and the residual are halves too). The order is the one of the graph the operator fuses: bias
// first, then the residual, then the activation, each rounded in float32.

#include "Functions.hlsl"

struct EpilogueParams
{
    uint Count;
    uint OffY;
    uint OffAcc;
    uint OffBias;
    uint OffRes;
    uint Plane;     // elements per channel plane (H * W)
    uint Channels;
    uint GridX;
    uint HasBias;
    uint HasRes;
    uint Act;       // 0 none, else the unary operation number + 1
    float P0;
    float P1;
};

[[vk::push_constant]] ConstantBuffer<EpilogueParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] RWStructuredBuffer<Elem> EpY;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> EpBias;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> EpRes;
#if F16
[[vk::binding(3, 0)]] StructuredBuffer<float> EpAcc;
#endif

float EpilogueValue(uint i)
{
#if F16
    float v = EpAcc[CP.OffAcc + i];
#else
    float v = EpY[CP.OffY + i];
#endif
    if (CP.HasBias != 0u)
    {
        uint c = (i / CP.Plane) % CP.Channels;
#if F16
        v += LoadH(EpBias, CP.OffBias + c);
#else
        v += EpBias[CP.OffBias + c];
#endif
    }

    if (CP.HasRes != 0u)
    {
#if F16
        v += LoadH(EpRes, CP.OffRes + i);
#else
        v += EpRes[CP.OffRes + i];
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
void ConvEpilogueOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    if (t >= HalfWords(CP.OffY, CP.Count))
    {
        return;
    }

    uint word = (CP.OffY >> 1) + t;
    uint res = EpY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(EpilogueValue(h - CP.OffY)) << (k * 16u));
        }
    }

    EpY[word] = res;
#else
    if (t < CP.Count)
    {
        EpY[CP.OffY + t] = EpilogueValue(t);
    }
#endif
}
