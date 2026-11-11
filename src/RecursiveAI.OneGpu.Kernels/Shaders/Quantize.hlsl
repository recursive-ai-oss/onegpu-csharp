// ONNX QuantizeLinear: y = saturate(round_half_even(x / scale) + zero_point) to uint8 or int8 (SIGNED), per tensor or per axis, from float32
// or, with F16, packed half input. The tensor is seen as [outer, AxisLen, inner]; Flags bit 0: per axis; bit 1: a zero point is given.
// The division is correctly rounded (DivExact, integer arithmetic: the GPU's own division is not promised to be), the rounding is to nearest
// even, the addition is a float32 addition, NaN goes to the lowest value, as the CPU kernel does. One thread owns one 32-bit word of
// four output bytes, so threads never write the same word.

#include "Common.hlsl"
#include "DivExact.hlsli"

struct QuantizeParams
{
    uint Count;
    uint OffX;
    uint OffO;      // element (byte) offset of the output
    uint OffS;
    uint OffZ;
    uint GridX;
    uint Flags;
    uint AxisLen;
    uint Inner;
};

[[vk::push_constant]] ConstantBuffer<QuantizeParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> X;
[[vk::binding(1, 0)]] StructuredBuffer<float> Scale;
[[vk::binding(2, 0)]] StructuredBuffer<uint> Zero;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Out;

float LoadX(uint i)
{
#if F16
    return LoadH(X, i);
#else
    return asfloat(X[i]);
#endif
}

int LoadZero(uint i)
{
#if SIGNED
    return ((int)(RawB(Zero, i) << 24)) >> 24;
#else
    return (int)RawB(Zero, i);
#endif
}

uint QuantizeOne(uint j)
{
    uint a = 0u;
    if ((P.Flags & 1u) != 0u)
    {
        a = (j / P.Inner) % P.AxisLen;
    }

    int zp = 0;
    if ((P.Flags & 2u) != 0u)
    {
        zp = LoadZero(P.OffZ + a);
    }

    float v = round(DivExact(LoadX(P.OffX + j), Scale[P.OffS + a])) + (float)zp;
#if SIGNED
    float lo = -128.0;
    float hi = 127.0;
#else
    float lo = 0.0;
    float hi = 255.0;
#endif
    if (IsNaN(v))
    {
        v = lo;
    }

    int q = (int)clamp(v, lo, hi);
    return (uint)q & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void QuantizeOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
    if (t >= BoolWords(P.OffO, P.Count))
    {
        return;
    }

    uint word = (P.OffO >> 2) + t;
    uint res = Out[word];
    [unroll]
    for (uint k = 0u; k < 4u; k++)
    {
        uint b = word * 4u + k;
        if (b >= P.OffO && b < P.OffO + P.Count)
        {
            res = WithByte(res, k, QuantizeOne(b - P.OffO));
        }
    }

    Out[word] = res;
}
