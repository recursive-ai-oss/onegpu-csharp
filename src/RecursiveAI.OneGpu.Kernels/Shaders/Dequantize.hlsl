// ONNX DequantizeLinear: y = (float)(x - zero_point) * scale, with one scale and zero point for the tensor or one per index of an axis.
// The tensor is seen as [outer, AxisLen, inner]; Flags bit 0: per axis; bit 1: a zero point is given. x is uint8, int8 (packed four to a
// 32-bit word, element 4k in the low byte of word k) or int32. The subtraction wraps in 32 bits and the conversion to float rounds to
// nearest even, then one multiply, as the CPU does. The output is float32 or, with F16, packed halves rounded to nearest even.
// KIND_U8, KIND_I8 and KIND_I32 select the input type.

#include "Common.hlsl"

struct DequantizeParams
{
    uint Count;
    uint OffX;
    uint OffO;
    uint OffS;
    uint OffZ;
    uint GridX;
    uint Flags;
    uint AxisLen;
    uint Inner;
};

[[vk::push_constant]] ConstantBuffer<DequantizeParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> X;
[[vk::binding(1, 0)]] StructuredBuffer<float> Scale;
[[vk::binding(2, 0)]] StructuredBuffer<uint> Zero;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Out;

int Load(StructuredBuffer<uint> buf, uint i)
{
#if KIND_U8
    return (int)RawB(buf, i);
#elif KIND_I8
    return ((int)(RawB(buf, i) << 24)) >> 24;
#else
    return asint(buf[i]);
#endif
}

float Dequantize(uint j)
{
    uint a = 0u;
    if ((P.Flags & 1u) != 0u)
    {
        a = (j / P.Inner) % P.AxisLen;
    }

    int zp = 0;
    if ((P.Flags & 2u) != 0u)
    {
        zp = Load(Zero, P.OffZ + a);
    }

    int q = Load(X, P.OffX + j);
    return (float)(q - zp) * Scale[P.OffS + a];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DequantizeOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
#if F16
    // One thread owns one 32-bit word of two halves, so two threads never write the same word; surplus threads stay off the buffer.
    if (t >= HalfWords(P.OffO, P.Count))
    {
        return;
    }

    uint word = (P.OffO >> 1) + t;
    uint res = Out[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= P.OffO && h < P.OffO + P.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(Dequantize(h - P.OffO)) << (k * 16u));
        }
    }

    Out[word] = res;
#else
    if (t < P.Count)
    {
        Out[P.OffO + t] = asuint(Dequantize(t));
    }
#endif
}
