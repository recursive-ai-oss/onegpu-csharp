// What follows an int8 product: from the int32 sums to the result the operator asks for.
//
//  Mode 0  int32 result: sum + optional int32 bias per column.
//  Mode 1  float32 result: float(sum + bias) * scale (MatMulInteger, Cast, Mul as one step).
//  Mode 3  adds a float32 bias per column to a result already in the output (the Add of the chain). It is a pass of its own because a
//          multiply followed by an add in one expression may be fused into one rounding by the target's compiler, which the unfused chain does not do.
//  Mode 2  8-bit result: clamp(round_half_even(float(sum + bias) * multiplier) + zeroPoint) to uint8 or int8, NaN to the smallest value
//          (QLinearMatMul). One thread owns one 32-bit word, so it writes up to four neighbouring results without touching the bytes around them.
//
// The scale or multiplier is one value or one per column. float(int32) rounds to nearest even. Flags: bit 0 int bias present, bit 1 scale
// per column, bit 3 output is uint8 (else int8).

#include "Common.hlsl"

struct Int8EpilogueParams
{
    uint Count;
    uint OffAcc;
    uint OffOut;
    uint GridX;
    uint N;
    uint Mode;
    uint Flags;
    uint OffBias;
    uint OffScale;
    uint OffFBias;
    int ZeroPoint;
};

[[vk::push_constant]] ConstantBuffer<Int8EpilogueParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Acc;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(2, 0)]] StructuredBuffer<uint> Bias;
[[vk::binding(3, 0)]] StructuredBuffer<uint> Scale;
[[vk::binding(4, 0)]] StructuredBuffer<uint> FBias;

int Sum(uint j)
{
    int acc = (int)Acc[P.OffAcc + j];
    if ((P.Flags & 1u) != 0u)
    {
        acc += (int)Bias[P.OffBias + (j % P.N)];
    }

    return acc;
}

float ScaleOf(uint j)
{
    return asfloat(Scale[P.OffScale + (((P.Flags & 2u) != 0u) ? (j % P.N) : 0u)]);
}

uint Quantized(uint j)
{
    bool u8 = (P.Flags & 8u) != 0u;
    int lo = u8 ? 0 : -128;
    int hi = u8 ? 255 : 127;
    float v = (float)Sum(j) * ScaleOf(j);
    if (IsNaN(v))
    {
        return (uint)lo & 0xFFu;
    }

    float r = round(v);
    r = clamp(r, (float)(lo - P.ZeroPoint), (float)(hi - P.ZeroPoint));
    return ((uint)((int)r + P.ZeroPoint)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void Int8EpilogueOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
    if (P.Mode == 2u)
    {
        // Threads count words of the output, which starts OffOut bytes in.
        uint words = ((P.OffOut + P.Count - 1u) >> 2) - (P.OffOut >> 2) + 1u;
        if (t >= words)
        {
            return;
        }

        uint word = (P.OffOut >> 2) + t;
        uint res = Out[word];
        [unroll]
        for (uint k = 0u; k < 4u; k++)
        {
            uint b = word * 4u + k;
            if (b >= P.OffOut && b < P.OffOut + P.Count)
            {
                res = (res & ~(0xFFu << (k * 8u))) | (Quantized(b - P.OffOut) << (k * 8u));
            }
        }

        Out[word] = res;
        return;
    }

    if (t >= P.Count)
    {
        return;
    }

    if (P.Mode == 0u)
    {
        Out[P.OffOut + t] = (uint)Sum(t);
        return;
    }

    if (P.Mode == 3u)
    {
        Out[P.OffOut + t] = asuint(asfloat(Out[P.OffOut + t]) + asfloat(FBias[P.OffFBias + (t % P.N)]));
        return;
    }

    Out[P.OffOut + t] = asuint((float)Sum(t) * ScaleOf(t));
}
