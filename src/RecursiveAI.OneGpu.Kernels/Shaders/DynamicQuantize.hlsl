// ONNX DynamicQuantizeLinear for float32 input and uint8 output, in three passes:
//  DqMinMax    each workgroup reduces a slice of the input to a minimum and a maximum (Part[2g], Part[2g + 1]); min and max do not depend on
//              the order, so the cut into workgroups cannot change the result.
//  DqFinish    one workgroup combines the partials and writes scale = (max(0, max) - min(0, min)) / 255 and zero point =
//              clamp(round_half_even(-(min / scale)), 0, 255); an all-zero input gives scale 1 and zero point 0.
//  DqQuantize  y = clamp(round_half_even(x / scale) + zeroPoint, 0, 255), NaN to 0.
// The division is a correctly rounded float32 division on the CPU, which the GPU's own division is not promised to be (Vulkan allows a few
// ulp, Metal compiles with fast math), so DivExact does it in integer arithmetic.

#include "Common.hlsl"
#include "DivExact.hlsli"

struct DqParams
{
    uint Count;
    uint OffX;
    uint OffY;
    uint GridX;
    uint Groups;        // workgroups of DqMinMax
    uint OffScale;
    uint OffZero;       // byte offset of the zero point in its buffer
};

[[vk::push_constant]] ConstantBuffer<DqParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<float> X;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Part;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> ScaleOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> ZeroOut;

groupshared float Smin[256];
groupshared float Smax[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void DqMinMax(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint g = gid.x;
    float mn = PosInf();
    float mx = NegInf();
    for (uint i = g * 256u + tid; i < P.Count; i += P.Groups * 256u)
    {
        float v = X[P.OffX + i];
        if (v < mn) { mn = v; }
        if (v > mx) { mx = v; }
    }

    Smin[tid] = mn;
    Smax[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            if (Smin[tid + s] < Smin[tid]) { Smin[tid] = Smin[tid + s]; }
            if (Smax[tid + s] > Smax[tid]) { Smax[tid] = Smax[tid + s]; }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        Part[2u * g] = asuint(Smin[0]);
        Part[2u * g + 1u] = asuint(Smax[0]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DqFinish(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    float mn = PosInf();
    float mx = NegInf();
    for (uint i = tid; i < P.Groups; i += 256u)
    {
        float a = asfloat(Part[2u * i]);
        float b = asfloat(Part[2u * i + 1u]);
        if (a < mn) { mn = a; }
        if (b > mx) { mx = b; }
    }

    Smin[tid] = mn;
    Smax[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            if (Smin[tid + s] < Smin[tid]) { Smin[tid] = Smin[tid + s]; }
            if (Smax[tid + s] > Smax[tid]) { Smax[tid] = Smax[tid + s]; }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        float lo = min(0.0, Smin[0]);
        float hi = max(0.0, Smax[0]);
        float scale = DivExact(hi - lo, 255.0);
        uint zp = 0u;
        if (scale != 0.0 && !IsNaN(scale) && (asuint(scale) & 0x7FFFFFFFu) < INF_BITS)
        {
            float z = round(0.0 - DivExact(lo, scale));
            zp = z < 0.0 ? 0u : (z > 255.0 ? 255u : (uint)z);
        }

        if (scale == 0.0)
        {
            scale = 1.0;
            zp = 0u;
        }

        ScaleOut[P.OffScale] = asuint(scale);
        InterlockedAnd(ZeroOut[P.OffZero >> 2], ~(0xFFu << ((P.OffZero & 3u) << 3)));
        InterlockedOr(ZeroOut[P.OffZero >> 2], zp << ((P.OffZero & 3u) << 3));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DqQuantize(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
    uint words = ((P.OffY + P.Count - 1u) >> 2) - (P.OffY >> 2) + 1u;
    // Keeps the declared read-write access on the two parameter buffers this pass only reads (see DqFinish).
    if (P.Count == 0xFFFFFFFFu)
    {
        ScaleOut[0] = 0u;
        ZeroOut[0] = 0u;
    }

    if (t >= words)
    {
        return;
    }

    float scale = asfloat(ScaleOut[P.OffScale]);
    float zp = (float)((ZeroOut[P.OffZero >> 2] >> ((P.OffZero & 3u) << 3)) & 0xFFu);
    uint word = (P.OffY >> 2) + t;
    uint res = Y[word];
    [unroll]
    for (uint k = 0u; k < 4u; k++)
    {
        uint b = word * 4u + k;
        if (b >= P.OffY && b < P.OffY + P.Count)
        {
            float v = round(DivExact(X[P.OffX + (b - P.OffY)], scale)) + zp;
            uint q = IsNaN(v) ? 0u : (uint)clamp(v, 0.0, 255.0);
            res = (res & ~(0xFFu << (k * 8u))) | (q << (k * 8u));
        }
    }

    Y[word] = res;
}
