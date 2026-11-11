// GlobalAveragePool and GlobalMaxPool: one workgroup per (image, channel) plane. Thread t folds the elements t, t + 256, t + 512, ... in
// increasing order, then the 256 partial results are combined by a halving tree in shared memory. The shape of the tree depends only on
// the plane size, never on the device or on scheduling, so the result is the same on every run.
//
// Max starts from the lowest finite float and is replaced only by strictly greater values, so NaN taps are ignored and a plane of NaN or
// -infinity gives the lowest finite float. Average divides the sum by the plane size.

#include "Common.hlsl"

struct GlobalPoolParams
{
    uint Planes;
    uint Plane;     // elements per plane
    uint OffX;
    uint OffY;
    uint GridX;
    uint Average;
};

[[vk::push_constant]] ConstantBuffer<GlobalPoolParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> GpX;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> GpY;

groupshared float gsPart[256];

// The result for one plane. Every thread of the workgroup calls this with the same plane, so the barriers are uniform.
float PlaneResult(uint plane, uint tid)
{
    uint xBase = CP.OffX + plane * CP.Plane;
#if F16
    float v = CP.Average != 0u ? 0.0 : -65504.0;
#else
    // The lowest finite float, from its bits with a term that is zero at run time (Average is 0 or 1): the shader compiler prints a folded constant as 3.4028235e38, which WGSL rejects as
    // not representable.
    float v = CP.Average != 0u ? 0.0 : asfloat(0xff7fffffu | (CP.Average >> 8));
#endif
    for (uint i = tid; i < CP.Plane; i += WG)
    {
#if F16
        float x = LoadH(GpX, xBase + i);
#else
        float x = GpX[xBase + i];
#endif
        if (CP.Average != 0u)
        {
            v += x;
        }
        else if (x > v)
        {
            v = x;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    gsPart[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            float a = gsPart[tid];
            float b = gsPart[tid + s];
            gsPart[tid] = CP.Average != 0u ? a + b : (b > a ? b : a);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float r = gsPart[0];
    return CP.Average != 0u ? r / (float)CP.Plane : r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void GlobalPoolOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint w = gid.y * CP.GridX + gid.x;
#if F16
    // A workgroup owns one 32-bit word of output, which holds two planes' results (or one, at an odd offset or at the end), so that no two
    // workgroups write the same word.
    uint word = (CP.OffY >> 1) + w;
    uint res = tid == 0u ? GpY[word] : 0u;
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Planes)
        {
            float r = PlaneResult(h - CP.OffY, tid);
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(r) << (k * 16u));
        }
    }

    if (tid == 0u)
    {
        GpY[word] = res;
    }
#else
    if (w < CP.Planes)
    {
        float r = PlaneResult(w, tid);
        if (tid == 0u)
        {
            GpY[CP.OffY + w] = r;
        }
    }
#endif
}
