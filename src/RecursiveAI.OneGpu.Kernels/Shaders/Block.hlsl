// Workgroup-wide reductions over 256 threads.
//
// Two builds. Without WAVE the combine is a binary tree in shared memory whose shape and order are fixed by the code (thread i combines with
// thread i + s for s = 128, 64, ..., 1), so the result does not depend on the device, its subgroup size or scheduling. With WAVE the subgroups
// reduce with the hardware's subgroup operations and one thread combines the per-subgroup results in index order; the order inside a subgroup
// is the driver's, so those results may differ in the last bits between devices.

#include "Common.hlsl"

groupshared float gsF[256];
groupshared float gsW[64];

// Combine kinds: 0 add, 1 max, 2 min, 3 multiply. Max and min propagate NaN.
float Combine(uint kind, float a, float b)
{
    if (kind == 0u)
    {
        return a + b;
    }

    if (kind == 3u)
    {
        return a * b;
    }

    if (IsNaN(a))
    {
        return a;
    }

    if (IsNaN(b))
    {
        return b;
    }

    if (kind == 1u)
    {
        return (b > a) ? b : a;
    }

    return (b < a) ? b : a;
}

float Identity(uint kind)
{
    return kind == 0u ? 0.0 : (kind == 1u ? NegInf() : (kind == 2u ? PosInf() : 1.0));
}

// The reduction of v over the workgroup, returned to every thread. Every thread of the group must call it.
float BlockReduce(uint kind, float v, uint tid)
{
#if WAVE
    float w;
    if (kind == 0u)
    {
        w = WaveActiveSum(v);
    }
    else if (kind == 1u)
    {
        w = WaveActiveMax(v);
    }
    else if (kind == 2u)
    {
        w = WaveActiveMin(v);
    }
    else
    {
        w = WaveActiveProduct(v);
    }

    if (kind == 1u || kind == 2u)
    {
        // Hardware min and max may ignore NaN; the kernels promise to propagate it.
        if (WaveActiveAnyTrue(IsNaN(v)))
        {
            w = QuietNaN();
        }
    }

    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
    {
        gsW[tid / lanes] = w;
    }

    GroupMemoryBarrierWithGroupSync();
    uint count = WG / lanes;
    float r = gsW[0];
    for (uint i = 1u; i < count; i++)
    {
        r = Combine(kind, r, gsW[i]);
    }

    GroupMemoryBarrierWithGroupSync();
    return r;
#else
    gsF[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u)
    {
        if (tid < s)
        {
            gsF[tid] = Combine(kind, gsF[tid], gsF[tid + s]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float r = gsF[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
#endif
}
