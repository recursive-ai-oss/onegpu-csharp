// ONNX ScatterElements: out[.., index[i0, .., ik], .., ik] = updates[i0, .., ik] along one axis; indices and updates share a shape.
// Dimensions are given innermost first (D0 the last) with the output stride of each (S0..S7); the axis dimension takes its position
// from the index value, scaled by AxisStride.
//
// Like ScatterND there are two ways to run (Flags bit 1: the caller guarantees distinct targets, one thread per update; otherwise the
// target elements are split into Buckets groups by element number and each thread walks all updates in order and applies those that
// fall into its bucket, so duplicates combine in index order without atomics on the data).
// Flags: bit 0 int64 indices, bit 1 distinct, bit 2 float32 (W32 only).

#include "ScatterCommon.hlsli"

struct ScatterElementsParams
{
    uint Count;         // threads
    uint OffUpd;
    uint OffIdx;
    uint OffOut;
    uint GridX;
    uint Flags;
    uint Op;
    uint Rank;
    uint AxisPos;
    uint AxisLen;
    uint AxisStride;
    uint Updates;       // number of update elements
    uint Buckets;
    uint D0, D1, D2, D3, D4, D5, D6, D7;
    uint S0, S1, S2, S3, S4, S5, S6, S7;
};

[[vk::push_constant]] ConstantBuffer<ScatterElementsParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Upd;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Status;

#include "ScatterApply.hlsli"

#define SE_STEP(R) \
    if (P.Rank > R) \
    { \
        uint c = rem; \
        if (P.Rank != R + 1u) { c = rem % P.D##R; rem = rem / P.D##R; } \
        if (P.AxisPos != R) { off += c * P.S##R; } \
    }

// Element index (without OffOut) that update e writes.
bool Target(uint e, bool wide, out uint off)
{
    uint rem = e;
    off = 0u;
    SE_STEP(0)
    SE_STEP(1)
    SE_STEP(2)
    SE_STEP(3)
    SE_STEP(4)
    SE_STEP(5)
    SE_STEP(6)
    SE_STEP(7)
    int v;
    uint a;
    if (!ReadIndex(Idx, P.OffIdx + e, wide, v) || !Normalize(v, P.AxisLen, a))
    {
        InterlockedMin(Status[0], e);
        return false;
    }

    off += a * P.AxisStride;
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ScatterElementsOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint g = (gid.y * P.GridX + gid.x) * WG + tid;
    if (g >= P.Count)
    {
        return;
    }

    bool wide = (P.Flags & 1u) != 0u;
    bool isFloat = (P.Flags & 4u) != 0u;
    if ((P.Flags & 2u) != 0u)
    {
        uint off;
        if (Target(g, wide, off))
        {
            Apply(P.OffOut + off, P.OffUpd + g, P.Op, isFloat);
        }

        return;
    }

    for (uint e = 0u; e < P.Updates; e++)
    {
        uint off;
        if (Target(e, wide, off) && (off % P.Buckets) == g)
        {
            Apply(P.OffOut + off, P.OffUpd + e, P.Op, isFloat);
        }
    }
}
