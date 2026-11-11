// ONNX GatherElements: out[i0, .., ik] = data[i0, .., index[i0, .., ik], .., ik] along one axis. The output has the indices' shape. The
// host gives the indices' dimensions innermost first (D0 is the last) with the stride in the data of each (S0..S7); the axis dimension
// takes its position from the index value instead, scaled by AxisStride. Flags bit 0: the indices are int64.

#include "IndexCommon.hlsli"

struct GatherElementsParams
{
    uint Count;
    uint OffData;
    uint OffOut;
    uint OffIdx;
    uint GridX;
    uint Flags;
    uint Rank;
    uint AxisPos;       // counted innermost first
    uint AxisLen;
    uint AxisStride;
    uint D0, D1, D2, D3, D4, D5, D6, D7;
    uint S0, S1, S2, S3, S4, S5, S6, S7;
};

[[vk::push_constant]] ConstantBuffer<GatherElementsParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Data;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Status;

#define GE_STEP(R) \
    if (P.Rank > R) \
    { \
        uint c = rem; \
        if (P.Rank != R + 1u) { c = rem % P.D##R; rem = rem / P.D##R; } \
        if (P.AxisPos != R) { off += c * P.S##R; } \
    }

bool Src(uint j, out uint s)
{
    uint rem = j;
    uint off = P.OffData;
    GE_STEP(0)
    GE_STEP(1)
    GE_STEP(2)
    GE_STEP(3)
    GE_STEP(4)
    GE_STEP(5)
    GE_STEP(6)
    GE_STEP(7)
    int v;
    uint a;
    s = 0u;
    if (!ReadIndex(Idx, P.OffIdx + j, (P.Flags & 1u) != 0u, v) || !Normalize(v, P.AxisLen, a))
    {
        InterlockedMin(Status[0], j);
        return false;
    }

    s = off + a * P.AxisStride;
    return true;
}

#include "GatherDriver.hlsli"

[shader("compute")]
[numthreads(256, 1, 1)]
void GatherElementsOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    RunGather(gid, tid);
}
