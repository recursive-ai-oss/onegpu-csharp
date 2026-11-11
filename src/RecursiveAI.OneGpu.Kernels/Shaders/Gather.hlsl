// ONNX Gather along one axis: the data tensor is seen as [outer, axisLen, inner] and the output as [outer, indexCount, inner].
// out[o, i, n] = data[o, index[i], n]. Flags bit 0: the indices are int64.

#include "IndexCommon.hlsli"

struct GatherParams
{
    uint Count;     // output elements
    uint OffData;
    uint OffOut;
    uint OffIdx;
    uint GridX;
    uint Flags;
    uint Inner;
    uint IdxCount;
    uint AxisLen;
};

[[vk::push_constant]] ConstantBuffer<GatherParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Data;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Status;

bool Src(uint j, out uint s)
{
    uint row = j / P.Inner;
    uint n = j - row * P.Inner;
    uint o = row / P.IdxCount;
    uint i = row - o * P.IdxCount;
    int v;
    uint a;
    s = 0u;
    if (!ReadIndex(Idx, P.OffIdx + i, (P.Flags & 1u) != 0u, v) || !Normalize(v, P.AxisLen, a))
    {
        InterlockedMin(Status[0], i);
        return false;
    }

    s = P.OffData + (o * P.AxisLen + a) * P.Inner + n;
    return true;
}

#include "GatherDriver.hlsli"

[shader("compute")]
[numthreads(256, 1, 1)]
void GatherOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    RunGather(gid, tid);
}
