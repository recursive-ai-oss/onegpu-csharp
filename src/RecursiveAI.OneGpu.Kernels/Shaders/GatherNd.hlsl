// ONNX GatherND: indices [.., Last] address slices of Slice elements of the data, after batch_dims leading dimensions shared by data
// and indices. Tuple t of the indices belongs to batch t / PerBatch. D0..D5 are the data dimensions each index component addresses
// (in component order) and S0..S5 their strides in the data. Flags bit 0: the indices are int64.

#include "IndexCommon.hlsli"

struct GatherNdParams
{
    uint Count;
    uint OffData;
    uint OffOut;
    uint OffIdx;
    uint GridX;
    uint Flags;
    uint Last;
    uint Slice;
    uint PerBatch;
    uint BatchStride;
    uint D0, D1, D2, D3, D4, D5;
    uint S0, S1, S2, S3, S4, S5;
};

[[vk::push_constant]] ConstantBuffer<GatherNdParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Data;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Status;

#define GND_STEP(R) \
    if (P.Last > R) \
    { \
        int v; \
        uint a; \
        uint at = t * P.Last + R; \
        if (!ReadIndex(Idx, P.OffIdx + at, wide, v) || !Normalize(v, P.D##R, a)) \
        { \
            InterlockedMin(Status[0], at); \
            return false; \
        } \
        off += a * P.S##R; \
    }

bool Src(uint j, out uint s)
{
    s = 0u;
    bool wide = (P.Flags & 1u) != 0u;
    uint t = j / P.Slice;
    uint r = j - t * P.Slice;
    uint off = P.OffData + (t / P.PerBatch) * P.BatchStride + r;
    GND_STEP(0)
    GND_STEP(1)
    GND_STEP(2)
    GND_STEP(3)
    GND_STEP(4)
    GND_STEP(5)
    s = off;
    return true;
}

#include "GatherDriver.hlsli"

[shader("compute")]
[numthreads(256, 1, 1)]
void GatherNdOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    RunGather(gid, tid);
}
