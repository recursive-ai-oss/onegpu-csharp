// ONNX ScatterND: indices [.., Last] address slices of Slice elements of the output; update tuple t writes its slice.
//
// Two ways to run, chosen by the host (Flags bit 1):
//  * distinct (the caller says no two tuples address the same slice): one thread per update element.
//  * general: the output slices are split into Buckets groups by slice number; the thread for (bucket, element of the slice) walks all
//    tuples in order and applies those that fall into its bucket. Every output element is written by one thread only, and applies its
//    updates in tuple order, so duplicates combine in index order with no atomics on the data and the result never depends on scheduling.
//    (Two tuples either hit the same slice completely or not at all, because slices are aligned blocks.)
// D0..D5 are the output dimensions each index component addresses (component order), S0..S5 their strides.
// Flags: bit 0 int64 indices, bit 1 distinct, bit 2 float32 (W32 only).

#include "ScatterCommon.hlsli"

struct ScatterNdParams
{
    uint Count;         // threads
    uint OffUpd;
    uint OffIdx;
    uint OffOut;
    uint GridX;
    uint Flags;
    uint Op;
    uint Last;
    uint Slice;
    uint PerBatch;
    uint BatchStride;
    uint Tuples;
    uint Buckets;
    uint D0, D1, D2, D3, D4, D5;
    uint S0, S1, S2, S3, S4, S5;
};

[[vk::push_constant]] ConstantBuffer<ScatterNdParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Upd;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> Status;

#include "ScatterApply.hlsli"

#define SND_STEP(R) \
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

// Element index (without OffOut) of the first element of tuple t's slice.
bool Base(uint t, bool wide, out uint off)
{
    off = (t / P.PerBatch) * P.BatchStride;
    SND_STEP(0)
    SND_STEP(1)
    SND_STEP(2)
    SND_STEP(3)
    SND_STEP(4)
    SND_STEP(5)
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ScatterNdOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint g = (gid.y * P.GridX + gid.x) * WG + tid;
    if (g >= P.Count)
    {
        return;
    }

    bool wide = (P.Flags & 1u) != 0u;
    bool isFloat = (P.Flags & 4u) != 0u;
    uint q = g / P.Slice;
    uint r = g - q * P.Slice;
    if ((P.Flags & 2u) != 0u)
    {
        uint off;
        if (Base(q, wide, off))
        {
            Apply(P.OffOut + off + r, P.OffUpd + q * P.Slice + r, P.Op, isFloat);
        }

        return;
    }

    for (uint t = 0u; t < P.Tuples; t++)
    {
        uint off;
        if (Base(t, wide, off) && ((off / P.Slice) % P.Buckets) == q)
        {
            Apply(P.OffOut + off + r, P.OffUpd + t * P.Slice + r, P.Op, isFloat);
        }
    }
}
