// Softmax and LogSoftmax along one axis. A row is Len elements Inner apart; row r starts at (r / Inner) * Len * Inner + r % Inner.
//
// The row statistics use the online normaliser of Milakov and Gimelshein ("Online normalizer calculation for softmax", 2018): one pass keeps
// the running maximum m and the running sum d of exp(x - m), rescaling d whenever the maximum grows. Partial results are merged with the
// same rule. Rows that are entirely -infinity, or that hold +infinity or NaN, give NaN, which is what the textbook formula gives.
//
// SoftmaxBlock gives a row to a workgroup: every thread folds a fixed strided slice of the row, then the 256 partial (m, d) pairs are merged.
// Without WAVE the merge is a fixed halving tree in shared memory (deterministic); with WAVE the maxima and sums go through subgroup
// operations. SoftmaxThread gives a row to one thread (short rows).
//
// F16 builds read halves anywhere but write whole words, so they need Inner = 1 and even Len and offsets (the host checks).

#include "Common.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct SoftmaxParams
{
    uint Rows;
    uint Len;
    uint InOff;
    uint OutOff;
    uint Inner;
    uint GridX;
    uint Log;
    uint Pad;
};

[[vk::push_constant]] ConstantBuffer<SoftmaxParams> SP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> SIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> SOut;

groupshared float gsM[256];
groupshared float gsD[256];

uint RowBase(uint r)
{
    return (r / SP.Inner) * SP.Len * SP.Inner + (r % SP.Inner);
}

float LoadX(uint i)
{
#if F16
    return LoadH(SIn, i);
#else
    return SIn[i];
#endif
}

void Fold(float x, inout float m, inout float d)
{
    if (IsNaN(x))
    {
        m = QuietNaN();
        d = QuietNaN();
    }
    else if (x > m)
    {
        d = d * exp(m - x) + 1.0;
        m = x;
    }
    else if (x != NegInf())
    {
        d += exp(x - m);
    }
}

void Merge(float m1, float d1, float m2, float d2, out float m, out float d)
{
    if (IsNaN(m1) || IsNaN(m2))
    {
        m = QuietNaN();
        d = QuietNaN();
        return;
    }

    m = (m2 > m1) ? m2 : m1;
    float a = (m1 == NegInf()) ? 0.0 : d1 * exp(m1 - m);
    float b = (m2 == NegInf()) ? 0.0 : d2 * exp(m2 - m);
    d = a + b;
}

float Result(float x, float m, float d)
{
    if (IsNaN(m) || m == PosInf() || m == NegInf())
    {
        return QuietNaN();
    }

    return (SP.Log != 0u) ? ((x - m) - log(d)) : (exp(x - m) / d);
}

void WriteRow(uint rowBase, uint first, uint step, float m, float d)
{
#if F16
    // Words of two halves; the row is word aligned.
    uint words = SP.Len >> 1;
    for (uint w = first; w < words; w += step)
    {
        uint i = rowBase + 2u * w;
        uint lo = F2H(Result(LoadX(SP.InOff + i), m, d));
        uint hi = F2H(Result(LoadX(SP.InOff + i + 1u), m, d));
        SOut[(SP.OutOff + i) >> 1] = lo | (hi << 16);
    }
#else
    for (uint i = first; i < SP.Len; i += step)
    {
        SOut[SP.OutOff + rowBase + i * SP.Inner] = Result(SIn[SP.InOff + rowBase + i * SP.Inner], m, d);
    }
#endif
}

// One level of the halving tree, written out so that every barrier is visibly in uniform control flow (WGSL requires it).
#define MERGE_STEP(S) \
    if (tid < (S)) \
    { \
        float nm, nd; \
        Merge(gsM[tid], gsD[tid], gsM[tid + (S)], gsD[tid + (S)], nm, nd); \
        gsM[tid] = nm; \
        gsD[tid] = nd; \
    } \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(256, 1, 1)]
void SoftmaxBlock(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // No early return before the barriers (see ReduceBlock): a surplus workgroup recomputes row 0 and skips the writes.
    uint row = gid.y * SP.GridX + gid.x;
    bool live = row < SP.Rows;
    if (!live)
    {
        row = 0u;
    }

    uint rowBase = RowBase(row);
    float m = NegInf();
    float d = 0.0;
    for (uint i = tid; i < SP.Len; i += WG)
    {
        Fold(LoadX(SP.InOff + rowBase + i * SP.Inner), m, d);
    }

#if WAVE
    float mw = WaveActiveMax(m);
    if (WaveActiveAnyTrue(IsNaN(m)))
    {
        mw = QuietNaN();
    }

    float dw = (m == NegInf()) ? 0.0 : d * exp(m - mw);
    dw = WaveActiveSum(dw);
    uint lanes = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u)
    {
        gsM[tid / lanes] = mw;
        gsD[tid / lanes] = dw;
    }

    GroupMemoryBarrierWithGroupSync();
    uint count = WG / lanes;
    float rm = gsM[0];
    float rd = gsD[0];
    for (uint s = 1u; s < count; s++)
    {
        float nm, nd;
        Merge(rm, rd, gsM[s], gsD[s], nm, nd);
        rm = nm;
        rd = nd;
    }
#else
    gsM[tid] = m;
    gsD[tid] = d;
    GroupMemoryBarrierWithGroupSync();
    MERGE_STEP(128u)
    MERGE_STEP(64u)
    MERGE_STEP(32u)
    MERGE_STEP(16u)
    MERGE_STEP(8u)
    MERGE_STEP(4u)
    MERGE_STEP(2u)
    MERGE_STEP(1u)

    float rm = gsM[0];
    float rd = gsD[0];
#endif
    if (live)
    {
        WriteRow(rowBase, tid, WG, rm, rd);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void SoftmaxThread(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint row = (gid.y * SP.GridX + gid.x) * WG + tid;
    if (row >= SP.Rows)
    {
        return;
    }

    uint rowBase = RowBase(row);
    float m = NegInf();
    float d = 0.0;
    for (uint i = 0u; i < SP.Len; i++)
    {
        Fold(LoadX(SP.InOff + rowBase + i * SP.Inner), m, d);
    }

    WriteRow(rowBase, 0u, 1u, m, d);
}
