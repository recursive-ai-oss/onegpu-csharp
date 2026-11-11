// Reductions over any set of axes. The input is described as kept dimensions (one output element per index) and reduced dimensions, each
// with its stride in the input, innermost first, so any axis set of a dense tensor maps to it. Operation numbers match the host enum.
//
// ReduceBlock gives each output element a workgroup (long reductions); ReduceThread gives it one thread that walks the reduced elements in
// order (short ones, and reductions over outer axes where neighbouring threads read neighbouring addresses).
//
// A long reduction with few outputs would leave most of the GPU idle, so it runs in two passes: ReduceSplit gives each of Split fixed,
// contiguous chunks of the reduced elements to a workgroup, which leaves its partial result in the scratch buffer (bound as the output,
// float bits); ReduceCombine then merges the Split partials of each output with the same workgroup tree. Split and Chunk depend only on
// the shape, so the partition and the order of combination are the same on every device and run.

#include "Block.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct ReduceParams
{
    uint NumOut;
    uint R;
    uint InOff;
    uint OutOff;
    uint Op;
    uint GridX;
    uint KRank;
    uint RRank;
    uint K0, K1, K2, K3;
    uint KS0, KS1, KS2, KS3;
    uint R0, R1, R2, R3;
    uint RS0, RS1, RS2, RS3;
    uint Split;
    uint Chunk;
    uint TotalR;
    uint Pad;
};

[[vk::push_constant]] ConstantBuffer<ReduceParams> RP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> RIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> ROut;

uint KindOf(uint op)
{
    // Sum, Mean, L2, SumSquare, L1, LogSum add; Max and Min compare; Prod multiplies.
    return op == 2u ? 1u : (op == 3u ? 2u : (op == 4u ? 3u : 0u));
}

float Transform(uint op, float x)
{
    if (op == 5u || op == 6u)
    {
        return x * x;
    }

    return op == 7u ? abs(x) : x;
}

float Finalize(uint op, float acc, uint n)
{
    if (op == 1u)
    {
        return acc / (float)n;
    }

    if (op == 5u)
    {
        return sqrt(acc);
    }

    return op == 8u ? log(acc) : acc;
}

#define KSTEP(R, D, S) \
    if (RP.KRank == R + 1u) { off += rem * RP.S; } \
    else if (RP.KRank > R) { uint c = rem % RP.D; rem /= RP.D; off += c * RP.S; }
#define RSTEP(R, D, S) \
    if (RP.RRank == R + 1u) { off += rem * RP.S; } \
    else if (RP.RRank > R) { uint c = rem % RP.D; rem /= RP.D; off += c * RP.S; }

uint KeptOffset(uint o)
{
    uint rem = o;
    uint off = RP.InOff;
    KSTEP(0u, K0, KS0)
    KSTEP(1u, K1, KS1)
    KSTEP(2u, K2, KS2)
    KSTEP(3u, K3, KS3)
    return off;
}

// RRANK = R1: the reduced elements form one run, so the offset needs no division (integer division is slow on GPUs without hardware for it
// and the loops below call this per element).
uint ReducedOffset(uint r)
{
#if RRANK_R1
    return r * RP.RS0;
#else
    uint rem = r;
    uint off = 0u;
    RSTEP(0u, R0, RS0)
    RSTEP(1u, R1, RS1)
    RSTEP(2u, R2, RS2)
    RSTEP(3u, R3, RS3)
    return off;
#endif
}

float LoadIn(uint i)
{
#if F16
    return LoadH(RIn, i);
#else
    return RIn[i];
#endif
}

void StoreOut(uint o, float v)
{
#if F16
    // Neighbouring outputs share a word and are written by different threads or groups: two atomic integer operations on disjoint bits.
    uint h = RP.OutOff + o;
    uint sh = (h & 1u) << 4;
    InterlockedAnd(ROut[h >> 1], ~(0xFFFFu << sh));
    InterlockedOr(ROut[h >> 1], F2H(v) << sh);
#else
    ROut[RP.OutOff + o] = v;
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ReduceBlock(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // A surplus workgroup of the rounded-up grid leaves before any barrier; the workgroup id is uniform, so the barriers stay valid.
    uint o = gid.y * RP.GridX + gid.x;
    if (o >= RP.NumOut)
    {
        return;
    }

    uint kind = KindOf(RP.Op);
    uint baseOff = KeptOffset(o);
    float acc = Identity(kind);
    float run = acc;
    uint inRun = 0u;
    for (uint r = tid; r < RP.R; r += WG)
    {
        // Short runs of 32 are summed first and then added to the total: the rounding error grows with the length of the chain of
        // additions, and this keeps it near that of a pairwise sum for long rows.
        run = Combine(kind, run, Transform(RP.Op, LoadIn(baseOff + ReducedOffset(r))));
        inRun++;
        if (inRun == 32u)
        {
            acc = Combine(kind, acc, run);
            run = Identity(kind);
            inRun = 0u;
        }
    }

    acc = Combine(kind, acc, run);

    float total = BlockReduce(kind, acc, tid);
    if (tid == 0u)
    {
        StoreOut(o, Finalize(RP.Op, total, RP.R));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ReduceThread(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint o = (gid.y * RP.GridX + gid.x) * WG + tid;
    if (o >= RP.NumOut)
    {
        return;
    }

    uint kind = KindOf(RP.Op);
    uint baseOff = KeptOffset(o);
    float acc = Identity(kind);
    float run = acc;
    uint inRun = 0u;
    for (uint r = 0u; r < RP.R; r++)
    {
        run = Combine(kind, run, Transform(RP.Op, LoadIn(baseOff + ReducedOffset(r))));
        inRun++;
        if (inRun == 32u)
        {
            acc = Combine(kind, acc, run);
            run = Identity(kind);
            inRun = 0u;
        }
    }

    acc = Combine(kind, acc, run);

    StoreOut(o, Finalize(RP.Op, acc, RP.R));
}

float LoadPartial(uint i)
{
#if F16
    return asfloat(RIn[i]);
#else
    return RIn[i];
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ReduceSplit(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint g = gid.y * RP.GridX + gid.x;
    uint o = g / RP.Split;
    uint s = g - o * RP.Split;
    if (o >= RP.NumOut)
    {
        return;
    }

    uint kind = KindOf(RP.Op);
    uint baseOff = KeptOffset(o);
    uint first = s * RP.Chunk;
    uint last = min(first + RP.Chunk, RP.R);
    float acc = Identity(kind);
    float run = acc;
    uint inRun = 0u;
    for (uint r = first + tid; r < last; r += WG)
    {
        run = Combine(kind, run, Transform(RP.Op, LoadIn(baseOff + ReducedOffset(r))));
        inRun++;
        if (inRun == 32u)
        {
            acc = Combine(kind, acc, run);
            run = Identity(kind);
            inRun = 0u;
        }
    }

    acc = Combine(kind, acc, run);

    float total = BlockReduce(kind, acc, tid);
    if (tid == 0u)
    {
#if F16
        ROut[g] = asuint(total);
#else
        ROut[g] = total;
#endif
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ReduceCombine(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint o = gid.y * RP.GridX + gid.x;
    if (o >= RP.NumOut)
    {
        return;
    }

    uint kind = KindOf(RP.Op);
    float v = Identity(kind);
    if (tid < RP.Split)
    {
        v = LoadPartial(o * RP.Split + tid);
    }

    float total = BlockReduce(kind, v, tid);
    if (tid == 0u)
    {
        StoreOut(o, Finalize(RP.Op, total, RP.TotalR));
    }
}
