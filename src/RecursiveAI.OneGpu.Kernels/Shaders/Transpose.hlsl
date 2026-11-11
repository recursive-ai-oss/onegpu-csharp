// N-dimensional permutation copy. The output is described by up to six dimensions, innermost first, each with the stride in the input that
// one step along it takes, so any permutation of a dense tensor is a list of (dimension, input stride) pairs. Values are copied bit for bit.
// ELEM selects the element size: W16 (half, read and written as halves inside words), W32 (4 bytes) or W64 (8 bytes, two words).

#include "Common.hlsl"

struct TransposeParams
{
    uint Count;
    uint OffI;
    uint OffO;
    uint GridX;
    uint Rank;
    uint D0, D1, D2, D3, D4, D5;
    uint S0, S1, S2, S3, S4, S5;
};

[[vk::push_constant]] ConstantBuffer<TransposeParams> TP;

[[vk::binding(0, 0)]] StructuredBuffer<uint> TIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> TOut;

#define TSTEP(R, D, S) \
    if (TP.Rank == R + 1u) { off += rem * TP.S; } \
    else if (TP.Rank > R) { uint c = rem % TP.D; rem /= TP.D; off += c * TP.S; }

// Element index in the input of output element j (counted from the first output element).
uint Source(uint j)
{
    uint rem = j;
    uint off = TP.OffI;
    TSTEP(0u, D0, S0)
    TSTEP(1u, D1, S1)
    TSTEP(2u, D2, S2)
    TSTEP(3u, D3, S3)
    TSTEP(4u, D4, S4)
    TSTEP(5u, D5, S5)
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void TransposeOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * TP.GridX + gid.x) * WG + tid;
#if ELEM_W16
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word, and a surplus thread would then
    // write that word back with stale contents.
    if (t >= HalfWords(TP.OffO, TP.Count))
    {
        return;
    }

    uint word = (TP.OffO >> 1) + t;
    uint res = TOut[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= TP.OffO && h < TP.OffO + TP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (RawH(TIn, Source(h - TP.OffO)) << (k * 16u));
        }
    }

    TOut[word] = res;
#elif ELEM_W64
    if (t < TP.Count)
    {
        uint s = Source(t);
        TOut[2u * (TP.OffO + t)] = TIn[2u * s];
        TOut[2u * (TP.OffO + t) + 1u] = TIn[2u * s + 1u];
    }
#else
    if (t < TP.Count)
    {
        TOut[TP.OffO + t] = TIn[Source(t)];
    }
#endif
}
