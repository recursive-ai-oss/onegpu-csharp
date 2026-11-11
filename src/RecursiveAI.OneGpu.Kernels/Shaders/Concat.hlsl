// ONNX Concat along one axis, up to four inputs per pass. The output is seen as [outer, TotalAxis, inner]; input i covers axis
// positions [Start_i, Start_i + Len_i) and is [outer, Len_i, inner]. A pass writes only the output elements of its own inputs and leaves the
// rest (other passes' inputs) as they are, which for halves means one thread owns a 32-bit word and keeps the half it does not write.

#include "Common.hlsl"

struct ConcatParams
{
    uint Count;         // output elements
    uint OffOut;
    uint GridX;
    uint Inner;
    uint TotalAxis;
    uint Inputs;        // inputs in this pass, 1..4
    uint Off0, Off1, Off2, Off3;
    uint Start0, Start1, Start2, Start3;
    uint Len0, Len1, Len2, Len3;
};

[[vk::push_constant]] ConstantBuffer<ConcatParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> In0;
[[vk::binding(1, 0)]] StructuredBuffer<uint> In1;
[[vk::binding(2, 0)]] StructuredBuffer<uint> In2;
[[vk::binding(3, 0)]] StructuredBuffer<uint> In3;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> Out;

// Which input holds output element j, and where in it; false when the element belongs to another pass.
bool Find(uint j, out uint which, out uint s)
{
    uint row = j / P.Inner;
    uint n = j - row * P.Inner;
    uint o = row / P.TotalAxis;
    uint a = row - o * P.TotalAxis;
    which = 0u;
    s = 0u;
    if (P.Inputs > 0u && a >= P.Start0 && a - P.Start0 < P.Len0) { which = 0u; s = P.Off0 + (o * P.Len0 + (a - P.Start0)) * P.Inner + n; return true; }
    if (P.Inputs > 1u && a >= P.Start1 && a - P.Start1 < P.Len1) { which = 1u; s = P.Off1 + (o * P.Len1 + (a - P.Start1)) * P.Inner + n; return true; }
    if (P.Inputs > 2u && a >= P.Start2 && a - P.Start2 < P.Len2) { which = 2u; s = P.Off2 + (o * P.Len2 + (a - P.Start2)) * P.Inner + n; return true; }
    if (P.Inputs > 3u && a >= P.Start3 && a - P.Start3 < P.Len3) { which = 3u; s = P.Off3 + (o * P.Len3 + (a - P.Start3)) * P.Inner + n; return true; }
    return false;
}

uint Word(uint which, uint i)
{
    if (which == 0u) { return In0[i]; }
    if (which == 1u) { return In1[i]; }
    if (which == 2u) { return In2[i]; }
    return In3[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ConcatOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
#if ELEM_W16
    if (t >= HalfWords(P.OffOut, P.Count))
    {
        return;
    }

    uint word = (P.OffOut >> 1) + t;
    uint res = Out[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        uint which;
        uint s;
        if (h >= P.OffOut && h < P.OffOut + P.Count && Find(h - P.OffOut, which, s))
        {
            uint v = (Word(which, s >> 1) >> ((s & 1u) << 4)) & 0xFFFFu;
            res = (res & ~(0xFFFFu << (k * 16u))) | (v << (k * 16u));
        }
    }

    Out[word] = res;
#elif ELEM_W64
    uint which;
    uint s;
    if (t < P.Count && Find(t, which, s))
    {
        Out[2u * (P.OffOut + t)] = Word(which, 2u * s);
        Out[2u * (P.OffOut + t) + 1u] = Word(which, 2u * s + 1u);
    }
#else
    uint which;
    uint s;
    if (t < P.Count && Find(t, which, s))
    {
        Out[P.OffOut + t] = Word(which, s);
    }
#endif
}
