// out = activation(a op b) with broadcasting. The activation is optional (Act = 0 for none, else the unary operation number + 1).

#include "Broadcast2.hlsl"

[[vk::binding(2, 0)]] RWStructuredBuffer<Elem> BinO;

float BinaryValue(uint i)
{
    float a, b;
    BinaryLoad(i, a, b);
    float v = Binary(BP.Op, a, b);
#if ACT
    if (BP.Act != 0u)
    {
        v = Unary(BP.Act - 1u, v, BP.P0, BP.P1);
    }
#endif

    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void BinaryOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * BP.GridX + gid.x) * WG + tid;
#if F16
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word, and a surplus thread would then
    // write that word back with stale contents.
    // One thread owns one 32-bit word, so two threads never write the same word.
    if (t >= HalfWords(BP.OffO, BP.Count))
    {
        return;
    }

    uint word = (BP.OffO >> 1) + t;
    uint res = BinO[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= BP.OffO && h < BP.OffO + BP.Count)
        {
            uint code = F2H(BinaryValue(h - BP.OffO));
            res = (res & ~(0xFFFFu << (k * 16u))) | (code << (k * 16u));
        }
    }

    BinO[word] = res;
#else
    if (t < BP.Count)
    {
        BinO[BP.OffO + t] = BinaryValue(t);
    }
#endif
}
