// out = (a op b) as a bool byte (0 or 1) per element, with broadcasting.

#include "Broadcast2.hlsl"

[[vk::binding(2, 0)]] RWStructuredBuffer<uint> CmpO;

[shader("compute")]
[numthreads(256, 1, 1)]
void CompareOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * BP.GridX + gid.x) * WG + tid;
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word.
    if (t >= BoolWords(BP.OffO, BP.Count))
    {
        return;
    }

    uint word = (BP.OffO >> 2) + t;
    uint res = CmpO[word];
    [unroll]
    for (uint k = 0u; k < 4u; k++)
    {
        uint h = word * 4u + k;
        if (h >= BP.OffO && h < BP.OffO + BP.Count)
        {
            float a, b;
            BinaryLoad(h - BP.OffO, a, b);
            res = WithByte(res, k, Compare(BP.Op, a, b) ? 1u : 0u);
        }
    }

    CmpO[word] = res;
}
