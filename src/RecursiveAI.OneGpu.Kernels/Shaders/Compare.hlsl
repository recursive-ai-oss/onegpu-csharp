// out = (a op b) as 0 or 1 in a 32-bit word per element, with broadcasting.

#include "Broadcast2.hlsl"

[[vk::binding(2, 0)]] RWStructuredBuffer<uint> CmpO;

[shader("compute")]
[numthreads(256, 1, 1)]
void CompareOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * BP.GridX + gid.x) * WG + tid;
    if (t < BP.Count)
    {
        float a, b;
        BinaryLoad(t, a, b);
        CmpO[BP.OffO + t] = Compare(BP.Op, a, b) ? 1u : 0u;
    }
}
