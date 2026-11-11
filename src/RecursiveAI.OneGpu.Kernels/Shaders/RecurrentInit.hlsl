// Starts a recurrent layer: writes the first hidden state (copy 0 of H) and the cell state of the scratch state buffer from the optional
// initial_h and initial_c tensors, or zeros. One thread per (direction, batch row, hidden unit). The scratch state is [H0, H1, C, ...] with
// each part dirs * batch * hidden floats. initial_h and initial_c are [dirs, batch, hidden] (layout 0) or [batch, dirs, hidden] (layout 1);
// F16 reads them as packed halves.

#include "Common.hlsl"

struct InitParams
{
    uint Count;     // dirs * batch * hidden
    uint Hid;
    uint Batch;
    uint Dirs;
    uint Flags;     // bit 0 initial_h given, bit 1 initial_c given, bit 2 layout 1
    uint OffH0;
    uint OffC0;
    uint OffState;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<InitParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> H0;
[[vk::binding(1, 0)]] StructuredBuffer<uint> C0;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> State;

float Load(StructuredBuffer<uint> buf, uint i)
{
#if F16
    return LoadH(buf, i);
#else
    return asfloat(buf[i]);
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void InitOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint e = (gid.y * P.GridX + gid.x) * WG + tid;
    if (e >= P.Count)
    {
        return;
    }

    uint j = e % P.Hid;
    uint b = (e / P.Hid) % P.Batch;
    uint d = e / (P.Hid * P.Batch);
    uint src = ((P.Flags & 4u) != 0u) ? (b * P.Dirs + d) * P.Hid + j : e;
    State[P.OffState + e] = ((P.Flags & 1u) != 0u) ? Load(H0, P.OffH0 + src) : 0.0;
    State[P.OffState + 2u * P.Count + e] = ((P.Flags & 2u) != 0u) ? Load(C0, P.OffC0 + src) : 0.0;
}
