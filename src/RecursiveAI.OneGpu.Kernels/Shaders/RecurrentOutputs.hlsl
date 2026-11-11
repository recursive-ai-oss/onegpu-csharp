// Writes the outputs of a recurrent layer from the scratch state: Mode 0 is Y (every step; zero at steps past a row's sequence length),
// mode 1 is Y_h and mode 2 is Y_c (the state after the row's last step; zero for a row whose length is 0, as onemodel's CPU kernel and the
// reference it follows do). Output layouts: layout 0 Y [seq, dirs, batch, hidden], states [dirs, batch, hidden]; layout 1 Y [batch, seq, dirs,
// hidden], states [batch, dirs, hidden]. The scratch keeps the steps as [dir][t][batch][hidden]. F16 writes packed halves, one thread per
// 32-bit word so two threads never write the same word, rounding to nearest even.

#include "Common.hlsl"

struct OutParams
{
    uint Count;     // elements of the output
    uint Mode;
    uint Hid;
    uint Batch;
    uint Seq;
    uint Dirs;
    uint Flags;     // bit 0 layout 1, bit 1 sequence lengths given
    uint OffOut;
    uint OffLens;
    uint OffState;
    uint Final;     // which copy of H holds the last state (0 or 1)
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<OutParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<float> State;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Lens;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Out;

float Value(uint e)
{
    bool layout1 = (P.Flags & 1u) != 0u;
    bool hasLens = (P.Flags & 2u) != 0u;
    uint dbh = P.Dirs * P.Batch * P.Hid;
    uint j = e % P.Hid;
    uint b, d, t = 0u;
    if (P.Mode == 0u)
    {
        if (layout1)
        {
            d = (e / P.Hid) % P.Dirs;
            t = (e / (P.Hid * P.Dirs)) % P.Seq;
            b = e / (P.Hid * P.Dirs * P.Seq);
        }
        else
        {
            b = (e / P.Hid) % P.Batch;
            d = (e / (P.Hid * P.Batch)) % P.Dirs;
            t = e / (P.Hid * P.Batch * P.Dirs);
        }
    }
    else if (layout1)
    {
        d = (e / P.Hid) % P.Dirs;
        b = e / (P.Hid * P.Dirs);
    }
    else
    {
        b = (e / P.Hid) % P.Batch;
        d = e / (P.Hid * P.Batch);
    }

    uint len = hasLens ? min(Lens[P.OffLens + b], P.Seq) : P.Seq;
    if (P.Mode == 0u)
    {
        if (t >= len)
        {
            return 0.0;
        }

        return State[P.OffState + 5u * dbh + ((d * P.Seq + t) * P.Batch + b) * P.Hid + j];
    }

    // A row of length 0 never ran, so its final state is zero; with nothing to step through (no sequence) the state is the initial one.
    if (hasLens && P.Seq > 0u && len == 0u)
    {
        return 0.0;
    }

    uint at = (d * P.Batch + b) * P.Hid + j;
    return State[P.OffState + (P.Mode == 1u ? P.Final * dbh : 2u * dbh) + at];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void OutputsOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
#if F16
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
        if (h >= P.OffOut && h < P.OffOut + P.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(Value(h - P.OffOut)) << (k * 16u));
        }
    }

    Out[word] = res;
#else
    if (t < P.Count)
    {
        Out[P.OffOut + t] = asuint(Value(t));
    }
#endif
}
