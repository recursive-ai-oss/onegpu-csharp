// The second half of a transposed convolution. The matrix product W^T * x has written, for every image and group, one row per
// (output channel of the group, kh, kw) and one column per input position; this pass adds each output element's contributions: output
// (oy, ox) receives the entry of input position (ih, iw) for every tap (kh, kw) with oy + PH = ih * SH + kh * DH and ox + PW = iw * SW + kw * DW.
// Gathering per output element, rather than scattering per input element, gives every thread its own output (no atomics) and a fixed order:
// taps in increasing (kh, kw), summed from zero in float32. The bias and the activation follow (bias, then activation).
//
// F16 writes halves, one thread per 32-bit word; the matrix is always float32 scratch.

#include "Functions.hlsl"

struct Col2ImParams
{
    uint Count;
    uint OffCol;
    uint OffBias;
    uint OffY;
    uint GridX;
    uint Cout;      // total output channels = Groups * CoutG
    uint Groups;
    uint H, W;      // input extent (the columns of the matrix)
    uint OH, OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;    // padding removed from the beginning of the full output
    uint HasBias;
    uint Act;
    float P0;
    float P1;
};

[[vk::push_constant]] ConstantBuffer<Col2ImParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<float> CiCol;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> CiBias;
[[vk::binding(2, 0)]] RWStructuredBuffer<Elem> CiY;

float Col2ImValue(uint e)
{
    uint ox = e % CP.OW;
    uint r = e / CP.OW;
    uint oy = r % CP.OH;
    uint plane = r / CP.OH;
    uint co = plane % CP.Cout;
    uint n = plane / CP.Cout;
    uint coutG = CP.Cout / CP.Groups;
    uint g = co / coutG;
    uint cl = co - g * coutG;
    uint hw = CP.H * CP.W;
    uint rows = coutG * CP.KH * CP.KW;
    uint baseCol = CP.OffCol + (n * CP.Groups + g) * rows * hw;
    float acc = 0.0;
    for (uint kh = 0u; kh < CP.KH; kh++)
    {
        int ty = (int)(oy + CP.PH) - (int)(kh * CP.DH);
        if (ty < 0 || (uint)ty % CP.SH != 0u)
        {
            continue;
        }

        uint ih = (uint)ty / CP.SH;
        if (ih >= CP.H)
        {
            continue;
        }

        for (uint kw = 0u; kw < CP.KW; kw++)
        {
            int tx = (int)(ox + CP.PW) - (int)(kw * CP.DW);
            if (tx < 0 || (uint)tx % CP.SW != 0u)
            {
                continue;
            }

            uint iw = (uint)tx / CP.SW;
            if (iw < CP.W)
            {
                acc += CiCol[baseCol + ((cl * CP.KH + kh) * CP.KW + kw) * hw + ih * CP.W + iw];
            }
        }
    }

    float v = acc;
    if (CP.HasBias != 0u)
    {
#if F16
        v += LoadH(CiBias, CP.OffBias + co);
#else
        v += CiBias[CP.OffBias + co];
#endif
    }

    if (CP.Act != 0u)
    {
        v = Unary(CP.Act - 1u, v, CP.P0, CP.P1);
    }

    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void Col2ImOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    if (t >= HalfWords(CP.OffY, CP.Count))
    {
        return;
    }

    uint word = (CP.OffY >> 1) + t;
    uint res = CiY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffY && h < CP.OffY + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(Col2ImValue(h - CP.OffY)) << (k * 16u));
        }
    }

    CiY[word] = res;
#else
    if (t < CP.Count)
    {
        CiY[CP.OffY + t] = Col2ImValue(t);
    }
#endif
}
