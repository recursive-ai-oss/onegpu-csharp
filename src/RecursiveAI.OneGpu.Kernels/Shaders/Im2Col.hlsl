// Unfolds the input windows of a convolution into the columns of a matrix (Chellapilla, Puri and Simard, "High Performance Convolutional
// Neural Networks for Document Processing", 2006), so that the convolution becomes one matrix product per image.
//
// For image n the matrix has K = Channels * KH * KW rows and OH * OW columns: row ((c * KH) + kh) * KW + kw, column oh * OW + ow holds
// x[n, c, oh * SH - PH + kh * DH, ow * SW - PW + kw * DW], or zero where that position lies in the padding. The rows of one channel group
// are therefore contiguous, which lets a grouped convolution use one matrix per group without copying.
//
// F16 reads and writes halves; a thread then owns one 32-bit word (two neighbouring elements of the matrix).

#include "Common.hlsl"

struct Im2ColParams
{
    uint Count;     // elements of the matrix to write
    uint OffX;
    uint OffO;
    uint GridX;
    uint Channels;
    uint H, W;
    uint OH, OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;    // padding before the input, per axis
};

[[vk::push_constant]] ConstantBuffer<Im2ColParams> CP;

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> ColX;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> ColO;

float ColValue(uint e)
{
    uint ohw = CP.OH * CP.OW;
    uint taps = CP.KH * CP.KW;
    uint rows = CP.Channels * taps;
    uint s = e % ohw;
    uint r = e / ohw;
    uint k = r % rows;
    uint n = r / rows;
    uint c = k / taps;
    uint kk = k - c * taps;
    uint kh = kk / CP.KW;
    uint kw = kk - kh * CP.KW;
    uint oh = s / CP.OW;
    uint ow = s - oh * CP.OW;
    int ih = (int)(oh * CP.SH + kh * CP.DH) - (int)CP.PH;
    int iw = (int)(ow * CP.SW + kw * CP.DW) - (int)CP.PW;
    if ((uint)ih >= CP.H || (uint)iw >= CP.W)
    {
        return 0.0;
    }

    uint at = CP.OffX + ((n * CP.Channels + c) * CP.H + (uint)ih) * CP.W + (uint)iw;
#if F16
    return LoadH(ColX, at);
#else
    return ColX[at];
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void Im2ColOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
#if F16
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word.
    if (t >= HalfWords(CP.OffO, CP.Count))
    {
        return;
    }

    uint word = (CP.OffO >> 1) + t;
    uint res = ColO[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= CP.OffO && h < CP.OffO + CP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(ColValue(h - CP.OffO)) << (k * 16u));
        }
    }

    ColO[word] = res;
#else
    if (t < CP.Count)
    {
        ColO[CP.OffO + t] = ColValue(t);
    }
#endif
}
