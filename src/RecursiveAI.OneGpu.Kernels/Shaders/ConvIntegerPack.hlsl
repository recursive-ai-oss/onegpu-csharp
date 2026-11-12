// Unfolds the input windows of an 8-bit convolution into the columns of a byte matrix (Chellapilla, Puri and Simard, "High Performance
// Convolutional Neural Networks for Document Processing", 2006), so that ConvInteger becomes one 8-bit matrix product per image and group.
//
// For image n and group g the matrix has K = (Cin / Groups) * KH * KW rows and OH * OW columns, stored with rows of Ldb = 4 * Ldb4 bytes so that every
// row starts on a word boundary: row (c * KH + kh) * KW + kw, column oh * OW + ow holds x[n, g * CinG + c, oh * SH - PH + kh * DH, ow * SW - PW + kw * DW],
// or the zero point of x where that position lies in the padding. Subtracting the zero point in the product then makes a padded tap contribute nothing,
// which is what ONNX means by padding a quantized input with zeros (the real value zero is the quantized value x_zero_point).
//
// The matrices of all images and groups follow one another, image-major. One thread owns one 32-bit word (four neighbouring columns of one row), so
// no two threads write the same word. The bytes are copied as they are, uint8 or int8, so the same shader serves both element types.

#include "Common.hlsl"

struct PackParams
{
    uint Count;     // words of the matrix to write
    uint OffX;      // byte offset of x
    uint OffO;      // word offset of the output
    uint GridX;
    uint CinG;      // input channels per group
    uint Cin;
    uint Groups;
    uint H, W;
    uint OH, OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;    // padding before the input, per axis
    uint Ldb4;      // words per matrix row
    uint HasZero;   // nonzero when the zero point of x is read from Zero
    uint OffZero;   // byte offset of the zero point
};

[[vk::push_constant]] ConstantBuffer<PackParams> CP;

[[vk::binding(0, 0)]] StructuredBuffer<uint> PackX;
[[vk::binding(1, 0)]] StructuredBuffer<uint> Zero;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> PackO;

[shader("compute")]
[numthreads(256, 1, 1)]
void ConvIntegerPackOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word.
    if (t >= CP.Count)
    {
        return;
    }

    uint taps = CP.KH * CP.KW;
    uint rows = CP.CinG * taps;
    uint col4 = t % CP.Ldb4;
    uint r = t / CP.Ldb4;
    uint k = r % rows;
    uint ig = r / rows;
    uint img = ig / CP.Groups;
    uint g = ig - img * CP.Groups;
    uint c = k / taps;
    uint kk = k - c * taps;
    uint kh = kk / CP.KW;
    uint kw = kk - kh * CP.KW;
    uint plane = CP.OffX + (img * CP.Cin + g * CP.CinG + c) * CP.H * CP.W;
    uint pad = CP.HasZero != 0u ? RawB(Zero, CP.OffZero) : 0u;
    uint ncols = CP.OH * CP.OW;

    uint word = 0u;
    [unroll]
    for (uint e = 0u; e < 4u; e++)
    {
        uint n = col4 * 4u + e;
        uint v = pad;
        if (n < ncols)
        {
            uint oh = n / CP.OW;
            uint ow = n - oh * CP.OW;
            int ih = (int)(oh * CP.SH + kh * CP.DH) - (int)CP.PH;
            int iw = (int)(ow * CP.SW + kw * CP.DW) - (int)CP.PW;
            if ((uint)ih < CP.H && (uint)iw < CP.W)
            {
                v = RawB(PackX, plane + (uint)ih * CP.W + (uint)iw);
            }
        }

        word |= v << (e * 8u);
    }

    PackO[CP.OffO + t] = word;
}
