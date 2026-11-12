// Convolution as an implicit matrix product (Chellapilla et al., "High Performance Convolutional Neural Networks for Document Processing", 2006,
// without ever building the column matrix): for image n and group g, C[m, p] = sum over k of W[g, m, k] * X(k, p), where m is an output channel of the
// group, p an output pixel and k = (input channel, kh, kw); X(k, p) is read straight from the input tensor, or is zero in the padding.
//
// The structure is the portable tiled matrix product (see GemmTiled.hlsli): a 64 x 64 tile of C per workgroup, the k loop in steps of 16 staged
// through workgroup memory, a 4 x 4 register tile per thread, every output element summed by one thread in increasing k. The result does not
// depend on scheduling, occupancy or subgroup size, so one kernel serves both the fast and the deterministic mode.
//
// With float32 inputs the bias, the residual and the activation are applied as each element is stored, so the output is written once. With F16
// inputs (halves read as pairs inside 32-bit words) the kernel writes the raw float32 sums and the epilogue pass converts them: two threads of
// different tiles can share a half-precision word, so a tiled kernel cannot write halves directly.
//
// Grid: x = tiles of pixels, y = tiles of output channels, z = images * groups.

#include "Functions.hlsl"
#include "TileAccum.hlsli"

#if F16
typedef uint Elem;
float LoadElem(StructuredBuffer<uint> buf, uint i)
{
    uint w = buf[i >> 1];
    return f16tof32((i & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu));
}
#else
typedef float Elem;
float LoadElem(StructuredBuffer<float> buf, uint i) { return buf[i]; }
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> ConvX;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> ConvW;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> ConvY;
[[vk::binding(3, 0)]] StructuredBuffer<float> ConvBias;
[[vk::binding(4, 0)]] StructuredBuffer<float> ConvRes;

struct ConvGemmParams
{
    uint M;         // output channels per group
    uint N;         // output pixels per image
    uint K;         // input channels per group * KH * KW
    uint Groups;
    uint Cin, H, W;
    uint OW;
    uint KH, KW;
    uint SH, SW;
    uint DH, DW;
    uint PH, PW;
    uint OffX, OffW, OffY, OffBias, OffRes;
    uint Cout;
    uint Flags;     // bit 0: add the bias, bit 1: add the residual, bit 2: pointwise (1x1, stride 1, no padding)
    uint Act;       // 0 none, else the unary operation number + 1
    float P0;
    float P1;
};

[[vk::push_constant]] ConstantBuffer<ConvGemmParams> P;

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TILE_THREADS = 256;

groupshared float As[BK][BM + 1];
groupshared float Bs[BK][BN + 1];

static uint gYBase;     // first output element of this image and group
static uint gChannel0;  // first output channel of this group, in the whole tensor
static uint gImage;

// The activations a convolution fuses. The host sends any other function through the separate epilogue pass.
float ConvAct(uint act, float x)
{
    switch (act)
    {
        case 8u: return Tanh(x);
        case 10u: return Sigmoid(x);
        case 11u: return IsNaN(x) ? x : ((x < 0.0) ? 0.0 : x);
        case 14u: return x * Sigmoid(x);
        case 15u: return IsNaN(x) ? x : max(0.0, min(1.0, P.P0 * x + P.P1));
        case 16u: return IsNaN(x) ? x : x * max(0.0, min(1.0, x * (1.0 / 6.0) + 0.5));
        case 17u: return IsNaN(x) ? x : min(max(x, P.P0), P.P1);
        case 18u: return IsNaN(x) ? x : ((x < 0.0) ? P.P0 * x : x);
        default: return x;
    }
}

void StoreElement(uint row, uint col, float value)
{
    if (row < P.M && col < P.N)
    {
        float v = value;
        uint co = gChannel0 + row;
        if ((P.Flags & 1u) != 0u) { v += ConvBias[P.OffBias + co]; }
        if ((P.Flags & 2u) != 0u) { v += ConvRes[P.OffRes + (gImage * P.Cout + co) * P.N + col]; }
        if (P.Act != 0u) { v = ConvAct(P.Act, v); }
        ConvY[gYBase + row * P.N + col] = v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ConvGemmTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    uint image = group.z / P.Groups;
    uint g = group.z - image * P.Groups;
    uint cinG = P.Cin / P.Groups;
    uint xBase = P.OffX + (image * P.Cin + g * cinG) * P.H * P.W;
    uint wBase = P.OffW + g * P.M * P.K;
    gImage = image;
    gChannel0 = g * P.M;
    gYBase = P.OffY + (image * P.Cout + g * P.M) * P.N;
    bool pointwise = (P.Flags & 4u) != 0u;
    uint taps = P.KH * P.KW;

    uint tx = tid & 15u;
    uint ty = tid >> 4;

    // The pixel this thread stages into the B tile is the same for every k step: the thread-to-element map is column = tid % 64.
    uint bc = tid & 63u;
    uint bk0 = tid >> 6;
    uint gn = n0 + bc;
    bool pixelValid = gn < P.N;
    uint oh = gn / P.OW;
    uint ow = gn - oh * P.OW;
    int ih0 = (int)(oh * P.SH) - (int)P.PH;
    int iw0 = (int)(ow * P.SW) - (int)P.PW;

    // Sixteen named scalars rather than an array: the Metal output of the shader compiler keeps indexed arrays in thread-local memory.
    TILE_DECLARE(c)
    TILE_DECLARE(u)
    uint groupSteps = 0;

    for (uint k0 = 0; k0 < P.K; k0 += 2 * BK)
    {
        // Two staged steps (32 terms) chain into one block sum t.
        TILE_DECLARE(t)
        [unroll] for (uint sub = 0; sub < 2; sub++)
        {
        uint kb = k0 + sub * BK;
        [unroll] for (uint e = 0; e < (BM * BK) / TILE_THREADS; e++)
        {
            uint idx = tid + e * TILE_THREADS;
            uint r = idx / BK;
            uint k = idx % BK;
            uint gm = m0 + r;
            uint gk = kb + k;
            float v = 0.0;
            if (gm < P.M && gk < P.K)
            {
                v = LoadElem(ConvW, wBase + gm * P.K + gk);
            }
            As[k][r] = v;
        }

        [unroll] for (uint f = 0; f < (BK * BN) / TILE_THREADS; f++)
        {
            uint k = bk0 + f * 4u;
            uint gk = kb + k;
            float v = 0.0;
            if (pixelValid && gk < P.K)
            {
                if (pointwise)
                {
                    v = LoadElem(ConvX, xBase + gk * P.N + gn);
                }
                else
                {
                    uint ci = gk / taps;
                    uint kk = gk - ci * taps;
                    uint kh = kk / P.KW;
                    uint kw = kk - kh * P.KW;
                    int ih = ih0 + (int)(kh * P.DH);
                    int iw = iw0 + (int)(kw * P.DW);
                    if ((uint)ih < P.H && (uint)iw < P.W)
                    {
                        v = LoadElem(ConvX, xBase + (ci * P.H + (uint)ih) * P.W + (uint)iw);
                    }
                }
            }
            Bs[k][bc] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk2 = 0; kk2 < BK; kk2++)
        {
            float a0 = As[kk2][ty + 0];
            float a1 = As[kk2][ty + 16];
            float a2 = As[kk2][ty + 32];
            float a3 = As[kk2][ty + 48];
            float b0 = Bs[kk2][tx + 0];
            float b1 = Bs[kk2][tx + 16];
            float b2 = Bs[kk2][tx + 32];
            float b3 = Bs[kk2][tx + 48];
            TILE_STEP(t, a0, a1, a2, a3, b0, b1, b2, b3)
        }

        GroupMemoryBarrierWithGroupSync();
        }

        TILE_ADD(u, t)
        if (++groupSteps == TILE_STEPS_PER_GROUP)
        {
            TILE_ADD(c, u)
            TILE_ZERO(u)
            groupSteps = 0;
        }
    }

    TILE_ADD(c, u)

    StoreElement(m0 + ty + 0, n0 + tx + 0, c00);
    StoreElement(m0 + ty + 0, n0 + tx + 16, c01);
    StoreElement(m0 + ty + 0, n0 + tx + 32, c02);
    StoreElement(m0 + ty + 0, n0 + tx + 48, c03);
    StoreElement(m0 + ty + 16, n0 + tx + 0, c10);
    StoreElement(m0 + ty + 16, n0 + tx + 16, c11);
    StoreElement(m0 + ty + 16, n0 + tx + 32, c12);
    StoreElement(m0 + ty + 16, n0 + tx + 48, c13);
    StoreElement(m0 + ty + 32, n0 + tx + 0, c20);
    StoreElement(m0 + ty + 32, n0 + tx + 16, c21);
    StoreElement(m0 + ty + 32, n0 + tx + 32, c22);
    StoreElement(m0 + ty + 32, n0 + tx + 48, c23);
    StoreElement(m0 + ty + 48, n0 + tx + 0, c30);
    StoreElement(m0 + ty + 48, n0 + tx + 16, c31);
    StoreElement(m0 + ty + 48, n0 + tx + 32, c32);
    StoreElement(m0 + ty + 48, n0 + tx + 48, c33);
}
