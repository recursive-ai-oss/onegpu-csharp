// Convolution as an implicit matrix product on matrix hardware: the structure of GemmCoop.hlsli (a 64 x 64 tile of C per workgroup, the k loop in steps of
// 16, tiles staged through workgroup memory in the element type, cooperative 16 x 16 fragments multiplied per 32-wide subgroup, accumulators leaving through
// workgroup memory) with the B tile gathered straight from the input tensor, as ConvGemm.hlsl does for the portable kernel, so no column matrix is built.
//
// For image n and group g, C[m, p] = sum over k of W[g, m, k] * X(k, p): m an output channel of the group, p an output pixel, k = (input channel, kh, kw);
// X(k, p) is zero in the padding. Grid: x = tiles of pixels, y = tiles of output channels, z = images * groups.
//
// The including file defines GEMM_F16 or GEMM_F32 (Metal only: no Vulkan device lists float32 fragments) and optionally GEMM_WIDE (eight subgroups).
// Float32 operands fuse the bias, the residual and the activation into the store; float16 operands write raw float32 sums for the epilogue pass, because a
// half-precision word can be shared by two tiles.

#include "Functions.hlsl"

#if defined(GEMM_F32)
typedef float Elem;
#else
typedef float16_t Elem;
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
#if defined(GEMM_WIDE)
#define THREADS 256
#define SUBGROUPS 8
#else
#define THREADS 128
#define SUBGROUPS 4
#endif

static const uint A_STRIDE = 24;
static const uint B_STRIDE = 72;

groupshared Elem TileA[BM * A_STRIDE];
groupshared Elem TileB[BK * B_STRIDE];
groupshared float TileC[SUBGROUPS * 256];

typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

static uint gYBase;
static uint gChannel0;
static uint gImage;

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

// Moves one accumulator through workgroup memory and stores it. Every lane of the workgroup reaches the barriers, whatever the bounds are.
#define STORE_ACC(ACC, ROW0, COL0)                                                                   \
    ACC.Store(TileC, sg * 256, 16, MatrixLayout::RowMajor);                                          \
    GroupMemoryBarrierWithGroupSync();                                                               \
    [unroll] for (uint e = 0; e < 8; e++)                                                            \
    {                                                                                                \
        uint ei = lane + 32 * e;                                                                     \
        StoreElement((ROW0) + (ei >> 4), (COL0) + (ei & 15u), TileC[sg * 256 + ei]);                 \
    }                                                                                                \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void ConvGemmCoop(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
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

    uint sg = tid >> 5;
    uint lane = tid & 31u;
#if defined(GEMM_WIDE)
    uint sgRow = sg >> 2;
    uint sgCol = sg & 3u;
    uint colTile = sgCol * 16;
#else
    uint sgRow = sg >> 1;
    uint sgCol = sg & 1u;
    uint colTile = sgCol * 32;
#endif

    // The pixel this thread stages into the B tile is the same for every k step: the thread-to-element map is column = tid % 64.
    uint bc = tid & 63u;
    uint bk0 = tid >> 6;
    uint gn = n0 + bc;
    bool pixelValid = gn < P.N;
    uint oh = gn / P.OW;
    uint ow = gn - oh * P.OW;
    int ih0 = (int)(oh * P.SH) - (int)P.PH;
    int iw0 = (int)(ow * P.SW) - (int)P.PW;

    MatC acc00 = MatC::Zero();
    MatC acc10 = MatC::Zero();
#if !defined(GEMM_WIDE)
    MatC acc01 = MatC::Zero();
    MatC acc11 = MatC::Zero();
#endif

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        [unroll] for (uint e = 0; e < (BM * BK) / THREADS; e++)
        {
            uint idx = tid + e * THREADS;
            uint r = idx >> 4;
            uint k = idx & 15u;
            uint gm = m0 + r;
            uint gk = k0 + k;
            Elem v = (Elem)0.0f;
            if (gm < P.M && gk < P.K)
            {
                v = ConvW[wBase + gm * P.K + gk];
            }
            TileA[r * A_STRIDE + k] = v;
        }

        [unroll] for (uint f = 0; f < (BK * BN) / THREADS; f++)
        {
            uint k = bk0 + f * (THREADS / 64u);
            uint gk = k0 + k;
            Elem v = (Elem)0.0f;
            if (pixelValid && gk < P.K)
            {
                if (pointwise)
                {
                    v = ConvX[xBase + gk * P.N + gn];
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
                        v = ConvX[xBase + (ci * P.H + (uint)ih) * P.W + (uint)iw];
                    }
                }
            }
            TileB[k * B_STRIDE + bc] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        MatA a0;
        MatA a1;
        MatB b0;
        a0.Load(TileA, (sgRow * 32 + 0) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        a1.Load(TileA, (sgRow * 32 + 16) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        b0.Load(TileB, colTile, B_STRIDE, MatrixLayout::RowMajor);
        acc00 = MatC::MulAdd(a0, b0, acc00);
        acc10 = MatC::MulAdd(a1, b0, acc10);
#if !defined(GEMM_WIDE)
        MatB b1;
        b1.Load(TileB, colTile + 16, B_STRIDE, MatrixLayout::RowMajor);
        acc01 = MatC::MulAdd(a0, b1, acc01);
        acc11 = MatC::MulAdd(a1, b1, acc11);
#endif

        GroupMemoryBarrierWithGroupSync();
    }

    uint rowBase = m0 + sgRow * 32;
    uint colBase = n0 + colTile;
    STORE_ACC(acc00, rowBase + 0, colBase + 0)
    STORE_ACC(acc10, rowBase + 16, colBase + 0)
#if !defined(GEMM_WIDE)
    STORE_ACC(acc01, rowBase + 0, colBase + 16)
    STORE_ACC(acc11, rowBase + 16, colBase + 16)
#endif
}
