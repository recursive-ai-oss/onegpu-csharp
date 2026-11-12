// Convolution as an implicit matrix product on matrix hardware, float32 (built for Metal: simdgroup matrices are the only float32 fragments any
// device lists). The decomposition is the one of ConvGemm.hlsl (C[m, p] = sum over k of W[m, k] * X(k, p), X read from the input tensor with the
// window gathered on the fly, zero in the padding, so no column matrix is ever written), and the matrix products are those of GemmCoop.hlsli:
// one 64 x 64 tile of C per workgroup, four 32-wide subgroups arranged 2 x 2, each owning 32 x 32 as four 16 x 16 float accumulators, k in steps of 16.
//
// Two paths share the accumulators. A pointwise convolution (1x1, stride 1, no padding) whose tile lies inside C reads its fragments straight from
// the weights and the input for every whole 16-step of k, with no staging and no barrier. Everything else (windows, edge tiles, the ragged k tail)
// stages the weight tile and the gathered input tile through workgroup memory first.
//
// The bias, the residual and the activation are applied as each element is stored, so the output is written once and nothing else touches it.
// Each output element is a sum in an order the matrix hardware fixes; the kernel is the fast variant only.
//
// Split-k: a layer whose grid is too small for the GPU (the late layers of a classifier at batch 1: a few tiles, each 4608 deep) is cut along k. The
// z axis then runs split-major (z = split * images * groups + image * groups + group), every range sums its part of k in the same order the whole sum
// would, and a following pass adds the planes in range order and applies the epilogue (ConvEpilogue.hlsl, SPLIT). The cut depends only on the
// shape, so the result does not depend on how the GPU schedules the ranges.
//
// Grid: x = tiles of pixels, y = tiles of output channels, z = images * groups (times the number of ranges when k is split). The host pins the
// subgroup size to 32 and runs full subgroups.

#include "Functions.hlsl"
#if defined(NARROW)
#define COOP_FOUR 0
#else
#define COOP_FOUR 1
#endif
#include "TileAccum.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float> ConvX;
[[vk::binding(1, 0)]] StructuredBuffer<float> ConvW;
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
    uint KPerSplit; // 0: one pass over all of k. Else k is cut into ranges of this many steps (a multiple of 16); the z axis counts split-major,
                    // and each range writes its raw sums (no bias, residual or activation) into its own plane SplitStride elements after the previous one
    uint SplitStride;
    uint ImageGroups; // images * groups of this dispatch, when k is split
};

[[vk::push_constant]] ConstantBuffer<ConvGemmParams> P;

#define BM 64
#define BK 16
#define THREADS 128
#if defined(NARROW)
// A 64 x 32 tile (each subgroup owns 32 x 16): four times as many workgroups for the small images of the late layers, which a 64 x 64 tile leaves most of the GPU idle on.
#define BN 32
#define B_STRIDE 40
#define SG_COLS 16
#define B_ROW_STEP 4    // rows of the B tile between two elements one thread stages (THREADS / BN)
#else
#define BN 64
#define B_STRIDE 72
#define SG_COLS 32
#define B_ROW_STEP 2
#endif

// Rows are padded so fragment loads stay aligned and spread over the banks (as in GemmCoop.hlsli).
#define A_STRIDE 24

groupshared float TileA[BM * A_STRIDE];
groupshared float TileB[BK * B_STRIDE];
groupshared float TileC[4 * 256];

typedef CooperativeMatrix<float, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

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

// Steps (input channel, kernel row, kernel column) to the next position of the k axis.
void NextTap(inout uint ci, inout uint kh, inout uint kw)
{
    kw++;
    if (kw == P.KW)
    {
        kw = 0;
        kh++;
        if (kh == P.KH)
        {
            kh = 0;
            ci++;
        }
    }
}


// Moves one accumulator through workgroup memory and stores it with the epilogue for the elements inside the matrix. Every lane reaches the barriers.
#define STORE_ACC(ACC, ROW0, COL0)                                                                    \
    ACC.Store(TileC, sg * 256, 16, MatrixLayout::RowMajor);                                           \
    GroupMemoryBarrierWithGroupSync();                                                                \
    [unroll] for (uint e = 0; e < 8; e++)                                                             \
    {                                                                                                 \
        uint ei = lane + 32 * e;                                                                      \
        uint gm = (ROW0) + (ei >> 4);                                                                 \
        uint gn = (COL0) + (ei & 15u);                                                                \
        if (gm < P.M && gn < P.N)                                                                     \
        {                                                                                             \
            float v = TileC[sg * 256 + ei];                                                           \
            uint co = channel0 + gm;                                                                  \
            if ((P.Flags & 1u) != 0u) { v += ConvBias[P.OffBias + co]; }                              \
            if ((P.Flags & 2u) != 0u) { v += ConvRes[P.OffRes + (image * P.Cout + co) * P.N + gn]; }  \
            if (P.Act != 0u) { v = ConvAct(P.Act, v); }                                               \
            ConvY[yBase + gm * P.N + gn] = v;                                                         \
        }                                                                                             \
    }                                                                                                 \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void ConvCoop(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    uint split = 0;
    uint ig = group.z;
    uint kBegin = 0;
    uint kEnd = P.K;
    uint planeBase = 0;
    if (P.KPerSplit != 0u)
    {
        split = ig / P.ImageGroups;
        ig = ig - split * P.ImageGroups;
        kBegin = split * P.KPerSplit;
        kEnd = min(P.K, kBegin + P.KPerSplit);
        planeBase = split * P.SplitStride;
    }

    uint image = ig / P.Groups;
    uint g = ig - image * P.Groups;
    uint cinG = P.Cin / P.Groups;
    uint xBase = P.OffX + (image * P.Cin + g * cinG) * P.H * P.W;
    uint wBase = P.OffW + g * P.M * P.K;
    uint channel0 = g * P.M;
    uint yBase = P.OffY + planeBase + (image * P.Cout + g * P.M) * P.N;
    bool pointwise = (P.Flags & 4u) != 0u;
    uint taps = P.KH * P.KW;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint sgRow = sg >> 1;
    uint sgCol = sg & 1u;
    uint colTile = sgCol * SG_COLS;

    // The pixel this thread stages into the B tile is the same for every k step: the thread-to-element map is column = tid % 64.
#if defined(NARROW)
    uint bc = tid & 31u;
    uint bk0 = tid >> 5;
#else
    uint bc = tid & 63u;
    uint bk0 = tid >> 6;
#endif
    uint gn = n0 + bc;
    bool pixelValid = gn < P.N;
    uint oh = gn / P.OW;
    uint ow = gn - oh * P.OW;
    int ih0 = (int)(oh * P.SH) - (int)P.PH;
    int iw0 = (int)(ow * P.SW) - (int)P.PW;

    MatC acc00 = MatC::Zero();
    MatC acc10 = MatC::Zero();
#if !defined(NARROW)
    MatC acc01 = MatC::Zero();
    MatC acc11 = MatC::Zero();
#endif
    MatC part00 = MatC::Zero();
    MatC part10 = MatC::Zero();
#if !defined(NARROW)
    MatC part01 = MatC::Zero();
    MatC part11 = MatC::Zero();
#endif
    MatC mid00 = MatC::Zero();
    MatC mid10 = MatC::Zero();
#if !defined(NARROW)
    MatC mid01 = MatC::Zero();
    MatC mid11 = MatC::Zero();
#endif
    uint foldGroups = 0;
    uint foldSteps = 0;

    uint kStaged = kBegin;
    if (pointwise && m0 + BM <= P.M && n0 + BN <= P.N)
    {
        kStaged = kBegin + ((kEnd - kBegin) & ~(BK - 1u));
        uint rowA = m0 + sgRow * 32;
        for (uint kd = kBegin; kd < kStaged; kd += BK)
        {
            MatA a0;
            MatA a1;
            MatB b0;
            a0.Load(ConvW, wBase + rowA * P.K + kd, P.K, MatrixLayout::RowMajor);
            a1.Load(ConvW, wBase + (rowA + 16) * P.K + kd, P.K, MatrixLayout::RowMajor);
            b0.Load(ConvX, xBase + kd * P.N + n0 + colTile, P.N, MatrixLayout::RowMajor);
            part00 = MatC::MulAdd(a0, b0, part00);
            part10 = MatC::MulAdd(a1, b0, part10);
#if !defined(NARROW)
            MatB b1;
            b1.Load(ConvX, xBase + kd * P.N + n0 + colTile + 16, P.N, MatrixLayout::RowMajor);
            part01 = MatC::MulAdd(a0, b1, part01);
            part11 = MatC::MulAdd(a1, b1, part11);
#endif
COOP_STEP_DONE
        }
    }

    for (uint k0 = kStaged; k0 < kEnd; k0 += BK)
    {
        [unroll] for (uint e = 0; e < (BM * BK) / THREADS; e++)
        {
            uint idx = tid + e * THREADS;
            uint r = idx >> 4;
            uint k = idx & 15u;
            uint gm = m0 + r;
            uint gk = k0 + k;
            float v = 0.0;
            if (gm < P.M && gk < kEnd)
            {
                v = ConvW[wBase + gm * P.K + gk];
            }
            TileA[r * A_STRIDE + k] = v;
        }

        // This thread's rows of the B tile are k = bk0, bk0 + B_ROW_STEP, ...: the window position (input channel, kernel row, kernel column) of the
        // first one costs two divisions, and each next row is found by stepping, so the loop carries no divisions.
        uint gk = k0 + bk0;
        uint ci = 0;
        uint kh = 0;
        uint kw = 0;
        if (!pointwise)
        {
            ci = gk / taps;
            uint kk = gk - ci * taps;
            kh = kk / P.KW;
            kw = kk - kh * P.KW;
        }

        [unroll] for (uint f = 0; f < (BK * BN) / THREADS; f++)
        {
            float v = 0.0;
            if (pixelValid && gk < kEnd)
            {
                if (pointwise)
                {
                    v = ConvX[xBase + gk * P.N + gn];
                }
                else
                {
                    int ih = ih0 + (int)(kh * P.DH);
                    int iw = iw0 + (int)(kw * P.DW);
                    if ((uint)ih < P.H && (uint)iw < P.W)
                    {
                        v = ConvX[xBase + (ci * P.H + (uint)ih) * P.W + (uint)iw];
                    }
                }
            }
            TileB[(bk0 + f * B_ROW_STEP) * B_STRIDE + bc] = v;
            gk += B_ROW_STEP;
            NextTap(ci, kh, kw);
            NextTap(ci, kh, kw);
#if defined(NARROW)
            NextTap(ci, kh, kw);
            NextTap(ci, kh, kw);
#endif
        }

        GroupMemoryBarrierWithGroupSync();

        MatA a0;
        MatA a1;
        MatB b0;
        a0.Load(TileA, (sgRow * 32 + 0) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        a1.Load(TileA, (sgRow * 32 + 16) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        b0.Load(TileB, colTile, B_STRIDE, MatrixLayout::RowMajor);
        part00 = MatC::MulAdd(a0, b0, part00);
        part10 = MatC::MulAdd(a1, b0, part10);
#if !defined(NARROW)
        MatB b1;
        b1.Load(TileB, colTile + 16, B_STRIDE, MatrixLayout::RowMajor);
        part01 = MatC::MulAdd(a0, b1, part01);
        part11 = MatC::MulAdd(a1, b1, part11);
#endif
COOP_STEP_DONE

        GroupMemoryBarrierWithGroupSync();
    }

    uint rowBase = m0 + sgRow * 32;
    uint colBase = n0 + colTile;
    COOP_FOLD_ALL
    STORE_ACC(acc00, rowBase + 0, colBase + 0)
    STORE_ACC(acc10, rowBase + 16, colBase + 0)
#if !defined(NARROW)
    STORE_ACC(acc01, rowBase + 0, colBase + 16)
    STORE_ACC(acc11, rowBase + 16, colBase + 16)
#endif
}
