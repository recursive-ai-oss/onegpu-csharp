// Float32 convolution on simdgroup matrices with the window taps as the inner loop of k (Apple GPUs).
//
// ConvCoop.hlsli numbers k as (input channel, kernel row, kernel column), so every staged element needs its own tap arithmetic and bounds test. Here k
// runs in chunks of 16 input channels and, inside a chunk, over the taps of the window: each k step of 16 is one tap of 16 channels. Then the
// row and column of the tap, and so whether it lies in the padding, are the same for all 16 elements a thread stages, and the 16 channels'
// addresses differ by a constant (one input plane). A step costs one bounds test and eight independent loads per thread instead of eight
// divisions, eight tap updates and eight tests. The input planes of a chunk are read again by each tap and stay in cache.
//
// Requires the channels per group to be a multiple of 16 (the host checks). The sums are a different order from ConvCoop.hlsli's, with the same
// block structure (a second accumulator takes 8 steps = 128 terms), so the accuracy is the same and the bits differ: this is the fast variant only.
//
// Tile: 64 output channels x 64 pixels (NARROW: 32), four 32-wide subgroups arranged 2 x 2, k in steps of 16 staged through workgroup memory with the
// next step's global loads in flight (software pipelining). Bias, residual and activation applied as the result is stored. Split k and the grid are as in
// ConvCoop.hlsli.

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
    uint Flags;     // bit 0: add the bias, bit 1: add the residual
    uint Act;       // 0 none, else the unary operation number + 1
    float P0;
    float P1;
    uint KPerSplit; // 0: one pass over all of k. Else k is cut into ranges of this many terms (a multiple of 16); the z axis counts split-major,
                    // and each range writes its raw sums (no bias, residual or activation) into its own plane SplitStride elements after the previous one
    uint SplitStride;
    uint ImageGroups;
};

[[vk::push_constant]] ConstantBuffer<ConvGemmParams> P;

#define BM 64
#if defined(NARROW)
// A 64 x 32 tile (each subgroup owns 32 x 16) for grids too small to fill the GPU with 64 x 64 tiles.
#define BN 32
#define B_STRIDE 40
#define SG_COLS 16
#else
#define BN 64
#define B_STRIDE 72
#define SG_COLS 32
#endif
#define BK 16
#define THREADS 128
#define A_STRIDE 24

groupshared float TileA[BM * A_STRIDE];
groupshared float TileB[BK * B_STRIDE];
// The accumulators leave through the memory of the A tile (1536 floats, 1024 needed) once the k loop is done, which keeps a workgroup at 10.7 KB so three fit on a core.
#define TileC TileA

typedef CooperativeMatrix<float, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

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
void ConvTap(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
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
    uint hw = P.H * P.W;
    uint xBase = P.OffX + (image * P.Cin + g * cinG) * hw;
    uint wBase = P.OffW + g * P.M * P.K;
    uint channel0 = g * P.M;
    uint yBase = P.OffY + planeBase + (image * P.Cout + g * P.M) * P.N;
    uint taps = P.KH * P.KW;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint sgRow = sg >> 1;
    uint sgCol = sg & 1u;
    uint colTile = sgCol * SG_COLS;

    // B tile: this thread stages the pixel bc of the tile, rows bk0, bk0 + 2, ... (the 16 channels of the step).
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

    // A tile: this thread stages column ac of rows ar, ar + 8, ... ar + 56.
    uint ar = tid >> 4;
    uint ac = tid & 15u;
    uint aTap = ac * taps;                          // channel offset within a chunk of the weights (OIHW: the taps of a channel are consecutive)
    // Rows past the last output channel read the last row (their products land in rows that are never stored), so the loads need no test.
    uint lastRow = P.M - 1u;
    uint aRow0 = min(m0 + ar, lastRow) * P.K;
    uint aRow1 = min(m0 + ar + 8u, lastRow) * P.K;
    uint aRow2 = min(m0 + ar + 16u, lastRow) * P.K;
    uint aRow3 = min(m0 + ar + 24u, lastRow) * P.K;
    uint aRow4 = min(m0 + ar + 32u, lastRow) * P.K;
    uint aRow5 = min(m0 + ar + 40u, lastRow) * P.K;
    uint aRow6 = min(m0 + ar + 48u, lastRow) * P.K;
    uint aRow7 = min(m0 + ar + 56u, lastRow) * P.K;

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
    uint foldSteps = 0;

    // The step being staged: chunk of 16 channels, tap (kh, kw) within the window.
    uint steps = (kEnd - kBegin) >> 4;
    uint s0 = kBegin >> 4;
    uint chunk = s0 / taps;
    uint tap = s0 - chunk * taps;
    uint kh = tap / P.KW;
    uint kw = tap - kh * P.KW;

    float pa0 = 0.0, pa1 = 0.0, pa2 = 0.0, pa3 = 0.0, pa4 = 0.0, pa5 = 0.0, pa6 = 0.0, pa7 = 0.0;
#if defined(NARROW)
    float pb0 = 0.0, pb1 = 0.0, pb2 = 0.0, pb3 = 0.0;
#else
    float pb0 = 0.0, pb1 = 0.0, pb2 = 0.0, pb3 = 0.0, pb4 = 0.0, pb5 = 0.0, pb6 = 0.0, pb7 = 0.0;
#endif

#if defined(NARROW)
#define LOAD_B_ROWS pb0 = ConvX[xo]; pb1 = ConvX[xo + 4u * hw]; pb2 = ConvX[xo + 8u * hw]; pb3 = ConvX[xo + 12u * hw];
#define ZERO_B_ROWS pb0 = 0.0; pb1 = 0.0; pb2 = 0.0; pb3 = 0.0;
#define STORE_B_ROWS \
        TileB[bk0 * B_STRIDE + bc] = pb0; TileB[(bk0 + 4) * B_STRIDE + bc] = pb1; TileB[(bk0 + 8) * B_STRIDE + bc] = pb2; TileB[(bk0 + 12) * B_STRIDE + bc] = pb3;
#else
#define LOAD_B_ROWS pb0 = ConvX[xo]; pb1 = ConvX[xo + 2u * hw]; pb2 = ConvX[xo + 4u * hw]; pb3 = ConvX[xo + 6u * hw]; pb4 = ConvX[xo + 8u * hw]; pb5 = ConvX[xo + 10u * hw]; pb6 = ConvX[xo + 12u * hw]; pb7 = ConvX[xo + 14u * hw];
#define ZERO_B_ROWS pb0 = 0.0; pb1 = 0.0; pb2 = 0.0; pb3 = 0.0; pb4 = 0.0; pb5 = 0.0; pb6 = 0.0; pb7 = 0.0;
#define STORE_B_ROWS \
        TileB[bk0 * B_STRIDE + bc] = pb0; TileB[(bk0 + 2) * B_STRIDE + bc] = pb1; TileB[(bk0 + 4) * B_STRIDE + bc] = pb2; TileB[(bk0 + 6) * B_STRIDE + bc] = pb3; \
        TileB[(bk0 + 8) * B_STRIDE + bc] = pb4; TileB[(bk0 + 10) * B_STRIDE + bc] = pb5; TileB[(bk0 + 12) * B_STRIDE + bc] = pb6; TileB[(bk0 + 14) * B_STRIDE + bc] = pb7;
#endif

    // Loads the step (chunk, tap) into the registers and moves on to the next step. The input loads sit inside a branch that is uniform for nearly every
    // thread (only the taps in the padding and the pixels past the end of the image skip it); measured faster than clamped addresses with a select afterwards.
    // The weights of a chunk are consecutive in memory (OIHW), so the nine steps of a 3 x 3 chunk read the same cache lines.
#define LOAD_STEP                                                                                                        \
    {                                                                                                                    \
        uint aCol = wBase + chunk * 16u * taps + aTap + tap;                                                             \
        pa0 = ConvW[aCol + aRow0]; pa1 = ConvW[aCol + aRow1]; pa2 = ConvW[aCol + aRow2]; pa3 = ConvW[aCol + aRow3];     \
        pa4 = ConvW[aCol + aRow4]; pa5 = ConvW[aCol + aRow5]; pa6 = ConvW[aCol + aRow6]; pa7 = ConvW[aCol + aRow7];     \
        int ih = ih0 + (int)(kh * P.DH);                                                                                 \
        int iw = iw0 + (int)(kw * P.DW);                                                                                 \
        if (pixelValid && (uint)ih < P.H && (uint)iw < P.W)                                                              \
        {                                                                                                                \
            uint xo = xBase + (chunk * 16u + bk0) * hw + (uint)ih * P.W + (uint)iw;                                      \
            LOAD_B_ROWS                                                                                                  \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            ZERO_B_ROWS                                                                                                  \
        }                                                                                                                \
        tap++;                                                                                                           \
        kw++;                                                                                                            \
        if (kw == P.KW)                                                                                                  \
        {                                                                                                                \
            kw = 0;                                                                                                      \
            kh++;                                                                                                        \
            if (kh == P.KH)                                                                                              \
            {                                                                                                            \
                kh = 0;                                                                                                  \
                tap = 0;                                                                                                 \
                chunk++;                                                                                                 \
            }                                                                                                            \
        }                                                                                                                \
    }

    if (steps > 0)
    {
        LOAD_STEP
    }

    for (uint s = 0; s < steps; s++)
    {
        TileA[ar * A_STRIDE + ac] = pa0;
        TileA[(ar + 8) * A_STRIDE + ac] = pa1;
        TileA[(ar + 16) * A_STRIDE + ac] = pa2;
        TileA[(ar + 24) * A_STRIDE + ac] = pa3;
        TileA[(ar + 32) * A_STRIDE + ac] = pa4;
        TileA[(ar + 40) * A_STRIDE + ac] = pa5;
        TileA[(ar + 48) * A_STRIDE + ac] = pa6;
        TileA[(ar + 56) * A_STRIDE + ac] = pa7;
        STORE_B_ROWS
        GroupMemoryBarrierWithGroupSync();

        if (s + 1 < steps)
        {
            LOAD_STEP
        }

        MatA a0;
        MatA a1;
        MatB b0;
#if !defined(NARROW)
        MatB b1;
#endif
        a0.Load(TileA, (sgRow * 32 + 0) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        a1.Load(TileA, (sgRow * 32 + 16) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        b0.Load(TileB, colTile, B_STRIDE, MatrixLayout::RowMajor);
        part00 = MatC::MulAdd(a0, b0, part00);
        part10 = MatC::MulAdd(a1, b0, part10);
#if !defined(NARROW)
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
