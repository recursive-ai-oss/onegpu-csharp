// Convolution as an implicit matrix product on matrix hardware, float32 or float16 (built for Metal: simdgroup matrices are the only float32
// fragments any device lists). The decomposition is the one of ConvGemm.hlsl (C[m, p] = sum over k of W[m, k] * X(k, p), X read from the input
// tensor with the window gathered on the fly, zero in the padding, so no column matrix is ever written), and the matrix products are those of
// GemmCoop.hlsli: one 64 x 64 tile of C per workgroup, four 32-wide subgroups arranged 2 x 2, each owning 32 x 32 as four 16 x 16 float
// accumulators, k in steps of 16.
//
// The including file defines CONV_F32 or CONV_F16 (operands, bias and residual in that type; float32 accumulation either way), NARROW for a 64 x 32
// tile, and, with CONV_F16, SPLIT to leave raw float32 sums in a plane instead of storing halves.
//
// Two paths share the accumulators. A pointwise convolution (1x1, stride 1, no padding) whose tile lies inside C reads its fragments straight from
// the weights and the input for every whole 16-step of k, with no staging and no barrier. Everything else (windows, edge tiles, the ragged k tail)
// stages the weight tile and the gathered input tile through workgroup memory first.
//
// The staged loop is software pipelined: the global loads of the next k step are issued before the matrix products of the current one, into
// registers, and written to workgroup memory afterwards, so the latency of the gather (the slow part: one scattered load per element) overlaps
// the products instead of stalling every step. Float16 tiles are small enough to be double-buffered, which leaves one barrier per step.
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

// SK is the k extent staged through workgroup memory per barrier pair. Both element types stage 16 deep: 32 deep measured slower for halves on an Apple M2
// (ResNet-50 b8 74 against 65 ms); SK = 32 still builds for experiments.
#if defined(CONV_F16)
typedef float16_t Elem;
#ifndef SK
#define SK 16
#endif
#ifndef DOUBLE_BUFFER
#define DOUBLE_BUFFER 0
#endif
#else
typedef float Elem;
#define SK 16
#define DOUBLE_BUFFER 0
#endif

#if defined(CONV_F16) && !defined(SPLIT)
typedef float16_t OutElem;
#else
typedef float OutElem;
#endif

[[vk::binding(0, 0)]] StructuredBuffer<Elem> ConvX;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> ConvW;
[[vk::binding(2, 0)]] RWStructuredBuffer<OutElem> ConvY;
[[vk::binding(3, 0)]] StructuredBuffer<Elem> ConvBias;
[[vk::binding(4, 0)]] StructuredBuffer<Elem> ConvRes;

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
#define B_PER_THREAD (SK / 4)
#else
#define BN 64
#define B_STRIDE 72
#define SG_COLS 32
#define B_ROW_STEP 2
#define B_PER_THREAD (SK / 2)
#endif
#define A_PER_THREAD (SK / 2)
#if SK == 32
#define A_SHIFT 5
#define A_MASK 31u
#else
#define A_SHIFT 4
#define A_MASK 15u
#endif

// Rows are padded so fragment loads stay aligned and spread over the banks (as in GemmCoop.hlsli).
#define A_STRIDE (SK + 8)
#define A_SIZE (BM * A_STRIDE)
#define B_SIZE (SK * B_STRIDE)

groupshared Elem TileA[(DOUBLE_BUFFER + 1) * A_SIZE];
groupshared Elem TileB[(DOUBLE_BUFFER + 1) * B_SIZE];
groupshared float TileC[4 * 256];

typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

#if defined(CONV_F16)
// Float16 inputs keep one chain (the products are exact in float and the hardware sums each step before it reaches the accumulator): the second
// accumulator of float32 would cost more than the rounding it saves, which the input rounding dominates anyway.
#undef COOP_STEP_DONE
#undef COOP_FOLD_ALL
#define COOP_STEP_DONE
#define COOP_FOLD_ALL
#define part00 acc00
#define part10 acc10
#define part01 acc01
#define part11 acc11
#endif

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
        {                           \
            float v = TileC[sg * 256 + ei];                                                           \
            uint co = channel0 + gm;                                                                  \
            if ((P.Flags & 1u) != 0u) { v += (float)ConvBias[P.OffBias + co]; }                       \
            if ((P.Flags & 2u) != 0u) { v += (float)ConvRes[P.OffRes + (image * P.Cout + co) * P.N + gn]; } \
            if (P.Act != 0u) { v = ConvAct(P.Act, v); }                                               \
            ConvY[yBase + gm * P.N + gn] = (OutElem)v;                                                \
        }                                                                                             \
    }                                                                                                 \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void ConvCoop(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    bool pointwise = (P.Flags & 4u) != 0u;
    // The last pixel tile of a pointwise layer whose pixel count is not a multiple of the tile is moved back to end at the last pixel, so it is a whole
    // tile and takes the direct-load path; it recomputes pixels the tile before it also writes, with the same inputs in the same order, so the stores agree.
    if (pointwise && P.N >= BN && n0 + BN > P.N)
    {
        n0 = P.N - BN;
    }

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
    uint taps = P.KH * P.KW;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint sgRow = sg >> 1;
    uint sgCol = sg & 1u;
    uint colTile = sgCol * SG_COLS;

    // The pixel this thread stages into the B tile is the same for every k step: the thread-to-element map is column = tid % BN.
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
#if !defined(CONV_F16)
    MatC part00 = MatC::Zero();
    MatC part10 = MatC::Zero();
#if !defined(NARROW)
    MatC part01 = MatC::Zero();
    MatC part11 = MatC::Zero();
#endif
    uint foldSteps = 0;
#endif

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

    // The window position (input channel, tap within the window) of the first k of the step being staged; it advances by SK per step.
    uint ci0 = 0;
    uint tm0 = 0;
    if (!pointwise)
    {
        ci0 = kStaged / taps;
        tm0 = kStaged - ci0 * taps;
    }

    // The staging registers are named scalars: the Metal output keeps indexed arrays in thread-local memory.
    Elem pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8, pa9, pa10, pa11, pa12, pa13, pa14, pa15;
    Elem pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8, pb9, pb10, pb11, pb12, pb13, pb14, pb15;
    pa0 = pa1 = pa2 = pa3 = pa4 = pa5 = pa6 = pa7 = pa8 = pa9 = pa10 = pa11 = pa12 = pa13 = pa14 = pa15 = (Elem)0.0f;
    pb0 = pb1 = pb2 = pb3 = pb4 = pb5 = pb6 = pb7 = pb8 = pb9 = pb10 = pb11 = pb12 = pb13 = pb14 = pb15 = (Elem)0.0f;

    // One element of the weight tile: row r = idx / SK of this workgroup's 64, column k = idx % SK of the step that starts at K0.
#define LOAD_A(E, DST, K0)                                                                                               \
    {                                                                                                                    \
        uint idx = tid + (E) * THREADS;                                                                                  \
        uint gm = m0 + (idx >> A_SHIFT);                                                                                 \
        uint gk = (K0) + (idx & A_MASK);                                                                                 \
        DST = (Elem)0.0f;                                                                                                \
        if (gm < P.M && gk < kEnd)                                                                                       \
        {                                                                                                                \
            DST = ConvW[wBase + gm * P.K + gk];                                                                          \
        }                                                                                                                \
    }

    // One element of the gathered input tile at tap position (ci, kh, kw), then the next row this thread stages.
#define LOAD_B(DST)                                                                                                      \
    {                                                                                                                    \
        DST = (Elem)0.0f;                                                                                                \
        if (pixelValid && gk < kEnd)                                                                                     \
        {                                                                                                                \
            if (pointwise)                                                                                               \
            {                                                                                                            \
                DST = ConvX[xBase + gk * P.N + gn];                                                                      \
            }                                                                                                            \
            else                                                                                                         \
            {                                                                                                            \
                int ih = ih0 + (int)(kh * P.DH);                                                                         \
                int iw = iw0 + (int)(kw * P.DW);                                                                         \
                if ((uint)ih < P.H && (uint)iw < P.W)                                                                    \
                {                                                                                                        \
                    DST = ConvX[xBase + (ci * P.H + (uint)ih) * P.W + (uint)iw];                                         \
                }                                                                                                        \
            }                                                                                                            \
        }                                                                                                                \
        gk += B_ROW_STEP;                                                                                                \
        NextTap(ci, kh, kw);                                                                                             \
        NextTap(ci, kh, kw);                                                                                             \
        if (B_ROW_STEP == 4) { NextTap(ci, kh, kw); NextTap(ci, kh, kw); }                                               \
    }

#define LOAD_A8(K0)  LOAD_A(0, pa0, K0) LOAD_A(1, pa1, K0) LOAD_A(2, pa2, K0) LOAD_A(3, pa3, K0) LOAD_A(4, pa4, K0) LOAD_A(5, pa5, K0) LOAD_A(6, pa6, K0) LOAD_A(7, pa7, K0)
#define LOAD_A16(K0) LOAD_A8(K0) LOAD_A(8, pa8, K0) LOAD_A(9, pa9, K0) LOAD_A(10, pa10, K0) LOAD_A(11, pa11, K0) LOAD_A(12, pa12, K0) LOAD_A(13, pa13, K0) LOAD_A(14, pa14, K0) LOAD_A(15, pa15, K0)
#define LOAD_B4 LOAD_B(pb0) LOAD_B(pb1) LOAD_B(pb2) LOAD_B(pb3)
#define LOAD_B8 LOAD_B4 LOAD_B(pb4) LOAD_B(pb5) LOAD_B(pb6) LOAD_B(pb7)
#define LOAD_B16 LOAD_B8 LOAD_B(pb8) LOAD_B(pb9) LOAD_B(pb10) LOAD_B(pb11) LOAD_B(pb12) LOAD_B(pb13) LOAD_B(pb14) LOAD_B(pb15)
#if A_PER_THREAD == 16
#define LOAD_A_ALL(K0) LOAD_A16(K0)
#else
#define LOAD_A_ALL(K0) LOAD_A8(K0)
#endif
#if B_PER_THREAD == 16
#define LOAD_B_ALL LOAD_B16
#elif B_PER_THREAD == 8
#define LOAD_B_ALL LOAD_B8
#else
#define LOAD_B_ALL LOAD_B4
#endif

    // Loads the step that starts at k0 into the registers (and moves the window position on by SK).
#define LOAD_STEP(K0)                                                                                                    \
    {                                                                                                                    \
        LOAD_A_ALL(K0)                                                                                                   \
        uint gk = (K0) + bk0;                                                                                            \
        uint ci = 0;                                                                                                     \
        uint kh = 0;                                                                                                     \
        uint kw = 0;                                                                                                     \
        if (!pointwise)                                                                                                  \
        {                                                                                                                \
            ci = ci0;                                                                                                    \
            uint t = tm0 + bk0;                                                                                          \
            while (t >= taps) { t -= taps; ci++; }                                                                       \
            kh = t / P.KW;                                                                                               \
            kw = t - kh * P.KW;                                                                                          \
        }                                                                                                                \
        LOAD_B_ALL                                                                                                       \
        if (taps == 1u) { ci0 += SK; }                                                                                   \
        else { tm0 += SK; while (tm0 >= taps) { tm0 -= taps; ci0++; } }                                                  \
    }

    // Writes the registers into the workgroup tiles of buffer BUF.
#define STORE_A(E, SRC, BUF)                                                                                             \
    TileA[(BUF) * A_SIZE + ((tid + (E) * THREADS) >> A_SHIFT) * A_STRIDE + ((tid + (E) * THREADS) & A_MASK)] = SRC;
#define STORE_B(F, SRC, BUF)                                                                                             \
    TileB[(BUF) * B_SIZE + (bk0 + (F) * B_ROW_STEP) * B_STRIDE + bc] = SRC;
#define STORE_A8(BUF)  STORE_A(0, pa0, BUF) STORE_A(1, pa1, BUF) STORE_A(2, pa2, BUF) STORE_A(3, pa3, BUF) STORE_A(4, pa4, BUF) STORE_A(5, pa5, BUF) STORE_A(6, pa6, BUF) STORE_A(7, pa7, BUF)
#define STORE_A16(BUF) STORE_A8(BUF) STORE_A(8, pa8, BUF) STORE_A(9, pa9, BUF) STORE_A(10, pa10, BUF) STORE_A(11, pa11, BUF) STORE_A(12, pa12, BUF) STORE_A(13, pa13, BUF) STORE_A(14, pa14, BUF) STORE_A(15, pa15, BUF)
#define STORE_B4(BUF) STORE_B(0, pb0, BUF) STORE_B(1, pb1, BUF) STORE_B(2, pb2, BUF) STORE_B(3, pb3, BUF)
#define STORE_B8(BUF) STORE_B4(BUF) STORE_B(4, pb4, BUF) STORE_B(5, pb5, BUF) STORE_B(6, pb6, BUF) STORE_B(7, pb7, BUF)
#define STORE_B16(BUF) STORE_B8(BUF) STORE_B(8, pb8, BUF) STORE_B(9, pb9, BUF) STORE_B(10, pb10, BUF) STORE_B(11, pb11, BUF) STORE_B(12, pb12, BUF) STORE_B(13, pb13, BUF) STORE_B(14, pb14, BUF) STORE_B(15, pb15, BUF)
#if A_PER_THREAD == 16
#define STORE_A_ALL(BUF) STORE_A16(BUF)
#else
#define STORE_A_ALL(BUF) STORE_A8(BUF)
#endif
#if B_PER_THREAD == 16
#define STORE_B_ALL(BUF) STORE_B16(BUF)
#elif B_PER_THREAD == 8
#define STORE_B_ALL(BUF) STORE_B8(BUF)
#else
#define STORE_B_ALL(BUF) STORE_B4(BUF)
#endif

    // The products of one 16-deep slice of the staged step (K16 = 0 or 16) from buffer BUF.
#define MMA_SLICE(BUF, K16)                                                                                              \
    {                                                                                                                    \
        MatA a0;                                                                                                         \
        MatA a1;                                                                                                         \
        MatB b0;                                                                                                         \
        a0.Load(TileA, (BUF) * A_SIZE + (sgRow * 32 + 0) * A_STRIDE + (K16), A_STRIDE, MatrixLayout::RowMajor);          \
        a1.Load(TileA, (BUF) * A_SIZE + (sgRow * 32 + 16) * A_STRIDE + (K16), A_STRIDE, MatrixLayout::RowMajor);         \
        b0.Load(TileB, (BUF) * B_SIZE + (K16) * B_STRIDE + colTile, B_STRIDE, MatrixLayout::RowMajor);                   \
        part00 = MatC::MulAdd(a0, b0, part00);                                                                           \
        part10 = MatC::MulAdd(a1, b0, part10);                                                                           \
        MATRIX_SLICE_WIDE(BUF, K16)                                                                                      \
        COOP_STEP_DONE                                                                                                   \
    }
#if defined(NARROW)
#define MATRIX_SLICE_WIDE(BUF, K16)
#else
#define MATRIX_SLICE_WIDE(BUF, K16)                                                                                      \
        MatB b1;                                                                                                         \
        b1.Load(TileB, (BUF) * B_SIZE + (K16) * B_STRIDE + colTile + 16, B_STRIDE, MatrixLayout::RowMajor);              \
        part01 = MatC::MulAdd(a0, b1, part01);                                                                           \
        part11 = MatC::MulAdd(a1, b1, part11);
#endif

#if !defined(CONV_F16) && !defined(NARROW)
    // Float32 stages straight into the tiles: with a 4-byte element the register staging below costs occupancy in the wide tile and measured slower on an Apple M2 (ResNet-50 b8 about 127 against 110 ms); the narrow tile of the late layers still gains from it (b1 14.8 against 16.9 ms).
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
#else
    uint cur = 0;
    uint k0 = kStaged;
    for (; k0 < kEnd; k0 += SK)
    {
        LOAD_STEP(k0)
        STORE_A_ALL(cur)
        STORE_B_ALL(cur)
        GroupMemoryBarrierWithGroupSync();

        MMA_SLICE(cur, 0)
#if SK == 32
        MMA_SLICE(cur, 16)
#endif

#if DOUBLE_BUFFER
        cur ^= 1u;
#else
        GroupMemoryBarrierWithGroupSync();
#endif
    }
#endif

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
