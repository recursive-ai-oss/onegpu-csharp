// 8-bit integer matrix product for few rows (built for Metal; plain integer arithmetic, no matrix hardware): C[m,n] = sum over k of (A[m,k] - zeroA) * (B[k,n] - zeroB),
// the exact int32 sum of ONNX MatMulInteger with the operands, zero points, epilogue and flags of Int8GemmKernel.hlsl, for B stored K x N (not transposed).
//
// A product with a handful of rows (one token of a decoder, a few of a batch) reads every weight once and does almost no arithmetic per byte, so it is limited by
// how fast the weights stream in. The tiled kernels waste it: they hold 63 of 64 rows of each tile idle and walk k in 16-wide steps with two barriers each. This kernel
// gives a workgroup 32 output columns (and 4 rows of C): eight lanes cover the columns four at a time, so a lane reads one 32-bit word of four adjacent columns per k,
// and sixteen lane groups split k between them, each reading four consecutive k at a time (four independent loads in flight). The sixteen partial sums of an output
// are added through workgroup memory. Integer addition is exact and associative, so splitting k changes no bit of the result (with 32-bit wrap-around, like the definition).
//
// Grid: x = tiles of 32 columns, y = tiles of 4 rows, z = batch.

#include "Common.hlsl"

struct Int8GemmParams
{
    uint M;
    uint N;
    uint K;
    uint OffA;
    uint OffB;
    uint OffC;
    uint StrideA;
    uint StrideB;
    uint StrideC;
    uint Lda;
    uint Ldb;
    uint Flags;     // bit 0 A is int8, bit 1 B is int8, bit 3 zero point of A present, bit 4 one per row, bit 5 of B present, bit 6 one per column,
                    // bit 7 every row of A (offset, stride, Lda) starts on a word boundary, bit 8 the same for B (bit 2, B stored N x K, is never set here)
    uint OffZA;
    uint OffZB;
    uint Epi;       // see Int8GemmEpilogue.hlsli
    uint OffBias;
    uint OffScale;
    uint OffFBias;
    uint GridX;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> C;
[[vk::binding(3, 0)]] StructuredBuffer<uint> ZA;
[[vk::binding(4, 0)]] StructuredBuffer<uint> ZB;
[[vk::binding(5, 0)]] StructuredBuffer<uint> EBias;
[[vk::binding(6, 0)]] StructuredBuffer<uint> EScale;
[[vk::binding(7, 0)]] StructuredBuffer<uint> EFBias;

// Declared after the buffers so that Metal numbers the push constants last, as the other matrix kernels do.
[[vk::push_constant]] ConstantBuffer<Int8GemmParams> P;

#define EPILOGUE_BINDINGS_DECLARED
#include "Int8GemmEpilogue.hlsli"

#define TN 32
#define MB 4
#define KS 16
#define THREADS 128

groupshared int Red[KS * MB * TN];

int Byte(StructuredBuffer<uint> buf, uint i, bool isSigned)
{
    uint b = (buf[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
    return isSigned ? ((int)(b << 24)) >> 24 : (int)b;
}

// Up to four consecutive bytes starting at byte index i (valid says how many exist), a whole word when the row is aligned and four are valid.
uint Word4(StructuredBuffer<uint> buf, uint i, uint valid, bool aligned)
{
    if (aligned && valid >= 4u)
    {
        return buf[i >> 2];
    }

    uint w = 0;
    for (uint e = 0; e < 4u; e++)
    {
        if (e < valid)
        {
            w |= ((buf[(i + e) >> 2] >> (((i + e) & 3u) << 3)) & 0xFFu) << (e * 8u);
        }
    }

    return w;
}

int Lane(uint word, uint e, bool isSigned)
{
    uint b = (word >> (e * 8u)) & 0xFFu;
    return isSigned ? ((int)(b << 24)) >> 24 : (int)b;
}

// One k row R (0 to 3) of the four-deep group: the four columns of the B word against the R-th byte of each row's A word.
#define KROW(R, BW)                                                                                     \
    if ((R) < valid)                                                                                    \
    {                                                                                                   \
        int bx0 = Lane(BW, 0, bSigned) - zb0;                                                           \
        int bx1 = Lane(BW, 1, bSigned) - zb1;                                                           \
        int bx2 = Lane(BW, 2, bSigned) - zb2;                                                           \
        int bx3 = Lane(BW, 3, bSigned) - zb3;                                                           \
        int ax0 = Lane(aw0, R, aSigned) - za0;                                                          \
        int ax1 = Lane(aw1, R, aSigned) - za1;                                                          \
        int ax2 = Lane(aw2, R, aSigned) - za2;                                                          \
        int ax3 = Lane(aw3, R, aSigned) - za3;                                                          \
        c00 += ax0 * bx0; c01 += ax0 * bx1; c02 += ax0 * bx2; c03 += ax0 * bx3;                         \
        c10 += ax1 * bx0; c11 += ax1 * bx1; c12 += ax1 * bx2; c13 += ax1 * bx3;                         \
        c20 += ax2 * bx0; c21 += ax2 * bx1; c22 += ax2 * bx2; c23 += ax2 * bx3;                         \
        c30 += ax3 * bx0; c31 += ax3 * bx1; c32 += ax3 * bx2; c33 += ax3 * bx3;                         \
    }

#define PUT(M, J, V) Red[(kl * MB + (M)) * TN + cq * 4u + (J)] = (V);

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void Int8GemmRows(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint n0 = group.x * TN;
    uint m0 = group.y * MB;
    uint batch = group.z;
    uint baseA = P.OffA + batch * P.StrideA;
    uint baseB = P.OffB + batch * P.StrideB;
    uint baseC = P.OffC + batch * P.StrideC;
    bool aSigned = (P.Flags & 1u) != 0u;
    bool bSigned = (P.Flags & 2u) != 0u;
    bool hasZA = (P.Flags & 8u) != 0u;
    bool rowZA = (P.Flags & 16u) != 0u;
    bool hasZB = (P.Flags & 32u) != 0u;
    bool colZB = (P.Flags & 64u) != 0u;
    bool alignedA = (P.Flags & 128u) != 0u;
    bool alignedB = (P.Flags & 256u) != 0u;

    uint cq = tid & 7u;
    uint kl = tid >> 3;
    uint col0 = n0 + cq * 4u;

    int zaScalar = hasZA && !rowZA ? Byte(ZA, P.OffZA, aSigned) : 0;
    int zbScalar = hasZB && !colZB ? Byte(ZB, P.OffZB, bSigned) : 0;
    int za0 = (rowZA && m0 + 0u < P.M) ? Byte(ZA, P.OffZA + m0 + 0u, aSigned) : zaScalar;
    int za1 = (rowZA && m0 + 1u < P.M) ? Byte(ZA, P.OffZA + m0 + 1u, aSigned) : zaScalar;
    int za2 = (rowZA && m0 + 2u < P.M) ? Byte(ZA, P.OffZA + m0 + 2u, aSigned) : zaScalar;
    int za3 = (rowZA && m0 + 3u < P.M) ? Byte(ZA, P.OffZA + m0 + 3u, aSigned) : zaScalar;
    int zb0 = (colZB && col0 + 0u < P.N) ? Byte(ZB, P.OffZB + col0 + 0u, bSigned) : zbScalar;
    int zb1 = (colZB && col0 + 1u < P.N) ? Byte(ZB, P.OffZB + col0 + 1u, bSigned) : zbScalar;
    int zb2 = (colZB && col0 + 2u < P.N) ? Byte(ZB, P.OffZB + col0 + 2u, bSigned) : zbScalar;
    int zb3 = (colZB && col0 + 3u < P.N) ? Byte(ZB, P.OffZB + col0 + 3u, bSigned) : zbScalar;

    int c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    int c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    int c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    int c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    uint colsValid = col0 < P.N ? P.N - col0 : 0u;

    // The general step: up to four k rows from k on, whatever the alignment and the edges (valid says how many k rows exist).
#define GENERAL_STEP(KK)                                                                                                 \
    {                                                                                                                    \
        uint kk0 = (KK);                                                                                                   \
        uint valid = min(4u, P.K - kk0);                                                                                   \
        uint bw0 = 0, bw1 = 0, bw2 = 0, bw3 = 0;                                                                         \
        if (colsValid != 0u)                                                                                             \
        {                                                                                                                \
            uint rowB = baseB + kk0 * P.Ldb + col0;                                                                        \
            bw0 = Word4(B, rowB, colsValid, alignedB);                                                                   \
            if (valid > 1u) { bw1 = Word4(B, rowB + P.Ldb, colsValid, alignedB); }                                       \
            if (valid > 2u) { bw2 = Word4(B, rowB + 2u * P.Ldb, colsValid, alignedB); }                                  \
            if (valid > 3u) { bw3 = Word4(B, rowB + 3u * P.Ldb, colsValid, alignedB); }                                  \
        }                                                                                                                \
        uint aw0 = 0, aw1 = 0, aw2 = 0, aw3 = 0;                                                                         \
        if (m0 + 0u < P.M) { aw0 = Word4(A, baseA + (m0 + 0u) * P.Lda + kk0, valid, alignedA); }                           \
        if (m0 + 1u < P.M) { aw1 = Word4(A, baseA + (m0 + 1u) * P.Lda + kk0, valid, alignedA); }                           \
        if (m0 + 2u < P.M) { aw2 = Word4(A, baseA + (m0 + 2u) * P.Lda + kk0, valid, alignedA); }                           \
        if (m0 + 3u < P.M) { aw3 = Word4(A, baseA + (m0 + 3u) * P.Lda + kk0, valid, alignedA); }                           \
        KROW(0, bw0) KROW(1, bw1) KROW(2, bw2) KROW(3, bw3)                                                              \
    }

    // The common step: four whole k rows, every row of A and B word aligned, all four columns and rows present. Straight-line code, so the eight loads
    // are in flight together instead of one behind each branch; the loads of two steps are issued before either is used.
#define FAST_LOADS(KK, BW0, BW1, BW2, BW3, AW0, AW1, AW2, AW3)                                                          \
        uint BW0 = B[(baseB + (KK) * P.Ldb + col0) >> 2];                                                                \
        uint BW1 = B[(baseB + ((KK) + 1u) * P.Ldb + col0) >> 2];                                                         \
        uint BW2 = B[(baseB + ((KK) + 2u) * P.Ldb + col0) >> 2];                                                         \
        uint BW3 = B[(baseB + ((KK) + 3u) * P.Ldb + col0) >> 2];                                                         \
        uint AW0 = A[(baseA + (m0 + 0u) * P.Lda + (KK)) >> 2];                                                           \
        uint AW1 = A[(baseA + (m0 + 1u) * P.Lda + (KK)) >> 2];                                                           \
        uint AW2 = A[(baseA + (m0 + 2u) * P.Lda + (KK)) >> 2];                                                           \
        uint AW3 = A[(baseA + (m0 + 3u) * P.Lda + (KK)) >> 2];

#define FAST_USE(BW0, BW1, BW2, BW3, AW0, AW1, AW2, AW3)                                                                 \
    {                                                                                                                    \
        uint valid = 4u;                                                                                                 \
        uint aw0 = AW0, aw1 = AW1, aw2 = AW2, aw3 = AW3;                                                                 \
        KROW(0, BW0) KROW(1, BW1) KROW(2, BW2) KROW(3, BW3)                                                              \
    }

    uint kFull = P.K & ~3u;
    bool fast = alignedA && alignedB && colsValid >= 4u && m0 + MB <= P.M;
    uint k = kl * 4u;
    if (fast)
    {
        for (; k + KS * 4u + 4u <= kFull; k += 2u * KS * 4u)
        {
            FAST_LOADS(k, bw0, bw1, bw2, bw3, aw0, aw1, aw2, aw3)
            FAST_LOADS(k + KS * 4u, cw0, cw1, cw2, cw3, ew0, ew1, ew2, ew3)
            FAST_USE(bw0, bw1, bw2, bw3, aw0, aw1, aw2, aw3)
            FAST_USE(cw0, cw1, cw2, cw3, ew0, ew1, ew2, ew3)
        }

        if (k + 4u <= kFull)
        {
            FAST_LOADS(k, bw0, bw1, bw2, bw3, aw0, aw1, aw2, aw3)
            FAST_USE(bw0, bw1, bw2, bw3, aw0, aw1, aw2, aw3)
            k += KS * 4u;
        }

        // The last group of fewer than four k rows belongs to the lane group that the stride would give it.
        if (kFull < P.K && ((kFull >> 2) % KS) == kl)
        {
            GENERAL_STEP(kFull)
        }
    }
    else
    {
        for (; k < P.K; k += KS * 4u)
        {
            GENERAL_STEP(k)
        }
    }

    PUT(0, 0, c00) PUT(0, 1, c01) PUT(0, 2, c02) PUT(0, 3, c03)
    PUT(1, 0, c10) PUT(1, 1, c11) PUT(1, 2, c12) PUT(1, 3, c13)
    PUT(2, 0, c20) PUT(2, 1, c21) PUT(2, 2, c22) PUT(2, 3, c23)
    PUT(3, 0, c30) PUT(3, 1, c31) PUT(3, 2, c32) PUT(3, 3, c33)
    GroupMemoryBarrierWithGroupSync();

    // One output per thread: row tid / 32 of the four, column tid % 32 of the tile; the sixteen partial sums of k added in a fixed order.
    uint m = tid >> 5;
    uint col = tid & 31u;
    int total = 0;
    [unroll] for (uint s = 0; s < KS; s++)
    {
        total += Red[(s * MB + m) * TN + col];
    }

    if (m0 + m < P.M && n0 + col < P.N)
    {
        C[baseC + (m0 + m) * P.N + n0 + col] = StoreValue(total, n0 + col);
    }
}
