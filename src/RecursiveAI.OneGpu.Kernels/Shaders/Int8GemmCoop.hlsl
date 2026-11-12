// 8-bit integer matrix product on simdgroup matrices (built for Metal): C[m,n] = sum over k of (A[m,k] - zeroA) * (B[k,n] - zeroB), the exact int32
// sum of ONNX MatMulInteger, with the operands, zero points, epilogue and flags of Int8GemmKernel.hlsl.
//
// Why the matrix units give the exact integer sum. The tiles are staged as float16 after the zero points are subtracted: a value in -255..255
// (a byte minus a zero point) is an integer below 2^11, which float16 holds exactly. A matrix product of float16 operands multiplies exactly
// (two 11-bit significands give at most 22 bits, which float32 holds) and accumulates in float32. A float32 sum of integers is exact as long as
// every partial sum stays below 2^24 in magnitude, whatever order the hardware adds in. So a chunk of k terms whose products are at most
// maxA * maxB in magnitude is exact when chunk * maxA * maxB <= 2^24: the host puts the chunk length, in steps of 16, in bits 16 to 22 of Flags
// (16 steps = 256 terms for two operands of up to 255; 32 or 64 steps when an operand is a plain int8 with no zero point, |value| <= 128).
// After each chunk the float32 sums are converted to int32 (exact, they are integers below 2^24) and added to int32 registers, which wrap like
// two's complement exactly as the definition does (Jacob et al., 2018, section 2.3). The result is the same integer for every order, so the
// kernel is the same for both modes.
//
// The int32 totals leave through workgroup memory split in two float32 halves (value >> 12 and value & 4095, both exact in a float), so the
// register-to-position map of the matrix type never has to be known. One 64 x 64 tile of C per workgroup, four 32-wide subgroups arranged 2 x 2,
// each owning 32 x 32 as four 16 x 16 accumulators, k in steps of 16. The host pins the subgroup size to 32 and runs full subgroups.

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
    uint Flags;     // bit 0 A is int8, bit 1 B is int8, bit 2 B stored N x K, bit 3 zero point of A present, bit 4 one per row, bit 5 of B present, bit 6 one per column,
                    // bit 7 every row of A (offset, stride, Lda) starts on a word boundary, bit 8 the same for B,
                    // bit 9 the tiles are numbered row-major over a GridX-wide grid instead of x = column tile, y = row tile,
                    // bits 16 to 22 how many 16-wide steps of k add up exactly in a float32 (see above)
    uint OffZA;
    uint OffZB;
    uint Epi;       // see Int8GemmEpilogue.hlsli
    uint OffBias;
    uint OffScale;
    uint OffFBias;
    uint GridX;     // workgroups along x when Flags bit 9 folds the tile grid into one dimension
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

#define BM 64
#define BN 64
#define BK 16
#define THREADS 128

// Rows are padded so fragment loads stay aligned and spread over the banks.
#define A_STRIDE 24
#define B_STRIDE 72

groupshared float16_t TileA[BM * A_STRIDE];
groupshared float16_t TileB[BK * B_STRIDE];
groupshared float TileHi[4 * 256];
groupshared float TileLo[4 * 256];

typedef CooperativeMatrix<float16_t, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<float16_t, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

int Byte(StructuredBuffer<uint> buf, uint i, bool isSigned)
{
    uint b = (buf[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
    return isSigned ? ((int)(b << 24)) >> 24 : (int)b;
}

// Four consecutive bytes starting at byte index i, a whole word when the row is aligned and four bytes are valid.
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

// The float32 sums of a chunk join the int32 totals (exact: integers below 2^24), and the sums restart from zero.
#define FOLD1(PART, T)                                                                                                    \
    T##_0 += (int)PART[0]; T##_1 += (int)PART[1]; T##_2 += (int)PART[2]; T##_3 += (int)PART[3];                           \
    T##_4 += (int)PART[4]; T##_5 += (int)PART[5]; T##_6 += (int)PART[6]; T##_7 += (int)PART[7];                           \
    PART[0] = 0.0; PART[1] = 0.0; PART[2] = 0.0; PART[3] = 0.0; PART[4] = 0.0; PART[5] = 0.0; PART[6] = 0.0; PART[7] = 0.0;

// An accumulator's totals through workgroup memory, as two exact float32 halves, then stored with the epilogue.
#define STORE_TOTAL(T, ROW0, COL0)                                                                                        \
    {                                                                                                                     \
        MatC hi = MatC::Zero();                                                                                           \
        MatC lo = MatC::Zero();                                                                                           \
        hi[0] = (float)(T##_0 >> 12); hi[1] = (float)(T##_1 >> 12); hi[2] = (float)(T##_2 >> 12); hi[3] = (float)(T##_3 >> 12); \
        hi[4] = (float)(T##_4 >> 12); hi[5] = (float)(T##_5 >> 12); hi[6] = (float)(T##_6 >> 12); hi[7] = (float)(T##_7 >> 12); \
        lo[0] = (float)(T##_0 & 4095); lo[1] = (float)(T##_1 & 4095); lo[2] = (float)(T##_2 & 4095); lo[3] = (float)(T##_3 & 4095); \
        lo[4] = (float)(T##_4 & 4095); lo[5] = (float)(T##_5 & 4095); lo[6] = (float)(T##_6 & 4095); lo[7] = (float)(T##_7 & 4095); \
        hi.Store(TileHi, sg * 256, 16, MatrixLayout::RowMajor);                                                           \
        lo.Store(TileLo, sg * 256, 16, MatrixLayout::RowMajor);                                                           \
        GroupMemoryBarrierWithGroupSync();                                                                                \
        [unroll] for (uint e = 0; e < 8; e++)                                                                             \
        {                                                                                                                 \
            uint ei = lane + 32 * e;                                                                                      \
            uint gm = (ROW0) + (ei >> 4);                                                                                 \
            uint gn = (COL0) + (ei & 15u);                                                                                \
            if (gm < P.M && gn < P.N)                                                                                     \
            {                                                                                                             \
                int total = (((int)TileHi[sg * 256 + ei]) << 12) + (int)TileLo[sg * 256 + ei];                           \
                C[baseC + gm * P.N + gn] = StoreValue(total, gn);                                                         \
            }                                                                                                             \
        }                                                                                                                 \
        GroupMemoryBarrierWithGroupSync();                                                                                \
    }

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void Int8GemmMatrix(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    if ((P.Flags & 512u) != 0u)
    {
        // A product with more tiles than one grid axis can hold: the host lays the tiles out over x and y, numbered row-major.
        // Which tile a workgroup owns depends only on its id, and surplus workgroups of the last row leave before touching anything.
        uint tilesN = (P.N + BN - 1u) / BN;
        uint flat = group.y * P.GridX + group.x;
        m0 = (flat / tilesN) * BM;
        n0 = (flat % tilesN) * BN;
        if (m0 >= P.M)
        {
            return;
        }
    }

    uint batch = group.z;
    uint baseA = P.OffA + batch * P.StrideA;
    uint baseB = P.OffB + batch * P.StrideB;
    uint baseC = P.OffC + batch * P.StrideC;
    bool aSigned = (P.Flags & 1u) != 0u;
    bool bSigned = (P.Flags & 2u) != 0u;
    bool transB = (P.Flags & 4u) != 0u;
    bool hasZA = (P.Flags & 8u) != 0u;
    bool rowZA = (P.Flags & 16u) != 0u;
    bool hasZB = (P.Flags & 32u) != 0u;
    bool colZB = (P.Flags & 64u) != 0u;
    bool alignedA = (P.Flags & 128u) != 0u;
    bool alignedB = (P.Flags & 256u) != 0u;
    uint chunkLimit = max(1u, (P.Flags >> 16) & 127u);
    int zaScalar = hasZA && !rowZA ? Byte(ZA, P.OffZA, aSigned) : 0;
    int zbScalar = hasZB && !colZB ? Byte(ZB, P.OffZB, bSigned) : 0;

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint sgRow = sg >> 1;
    uint sgCol = sg & 1u;
    uint rowBase = m0 + sgRow * 32;
    uint colBase = n0 + sgCol * 32;
    uint colTile = sgCol * 32;

    MatC part00 = MatC::Zero();
    MatC part10 = MatC::Zero();
    MatC part01 = MatC::Zero();
    MatC part11 = MatC::Zero();
    int a00_0 = 0, a00_1 = 0, a00_2 = 0, a00_3 = 0, a00_4 = 0, a00_5 = 0, a00_6 = 0, a00_7 = 0;
    int a10_0 = 0, a10_1 = 0, a10_2 = 0, a10_3 = 0, a10_4 = 0, a10_5 = 0, a10_6 = 0, a10_7 = 0;
    int a01_0 = 0, a01_1 = 0, a01_2 = 0, a01_3 = 0, a01_4 = 0, a01_5 = 0, a01_6 = 0, a01_7 = 0;
    int a11_0 = 0, a11_1 = 0, a11_2 = 0, a11_3 = 0, a11_4 = 0, a11_5 = 0, a11_6 = 0, a11_7 = 0;
    uint chunkSteps = 0;

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        // A tile: 64 rows x 4 words of four consecutive k; two words per thread.
        [unroll] for (uint e = 0; e < 2u; e++)
        {
            uint w = tid + e * THREADS;
            uint r = w >> 2;
            uint kq = (w & 3u) << 2;
            uint gm = m0 + r;
            uint gk = k0 + kq;
            uint word = 0;
            int z = 0;
            if (gm < P.M && gk < P.K)
            {
                word = Word4(A, baseA + gm * P.Lda + gk, P.K - gk, alignedA);
                z = rowZA ? Byte(ZA, P.OffZA + gm, aSigned) : zaScalar;
            }

            [unroll] for (uint j = 0; j < 4u; j++)
            {
                TileA[r * A_STRIDE + kq + j] = (float16_t)((float)((gm < P.M && gk + j < P.K) ? Lane(word, j, aSigned) - z : 0));
            }
        }

        // B tile: 16 k x 64 columns. Not transposed: a word is four consecutive columns of one k; transposed: four consecutive k of one column.
        [unroll] for (uint e = 0; e < 2u; e++)
        {
            uint w = tid + e * THREADS;
            if (transB)
            {
                uint c = w >> 2;
                uint kq = (w & 3u) << 2;
                uint gn = n0 + c;
                uint gk = k0 + kq;
                uint word = 0;
                int z = 0;
                if (gn < P.N && gk < P.K)
                {
                    word = Word4(B, baseB + gn * P.Ldb + gk, P.K - gk, alignedB);
                    z = colZB ? Byte(ZB, P.OffZB + gn, bSigned) : zbScalar;
                }

                [unroll] for (uint j = 0; j < 4u; j++)
                {
                    TileB[(kq + j) * B_STRIDE + c] = (float16_t)((float)((gn < P.N && gk + j < P.K) ? Lane(word, j, bSigned) - z : 0));
                }
            }
            else
            {
                uint k = w >> 4;
                uint cq = (w & 15u) << 2;
                uint gn = n0 + cq;
                uint gk = k0 + k;
                uint word = 0;
                if (gn < P.N && gk < P.K)
                {
                    word = Word4(B, baseB + gk * P.Ldb + gn, P.N - gn, alignedB);
                }

                [unroll] for (uint j = 0; j < 4u; j++)
                {
                    int z = (colZB && gn + j < P.N) ? Byte(ZB, P.OffZB + gn + j, bSigned) : zbScalar;
                    TileB[k * B_STRIDE + cq + j] = (float16_t)((float)((gn + j < P.N && gk < P.K) ? Lane(word, j, bSigned) - z : 0));
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        MatA a0;
        MatA a1;
        MatB b0;
        MatB b1;
        a0.Load(TileA, (sgRow * 32 + 0) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        a1.Load(TileA, (sgRow * 32 + 16) * A_STRIDE, A_STRIDE, MatrixLayout::RowMajor);
        b0.Load(TileB, colTile, B_STRIDE, MatrixLayout::RowMajor);
        b1.Load(TileB, colTile + 16, B_STRIDE, MatrixLayout::RowMajor);
        part00 = MatC::MulAdd(a0, b0, part00);
        part10 = MatC::MulAdd(a1, b0, part10);
        part01 = MatC::MulAdd(a0, b1, part01);
        part11 = MatC::MulAdd(a1, b1, part11);

        chunkSteps++;
        if (chunkSteps == chunkLimit)
        {
            FOLD1(part00, a00) FOLD1(part10, a10) FOLD1(part01, a01) FOLD1(part11, a11)
            chunkSteps = 0;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    FOLD1(part00, a00) FOLD1(part10, a10) FOLD1(part01, a01) FOLD1(part11, a11)
    STORE_TOTAL(a00, rowBase + 0, colBase + 0)
    STORE_TOTAL(a10, rowBase + 16, colBase + 0)
    STORE_TOTAL(a01, rowBase + 0, colBase + 16)
    STORE_TOTAL(a11, rowBase + 16, colBase + 16)
}
