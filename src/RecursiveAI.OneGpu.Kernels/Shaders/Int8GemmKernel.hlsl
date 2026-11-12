// 8-bit integer matrix product with 32-bit accumulation: C[m,n] = sum over k of (A[m,k] - zeroA) * (B[k,n] - zeroB), as in ONNX MatMulInteger.
//
// A and B hold uint8 or int8 values (Flags bits 0 and 1 say which), read as bytes out of 32-bit words. The zero points are subtracted
// as the tiles are staged, so the arithmetic is the exact integer sum the definition gives, with 32-bit wrap-around like any
// two's-complement accumulation (Jacob et al., 2018, section 2.3 rearranges the same sum; the two agree modulo 2^32, so the results match bit for bit).
// A zero point is one value (read from the first byte at its offset) or a vector: one per row of A, one per column of B.
// This is the portable variant: bytes are unpacked with integer arithmetic. A packed-dot variant replaces the staging and the inner
// loop where the device has an 8-bit dot product; the sums it gives are the same integers.
//
// Tiles follow the float kernel (64x64 per workgroup, k in steps of 16, a 4x4 register tile per thread). A and a transposed B are staged four
// bytes at a time when their rows start on a word boundary (Flags bits 7 and 8). The products run in float: two values in -255..255 multiply to
// less than 65025, so up to 256 of them add up exactly in a float (every partial sum is an integer below 2^24), and the float units are the
// fastest multipliers on most GPUs. The running sums are folded into int32 every 128 values of k, which keeps the result the exact integer
// sum with the 32-bit wrap-around of two's complement. Every output element is summed by one thread in increasing k; integer addition is
// associative, so no order could change the bits anyway.

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
                    // bit 9 the tiles are numbered row-major over a GridX-wide grid instead of x = column tile, y = row tile
    uint OffZA;
    uint OffZB;
    uint Epi;       // see Int8GemmEpilogue.hlsli
    uint OffBias;
    uint OffScale;
    uint OffFBias;
    uint GridX;     // workgroups along x when Flags bit 9 folds the tile grid into one dimension
};

[[vk::push_constant]] ConstantBuffer<Int8GemmParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> C;
[[vk::binding(3, 0)]] StructuredBuffer<uint> ZA;
[[vk::binding(4, 0)]] StructuredBuffer<uint> ZB;

#include "Int8GemmEpilogue.hlsli"

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TILE_THREADS = 256;

// Products of two values in -255..255 are below 65025, so up to 256 of them add up exactly in a float (all partial sums are integers
// below 2^24); the float units are the fastest multipliers on most GPUs. The running float sums are folded into int32 every 128 values of k.
typedef float TileT;
typedef float AccT;
groupshared TileT As[BK][BM + 1];
groupshared TileT Bs[BK][BN + 1];

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

[shader("compute")]
[numthreads(256, 1, 1)]
void Int8GemmTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
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
    int zaScalar = hasZA && !rowZA ? Byte(ZA, P.OffZA, aSigned) : 0;
    int zbScalar = hasZB && !colZB ? Byte(ZB, P.OffZB, bSigned) : 0;

    uint tx = tid & 15u;
    uint ty = tid >> 4;

    // Sixteen named scalars rather than an array: indexed arrays live in thread-local memory on Metal.
    AccT c00 = 0; AccT c01 = 0; AccT c02 = 0; AccT c03 = 0; AccT c10 = 0; AccT c11 = 0; AccT c12 = 0; AccT c13 = 0; AccT c20 = 0; AccT c21 = 0; AccT c22 = 0; AccT c23 = 0; AccT c30 = 0; AccT c31 = 0; AccT c32 = 0; AccT c33 = 0;
    int i00 = 0; int i01 = 0; int i02 = 0; int i03 = 0; int i10 = 0; int i11 = 0; int i12 = 0; int i13 = 0; int i20 = 0; int i21 = 0; int i22 = 0; int i23 = 0; int i30 = 0; int i31 = 0; int i32 = 0; int i33 = 0;

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        // A: one word (four consecutive k) of one row per thread.
        if (tid < BM * 4u)
        {
            uint r = tid >> 2;
            uint gm = m0 + r;
            uint gk = k0 + ((tid & 3u) << 2);
            uint word = 0;
            int z = 0;
            if (gm < P.M && gk < P.K)
            {
                word = Word4(A, baseA + gm * P.Lda + gk, P.K - gk, alignedA);
                z = rowZA ? Byte(ZA, P.OffZA + gm, aSigned) : zaScalar;
            }

            [unroll] for (uint e = 0; e < 4u; e++)
            {
                As[(tid & 3u) * 4u + e][r] = (TileT)((gm < P.M && gk + e < P.K) ? Lane(word, e, aSigned) - z : 0);
            }
        }

        if (transB)
        {
            if (tid < BN * 4u)
            {
                uint c = tid >> 2;
                uint gn = n0 + c;
                uint gk = k0 + ((tid & 3u) << 2);
                uint word = 0;
                int z = 0;
                if (gn < P.N && gk < P.K)
                {
                    word = Word4(B, baseB + gn * P.Ldb + gk, P.K - gk, alignedB);
                    z = colZB ? Byte(ZB, P.OffZB + gn, bSigned) : zbScalar;
                }

                [unroll] for (uint e = 0; e < 4u; e++)
                {
                    Bs[(tid & 3u) * 4u + e][c] = (TileT)((gn < P.N && gk + e < P.K) ? Lane(word, e, bSigned) - z : 0);
                }
            }
        }
        else
        {
            [unroll] for (uint f = 0; f < (BK * BN) / TILE_THREADS; f++)
            {
                uint idx = tid + f * TILE_THREADS;
                uint c = idx % BN;
                uint k = idx / BN;
                uint gn = n0 + c;
                uint gk = k0 + k;
                int v = 0;
                if (gn < P.N && gk < P.K)
                {
                    int z = colZB ? Byte(ZB, P.OffZB + gn, bSigned) : zbScalar;
                    v = Byte(B, baseB + gk * P.Ldb + gn, bSigned) - z;
                }

                Bs[k][c] = (TileT)v;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < BK; kk++)
        {
            TileT a0 = As[kk][ty + 0];
            TileT a1 = As[kk][ty + 16];
            TileT a2 = As[kk][ty + 32];
            TileT a3 = As[kk][ty + 48];
            TileT b0 = Bs[kk][tx + 0];
            TileT b1 = Bs[kk][tx + 16];
            TileT b2 = Bs[kk][tx + 32];
            TileT b3 = Bs[kk][tx + 48];
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();

        if (((k0 >> 4) & 7u) == 7u)
        {
            i00 += (int)c00; c00 = 0; i01 += (int)c01; c01 = 0; i02 += (int)c02; c02 = 0; i03 += (int)c03; c03 = 0;
            i10 += (int)c10; c10 = 0; i11 += (int)c11; c11 = 0; i12 += (int)c12; c12 = 0; i13 += (int)c13; c13 = 0;
            i20 += (int)c20; c20 = 0; i21 += (int)c21; c21 = 0; i22 += (int)c22; c22 = 0; i23 += (int)c23; c23 = 0;
            i30 += (int)c30; c30 = 0; i31 += (int)c31; c31 = 0; i32 += (int)c32; c32 = 0; i33 += (int)c33; c33 = 0;
        }
    }

    i00 += (int)c00; c00 = 0; i01 += (int)c01; c01 = 0; i02 += (int)c02; c02 = 0; i03 += (int)c03; c03 = 0;
    i10 += (int)c10; c10 = 0; i11 += (int)c11; c11 = 0; i12 += (int)c12; c12 = 0; i13 += (int)c13; c13 = 0;
    i20 += (int)c20; c20 = 0; i21 += (int)c21; c21 = 0; i22 += (int)c22; c22 = 0; i23 += (int)c23; c23 = 0;
    i30 += (int)c30; c30 = 0; i31 += (int)c31; c31 = 0; i32 += (int)c32; c32 = 0; i33 += (int)c33; c33 = 0;

#define STORE(DR, DC, V) \
    { uint row = m0 + ty + DR; uint col = n0 + tx + DC; if (row < P.M && col < P.N) { C[baseC + row * P.N + col] = StoreValue((int)(V), col); } }
    STORE(0, 0, i00) STORE(0, 16, i01) STORE(0, 32, i02) STORE(0, 48, i03)
    STORE(16, 0, i10) STORE(16, 16, i11) STORE(16, 32, i12) STORE(16, 48, i13)
    STORE(32, 0, i20) STORE(32, 16, i21) STORE(32, 32, i22) STORE(32, 48, i23)
    STORE(48, 0, i30) STORE(48, 16, i31) STORE(48, 32, i32) STORE(48, 48, i33)
}
