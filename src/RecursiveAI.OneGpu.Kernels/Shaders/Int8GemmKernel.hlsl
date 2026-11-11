// 8-bit integer matrix product with 32-bit accumulation: C[m,n] = sum over k of (A[m,k] - zeroA) * (B[k,n] - zeroB), as in ONNX MatMulInteger.
//
// A and B hold uint8 or int8 values (Flags bits 0 and 1 say which), read as bytes out of 32-bit words. The zero points are subtracted
// as the tiles are staged, so the arithmetic is the exact integer sum the definition gives, with 32-bit wrap-around like any
// two's-complement accumulation (Jacob et al., 2018, section 2.3 rearranges the same sum; the two agree modulo 2^32, so the results match bit for bit).
// A zero point is one value (read from the first byte at its offset) or a vector: one per row of A, one per column of B.
// This is the portable variant: bytes are unpacked with integer arithmetic. A packed-dot variant replaces the staging and the inner
// loop where the device has an 8-bit dot product; the sums it gives are the same integers.
//
// Tiles follow the float kernel (64x64 per workgroup, k in steps of 16, a 4x4 register tile per thread). Every output element is summed
// by one thread in increasing k; integer addition is associative, so no order could change the bits anyway.

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
    uint Flags;     // bit 0 A is int8, bit 1 B is int8, bit 2 B stored N x K, bit 3 zero point of A present, bit 4 one per row, bit 5 of B present, bit 6 one per column
    uint OffZA;
    uint OffZB;
};

[[vk::push_constant]] ConstantBuffer<Int8GemmParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> C;
[[vk::binding(3, 0)]] StructuredBuffer<uint> ZA;
[[vk::binding(4, 0)]] StructuredBuffer<uint> ZB;

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TILE_THREADS = 256;

groupshared int As[BK][BM + 1];
groupshared int Bs[BK][BN + 1];

int Byte(StructuredBuffer<uint> buf, uint i, bool isSigned)
{
    uint b = (buf[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
    return isSigned ? ((int)(b << 24)) >> 24 : (int)b;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void Int8GemmTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
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
    int zaScalar = hasZA && !rowZA ? Byte(ZA, P.OffZA, aSigned) : 0;
    int zbScalar = hasZB && !colZB ? Byte(ZB, P.OffZB, bSigned) : 0;

    uint tx = tid & 15u;
    uint ty = tid >> 4;

    // Sixteen named scalars rather than an array: indexed arrays live in thread-local memory on Metal.
    int c00 = 0; int c01 = 0; int c02 = 0; int c03 = 0; int c10 = 0; int c11 = 0; int c12 = 0; int c13 = 0; int c20 = 0; int c21 = 0; int c22 = 0; int c23 = 0; int c30 = 0; int c31 = 0; int c32 = 0; int c33 = 0;

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        [unroll] for (uint e = 0; e < (BM * BK) / TILE_THREADS; e++)
        {
            uint idx = tid + e * TILE_THREADS;
            uint r = idx / BK;
            uint k = idx % BK;
            uint gm = m0 + r;
            uint gk = k0 + k;
            int v = 0;
            if (gm < P.M && gk < P.K)
            {
                int z = rowZA ? Byte(ZA, P.OffZA + gm, aSigned) : zaScalar;
                v = Byte(A, baseA + gm * P.Lda + gk, aSigned) - z;
            }

            As[k][r] = v;
        }

        [unroll] for (uint f = 0; f < (BK * BN) / TILE_THREADS; f++)
        {
            uint idx = tid + f * TILE_THREADS;
            uint c, k;
            if (transB) { c = idx / BK; k = idx % BK; }
            else { c = idx % BN; k = idx / BN; }
            uint gn = n0 + c;
            uint gk = k0 + k;
            int v = 0;
            if (gn < P.N && gk < P.K)
            {
                int z = colZB ? Byte(ZB, P.OffZB + gn, bSigned) : zbScalar;
                v = Byte(B, baseB + (transB ? gn * P.Ldb + gk : gk * P.Ldb + gn), bSigned) - z;
            }

            Bs[k][c] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < BK; kk++)
        {
            int a0 = As[kk][ty + 0];
            int a1 = As[kk][ty + 16];
            int a2 = As[kk][ty + 32];
            int a3 = As[kk][ty + 48];
            int b0 = Bs[kk][tx + 0];
            int b1 = Bs[kk][tx + 16];
            int b2 = Bs[kk][tx + 32];
            int b3 = Bs[kk][tx + 48];
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

#define STORE(DR, DC, V) \
    { uint row = m0 + ty + DR; uint col = n0 + tx + DC; if (row < P.M && col < P.N) { C[baseC + row * P.N + col] = (uint)(V); } }
    STORE(0, 0, c00) STORE(0, 16, c01) STORE(0, 32, c02) STORE(0, 48, c03)
    STORE(16, 0, c10) STORE(16, 16, c11) STORE(16, 32, c12) STORE(16, 48, c13)
    STORE(32, 0, c20) STORE(32, 16, c21) STORE(32, 32, c22) STORE(32, 48, c23)
    STORE(48, 0, c30) STORE(48, 16, c31) STORE(48, 32, c32) STORE(48, 48, c33)
}
