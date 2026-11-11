// Portable tiled matrix multiplication: C = alpha * op(A) * op(B) + beta * C, batched, with float accumulation.
//
// Blocking follows Goto and van de Geijn: a 64x64 tile of C per workgroup, the k loop in steps of 16 staged through
// workgroup memory, and a 4x4 register tile per thread. Every output element is summed by one thread in increasing k, so
// the result does not depend on scheduling, occupancy or the subgroup size: the same bits come out of every run on one
// device and driver. That is the property the deterministic mode promises, so both modes share this kernel.
//
// The including file defines one of GEMM_F32, GEMM_F16 or GEMM_BF16. Half-precision inputs are read as pairs packed in
// 32-bit words, so no 16-bit storage feature is needed on any target; bf16 is widened by placing its bits in the high half
// of a float.

#if defined(GEMM_F32)
[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] StructuredBuffer<float> B;
float LoadElement(StructuredBuffer<float> buf, uint i) { return buf[i]; }
#elif defined(GEMM_F16)
[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
float LoadElement(StructuredBuffer<uint> buf, uint i)
{
    uint w = buf[i >> 1];
    return f16tof32((i & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu));
}
#else
[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
float LoadElement(StructuredBuffer<uint> buf, uint i)
{
    uint w = buf[i >> 1];
    return asfloat((i & 1u) != 0u ? (w & 0xFFFF0000u) : (w << 16));
}
#endif

[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]] StructuredBuffer<float> Bias;   // read only when Flags bit 2 is set; declared before the push constants so Metal numbers it before them

struct GemmParams
{
    uint M;
    uint N;
    uint K;
    uint Lda;
    uint Ldb;
    uint Ldc;
    uint OffsetA;
    uint OffsetB;
    uint OffsetC;
    uint StrideA;
    uint StrideB;
    uint StrideC;
    uint Flags;     // bit 0: A is stored K x M (transposed); bit 1: B is stored N x K (transposed); bit 2: add Bias[BiasOffset + column]
    float Alpha;
    float Beta;
    uint BiasOffset;
    uint Act;
};

[[vk::push_constant]] ConstantBuffer<GemmParams> P;

#include "Functions.hlsl"
#include "GemmEpilogue.hlsli"

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TILE_THREADS = 256;

// One column of padding keeps the strided reads of the inner loop off the same banks.
groupshared float As[BK][BM + 1];
groupshared float Bs[BK][BN + 1];

// Writes alpha * value + beta * C at (row, col) when the element is inside the matrix.
void StoreElement(uint baseC, uint row, uint col, float value)
{
    if (row < P.M && col < P.N)
    {
        uint at = baseC + row * P.Ldc + col;
        float r = P.Alpha * value;
        if (P.Beta != 0.0) { r = mad(P.Beta, C[at], r); }
        C[at] = GemmEpilogue(r, col);
    }
}

void GemmTiledBody(uint3 group, uint tid)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    uint batch = group.z;
    uint baseA = P.OffsetA + batch * P.StrideA;
    uint baseB = P.OffsetB + batch * P.StrideB;
    uint baseC = P.OffsetC + batch * P.StrideC;
    bool transA = (P.Flags & 1u) != 0u;
    bool transB = (P.Flags & 2u) != 0u;

    uint tx = tid & 15u;
    uint ty = tid >> 4;

    // The 4 x 4 register tile is sixteen named scalars rather than an array: the Metal output of the shader compiler keeps indexed
    // arrays in thread-local memory, which made the kernel more than ten times slower there.
    float c00 = 0.0; float c01 = 0.0; float c02 = 0.0; float c03 = 0.0; float c10 = 0.0; float c11 = 0.0; float c12 = 0.0; float c13 = 0.0; float c20 = 0.0; float c21 = 0.0; float c22 = 0.0; float c23 = 0.0; float c30 = 0.0; float c31 = 0.0; float c32 = 0.0; float c33 = 0.0;

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        // Stage the A tile (BM x BK) and the B tile (BK x BN). The thread-to-element map follows the memory layout so a
        // warp reads consecutive addresses in either orientation.
        [unroll] for (uint e = 0; e < (BM * BK) / TILE_THREADS; e++)
        {
            uint idx = tid + e * TILE_THREADS;
            uint r, k;
            if (transA) { r = idx % BM; k = idx / BM; }
            else { r = idx / BK; k = idx % BK; }
            uint gm = m0 + r;
            uint gk = k0 + k;
            float v = 0.0;
            if (gm < P.M && gk < P.K)
            {
                v = LoadElement(A, baseA + (transA ? gk * P.Lda + gm : gm * P.Lda + gk));
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
            float v = 0.0;
            if (gn < P.N && gk < P.K)
            {
                v = LoadElement(B, baseB + (transB ? gn * P.Ldb + gk : gk * P.Ldb + gn));
            }
            Bs[k][c] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < BK; kk++)
        {
            float a0 = As[kk][ty + 0];
            float a1 = As[kk][ty + 16];
            float a2 = As[kk][ty + 32];
            float a3 = As[kk][ty + 48];
            float b0 = Bs[kk][tx + 0];
            float b1 = Bs[kk][tx + 16];
            float b2 = Bs[kk][tx + 32];
            float b3 = Bs[kk][tx + 48];
            c00 = mad(a0, b0, c00);
            c01 = mad(a0, b1, c01);
            c02 = mad(a0, b2, c02);
            c03 = mad(a0, b3, c03);
            c10 = mad(a1, b0, c10);
            c11 = mad(a1, b1, c11);
            c12 = mad(a1, b2, c12);
            c13 = mad(a1, b3, c13);
            c20 = mad(a2, b0, c20);
            c21 = mad(a2, b1, c21);
            c22 = mad(a2, b2, c22);
            c23 = mad(a2, b3, c23);
            c30 = mad(a3, b0, c30);
            c31 = mad(a3, b1, c31);
            c32 = mad(a3, b2, c32);
            c33 = mad(a3, b3, c33);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // beta == 0 never reads C, so uninitialised output memory (even NaN) is fine, as in BLAS. The 16 stores are straight-line calls
    // because the register tile is sixteen scalars; each one tests its own bounds.
    StoreElement(baseC, m0 + ty + 0, n0 + tx + 0, c00);
    StoreElement(baseC, m0 + ty + 0, n0 + tx + 16, c01);
    StoreElement(baseC, m0 + ty + 0, n0 + tx + 32, c02);
    StoreElement(baseC, m0 + ty + 0, n0 + tx + 48, c03);
    StoreElement(baseC, m0 + ty + 16, n0 + tx + 0, c10);
    StoreElement(baseC, m0 + ty + 16, n0 + tx + 16, c11);
    StoreElement(baseC, m0 + ty + 16, n0 + tx + 32, c12);
    StoreElement(baseC, m0 + ty + 16, n0 + tx + 48, c13);
    StoreElement(baseC, m0 + ty + 32, n0 + tx + 0, c20);
    StoreElement(baseC, m0 + ty + 32, n0 + tx + 16, c21);
    StoreElement(baseC, m0 + ty + 32, n0 + tx + 32, c22);
    StoreElement(baseC, m0 + ty + 32, n0 + tx + 48, c23);
    StoreElement(baseC, m0 + ty + 48, n0 + tx + 0, c30);
    StoreElement(baseC, m0 + ty + 48, n0 + tx + 16, c31);
    StoreElement(baseC, m0 + ty + 48, n0 + tx + 32, c32);
    StoreElement(baseC, m0 + ty + 48, n0 + tx + 48, c33);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void GemmTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    GemmTiledBody(group, tid);
}
