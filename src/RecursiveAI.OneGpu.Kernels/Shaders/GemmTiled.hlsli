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
    uint Flags;     // bit 0: A is stored K x M (transposed); bit 1: B is stored N x K (transposed)
    float Alpha;
    float Beta;
};

[[vk::push_constant]] ConstantBuffer<GemmParams> P;

static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TILE_THREADS = 256;

// One column of padding keeps the strided reads of the inner loop off the same banks.
groupshared float As[BK][BM + 1];
groupshared float Bs[BK][BN + 1];

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

    float acc[4][4];
    [unroll] for (uint i = 0; i < 4; i++)
    {
        [unroll] for (uint j = 0; j < 4; j++)
        {
            acc[i][j] = 0.0;
        }
    }

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
            float a[4];
            float b[4];
            [unroll] for (uint i = 0; i < 4; i++) { a[i] = As[kk][ty + 16 * i]; }
            [unroll] for (uint j = 0; j < 4; j++) { b[j] = Bs[kk][tx + 16 * j]; }
            [unroll] for (uint i2 = 0; i2 < 4; i2++)
            {
                [unroll] for (uint j2 = 0; j2 < 4; j2++)
                {
                    acc[i2][j2] = mad(a[i2], b[j2], acc[i2][j2]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // beta == 0 never reads C, so uninitialised output memory (even NaN) is fine, as in BLAS. The bounds are tested with positive
    // conditions around the store rather than a continue: the Metal output of the shader compiler dropped the
    // continue inside the unrolled loops, so the guard never ran.
    [unroll] for (uint i3 = 0; i3 < 4; i3++)
    {
        uint gm = m0 + ty + 16 * i3;
        [unroll] for (uint j3 = 0; j3 < 4; j3++)
        {
            uint gn = n0 + tx + 16 * j3;
            if (gm < P.M && gn < P.N)
            {
                uint at = baseC + gm * P.Ldc + gn;
                float r = P.Alpha * acc[i3][j3];
                if (P.Beta != 0.0) { r = mad(P.Beta, C[at], r); }
                C[at] = r;
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void GemmTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    GemmTiledBody(group, tid);
}
