// Matrix multiplication on matrix hardware: C = alpha * op(A) * op(B) + beta * C, batched, 16-bit inputs, float accumulation and output.
//
// One 64 x 64 tile of C per workgroup. Four 32-wide subgroups (128 invocations) arranged 2 x 2 each own 32 x 32 of C as four 16 x 16
// float accumulators; with GEMM_WIDE, eight subgroups (256 invocations) arranged 2 x 4 each own 32 x 16 as two accumulators, which
// keeps fewer values live per invocation and suits GPUs that hide latency with more resident threads. The k loop runs in steps of 16. The A and B tiles are staged through workgroup memory in the matrix
// element type, which is where bounds, transposes and odd leading dimensions are handled; each subgroup then multiplies its fragments
// with cooperative matrices (VK_KHR_cooperative_matrix on Vulkan, simdgroup_matrix on Metal). The accumulators leave through workgroup
// memory so alpha, beta and the matrix edges are ordinary code.
//
// The host picks this kernel only when the device offers 16 x 16 x 16 matrices of the element type with a float accumulator and runs
// 32-wide subgroups; it creates the pipeline with full subgroups, so subgroup n is invocations 32n .. 32n+31 of the workgroup.
//
// The including file defines GEMM_F16, GEMM_BF16 or GEMM_F32 (built for Metal only: no Vulkan device lists float32 fragments).
// Reading 16-bit elements from storage buffers needs 16-bit storage access, which every device with cooperative matrices has; the
// host checks the kernel's reflection against the device before choosing it, and the portable kernel stays the fallback.

#if defined(GEMM_F32)
typedef float Elem;
#elif defined(GEMM_F16)
typedef float16_t Elem;
#else
typedef bfloat16_t Elem;
#endif

// The operands are read in their own element type. Staging is then a plain copy into workgroup memory, with no conversion: a
// bfloat16 converted through float32 on the way in cost a third of the throughput on the first version of this kernel.
[[vk::binding(0, 0)]] StructuredBuffer<Elem> A;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> B;
Elem LoadElement(StructuredBuffer<Elem> buf, uint i) { return buf[i]; }
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
#if defined(GEMM_WIDE)
#define THREADS 256
#define SUBGROUPS 8
#else
#define THREADS 128
#define SUBGROUPS 4
#endif

// Rows are padded so fragment loads stay 16-byte aligned and spread over the banks.
static const uint A_STRIDE = 24;
static const uint B_STRIDE = 72;

groupshared Elem TileA[BM * A_STRIDE];
groupshared Elem TileB[BK * B_STRIDE];
groupshared float TileC[SUBGROUPS * 256];

typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::A> MatA;
typedef CooperativeMatrix<Elem, 16, 16, MatrixUse::B> MatB;
typedef CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> MatC;

// Moves one accumulator through workgroup memory and stores alpha * value + beta * C for the elements inside the matrix. Every lane of
// the workgroup reaches the barriers, whatever the bounds are.
#define STORE_ACC(ACC, ROW0, COL0)                                                                   \
    ACC.Store(TileC, sg * 256, 16, MatrixLayout::RowMajor);                                          \
    GroupMemoryBarrierWithGroupSync();                                                               \
    [unroll] for (uint e = 0; e < 8; e++)                                                            \
    {                                                                                                \
        uint ei = lane + 32 * e;                                                                     \
        uint gm = (ROW0) + (ei >> 4);                                                                \
        uint gn = (COL0) + (ei & 15u);                                                               \
        if (gm < P.M && gn < P.N)                                                                    \
        {                                                                                            \
            uint at = baseC + gm * P.Ldc + gn;                                                       \
            float r = P.Alpha * TileC[sg * 256 + ei];                                                \
            if (P.Beta != 0.0) { r = mad(P.Beta, C[at], r); }                                        \
            C[at] = GemmEpilogue(r, gn);                                                             \
        }                                                                                            \
    }                                                                                                \
    GroupMemoryBarrierWithGroupSync();

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void GemmCoop(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    uint batch = group.z;
    uint baseA = P.OffsetA + batch * P.StrideA;
    uint baseB = P.OffsetB + batch * P.StrideB;
    uint baseC = P.OffsetC + batch * P.StrideC;
    bool transA = (P.Flags & 1u) != 0u;
    bool transB = (P.Flags & 2u) != 0u;

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

    MatC acc00 = MatC::Zero();
    MatC acc10 = MatC::Zero();
#if !defined(GEMM_WIDE)
    MatC acc01 = MatC::Zero();
    MatC acc11 = MatC::Zero();
#endif

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        // Stage the A tile (BM x BK) and the B tile (BK x BN); the thread-to-element map follows the memory layout so a subgroup reads
        // consecutive addresses in either orientation.
        [unroll] for (uint e = 0; e < (BM * BK) / THREADS; e++)
        {
            uint idx = tid + e * THREADS;
            uint r, k;
            if (transA) { r = idx & 63u; k = idx >> 6; }
            else { r = idx >> 4; k = idx & 15u; }
            uint gm = m0 + r;
            uint gk = k0 + k;
            Elem v = (Elem)0.0f;
            if (gm < P.M && gk < P.K)
            {
                v = LoadElement(A, baseA + (transA ? gk * P.Lda + gm : gm * P.Lda + gk));
            }
            TileA[r * A_STRIDE + k] = v;
        }

        [unroll] for (uint f = 0; f < (BK * BN) / THREADS; f++)
        {
            uint idx = tid + f * THREADS;
            uint c, k;
            if (transB) { c = idx >> 4; k = idx & 15u; }
            else { c = idx & 63u; k = idx >> 6; }
            uint gn = n0 + c;
            uint gk = k0 + k;
            Elem v = (Elem)0.0f;
            if (gn < P.N && gk < P.K)
            {
                v = LoadElement(B, baseB + (transB ? gn * P.Ldb + gk : gk * P.Ldb + gn));
            }
            TileB[k * B_STRIDE + c] = v;
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

    // beta == 0 never reads C, so uninitialised output memory (even NaN) is fine, as in BLAS.
    uint rowBase = m0 + sgRow * 32;
    uint colBase = n0 + colTile;
    STORE_ACC(acc00, rowBase + 0, colBase + 0)
    STORE_ACC(acc10, rowBase + 16, colBase + 0)
#if !defined(GEMM_WIDE)
    STORE_ACC(acc01, rowBase + 0, colBase + 16)
    STORE_ACC(acc11, rowBase + 16, colBase + 16)
#endif
}
