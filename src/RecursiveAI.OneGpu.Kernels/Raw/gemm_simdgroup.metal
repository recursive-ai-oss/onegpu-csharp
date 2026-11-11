// Matrix multiplication with simdgroup matrices: C = alpha * op(A) * op(B) + beta * C, batched, float32 accumulation and output.
//
// The shader compiler used for the other kernels cannot express simdgroup_matrix yet, so this kernel is written in Metal Shading
// Language directly (Metal Shading Language Specification, "Matrix Data Types" and "SIMD-group Matrix Functions"). The host puts
// three definitions in front of this text:
//   GEMM_KIND   0 = float32 inputs, 1 = float16 inputs, 2 = bfloat16 inputs
//   GEMM_ELEM   the threadgroup tile element type: float, half or bfloat
//   GEMM_IN     the buffer element type A and B are read as: float, or uint for pairs of 16-bit values packed in 32-bit words
//
// One 64 x 64 tile of C per threadgroup of 256 invocations (eight 32-wide simdgroups, 2 x 4), each simdgroup owning 32 x 16 of C as
// eight 8 x 8 float accumulators. The k loop runs in steps of 16. Tiles are staged through threadgroup memory, which is where bounds,
// transposes and odd leading dimensions are handled. The buffer and parameter layout is the one the portable kernel uses.
#include <metal_stdlib>
using namespace metal;

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
    uint Flags;
    float Alpha;
    float Beta;
};

#if GEMM_KIND == 0
inline GEMM_ELEM load_element(device const float* p, uint i) { return p[i]; }
#elif GEMM_KIND == 1
inline GEMM_ELEM load_element(device const uint* p, uint i)
{
    uint w = p[i >> 1];
    return as_type<half>(ushort((i & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu)));
}
#else
inline GEMM_ELEM load_element(device const uint* p, uint i)
{
    uint w = p[i >> 1];
    return as_type<bfloat>(ushort((i & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu)));
}
#endif

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;
constant uint A_STRIDE = 24;   // padded rows keep simdgroup loads aligned and off the same banks
constant uint B_STRIDE = 72;

kernel void GemmTiled(
    device const GEMM_IN* A [[buffer(0)]],
    device const GEMM_IN* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& P [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup GEMM_ELEM As[BM * A_STRIDE];
    threadgroup GEMM_ELEM Bs[BK * B_STRIDE];
    threadgroup float Cs[8 * 128];

    const uint m0 = group.y * BM;
    const uint n0 = group.x * BN;
    const uint baseA = P.OffsetA + group.z * P.StrideA;
    const uint baseB = P.OffsetB + group.z * P.StrideB;
    const uint baseC = P.OffsetC + group.z * P.StrideC;
    const bool transA = (P.Flags & 1u) != 0u;
    const bool transB = (P.Flags & 2u) != 0u;
    const uint sgRow = sg >> 2;
    const uint sgCol = sg & 3u;

    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (uint i = 0; i < 4; i++)
    {
        for (uint j = 0; j < 2; j++)
        {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < P.K; k0 += BK)
    {
        for (uint e = 0; e < (BM * BK) / 256; e++)
        {
            uint idx = tid + e * 256;
            uint r = transA ? (idx % BM) : (idx / BK);
            uint k = transA ? (idx / BM) : (idx % BK);
            uint gm = m0 + r;
            uint gk = k0 + k;
            GEMM_ELEM v = GEMM_ELEM(0);
            if (gm < P.M && gk < P.K)
            {
                v = load_element(A, baseA + (transA ? gk * P.Lda + gm : gm * P.Lda + gk));
            }
            As[r * A_STRIDE + k] = v;
        }

        for (uint f = 0; f < (BK * BN) / 256; f++)
        {
            uint idx = tid + f * 256;
            uint c = transB ? (idx / BK) : (idx % BN);
            uint k = transB ? (idx % BK) : (idx / BN);
            uint gn = n0 + c;
            uint gk = k0 + k;
            GEMM_ELEM v = GEMM_ELEM(0);
            if (gn < P.N && gk < P.K)
            {
                v = load_element(B, baseB + (transB ? gn * P.Ldb + gk : gk * P.Ldb + gn));
            }
            Bs[k * B_STRIDE + c] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8)
        {
            simdgroup_matrix<GEMM_ELEM, 8, 8> a[4];
            simdgroup_matrix<GEMM_ELEM, 8, 8> b[2];
            for (uint i = 0; i < 4; i++)
            {
                simdgroup_load(a[i], As + (sgRow * 32 + 8 * i) * A_STRIDE + kk, A_STRIDE);
            }
            for (uint j = 0; j < 2; j++)
            {
                simdgroup_load(b[j], Bs + kk * B_STRIDE + sgCol * 16 + 8 * j, B_STRIDE);
            }
            for (uint i = 0; i < 4; i++)
            {
                for (uint j = 0; j < 2; j++)
                {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each simdgroup writes 8 rows of its 32 x 16 block at a time to its own scratch area, then its lanes apply alpha and beta and
    // store. Four passes keep the scratch at 4 KB for all eight simdgroups.
    threadgroup float* scratch = Cs + sg * 128;
    for (uint i = 0; i < 4; i++)
    {
        simdgroup_store(acc[i][0], scratch, 16);
        simdgroup_store(acc[i][1], scratch + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < 4; e++)
        {
            uint idx = lane + 32 * e;
            uint r = idx >> 4;
            uint c = idx & 15u;
            uint gm = m0 + sgRow * 32 + 8 * i + r;
            uint gn = n0 + sgCol * 16 + c;
            if (gm < P.M && gn < P.N)
            {
                uint at = baseC + gm * P.Ldc + gn;
                float result = P.Alpha * scratch[idx];
                if (P.Beta != 0.0f)
                {
                    result = fma(P.Beta, C[at], result);
                }
                C[at] = result;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
