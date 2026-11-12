// Matrix product with very few rows of A (M <= 4: decoding one to four tokens), float32 inputs, float32 accumulation: C = alpha * A * op(B) + beta * C.
//
// With so few rows a 64 x 64 tile wastes 60 of its 64 rows, and the product is limited by how fast the weights B stream from memory, not by arithmetic.
// This kernel streams B once and keeps up to four rows of A against each element. Both layouts of B cover k with several threads per output element
// so a small N still fills the GPU:
//
//   B stored K x N (not transposed): a workgroup owns 16 columns; its 256 threads are 16 columns x 16 slices of k. Thread (slice s, column c) chains k = s, s + 16, ...
//     in increasing order; a warp reads whole rows of 16 consecutive columns. Wide mode (Flags bit 5, for large N) is 32 columns x 8 slices.
//   B stored N x K (transposed): a workgroup owns 8 columns; 32 threads per column read consecutive k (a coalesced row), thread l chains k = l, l + 32, ... in increasing order.
//
// The partial sums of a column are added by one thread in increasing slice (or lane) order through workgroup memory, so the order is fixed by the shape and
// nothing depends on scheduling or the subgroup size. No atomics, no subgroup arithmetic: the same bits come out of every run, in both modes.
//
// Grid: x = tiles of columns, y = batch. Push constants as GemmTiled.hlsli. A must not be transposed and M must be at most 4 (the host checks).

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] StructuredBuffer<float> B;
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
    uint Flags;     // bit 1: B is stored N x K (transposed); bit 2: add Bias[BiasOffset + column]; bit 3: the bias is indexed by row instead
    float Alpha;
    float Beta;
    uint BiasOffset;
    uint Act;
};

[[vk::push_constant]] ConstantBuffer<GemmParams> P;

#include "Functions.hlsl"
#include "GemmEpilogue.hlsli"

groupshared float Part[4 * 256];

// Row m of A at k, or zero beyond the rows of this product.
float ARow(uint baseA, uint m, uint k)
{
    return (m < P.M) ? A[baseA + m * P.Lda + k] : 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void GemmSkinnyOp(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    uint batch = group.y;
    uint baseA = P.OffsetA + batch * P.StrideA;
    uint baseB = P.OffsetB + batch * P.StrideB;
    uint baseC = P.OffsetC + batch * P.StrideC;
    bool transB = (P.Flags & 2u) != 0u;
    float s0 = 0.0;
    float s1 = 0.0;
    float s2 = 0.0;
    float s3 = 0.0;

    // Not transposed: 16 columns x 16 slices, or with Flags bit 5 (wide, for large N: a warp then reads 128 bytes of a row at once) 32 columns x 8 slices.
    bool wide = (P.Flags & 32u) != 0u;
    uint cw = wide ? 32u : 16u;
    uint ks = 256u / cw;
    if (!transB)
    {
        uint c = t % cw;
        uint s = t / cw;
        uint n = group.x * cw + c;
        if (n < P.N)
        {
            uint k = s;
            // Four independent loads of B in flight per thread, then the four products in increasing k.
            for (; k + 3u * ks < P.K; k += 4u * ks)
            {
                float b0 = B[baseB + k * P.Ldb + n];
                float b1 = B[baseB + (k + ks) * P.Ldb + n];
                float b2 = B[baseB + (k + 2u * ks) * P.Ldb + n];
                float b3 = B[baseB + (k + 3u * ks) * P.Ldb + n];
                s0 = mad(ARow(baseA, 0u, k), b0, s0);
                s1 = mad(ARow(baseA, 1u, k), b0, s1);
                s2 = mad(ARow(baseA, 2u, k), b0, s2);
                s3 = mad(ARow(baseA, 3u, k), b0, s3);
                s0 = mad(ARow(baseA, 0u, k + ks), b1, s0);
                s1 = mad(ARow(baseA, 1u, k + ks), b1, s1);
                s2 = mad(ARow(baseA, 2u, k + ks), b1, s2);
                s3 = mad(ARow(baseA, 3u, k + ks), b1, s3);
                s0 = mad(ARow(baseA, 0u, k + 2u * ks), b2, s0);
                s1 = mad(ARow(baseA, 1u, k + 2u * ks), b2, s1);
                s2 = mad(ARow(baseA, 2u, k + 2u * ks), b2, s2);
                s3 = mad(ARow(baseA, 3u, k + 2u * ks), b2, s3);
                s0 = mad(ARow(baseA, 0u, k + 3u * ks), b3, s0);
                s1 = mad(ARow(baseA, 1u, k + 3u * ks), b3, s1);
                s2 = mad(ARow(baseA, 2u, k + 3u * ks), b3, s2);
                s3 = mad(ARow(baseA, 3u, k + 3u * ks), b3, s3);
            }

            for (; k < P.K; k += ks)
            {
                float b0 = B[baseB + k * P.Ldb + n];
                s0 = mad(ARow(baseA, 0u, k), b0, s0);
                s1 = mad(ARow(baseA, 1u, k), b0, s1);
                s2 = mad(ARow(baseA, 2u, k), b0, s2);
                s3 = mad(ARow(baseA, 3u, k), b0, s3);
            }
        }
    }
    else
    {
        uint col = t >> 5;
        uint lane = t & 31u;
        uint n = group.x * 8u + col;
        if (n < P.N)
        {
            uint rowB = baseB + n * P.Ldb;
            uint k = lane;
            for (; k + 96u < P.K; k += 128u)
            {
                float b0 = B[rowB + k];
                float b1 = B[rowB + k + 32u];
                float b2 = B[rowB + k + 64u];
                float b3 = B[rowB + k + 96u];
                s0 = mad(ARow(baseA, 0u, k), b0, s0);
                s1 = mad(ARow(baseA, 1u, k), b0, s1);
                s2 = mad(ARow(baseA, 2u, k), b0, s2);
                s3 = mad(ARow(baseA, 3u, k), b0, s3);
                s0 = mad(ARow(baseA, 0u, k + 32u), b1, s0);
                s1 = mad(ARow(baseA, 1u, k + 32u), b1, s1);
                s2 = mad(ARow(baseA, 2u, k + 32u), b1, s2);
                s3 = mad(ARow(baseA, 3u, k + 32u), b1, s3);
                s0 = mad(ARow(baseA, 0u, k + 64u), b2, s0);
                s1 = mad(ARow(baseA, 1u, k + 64u), b2, s1);
                s2 = mad(ARow(baseA, 2u, k + 64u), b2, s2);
                s3 = mad(ARow(baseA, 3u, k + 64u), b2, s3);
                s0 = mad(ARow(baseA, 0u, k + 96u), b3, s0);
                s1 = mad(ARow(baseA, 1u, k + 96u), b3, s1);
                s2 = mad(ARow(baseA, 2u, k + 96u), b3, s2);
                s3 = mad(ARow(baseA, 3u, k + 96u), b3, s3);
            }

            for (; k < P.K; k += 32u)
            {
                float b0 = B[rowB + k];
                s0 = mad(ARow(baseA, 0u, k), b0, s0);
                s1 = mad(ARow(baseA, 1u, k), b0, s1);
                s2 = mad(ARow(baseA, 2u, k), b0, s2);
                s3 = mad(ARow(baseA, 3u, k), b0, s3);
            }
        }
    }

    Part[t] = s0;
    Part[256u + t] = s1;
    Part[512u + t] = s2;
    Part[768u + t] = s3;
    GroupMemoryBarrierWithGroupSync();

    if (t < 4u * (transB ? 8u : cw))
    {
        // One thread per (row, column): its partial sums in increasing slice or lane order.
        uint m = t / (transB ? 8u : cw);
        uint c = t % (transB ? 8u : cw);
        uint n = group.x * (transB ? 8u : cw) + c;
        if (m < P.M && n < P.N)
        {
            float sum = 0.0;
            if (transB)
            {
                for (uint l = 0; l < 32u; l++) { sum += Part[m * 256u + c * 32u + l]; }
            }
            else
            {
                for (uint sl = 0; sl < ks; sl++) { sum += Part[m * 256u + sl * cw + c]; }
            }

            uint at = baseC + m * P.Ldc + n;
            float r = P.Alpha * sum;
            if (P.Beta != 0.0) { r = mad(P.Beta, C[at], r); }
            C[at] = GemmEpilogue(r, m, n);
        }
    }
}
