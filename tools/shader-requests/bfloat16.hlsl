// Requested HLSL surface: a bfloat16 element type (1 sign, 8 exponent, 7 mantissa bits), as storage and as a cooperative-matrix element.
//
//   bfloat16_t               scalar type; converts to and from float (float to bfloat16 rounds to nearest even)
//   StructuredBuffer<bfloat16_t>, groupshared bfloat16_t[]   2-byte elements
//   CooperativeMatrix<bfloat16_t, 16, 16, MatrixUse::A>      see cooperative_matrix.hlsl; accumulates in float
//
// Answered by 1.5.0. The portable GEMM kernel still reads bfloat16 weights from storage buffers of 32-bit words and widens with a shift,
// which needs no type; the cooperative-matrix kernel reads and stores the bfloat16 type directly.

StructuredBuffer<bfloat16_t> A : register(t0);
StructuredBuffer<bfloat16_t> B : register(t1);
RWStructuredBuffer<float> C : register(u2);

groupshared bfloat16_t TileA[16 * 16];
groupshared bfloat16_t TileB[16 * 16];

[shader("compute")]
[numthreads(32, 1, 1)]
void Bf16Product(uint tid : SV_GroupIndex)
{
    for (uint i = tid; i < 256; i += 32)
    {
        TileA[i] = A[i];
        TileB[i] = (bfloat16_t)((float)B[i] * 2.0f);   // float -> bfloat16 conversion
    }
    GroupMemoryBarrierWithGroupSync();

    CooperativeMatrix<bfloat16_t, 16, 16, MatrixUse::A> a;
    CooperativeMatrix<bfloat16_t, 16, 16, MatrixUse::B> b;
    a.Load(TileA, 0, 16, MatrixLayout::RowMajor);
    b.Load(TileB, 0, 16, MatrixLayout::RowMajor);
    CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator> acc = CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator>::Zero();
    acc = CooperativeMatrix<float, 16, 16, MatrixUse::Accumulator>::MulAdd(a, b, acc);
    acc.Store(C, 0, 16, MatrixLayout::RowMajor);
}
