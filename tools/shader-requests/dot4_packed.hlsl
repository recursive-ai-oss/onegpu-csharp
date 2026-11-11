// Requested HLSL surface: packed 8-bit dot products (Shader Model 6.4 semantics), needed by the int8 GEMM.
//
//   int  dot4add_i8packed(uint a, uint b, int acc)   four signed 8-bit lanes of a and b multiplied pairwise and summed into acc
//   uint dot4add_u8packed(uint a, uint b, uint acc)  the same for unsigned lanes
//   Lane 0 is the least significant byte. The sum wraps modulo 2^32 like the integer addition it stands for.

StructuredBuffer<uint> A : register(t0);
StructuredBuffer<uint> B : register(t1);
RWStructuredBuffer<int> S : register(u2);
RWStructuredBuffer<uint> U : register(u3);

[shader("compute")]
[numthreads(64, 1, 1)]
void Dot4(uint id : SV_DispatchThreadID)
{
    S[id] = dot4add_i8packed(A[id], B[id], S[id]);
    U[id] = dot4add_u8packed(A[id], B[id], U[id]);
}
