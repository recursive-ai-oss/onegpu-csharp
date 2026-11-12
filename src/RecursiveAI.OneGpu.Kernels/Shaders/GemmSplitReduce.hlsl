// The end of a matrix product whose k axis was cut into ranges (GemmTiled.hlsli with Flags bit 4): the sum of an element is its planes of partial sums added in
// range order, one float32 addition at a time, so the result depends on the shape and nothing else. Then alpha * sum + beta * C, the bias and the activation, as
// the whole-k kernel applies them.
//
// The planes are Splits matrices of Batch * M * N elements each, PlaneStride apart, laid out [range][batch][row][column] from element 0 of the scratch buffer.
// One thread per output element over the flattened batch, row and column. Push constants: see ReduceParams.

[[vk::binding(0, 0)]] StructuredBuffer<float> Planes;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(2, 0)]] StructuredBuffer<float> Bias;   // read only when Flags bit 2 is set

struct GemmParams
{
    uint Count;       // Batch * M * N
    uint N;
    uint MN;          // M * N
    uint Ldc;
    uint OffsetC;
    uint StrideC;
    uint Splits;
    uint PlaneStride;
    uint Flags;       // bit 2: add Bias[BiasOffset + column]; bit 3: the bias is indexed by row instead
    float Alpha;
    float Beta;
    uint BiasOffset;
    uint Act;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<GemmParams> P;

#include "Functions.hlsl"
#include "GemmEpilogue.hlsli"

[shader("compute")]
[numthreads(256, 1, 1)]
void GemmSplitReduceOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint i = (gid.y * P.GridX + gid.x) * 256u + tid;
    if (i >= P.Count)
    {
        return;
    }

    float v = Planes[i];
    for (uint s = 1u; s < P.Splits; s++)
    {
        v += Planes[s * P.PlaneStride + i];
    }

    uint batch = i / P.MN;
    uint rem = i - batch * P.MN;
    uint row = rem / P.N;
    uint col = rem - row * P.N;
    uint at = P.OffsetC + batch * P.StrideC + row * P.Ldc + col;
    float r = P.Alpha * v;
    if (P.Beta != 0.0) { r = mad(P.Beta, C[at], r); }
    C[at] = GemmEpilogue(r, row, col);
}
