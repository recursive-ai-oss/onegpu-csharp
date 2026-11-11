// A compute sample: element-wise addition of two arrays, once on 32-bit integers and once on 32-bit floats. The element count arrives as a
// push constant, so the last workgroup of a dispatch that does not divide evenly stays inside the arrays.

struct Params
{
    uint Count;
};

[[vk::push_constant]] ConstantBuffer<Params> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> AU;
[[vk::binding(1, 0)]] StructuredBuffer<uint> BU;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> SumU;

[shader("compute")]
[numthreads(64, 1, 1)]
void AddUint(uint3 id : SV_DispatchThreadID)
{
    if (id.x < P.Count)
    {
        SumU[id.x] = AU[id.x] + BU[id.x];
    }
}

[[vk::binding(3, 0)]] StructuredBuffer<float> AF;
[[vk::binding(4, 0)]] StructuredBuffer<float> BF;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> SumF;

[shader("compute")]
[numthreads(64, 1, 1)]
void AddFloat(uint3 id : SV_DispatchThreadID)
{
    if (id.x < P.Count)
    {
        SumF[id.x] = AF[id.x] + BF[id.x];
    }
}
