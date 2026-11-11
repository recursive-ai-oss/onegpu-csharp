// Adds one to every element: the smallest kernel that proves a dispatch ran and its result came back.

struct Params
{
    uint Count;
};

[[vk::push_constant]] ConstantBuffer<Params> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> Input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Output;

[shader("compute")]
[numthreads(64, 1, 1)]
void Increment(uint3 id : SV_DispatchThreadID)
{
    if (id.x < P.Count)
    {
        Output[id.x] = Input[id.x] + 1;
    }
}
