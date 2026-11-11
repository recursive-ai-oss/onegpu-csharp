// A compute sample: writes Value + index into Count elements starting at First, and touches nothing else.

struct FillParams
{
    uint Value;
    uint Count;
    uint First;
};

[[vk::push_constant]] ConstantBuffer<FillParams> P;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> Data;

[shader("compute")]
[numthreads(32, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x < P.Count)
    {
        Data[P.First + id.x] = P.Value + id.x;
    }
}
