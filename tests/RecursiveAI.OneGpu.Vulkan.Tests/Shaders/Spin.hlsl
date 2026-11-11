// Busy work of a chosen length, for timing tests: a dependent chain of multiply-adds per thread that the compiler cannot fold away because
// the iteration count arrives at run time.

struct Params
{
    uint Iterations;
};

[[vk::push_constant]] ConstantBuffer<Params> P;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> Result;

[shader("compute")]
[numthreads(64, 1, 1)]
void Burn(uint3 id : SV_DispatchThreadID)
{
    float x = (float)id.x * 0.001f + 1.0f;
    for (uint i = 0; i < P.Iterations; i++)
    {
        x = mad(x, 0.9999f, 0.0001f);
    }

    Result[id.x] = x;
}
