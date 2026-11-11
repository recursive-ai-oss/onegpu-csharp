// y = f(x) over a dense range.

#include "Functions.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct UnaryParams
{
    uint Count;
    uint OffX;
    uint OffY;
    uint Op;
    float P0;
    float P1;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<UnaryParams> UP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> UnX;
[[vk::binding(1, 0)]] RWStructuredBuffer<Elem> UnY;

[shader("compute")]
[numthreads(256, 1, 1)]
void UnaryOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * UP.GridX + gid.x) * WG + tid;
#if F16
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word, and a surplus thread would then
    // write that word back with stale contents.
    if (t >= HalfWords(UP.OffY, UP.Count))
    {
        return;
    }

    uint word = (UP.OffY >> 1) + t;
    uint res = UnY[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= UP.OffY && h < UP.OffY + UP.Count)
        {
            uint code = F2H(Unary(UP.Op, LoadH(UnX, UP.OffX + (h - UP.OffY)), UP.P0, UP.P1));
            res = (res & ~(0xFFFFu << (k * 16u))) | (code << (k * 16u));
        }
    }

    UnY[word] = res;
#else
    if (t < UP.Count)
    {
        UnY[UP.OffY + t] = Unary(UP.Op, UnX[UP.OffX + t], UP.P0, UP.P1);
    }
#endif
}
