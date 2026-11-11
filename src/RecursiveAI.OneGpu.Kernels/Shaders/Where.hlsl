// out = cond ? a : b with broadcasting. cond is a 32-bit word per element (0 or 1). Values are copied bit for bit.

#include "Common.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct WhereParams
{
    uint Count;
    uint OffC;
    uint OffA;
    uint OffB;
    uint OffO;
    uint GridX;
    uint Rank;
    uint D0, D1, D2, D3, D4, D5;
    uint C0, C1, C2, C3, C4, C5;
    uint A0, A1, A2, A3, A4, A5;
    uint B0, B1, B2, B3, B4, B5;
};

[[vk::push_constant]] ConstantBuffer<WhereParams> WP;

[[vk::binding(0, 0)]] StructuredBuffer<uint> WhC;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> WhA;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> WhB;
[[vk::binding(3, 0)]] RWStructuredBuffer<Elem> WhO;

#define WSTEP(R, D, C, A, B) if (WP.Rank > R) { uint c = rem % WP.D; rem /= WP.D; ic += c * WP.C; ia += c * WP.A; ib += c * WP.B; }

void WhereLocate(uint i, out uint ic, out uint ia, out uint ib)
{
    uint rem = i;
    ic = WP.OffC;
    ia = WP.OffA;
    ib = WP.OffB;
    WSTEP(0u, D0, C0, A0, B0)
    WSTEP(1u, D1, C1, A1, B1)
    WSTEP(2u, D2, C2, A2, B2)
    WSTEP(3u, D3, C3, A3, B3)
    WSTEP(4u, D4, C4, A4, B4)
    WSTEP(5u, D5, C5, A5, B5)
}

#if F16
uint Pick(uint i)
{
    uint ic, ia, ib;
    WhereLocate(i, ic, ia, ib);
    return (WhC[ic] != 0u) ? RawH(WhA, ia) : RawH(WhB, ib);
}
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void WhereOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * WP.GridX + gid.x) * WG + tid;
#if F16
    uint word = (WP.OffO >> 1) + t;
    uint res = WhO[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= WP.OffO && h < WP.OffO + WP.Count)
        {
            res = (res & ~(0xFFFFu << (k * 16u))) | (Pick(h - WP.OffO) << (k * 16u));
        }
    }

    WhO[word] = res;
#else
    if (t < WP.Count)
    {
        uint ic, ia, ib;
        WhereLocate(t, ic, ia, ib);
        WhO[WP.OffO + t] = (WhC[ic] != 0u) ? WhA[ia] : WhB[ib];
    }
#endif
}
