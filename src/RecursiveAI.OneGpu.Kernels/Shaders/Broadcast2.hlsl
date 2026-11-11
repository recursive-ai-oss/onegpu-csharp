// Two broadcast operands described by up to six dimensions, innermost first, with a stride per operand (0 along a broadcast dimension).

#include "Functions.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct BinaryParams
{
    uint Count;
    uint OffA;
    uint OffB;
    uint OffO;
    uint Op;
    uint Act;
    float P0;
    float P1;
    uint GridX;
    uint Rank;
    uint D0, D1, D2, D3, D4, D5;
    uint A0, A1, A2, A3, A4, A5;
    uint B0, B1, B2, B3, B4, B5;
};

[[vk::push_constant]] ConstantBuffer<BinaryParams> BP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> BinA;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> BinB;

#define BSTEP(R, D, A, B) if (BP.Rank > R) { uint c = rem % BP.D; rem /= BP.D; ia += c * BP.A; ib += c * BP.B; }

void BinaryLocate(uint i, out uint ia, out uint ib)
{
    uint rem = i;
    ia = BP.OffA;
    ib = BP.OffB;
    BSTEP(0u, D0, A0, B0)
    BSTEP(1u, D1, A1, B1)
    BSTEP(2u, D2, A2, B2)
    BSTEP(3u, D3, A3, B3)
    BSTEP(4u, D4, A4, B4)
    BSTEP(5u, D5, A5, B5)
}

void BinaryLoad(uint i, out float a, out float b)
{
    uint ia, ib;
    BinaryLocate(i, ia, ib);
#if F16
    a = LoadH(BinA, ia);
    b = LoadH(BinB, ib);
#else
    a = BinA[ia];
    b = BinB[ib];
#endif
}
