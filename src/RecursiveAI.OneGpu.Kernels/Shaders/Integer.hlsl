// Integer and bool tensors with ONNX broadcasting: arithmetic, comparison, logic and the unary Neg, Abs and Not.
//
// Representation: int32 is one 32-bit word per element, int64 is two words (low, then high), bool is one byte per element, four to a
// word. No 64-bit integer support is needed in the shader. Every operand is widened to a signed 64-bit value (int32 and bool by sign
// extension, which leaves bool as 0 or 1), the operation runs on that, and the low 32 bits are stored for int32. For add, subtract,
// multiply, negate, absolute value, division and modulo that gives exactly the wrapping 32-bit result, so one code path serves every type.
//
// Division by zero gives 0 and x / -1 is -x (wrapping), as in the CPU reference. Mod follows the divisor's sign (floor modulus),
// Fmod the dividend's (truncating remainder).

#include "Common.hlsl"

struct IntParams
{
    uint Count;
    uint OffA;
    uint OffB;
    uint OffO;
    uint Op;
    uint KindA;   // input type: 0 int32, 1 int64, 2 bool
    uint KindO;   // output type, same codes
    uint GridX;
    uint Rank;
    uint D0, D1, D2, D3, D4, D5;
    uint A0, A1, A2, A3, A4, A5;
    uint B0, B1, B2, B3, B4, B5;
};

[[vk::push_constant]] ConstantBuffer<IntParams> IP;

[[vk::binding(0, 0)]] StructuredBuffer<uint> IntA;
[[vk::binding(1, 0)]] StructuredBuffer<uint> IntB;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> IntO;

struct I64
{
    uint lo;
    uint hi;
};

I64 Make(uint lo, uint hi)
{
    I64 r;
    r.lo = lo;
    r.hi = hi;
    return r;
}

bool IsNeg(I64 a)
{
    return (a.hi >> 31) != 0u;
}

bool IsZero(I64 a)
{
    return (a.lo | a.hi) == 0u;
}

I64 Neg(I64 a)
{
    uint lo = ~a.lo + 1u;
    return Make(lo, ~a.hi + ((lo == 0u) ? 1u : 0u));
}

I64 Add(I64 a, I64 b)
{
    uint lo = a.lo + b.lo;
    return Make(lo, a.hi + b.hi + ((lo < a.lo) ? 1u : 0u));
}

I64 Sub(I64 a, I64 b)
{
    return Add(a, Neg(b));
}

// The full 64-bit product of two 32-bit values, from 16-bit limbs.
I64 MulWide(uint a, uint b)
{
    uint a0 = a & 0xFFFFu;
    uint a1 = a >> 16;
    uint b0 = b & 0xFFFFu;
    uint b1 = b >> 16;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return Make((p00 & 0xFFFFu) | (mid << 16), p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16));
}

// The low 64 bits of the product.
I64 Mul(I64 a, I64 b)
{
    I64 p = MulWide(a.lo, b.lo);
    return Make(p.lo, p.hi + a.lo * b.hi + a.hi * b.lo);
}

bool Less(I64 a, I64 b)
{
    int ah = asint(a.hi);
    int bh = asint(b.hi);
    return (ah < bh) || (a.hi == b.hi && a.lo < b.lo);
}

bool Equal(I64 a, I64 b)
{
    return a.lo == b.lo && a.hi == b.hi;
}

bool GreaterEqualU(I64 a, I64 b)
{
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
}

// Unsigned division and remainder by shift and subtract, one bit at a time from the top bit of the dividend down: the dividend is shifted left
// through the remainder, every shift is by one and the loop count is fixed, and the conditional subtract is a masked subtract. Operands that fit
// 32 bits use the hardware divide. DivOps calls this once for Div, Mod and Fmod: three copies of it in one kernel crashed one vendor's driver compiler.
void UDivMod(I64 n, I64 d, out I64 q, out I64 r)
{
    uint nl = n.lo;
    uint nh = n.hi;
    uint ql = 0u;
    uint qh = 0u;
    uint rl = 0u;
    uint rh = 0u;
    if (n.hi == 0u && d.hi == 0u)
    {
        ql = n.lo / d.lo;
        rl = n.lo - ql * d.lo;
    }
    else
    {
        [loop]
        for (uint k = 0u; k < 64u; k++)
        {
            uint carry = rh >> 31;
            rh = (rh << 1) | (rl >> 31);
            rl = (rl << 1) | (nh >> 31);
            nh = (nh << 1) | (nl >> 31);
            nl = nl << 1;
            uint ge = (carry != 0u || rh > d.hi || (rh == d.hi && rl >= d.lo)) ? 1u : 0u;
            uint m = 0u - ge;
            uint sl = d.lo & m;
            uint sh = d.hi & m;
            uint nrl = rl - sl;
            rh = rh - sh - ((rl < sl) ? 1u : 0u);
            rl = nrl;
            qh = (qh << 1) | (ql >> 31);
            ql = (ql << 1) | ge;
        }
    }

    q = Make(ql, qh);
    r = Make(rl, rh);
}

I64 Abs(I64 a)
{
    return IsNeg(a) ? Neg(a) : a;
}


// Division, Mod (the divisor's sign) and Fmod (the dividend's sign) share one unsigned division. A zero divisor gives 0 and a divisor of -1
// gives -a for division and 0 for the remainders.
I64 DivOps(uint op, I64 a, I64 b)
{
    bool bz = IsZero(b);
    bool bm1 = (b.lo == 0xFFFFFFFFu && b.hi == 0xFFFFFFFFu);
    I64 q, r;
    UDivMod(Abs(a), bz ? Make(1u, 0u) : Abs(b), q, r);
    if (op == 3u)
    {
        return bz ? Make(0u, 0u) : (bm1 ? Neg(a) : ((IsNeg(a) != IsNeg(b)) ? Neg(q) : q));
    }

    I64 m = (bz || bm1) ? Make(0u, 0u) : (IsNeg(a) ? Neg(r) : r);
    if (op == 4u && !IsZero(m) && (IsNeg(m) != IsNeg(b)))
    {
        m = Add(m, b);
    }

    return m;
}

// Element i of a packed array of the given kind, widened to 64 bits.
I64 Load(StructuredBuffer<uint> buf, uint kind, uint i)
{
    if (kind == 1u)
    {
        return Make(buf[2u * i], buf[2u * i + 1u]);
    }

    uint v = (kind == 2u) ? RawB(buf, i) : buf[i];
    return Make(v, ((kind == 0u) && ((v >> 31) != 0u)) ? 0xFFFFFFFFu : 0u);
}

I64 Apply(uint op, I64 a, I64 b)
{
    if (op >= 3u && op <= 5u)
    {
        return DivOps(op, a, b);
    }

    switch (op)
    {
        case 0u: return Add(a, b);
        case 1u: return Sub(a, b);
        case 2u: return Mul(a, b);
        case 6u: return Less(b, a) ? b : a;
        case 7u: return Less(a, b) ? b : a;
        case 8u: return Make(Equal(a, b) ? 1u : 0u, 0u);
        case 9u: return Make(Less(a, b) ? 1u : 0u, 0u);
        case 10u: return Make(Less(b, a) ? 1u : 0u, 0u);
        case 11u: return Make(Less(b, a) ? 0u : 1u, 0u);
        case 12u: return Make(Less(a, b) ? 0u : 1u, 0u);
        case 13u: return Make((!IsZero(a) && !IsZero(b)) ? 1u : 0u, 0u);
        case 14u: return Make((!IsZero(a) || !IsZero(b)) ? 1u : 0u, 0u);
        case 15u: return Make((IsZero(a) != IsZero(b)) ? 1u : 0u, 0u);
        case 16u: return Neg(a);
        case 17u: return Abs(a);
        case 18u: return Make(IsZero(a) ? 1u : 0u, 0u);
        default: return Make(0u, 0u);
    }
}

#define ISTEP(R, D, A, B) \
    if (IP.Rank == R + 1u) { ia += rem * IP.A; ib += rem * IP.B; } \
    else if (IP.Rank > R) { uint c = rem % IP.D; rem /= IP.D; ia += c * IP.A; ib += c * IP.B; }

I64 Element(uint i)
{
    uint rem = i;
    uint ia = IP.OffA;
    uint ib = IP.OffB;
    ISTEP(0u, D0, A0, B0)
    ISTEP(1u, D1, A1, B1)
    ISTEP(2u, D2, A2, B2)
    ISTEP(3u, D3, A3, B3)
    ISTEP(4u, D4, A4, B4)
    ISTEP(5u, D5, A5, B5)
    return Apply(IP.Op, Load(IntA, IP.KindA, ia), Load(IntB, IP.KindA, ib));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void IntegerOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * IP.GridX + gid.x) * WG + tid;
    if (IP.KindO == 2u)
    {
        // One thread per output word; surplus threads must not touch the buffer.
        if (t >= BoolWords(IP.OffO, IP.Count))
        {
            return;
        }

        uint word = (IP.OffO >> 2) + t;
        uint res = IntO[word];
        [unroll]
        for (uint k = 0u; k < 4u; k++)
        {
            uint h = word * 4u + k;
            if (h >= IP.OffO && h < IP.OffO + IP.Count)
            {
                res = WithByte(res, k, Element(h - IP.OffO).lo);
            }
        }

        IntO[word] = res;
        return;
    }

    if (t >= IP.Count)
    {
        return;
    }

    I64 v = Element(t);
    if (IP.KindO == 1u)
    {
        IntO[2u * (IP.OffO + t)] = v.lo;
        IntO[2u * (IP.OffO + t) + 1u] = v.hi;
    }
    else
    {
        IntO[IP.OffO + t] = v.lo;
    }
}
