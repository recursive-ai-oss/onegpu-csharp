// Correctly rounded float32 division in integer arithmetic, shared by the quantizers. Include after Common.hlsl.

// Splits a finite nonzero float into a 24-bit mantissa with its top bit set and an exponent, normalising subnormals.
void Unpack(uint bits, out uint m, out int e)
{
    e = (int)((bits >> 23) & 255u);
    m = bits & 0x7FFFFFu;
    if (e == 0)
    {
        e = 1;
        while (m < 0x800000u)
        {
            m <<= 1;
            e -= 1;
        }
    }
    else
    {
        m |= 0x800000u;
    }
}

// x / s rounded to nearest even, in integer arithmetic: long division of the mantissas, 26 quotient bits plus a sticky bit from the
// remainder, then rounding. The target's own division is not promised to be correctly rounded (Vulkan allows a few ulp, Metal compiles with
// fast math) and a fused multiply-add correction is not promised to be fused, but integer arithmetic is the same everywhere.
// Zero, infinities, NaNs and results outside the normal range use the target's division; the quantizer meets none of them
// in practice (the scale is a range divided by 255, so a quotient is below 2^8 in size).
float DivExact(float x, float s)
{
    uint xb = asuint(x);
    uint sb = asuint(s);
    uint xa = xb & 0x7FFFFFFFu;
    uint sa = sb & 0x7FFFFFFFu;
    if (xa == 0u && sa != 0u && sa <= INF_BITS)
    {
        return asfloat((xb ^ sb) & 0x80000000u);
    }

    if (xa == 0u || sa == 0u || xa >= INF_BITS || sa >= INF_BITS)
    {
        return x / s;
    }

    uint ma, mb;
    int ea, eb;
    Unpack(xa, ma, ea);
    Unpack(sa, mb, eb);
    uint r = ma;
    uint q = 0u;
    if (r >= mb)
    {
        q = 1u;
        r -= mb;
    }

    for (uint i = 0u; i < 26u; i++)
    {
        r <<= 1;
        q <<= 1;
        if (r >= mb)
        {
            q |= 1u;
            r -= mb;
        }
    }

    // q is the quotient times 2^26, truncated: 27 bits when ma >= mb, 26 when ma < mb.
    int e = ea - eb + 127;
    uint shift = 2u;
    if (q >= 0x4000000u)
    {
        shift = 3u;
    }
    else
    {
        e -= 1;
    }

    uint m = q >> shift;
    uint rest = q & ((1u << shift) - 1u);
    uint half = 1u << (shift - 1u);
    bool sticky = r != 0u;
    if (rest > half || (rest == half && (sticky || (m & 1u) != 0u)))
    {
        m += 1u;
        if (m == 0x1000000u)
        {
            m >>= 1;
            e += 1;
        }
    }

    if (e < 1 || e > 254)
    {
        return x / s;
    }

    return asfloat(((xb ^ sb) & 0x80000000u) | ((uint)e << 23) | (m & 0x7FFFFFu));
}
