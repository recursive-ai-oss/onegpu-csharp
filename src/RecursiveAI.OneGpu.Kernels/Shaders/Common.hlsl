// Shared helpers of the kernel families.
//
// Half-precision tensors are stored as packed pairs in 32-bit words (element 2k in the low half of word k), so the kernels need no
// 16-bit storage support and run on every backend. Conversions are written out in integer arithmetic: the rounding is round to
// nearest even on every device, which the driver's own float-to-half conversion does not promise.

#define WG 256u
#define INF_BITS 0x7F800000u

// Metal compiles with fast math, which may assume a value is never NaN; testing the bits keeps the result right there too.
bool IsNaN(float x)
{
    return (asuint(x) & 0x7FFFFFFFu) > INF_BITS;
}

float PosInf()
{
    return asfloat(INF_BITS);
}

float NegInf()
{
    return asfloat(0xFF800000u);
}

float QuietNaN()
{
    return asfloat(0x7FC00000u);
}

// IEEE 754 binary16 to binary32 (exact).
float H2F(uint h)
{
    uint s = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 31u;
    uint m = h & 1023u;
    if (e == 0u)
    {
        // Zero or subnormal: m * 2^-24, exact in binary32.
        return asfloat(asuint((float)m * (1.0 / 16777216.0)) | s);
    }

    if (e == 31u)
    {
        return asfloat(s | INF_BITS | (m << 13));
    }

    return asfloat(s | ((e + 112u) << 23) | (m << 13));
}

// IEEE 754 binary32 to binary16, round to nearest even; overflow gives infinity, NaN gives a quiet NaN.
uint F2H(float f)
{
    uint x = asuint(f);
    uint s = (x >> 16) & 0x8000u;
    uint a = x & 0x7FFFFFFFu;
    if (a > INF_BITS)
    {
        return s | 0x7E00u;
    }

    if (a >= 0x477FF000u)
    {
        return s | 0x7C00u;
    }

    if (a < 0x38800000u)
    {
        if (a < 0x33000000u)
        {
            return s;
        }

        // Subnormal half: units of 2^-24.
        uint e = a >> 23;
        uint mant = (a & 0x7FFFFFu) | 0x800000u;
        uint sh = 126u - e;
        uint m = mant >> sh;
        uint rem = mant & ((1u << sh) - 1u);
        uint halfway = 1u << (sh - 1u);
        if (rem > halfway || (rem == halfway && (m & 1u) != 0u))
        {
            m += 1u;
        }

        return s | m;
    }

    uint r = a - 0x38000000u;
    r += 0xFFFu + ((r >> 13) & 1u);
    return s | (r >> 13);
}

// Number of 32-bit words that hold count halves starting at half index off.
uint HalfWords(uint off, uint count)
{
    return ((off + count - 1u) >> 1) - (off >> 1) + 1u;
}

// The 16 bits of element i of a packed half array.
uint RawH(StructuredBuffer<uint> buf, uint i)
{
    return (buf[i >> 1] >> ((i & 1u) << 4)) & 0xFFFFu;
}

float LoadH(StructuredBuffer<uint> buf, uint i)
{
    return H2F(RawH(buf, i));
}

// Bool tensors hold one byte per element (0 or 1), four to a 32-bit word with element 4k in the low byte of word k, like the packed
// halves above. A thread owns whole output words, so neighbouring threads never write the same word and no atomics are needed.

// Number of 32-bit words that hold count bytes starting at byte index off.
uint BoolWords(uint off, uint count)
{
    return ((off + count - 1u) >> 2) - (off >> 2) + 1u;
}

// Byte i of a packed bool array.
uint RawB(StructuredBuffer<uint> buf, uint i)
{
    return (buf[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
}

// Replaces byte k (0..3) of a word.
uint WithByte(uint word, uint k, uint value)
{
    return (word & ~(0xFFu << (k * 8u))) | ((value & 0xFFu) << (k * 8u));
}
