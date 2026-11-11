// Combining an update into the output, shared by ScatterND and ScatterElements. The output already holds a copy of the data.
//
// Op: 0 replace, 1 add, 2 mul, 3 min, 4 max (the ONNX `reduction` attribute). Updates that hit the same output element are applied in
// the order of their positions in the updates tensor, by one thread (see the kernels), so a float sum is the same sum run after run.
// Min and max keep the existing value on a tie and propagate NaN. Integers wrap around on add and mul.
//
// ELEM selects the element: W16 (float16, stored as halves in 32-bit words: the thread that owns a half writes it with atomic AND/OR so
// the other half of the word, owned by another thread, is left alone), W32 (Flags bit 2 says float32, otherwise int32) or W64 (int64).

#include "IndexCommon.hlsli"

uint Combine32(uint cur, uint upd, uint op, bool isFloat)
{
    if (op == 0u)
    {
        return upd;
    }

    if (isFloat)
    {
        float a = asfloat(cur);
        float b = asfloat(upd);
        float r = a;
        if (op == 1u) { r = a + b; }
        else if (op == 2u) { r = a * b; }
        else if (op == 3u) { r = (b < a || IsNaN(b)) ? b : a; }
        else { r = (b > a || IsNaN(b)) ? b : a; }
        return asuint(r);
    }

    int ia = (int)cur;
    int ib = (int)upd;
    if (op == 1u) { return cur + upd; }
    if (op == 2u) { return cur * upd; }
    if (op == 3u) { return ib < ia ? upd : cur; }
    return ib > ia ? upd : cur;
}

// High 32 bits of the 64-bit product of two unsigned 32-bit values.
uint MulHi(uint a, uint b)
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
    return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

void Combine64(inout uint lo, inout uint hi, uint ulo, uint uhi, uint op)
{
    if (op == 0u)
    {
        lo = ulo;
        hi = uhi;
    }
    else if (op == 1u)
    {
        uint nlo = lo + ulo;
        hi = hi + uhi + (nlo < lo ? 1u : 0u);
        lo = nlo;
    }
    else if (op == 2u)
    {
        uint nhi = MulHi(lo, ulo) + lo * uhi + hi * ulo;
        lo = lo * ulo;
        hi = nhi;
    }
    else
    {
        bool less = ((int)uhi < (int)hi) || (uhi == hi && ulo < lo);
        bool greater = ((int)uhi > (int)hi) || (uhi == hi && ulo > lo);
        if ((op == 3u && less) || (op == 4u && greater))
        {
            lo = ulo;
            hi = uhi;
        }
    }
}

