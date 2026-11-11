// Converts between float32, float16, int32 and int64 without needing 64-bit integer support in the shader: an int64 is two 32-bit words
// (low, high). Conversions follow IEEE 754 round to nearest even towards floats and truncate towards zero towards integers, saturating at
// the destination's range with NaN becoming 0.

#include "Common.hlsl"

struct CastParams
{
    uint Count;
    uint OffS;
    uint OffD;
    uint Src;   // 0 f32, 1 f16, 2 i32, 3 i64
    uint Dst;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<CastParams> CP;

[[vk::binding(0, 0)]] StructuredBuffer<uint> CastS;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> CastD;

float Int64ToFloat(uint lo, uint hi)
{
    bool neg = (hi >> 31) != 0u;
    uint ml = lo;
    uint mh = hi;
    if (neg)
    {
        ml = ~lo + 1u;
        mh = ~hi + (ml == 0u ? 1u : 0u);
    }

    if (ml == 0u && mh == 0u)
    {
        return 0.0;
    }

    uint p = (mh != 0u) ? (32u + firstbithigh(mh)) : firstbithigh(ml);
    uint top;
    uint sticky = 0u;
    if (p >= 32u)
    {
        uint s = p - 31u;   // 1..32
        if (s == 32u)
        {
            top = mh;
            sticky = (ml != 0u) ? 1u : 0u;
        }
        else
        {
            top = (ml >> s) | (mh << (32u - s));
            sticky = ((ml & ((1u << s) - 1u)) != 0u) ? 1u : 0u;
        }
    }
    else
    {
        top = ml << (31u - p);
    }

    // Round the 32 significant bits to 24; the sticky bit makes ties and near-ties come out right.
    top |= sticky;
    uint mant = top >> 8;
    uint rem = top & 0xFFu;
    if (rem > 0x80u || (rem == 0x80u && (mant & 1u) != 0u))
    {
        mant += 1u;
    }

    float v = (float)mant * asfloat((p - 23u + 127u) << 23);
    return neg ? -v : v;
}

void FloatToInt64(float x, out uint lo, out uint hi)
{
    lo = 0u;
    hi = 0u;
    if (IsNaN(x))
    {
        return;
    }

    bool neg = x < 0.0;
    uint bits = asuint(x) & 0x7FFFFFFFu;
    if (bits < 0x3F800000u)
    {
        return;
    }

    if (bits >= 0x5F000000u)
    {
        // 2^63 and beyond (including infinity): saturate.
        if (neg)
        {
            hi = 0x80000000u;
        }
        else
        {
            hi = 0x7FFFFFFFu;
            lo = 0xFFFFFFFFu;
        }

        return;
    }

    uint e = (bits >> 23) - 127u;
    uint mant = (bits & 0x7FFFFFu) | 0x800000u;
    if (e >= 23u)
    {
        uint s = e - 23u;
        if (s < 32u)
        {
            lo = mant << s;
            hi = (s == 0u) ? 0u : (mant >> (32u - s));
        }
        else
        {
            hi = mant << (s - 32u);
        }
    }
    else
    {
        lo = mant >> (23u - e);
    }

    if (neg)
    {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
}

uint FloatToInt32(float x)
{
    if (IsNaN(x))
    {
        return 0u;
    }

    if (x >= 2147483648.0)
    {
        return 0x7FFFFFFFu;
    }

    if (x <= -2147483648.0)
    {
        return 0x80000000u;
    }

    return (uint)(int)x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void CastOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * CP.GridX + gid.x) * WG + tid;
    uint first = t;
    uint last = t;
    if (CP.Dst == 1u)
    {
        // One thread per 32-bit word of half-precision output.
        uint word = (CP.OffD >> 1) + t;
        first = word * 2u;
        last = first + 1u;
        if (first < CP.OffD)
        {
            first = CP.OffD;
        }

        if (last >= CP.OffD + CP.Count)
        {
            last = CP.OffD + CP.Count - 1u;
        }
    }
    else if (t >= CP.Count)
    {
        return;
    }
    else
    {
        first = CP.OffD + t;
        last = first;
    }

    uint word = first >> 1;
    uint res = (CP.Dst == 1u) ? CastD[word] : 0u;
    for (uint h = first; h <= last; h++)
    {
        uint i = h - CP.OffD;
        float f = 0.0;
        uint lo = 0u;
        uint hi = 0u;
        bool isFloat = true;
        if (CP.Src == 0u)
        {
            f = asfloat(CastS[CP.OffS + i]);
        }
        else if (CP.Src == 1u)
        {
            f = H2F(RawH(CastS, CP.OffS + i));
        }
        else if (CP.Src == 2u)
        {
            isFloat = false;
            lo = CastS[CP.OffS + i];
            hi = ((lo >> 31) != 0u) ? 0xFFFFFFFFu : 0u;
        }
        else
        {
            isFloat = false;
            lo = CastS[2u * (CP.OffS + i)];
            hi = CastS[2u * (CP.OffS + i) + 1u];
        }

        if (CP.Dst <= 1u)
        {
            float v = isFloat ? f : Int64ToFloat(lo, hi);
            if (CP.Dst == 0u)
            {
                CastD[h] = asuint(v);
            }
            else
            {
                uint k = h & 1u;
                res = (res & ~(0xFFFFu << (k * 16u))) | (F2H(v) << (k * 16u));
            }
        }
        else if (CP.Dst == 2u)
        {
            CastD[h] = isFloat ? FloatToInt32(f) : lo;
        }
        else
        {
            if (isFloat)
            {
                FloatToInt64(f, lo, hi);
            }

            CastD[2u * h] = lo;
            CastD[2u * h + 1u] = hi;
        }
    }

    if (CP.Dst == 1u)
    {
        CastD[word] = res;
    }
}
