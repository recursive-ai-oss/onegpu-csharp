// Writes a constant (ConstantOfShape) or an arithmetic sequence (Range) into a tensor.
//
// Element types: 0 one 32-bit word (int32, float32, uint32), 1 two words (int64), 2 half (two to a word), 3 bool (four to a word).
// A thread owns whole output words, so no atomics are needed; the first and last word of a range at an odd offset keep the
// neighbouring elements they do not own.

#include "Common.hlsl"

struct GenParams
{
    uint Count;
    uint OffO;
    uint Mode;    // 0 fill, 1 range
    uint Kind;    // 0 word, 1 pair of words, 2 half, 3 bool
    uint Float;   // range of 32-bit values: 1 for float32, 0 for int32
    uint V0;      // fill value, or range start (low word)
    uint V1;      // high word
    uint D0;      // range delta (low word)
    uint D1;      // high word
    uint GridX;
    uint Keep;    // always 0xFFFFFFFF; see the float range below
};

[[vk::push_constant]] ConstantBuffer<GenParams> GP;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> GenO;

// out[i] = start + i * delta in 64 bits, wrapping.
void RangeValue(uint i, out uint lo, out uint hi)
{
    // The 64-bit product of i and delta's low word, plus i times its high word in the upper half.
    uint a0 = i & 0xFFFFu;
    uint a1 = i >> 16;
    uint b0 = GP.D0 & 0xFFFFu;
    uint b1 = GP.D0 >> 16;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    uint plo = (p00 & 0xFFFFu) | (mid << 16);
    uint phi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16) + i * GP.D1;
    lo = GP.V0 + plo;
    hi = GP.V1 + phi + ((lo < GP.V0) ? 1u : 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void GenerateOp(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * GP.GridX + gid.x) * WG + tid;
    uint shift = (GP.Kind == 3u) ? 2u : ((GP.Kind == 2u) ? 1u : 0u);
    uint first = GP.OffO + t;
    uint last = first;
    if (shift != 0u)
    {
        if (t >= (((GP.OffO + GP.Count - 1u) >> shift) - (GP.OffO >> shift) + 1u))
        {
            return;
        }

        first = ((GP.OffO >> shift) + t) << shift;
        last = first + (1u << shift) - 1u;
        if (first < GP.OffO)
        {
            first = GP.OffO;
        }

        if (last >= GP.OffO + GP.Count)
        {
            last = GP.OffO + GP.Count - 1u;
        }
    }
    else if (t >= GP.Count)
    {
        return;
    }

    uint word = first >> shift;
    uint res = (shift != 0u) ? GenO[word] : 0u;
    for (uint h = first; h <= last; h++)
    {
        uint i = h - GP.OffO;
        uint lo = GP.V0;
        uint hi = GP.V1;
        if (GP.Mode == 1u)
        {
            if (GP.Float != 0u)
            {
                // start + i * delta in float32 as two separately rounded operations, which is what the CPU reference does. A compiler may fuse
                // a multiply and an add into one rounding (and WGSL cannot forbid it), so the product goes through an integer mask that
                // is only known at run time: nothing can fuse across it.
                float p = asfloat(asuint((float)i * asfloat(GP.D0)) & GP.Keep);
                lo = asuint(asfloat(GP.V0) + p);
            }
            else
            {
                RangeValue(i, lo, hi);
            }
        }

        if (GP.Kind == 1u)
        {
            GenO[2u * h] = lo;
            GenO[2u * h + 1u] = hi;
        }
        else if (GP.Kind == 2u)
        {
            res = (res & ~(0xFFFFu << ((h & 1u) * 16u))) | ((lo & 0xFFFFu) << ((h & 1u) * 16u));
        }
        else if (GP.Kind == 3u)
        {
            res = WithByte(res, h & 3u, lo);
        }
        else
        {
            GenO[h] = lo;
        }
    }

    if (shift != 0u)
    {
        GenO[word] = res;
    }
}
