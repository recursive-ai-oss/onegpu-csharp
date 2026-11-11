// ONNX DynamicQuantizeLinear for float32 input and uint8 output, in three passes:
//  DqMinMax    each workgroup reduces a slice of the input to a minimum and a maximum (Part[2g], Part[2g + 1]); min and max do not depend on
//              the order, so the cut into workgroups cannot change the result.
//  DqFinish    one workgroup combines the partials and writes scale = (max(0, max) - min(0, min)) / 255 and zero point =
//              clamp(round_half_even(-(min / scale)), 0, 255); an all-zero input gives scale 1 and zero point 0.
//  DqQuantize  y = clamp(round_half_even(x / scale) + zeroPoint, 0, 255), NaN to 0.
// The division is a correctly rounded float32 division on the CPU, which the GPU's own division is not promised to be (Vulkan allows a few
// ulp, Metal compiles with fast math), so DivExact does it in integer arithmetic.

#include "Common.hlsl"

struct DqParams
{
    uint Count;
    uint OffX;
    uint OffY;
    uint GridX;
    uint Groups;        // workgroups of DqMinMax
    uint OffScale;
    uint OffZero;       // byte offset of the zero point in its buffer
};

[[vk::push_constant]] ConstantBuffer<DqParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<float> X;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Part;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Y;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> ScaleOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> ZeroOut;

groupshared float Smin[256];
groupshared float Smax[256];

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

[shader("compute")]
[numthreads(256, 1, 1)]
void DqMinMax(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint g = gid.x;
    float mn = PosInf();
    float mx = NegInf();
    for (uint i = g * 256u + tid; i < P.Count; i += P.Groups * 256u)
    {
        float v = X[P.OffX + i];
        if (v < mn) { mn = v; }
        if (v > mx) { mx = v; }
    }

    Smin[tid] = mn;
    Smax[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            if (Smin[tid + s] < Smin[tid]) { Smin[tid] = Smin[tid + s]; }
            if (Smax[tid + s] > Smax[tid]) { Smax[tid] = Smax[tid + s]; }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        Part[2u * g] = asuint(Smin[0]);
        Part[2u * g + 1u] = asuint(Smax[0]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DqFinish(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    float mn = PosInf();
    float mx = NegInf();
    for (uint i = tid; i < P.Groups; i += 256u)
    {
        float a = asfloat(Part[2u * i]);
        float b = asfloat(Part[2u * i + 1u]);
        if (a < mn) { mn = a; }
        if (b > mx) { mx = b; }
    }

    Smin[tid] = mn;
    Smax[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            if (Smin[tid + s] < Smin[tid]) { Smin[tid] = Smin[tid + s]; }
            if (Smax[tid + s] > Smax[tid]) { Smax[tid] = Smax[tid + s]; }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        float lo = min(0.0, Smin[0]);
        float hi = max(0.0, Smax[0]);
        float scale = DivExact(hi - lo, 255.0);
        uint zp = 0u;
        if (scale != 0.0 && !IsNaN(scale) && (asuint(scale) & 0x7FFFFFFFu) < INF_BITS)
        {
            float z = round(0.0 - DivExact(lo, scale));
            zp = z < 0.0 ? 0u : (z > 255.0 ? 255u : (uint)z);
        }

        if (scale == 0.0)
        {
            scale = 1.0;
            zp = 0u;
        }

        ScaleOut[P.OffScale] = asuint(scale);
        InterlockedAnd(ZeroOut[P.OffZero >> 2], ~(0xFFu << ((P.OffZero & 3u) << 3)));
        InterlockedOr(ZeroOut[P.OffZero >> 2], zp << ((P.OffZero & 3u) << 3));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void DqQuantize(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
    uint words = ((P.OffY + P.Count - 1u) >> 2) - (P.OffY >> 2) + 1u;
    // Keeps the declared read-write access on the two parameter buffers this pass only reads (see DqFinish).
    if (P.Count == 0xFFFFFFFFu)
    {
        ScaleOut[0] = 0u;
        ZeroOut[0] = 0u;
    }

    if (t >= words)
    {
        return;
    }

    float scale = asfloat(ScaleOut[P.OffScale]);
    float zp = (float)((ZeroOut[P.OffZero >> 2] >> ((P.OffZero & 3u) << 3)) & 0xFFu);
    uint word = (P.OffY >> 2) + t;
    uint res = Y[word];
    [unroll]
    for (uint k = 0u; k < 4u; k++)
    {
        uint b = word * 4u + k;
        if (b >= P.OffY && b < P.OffY + P.Count)
        {
            float v = round(DivExact(X[P.OffX + (b - P.OffY)], scale)) + zp;
            uint q = IsNaN(v) ? 0u : (uint)clamp(v, 0.0, 255.0);
            res = (res & ~(0xFFu << (k * 8u))) | (q << (k * 8u));
        }
    }

    Y[word] = res;
}
