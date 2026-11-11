// What the 8-bit products apply to each int32 sum as they store it, so that the float and biased forms need no pass of their own.
// Included after the product's push constants (P.Epi, P.OffBias, P.OffScale, P.OffFBias) and its first five bindings.
//
// P.Epi: bit 0 float32 result, bit 1 int32 bias per column, bit 2 scale per column (else one scale), bit 3 float32 bias per column.
// Zero stores the raw sum. The float result is float(sum + bias) * scale (+ float bias): the same two roundings as running the multiply and the
// add as passes of their own. A target may fuse a multiply that feeds an add into one rounding (fused multiply-add), which the separate passes
// never do, so the product passes through an OR with the top bit of P.Epi, which the host keeps zero but no compiler can know: the multiply's
// result is then not the operand of an addition.

[[vk::binding(5, 0)]] StructuredBuffer<uint> EBias;
[[vk::binding(6, 0)]] StructuredBuffer<uint> EScale;
[[vk::binding(7, 0)]] StructuredBuffer<uint> EFBias;

uint StoreValue(int sum, uint col)
{
    if ((P.Epi & 3u) == 0u)
    {
        return (uint)sum;
    }

    if ((P.Epi & 2u) != 0u)
    {
        sum += (int)EBias[P.OffBias + col];
    }

    if ((P.Epi & 1u) == 0u)
    {
        return (uint)sum;
    }

    float scale = asfloat(EScale[P.OffScale + (((P.Epi & 4u) != 0u) ? col : 0u)]);
    uint bits = asuint((float)sum * scale) | (P.Epi >> 31u);
    if ((P.Epi & 8u) != 0u)
    {
        bits = asuint(asfloat(bits) + asfloat(EFBias[P.OffFBias + col]));
    }

    return bits;
}
