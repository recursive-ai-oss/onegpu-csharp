// Epilogue shared by the matrix-product kernels: after alpha * product + beta * C, add the bias of the column, or of the row with Flags bit 3 (Flags bit 2 says a bias is given) and apply an
// activation (Act: 0 none, 1 Relu, 2 Gelu with the exact normal CDF, 3 Gelu with the tanh approximation). The functions are the ones the
// element-wise kernels use, so the fused result is the one the separate passes would give. The including file declares Bias and P
// (GemmParams) first, Bias before the push-constant block: Metal numbers resources in declaration order and the host binds A, B, C and Bias
// in slots 0 to 3.

float GemmEpilogue(float r, uint row, uint col)
{
    if ((P.Flags & 4u) != 0u)
    {
        r = r + Bias[P.BiasOffset + (((P.Flags & 8u) != 0u) ? row : col)];
    }

    if (P.Act == 1u) { r = ReluF(r); }
    else if (P.Act == 2u) { r = GeluErfF(r); }
    else if (P.Act == 3u) { r = GeluTanhF(r); }
    return r;
}
