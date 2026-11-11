namespace RecursiveAI.OneGpu.Kernels;

/// <summary>The activation a <see cref="Gemm"/> can apply to its result in the same dispatch.</summary>
public enum GemmActivation
{
    /// <summary>No activation.</summary>
    None = 0,

    /// <summary>max(x, 0); NaN passes through. Same function as <see cref="GpuUnaryOp.Relu"/>.</summary>
    Relu = 1,

    /// <summary>x * Phi(x) with the exact normal CDF. Same function as <see cref="GpuUnaryOp.GeluErf"/>.</summary>
    GeluErf = 2,

    /// <summary>The tanh approximation of GELU. Same function as <see cref="GpuUnaryOp.GeluTanh"/>.</summary>
    GeluTanh = 3,
}

/// <summary>Which output index a <see cref="GemmEpilogue"/> bias vector follows.</summary>
public enum GemmBiasAxis
{
    /// <summary>One value per column (N values): the bias of a fully connected layer.</summary>
    Column = 0,

    /// <summary>One value per row (M values): the bias per output channel when the weights are the left operand of a convolution lowered to a product.</summary>
    Row = 1,
}

/// <summary>
/// Work a <see cref="Gemm"/> does on each output element after <c>Alpha * product + Beta * C</c>: add the bias of the element's column (or row), then
/// apply an activation. The default value does nothing. The bias and the activation use the code of <see cref="ElementwiseKernels"/> (an
/// add, then the same GELU or ReLU function), so the result matches the product followed by those kernels; on a compiler that fuses
/// a multiply with the following add, the last bit of <c>Alpha * product + bias</c> can differ when <c>Alpha</c> is not 1 and not a power of two.
/// </summary>
public readonly struct GemmEpilogue
{
    /// <summary>Creates an epilogue.</summary>
    /// <param name="bias">A float32 vector added to every row of the result, element <c>j</c> to column <c>j</c>; null for no bias.</param>
    /// <param name="offsetBias">Element offset of the first bias value in <paramref name="bias"/>.</param>
    /// <param name="activation">Applied after the bias.</param>
    /// <param name="biasAxis">Whether element <c>i</c> of the bias belongs to column <c>i</c> (N values) or to row <c>i</c> (M values) of every product in the batch.</param>
    public GemmEpilogue(GpuBuffer<float>? bias, int offsetBias = 0, GemmActivation activation = GemmActivation.None, GemmBiasAxis biasAxis = GemmBiasAxis.Column)
    {
        BiasAxis = biasAxis;
        Bias = bias;
        OffsetBias = offsetBias;
        Activation = activation;
    }

    /// <summary>The bias vector, or null.</summary>
    public GpuBuffer<float>? Bias { get; }

    /// <summary>Element offset of the first bias value.</summary>
    public int OffsetBias { get; }

    /// <summary>Whether the bias follows columns or rows.</summary>
    public GemmBiasAxis BiasAxis { get; }

    /// <summary>The activation applied after the bias.</summary>
    public GemmActivation Activation { get; }

    /// <summary>True when the epilogue changes nothing.</summary>
    public bool IsEmpty => Bias is null && Activation == GemmActivation.None;
}
