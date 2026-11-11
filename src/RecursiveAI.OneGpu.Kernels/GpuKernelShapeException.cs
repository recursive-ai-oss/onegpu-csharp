namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// An operand's shape, stride, offset or size is invalid for a kernel: negative, inconsistent, out of the buffer's range or larger than the
/// device can dispatch. Raised before anything is recorded, so a bad shape never reaches the driver.
/// </summary>
public sealed class GpuKernelShapeException : GpuException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="operation">The operation, for example <c>Gemm</c>.</param>
    /// <param name="dimension">The offending dimension or operand, for example <c>K</c> or <c>OffsetA</c>.</param>
    /// <param name="message">What is wrong, with the values.</param>
    public GpuKernelShapeException(string operation, string dimension, string message)
        : base($"{operation}: {dimension}: {message}")
    {
        Operation = operation;
        Dimension = dimension;
    }

    /// <summary>The operation that rejected the shape.</summary>
    public string Operation { get; }

    /// <summary>The dimension or operand that is wrong.</summary>
    public string Dimension { get; }
}
