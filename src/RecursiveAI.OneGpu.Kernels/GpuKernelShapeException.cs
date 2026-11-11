namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// A shape, stride, offset or size given to a kernel is invalid for the operation: dimensions that cannot broadcast, an axis out of range,
/// a range that reaches past a buffer, or a size beyond what the kernels address. Raised before any buffer is created or work recorded.
/// </summary>
public sealed class GpuKernelShapeException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="operation">The operation, for example <c>Add</c> or <c>Softmax</c>.</param>
    /// <param name="argument">The argument or dimension at fault, for example <c>shapeB[2]</c> or <c>axis</c>.</param>
    /// <param name="message">What is wrong, with the values involved.</param>
    public GpuKernelShapeException(string operation, string argument, string message)
        : base($"{operation}: {argument}: {message}", argument)
    {
        Operation = operation;
        Argument = argument;
    }

    /// <summary>The operation that rejected its arguments.</summary>
    public string Operation { get; }

    /// <summary>The argument or dimension at fault.</summary>
    public string Argument { get; }
}
