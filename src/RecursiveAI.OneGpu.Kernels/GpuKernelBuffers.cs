namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Buffer sizing rules of the kernels.</summary>
public static class GpuKernelBuffers
{
    /// <summary>
    /// The number of <see cref="Half"/> elements to allocate for <paramref name="count"/> values. The kernels read and write half-precision
    /// data in pairs inside 32-bit words, so a buffer they touch must end on a word boundary: one spare element when <paramref name="count"/> is odd.
    /// </summary>
    /// <param name="count">Number of values, at least 0.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative or too large.</exception>
    public static int PaddedHalfLength(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, int.MaxValue - 1);
        return (count + 1) & ~1;
    }

    /// <summary>
    /// The number of bool elements to allocate for <paramref name="count"/> values. Bool tensors are one byte per element (0 or 1), and the
    /// kernels read and write them four to a 32-bit word, so a buffer they touch must end on a word boundary.
    /// </summary>
    /// <param name="count">Number of values, at least 0.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative or too large.</exception>
    public static int PaddedBoolLength(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, int.MaxValue - 3);
        return (count + 3) & ~3;
    }
}
