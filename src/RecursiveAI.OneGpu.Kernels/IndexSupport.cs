namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Element and index handling shared by the gather, scatter and layout kernels.</summary>
internal static class IndexSupport
{
    /// <summary>Selects the element width of a payload: 0 for halves, 1 for 4-byte values, 2 for 8-byte values.</summary>
    public static int Width<T>(string operation, bool allowUnsigned = true)
        where T : unmanaged
    {
        var kind = KernelTypes.Of<T>(operation);
        if (kind == ElementKind.U32 && !allowUnsigned)
        {
            throw new GpuKernelShapeException(operation, "T", "uint is not supported here; use int");
        }

        return kind switch
        {
            ElementKind.F16 => 0,
            ElementKind.I64 => 2,
            _ => 1,
        };
    }

    /// <summary>True for int64 indices, false for int32; anything else is refused.</summary>
    public static bool WideIndex<TIndex>(string operation)
        where TIndex : unmanaged
    {
        if (typeof(TIndex) == typeof(int))
        {
            return false;
        }

        if (typeof(TIndex) == typeof(long))
        {
            return true;
        }

        throw new GpuKernelShapeException(operation, "TIndex", $"index type {typeof(TIndex).Name} is not supported; use int or long");
    }

    /// <summary>Normalises an axis in [-rank, rank).</summary>
    public static int Axis(string operation, int axis, int rank)
    {
        if (axis < -rank || axis >= rank)
        {
            throw new GpuKernelShapeException(operation, "axis", $"axis {axis} is outside [{-rank}, {rank})");
        }

        return axis < 0 ? axis + rank : axis;
    }

    /// <summary>The element count of one buffer range in 32-bit-word units a half buffer needs: threads for a dense output of <paramref name="count"/> elements.</summary>
    public static long Threads(int width, int offset, long count) =>
        width == 0 ? (((long)offset + count - 1) >> 1) - (offset >> 1) + 1 : count;

    public static int RowMajorStrides(ReadOnlySpan<int> shape, Span<long> strides)
    {
        long run = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = run;
            run *= shape[i];
        }

        return shape.Length;
    }
}
