namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// ONNX multidirectional broadcasting (numpy rules: shapes are right-aligned, a dimension of 1 stretches) turned into the form the
/// kernels read: dimensions innermost first, with adjacent dimensions merged whenever every operand steps through them contiguously.
/// </summary>
internal static class Broadcasting
{
    /// <summary>The most dimensions a kernel receives after merging.</summary>
    public const int MaxRank = 6;

    /// <summary>The largest input rank accepted before merging.</summary>
    const int MaxInputRank = 8;

    /// <summary>The broadcast result shape of the operands, or throws when they do not broadcast.</summary>
    public static int[] ResultShape(string operation, ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        var rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            var da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            var db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            result[i] = Merge(operation, i, da, db);
        }

        return result;
    }

    static int Merge(string operation, int axis, int da, int db)
    {
        if (da < 0 || db < 0)
        {
            throw new GpuKernelShapeException(operation, $"shape[{axis}]", $"dimension {Math.Min(da, db)} is negative");
        }

        if (da == db || db == 1)
        {
            return da;
        }

        if (da == 1)
        {
            return db;
        }

        throw new GpuKernelShapeException(operation, $"shape[{axis}]", $"dimensions {da} and {db} do not broadcast");
    }

    /// <summary>
    /// Plans a kernel over up to three operands. Fills <paramref name="dims"/> (innermost first) and one stride array per operand and
    /// returns the merged rank. The second and third shapes may be empty arrays to mean a scalar, so pass <paramref name="operands"/>
    /// to say how many are real.
    /// </summary>
    public static int Plan(string operation, int operands, ReadOnlySpan<int> s0, ReadOnlySpan<int> s1, ReadOnlySpan<int> s2,
        Span<uint> dims, Span<uint> st0, Span<uint> st1, Span<uint> st2, out long count)
    {
        var rank = Math.Max(s0.Length, Math.Max(operands > 1 ? s1.Length : 0, operands > 2 ? s2.Length : 0));
        if (rank > MaxInputRank)
        {
            throw new GpuKernelShapeException(operation, "shape", $"rank {rank} is above the supported {MaxInputRank}");
        }

        Span<long> d = stackalloc long[MaxInputRank];
        Span<long> a = stackalloc long[MaxInputRank];
        Span<long> b = stackalloc long[MaxInputRank];
        Span<long> c = stackalloc long[MaxInputRank];
        long n0 = 1, n1 = 1, n2 = 1;
        long total = 1;
        for (var i = 0; i < rank; i++)
        {
            // i counts from the innermost dimension.
            var x0 = Dim(s0, i);
            var x1 = operands > 1 ? Dim(s1, i) : 1;
            var x2 = operands > 2 ? Dim(s2, i) : 1;
            var axis = rank - 1 - i;
            var r = Merge(operation, axis, Merge(operation, axis, x0, x1), x2);
            d[i] = r;
            a[i] = x0 == 1 ? 0 : n0;
            b[i] = x1 == 1 ? 0 : n1;
            c[i] = x2 == 1 ? 0 : n2;
            n0 = checked(n0 * x0);
            n1 = checked(n1 * x1);
            n2 = checked(n2 * x2);
            total = checked(total * r);
            if (total > Check.MaxElements)
            {
                throw new GpuKernelShapeException(operation, $"shape[{axis}]", $"the result holds more than {Check.MaxElements} elements");
            }
        }

        count = total;

        // Drop dimensions of 1 and merge neighbours that every operand walks contiguously.
        var m = 0;
        for (var i = 0; i < rank; i++)
        {
            if (d[i] == 1)
            {
                continue;
            }

            if (m > 0 && a[i] == a[m - 1] * d[m - 1] && b[i] == b[m - 1] * d[m - 1] && c[i] == c[m - 1] * d[m - 1])
            {
                d[m - 1] *= d[i];
                continue;
            }

            d[m] = d[i];
            a[m] = a[i];
            b[m] = b[i];
            c[m] = c[i];
            m++;
        }

        if (m > MaxRank)
        {
            throw new GpuKernelShapeException(operation, "shape", $"the operands need {m} dimensions after merging, above the supported {MaxRank}");
        }

        for (var i = 0; i < MaxRank; i++)
        {
            dims[i] = i < m ? (uint)d[i] : 1u;
            st0[i] = i < m ? (uint)a[i] : 0u;
            st1[i] = i < m ? (uint)b[i] : 0u;
            st2[i] = i < m ? (uint)c[i] : 0u;
        }

        return m;
    }

    static int Dim(ReadOnlySpan<int> shape, int fromInner) => fromInner < shape.Length ? shape[shape.Length - 1 - fromInner] : 1;
}
