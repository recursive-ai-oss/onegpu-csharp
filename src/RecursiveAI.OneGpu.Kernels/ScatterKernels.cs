using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>How an update combines with the value already in the output (the ONNX <c>reduction</c> attribute).</summary>
public enum GpuScatterReduction
{
    /// <summary>The update replaces the value; with duplicate targets the last update in index order wins.</summary>
    None = 0,

    /// <summary>The update is added.</summary>
    Add = 1,

    /// <summary>The update is multiplied in.</summary>
    Mul = 2,

    /// <summary>The smaller value is kept; on a tie the existing value stays; NaN propagates.</summary>
    Min = 3,

    /// <summary>The larger value is kept; on a tie the existing value stays; NaN propagates.</summary>
    Max = 4,
}

/// <summary>
/// The ONNX scatter operators <c>ScatterND</c> and <c>ScatterElements</c>, with every reduction of <see cref="GpuScatterReduction"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Determinism.</b> Updates that hit the same output element are applied in the order of their positions in the updates tensor, in both
/// <see cref="GpuKernelMode"/> values, with no atomic operation on the data: the output elements are split into groups, one thread owns every
/// element of its group and walks the updates in order. A float sum of duplicates is therefore the same sequential sum on every run, on every device.
/// When the caller knows the targets are distinct (<c>indicesAreDistinct</c>), every update gets a thread of its own and the order cannot matter; the kernel
/// does not check the promise, and duplicates then give an unspecified (but memory-safe) result.
/// </para>
/// <para>
/// Payload types are <see cref="float"/>, <see cref="Half"/>, <see cref="int"/> and <see cref="long"/> (integers wrap around on Add and Mul; Half
/// arithmetic is done in float32 and rounded to half after every update, which is what a sequential half update does). Indices are
/// <see cref="int"/> or <see cref="long"/>, negative values counting from the end of the axis. An invalid index is skipped and recorded in the
/// <see cref="GpuIndexStatus"/> (see <see cref="GatherKernels"/>). The output is first filled with a copy of the data (one extra pass), and may
/// be the data buffer at the same offset to update in place. Not thread-safe.
/// </para>
/// </remarks>
public sealed class ScatterKernels : IDisposable
{
    const int NdWords = 25;
    const int ElementsWords = 29;
    const long NdThreadsPerBucketTarget = 8192;
    const long MaxElementBuckets = 4096;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly TransposeKernels _copy;
    readonly WideKernelPipeline?[] _nd = new WideKernelPipeline?[4];
    readonly WideKernelPipeline?[] _elements = new WideKernelPipeline?[4];
    GpuBuffer<uint>? _scratchStatus;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public ScatterKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        _copy = new TransposeKernels(device, _maxBindGroups);
    }

    /// <summary>
    /// Records ONNX <c>ScatterND</c>: <c>output = data</c>, then for each index tuple the slice of the output it addresses is combined with the matching
    /// slice of <paramref name="updates"/>.
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <typeparam name="TIndex"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="data">The tensor to update.</param>
    /// <param name="offsetData">Element offset of the tensor.</param>
    /// <param name="dataShape">Shape of the data (and of the output).</param>
    /// <param name="indices">The index tuples, dense row-major.</param>
    /// <param name="offsetIndices">Element offset of the first index.</param>
    /// <param name="indicesShape">Shape of the indices; the last dimension is the tuple length, at most 6 and at most the data's rank.</param>
    /// <param name="updates">The updates: shape <c>indices.shape[:-1] + data.shape[last:]</c>, dense row-major.</param>
    /// <param name="offsetUpdates">Element offset of the first update.</param>
    /// <param name="output">Receives the result; it may be <paramref name="data"/> at the same offset.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="reduction">How updates combine with the existing value.</param>
    /// <param name="indicesAreDistinct">True when no two tuples address the same slice; then every update runs in parallel.</param>
    /// <param name="status">Receives the position of an invalid index; null to skip them silently.</param>
    /// <exception cref="GpuKernelShapeException">A shape, type or range is invalid, or the output overlaps the data without being the same range.</exception>
    public void EncodeScatterNd<T, TIndex>(GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, ReadOnlySpan<int> dataShape, GpuBuffer<TIndex> indices, int offsetIndices, ReadOnlySpan<int> indicesShape, GpuBuffer<T> updates, int offsetUpdates, GpuBuffer<T> output, int offsetOutput, GpuScatterReduction reduction = GpuScatterReduction.None, bool indicesAreDistinct = false, GpuIndexStatus? status = null)
        where T : unmanaged
        where TIndex : unmanaged
    {
        const string Name = "ScatterND";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(updates);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name, allowUnsigned: false);
        var wide = IndexSupport.WideIndex<TIndex>(Name);
        CheckReduction(Name, reduction);
        GatherKernels.NdGeometry(Name, dataShape, indicesShape, 0, out var last, out var tuples, out var slice, out _, out _);
        var dataCount = Check.Product(Name, nameof(dataShape), dataShape);
        var updateCount = checked(tuples * slice);
        var indexCount = checked(tuples * last);
        var half = width == 0;
        Check.Range(Name, nameof(data), data.Length, offsetData, dataCount, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, dataCount, half);
        Check.Range(Name, nameof(indices), indices.Length, offsetIndices, indexCount);
        Check.Range(Name, nameof(updates), updates.Length, offsetUpdates, updateCount, half);
        GatherKernels.CheckAddressable(Name, dataCount * (width == 2 ? 2 : 1), updateCount * (width == 2 ? 2 : 1), indexCount * (wide ? 2 : 1));
        CopyData(Name, encoder, data, offsetData, output, offsetOutput, dataCount);
        if (updateCount == 0 || dataCount == 0)
        {
            return;
        }

        status?.Note(Name);
        long buckets = 1;
        long threads;
        if (indicesAreDistinct)
        {
            threads = updateCount;
        }
        else
        {
            buckets = Math.Min(tuples, Math.Max(1, NdThreadsPerBucketTarget / slice));
            threads = buckets * slice;
        }

        Span<uint> w = stackalloc uint[NdWords];
        var (gx, gy) = Grid.For(_device, (threads + 255) / 256);
        w[0] = (uint)threads;
        w[1] = (uint)offsetUpdates;
        w[2] = (uint)offsetIndices;
        w[3] = (uint)offsetOutput;
        w[4] = gx;
        w[5] = Flags<T>(wide, indicesAreDistinct);
        w[6] = (uint)reduction;
        w[7] = (uint)last;
        w[8] = (uint)slice;
        w[9] = 1;
        w[10] = 0;
        w[11] = (uint)tuples;
        w[12] = (uint)buckets;
        GatherKernels.FillNdDims(w[13..], dataShape, 0, last);
        var pipeline = _nd[width] ??= CreateNd(width);
        pipeline.Record(encoder, w, gx, gy, 1, [updates.Buffer, indices.Buffer, output.Buffer, StatusBuffer(status)]);
    }

    /// <summary>
    /// Records ONNX <c>ScatterElements</c> along <paramref name="axis"/>: <c>output = data</c>, then
    /// <c>output[i0, .., indices[i0, .., ik], .., ik]</c> is combined with <c>updates[i0, .., ik]</c> for every position of the indices.
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <typeparam name="TIndex"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="data">The tensor to update.</param>
    /// <param name="offsetData">Element offset of the tensor.</param>
    /// <param name="dataShape">Shape of the data (rank 1 to 8).</param>
    /// <param name="axis">The axis, in [-rank, rank).</param>
    /// <param name="indices">The indices, dense row-major.</param>
    /// <param name="offsetIndices">Element offset of the first index.</param>
    /// <param name="indicesShape">Shape of the indices and the updates; no larger than the data off the axis.</param>
    /// <param name="updates">The updates, dense row-major, with the indices' shape.</param>
    /// <param name="offsetUpdates">Element offset of the first update.</param>
    /// <param name="output">Receives the result; it may be <paramref name="data"/> at the same offset.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="reduction">How updates combine with the existing value.</param>
    /// <param name="indicesAreDistinct">True when no two updates address the same element; then every update runs in parallel.</param>
    /// <param name="status">Receives the position of an invalid index; null to skip them silently.</param>
    /// <exception cref="GpuKernelShapeException">A shape, type or range is invalid.</exception>
    public void EncodeScatterElements<T, TIndex>(GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, ReadOnlySpan<int> dataShape, int axis, GpuBuffer<TIndex> indices, int offsetIndices, ReadOnlySpan<int> indicesShape, GpuBuffer<T> updates, int offsetUpdates, GpuBuffer<T> output, int offsetOutput, GpuScatterReduction reduction = GpuScatterReduction.None, bool indicesAreDistinct = false, GpuIndexStatus? status = null)
        where T : unmanaged
        where TIndex : unmanaged
    {
        const string Name = "ScatterElements";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(updates);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name, allowUnsigned: false);
        var wide = IndexSupport.WideIndex<TIndex>(Name);
        CheckReduction(Name, reduction);
        var rank = dataShape.Length;
        if (rank == 0 || rank > 8)
        {
            throw new GpuKernelShapeException(Name, nameof(dataShape), $"rank {rank} is outside 1..8");
        }

        if (indicesShape.Length != rank)
        {
            throw new GpuKernelShapeException(Name, nameof(indicesShape), $"rank {indicesShape.Length} differs from the data's rank {rank}");
        }

        var ax = IndexSupport.Axis(Name, axis, rank);
        for (var i = 0; i < rank; i++)
        {
            if (i != ax && indicesShape[i] > dataShape[i])
            {
                throw new GpuKernelShapeException(Name, $"{nameof(indicesShape)}[{i}]", $"dimension {indicesShape[i]} is larger than the data's {dataShape[i]}");
            }
        }

        var dataCount = Check.Product(Name, nameof(dataShape), dataShape);
        var count = Check.Product(Name, nameof(indicesShape), indicesShape);
        var half = width == 0;
        Check.Range(Name, nameof(data), data.Length, offsetData, dataCount, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, dataCount, half);
        Check.Range(Name, nameof(indices), indices.Length, offsetIndices, count);
        Check.Range(Name, nameof(updates), updates.Length, offsetUpdates, count, half);
        GatherKernels.CheckAddressable(Name, dataCount * (width == 2 ? 2 : 1), count * (width == 2 ? 2 : 1), count * (wide ? 2 : 1));
        CopyData(Name, encoder, data, offsetData, output, offsetOutput, dataCount);
        if (count == 0 || dataCount == 0)
        {
            return;
        }

        status?.Note(Name);
        long buckets = 1;
        long threads = count;
        if (!indicesAreDistinct)
        {
            buckets = Math.Clamp(count / 16, 1, MaxElementBuckets);
            threads = buckets;
        }

        Span<long> strides = stackalloc long[8];
        IndexSupport.RowMajorStrides(dataShape, strides);
        Span<uint> w = stackalloc uint[ElementsWords];
        var (gx, gy) = Grid.For(_device, (threads + 255) / 256);
        w[0] = (uint)threads;
        w[1] = (uint)offsetUpdates;
        w[2] = (uint)offsetIndices;
        w[3] = (uint)offsetOutput;
        w[4] = gx;
        w[5] = Flags<T>(wide, indicesAreDistinct);
        w[6] = (uint)reduction;
        w[7] = (uint)rank;
        w[8] = (uint)(rank - 1 - ax);
        w[9] = (uint)dataShape[ax];
        w[10] = (uint)strides[ax];
        w[11] = (uint)count;
        w[12] = (uint)buckets;
        for (var i = 0; i < 8; i++)
        {
            var d = rank - 1 - i;
            w[13 + i] = i < rank ? (uint)indicesShape[d] : 1u;
            w[21 + i] = i < rank ? (uint)strides[d] : 0u;
        }

        var pipeline = _elements[width] ??= CreateElements(width);
        pipeline.Record(encoder, w, gx, gy, 1, [updates.Buffer, indices.Buffer, output.Buffer, StatusBuffer(status)]);
    }

    internal int BindGroupsCreated => WideKernelPipeline.Total(_nd, _elements) + _copy.BindGroupsCreated;

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _nd, _elements })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.ReleaseBindings();
            }
        }

        _copy.ReleaseBindings();
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        foreach (var set in new[] { _nd, _elements })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }

        _copy.Dispose();
        _scratchStatus?.Dispose();
        _scratchStatus = null;
    }

    static uint Flags<T>(bool wide, bool distinct)
        where T : unmanaged =>
        (wide ? 1u : 0u) | (distinct ? 2u : 0u) | (typeof(T) == typeof(float) ? 4u : 0u);

    static void CheckReduction(string name, GpuScatterReduction reduction)
    {
        if (reduction is < GpuScatterReduction.None or > GpuScatterReduction.Max)
        {
            throw new GpuKernelShapeException(name, nameof(reduction), $"{reduction} is not a reduction");
        }
    }

    void CopyData<T>(string name, GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, GpuBuffer<T> output, int offsetOutput, long count)
        where T : unmanaged
    {
        if (count == 0)
        {
            return;
        }

        if (ReferenceEquals(data.Buffer, output.Buffer))
        {
            if (offsetData == offsetOutput)
            {
                return;
            }

            if (offsetData < offsetOutput + count && offsetOutput < offsetData + count)
            {
                throw new GpuKernelShapeException(name, "output", "the output overlaps the data without being the same range");
            }
        }

        _copy.Encode(encoder, data, offsetData, [(int)count], [0], output, offsetOutput);
    }

    GpuBuffer StatusBuffer(GpuIndexStatus? status)
    {
        if (status is not null)
        {
            return status.Buffer.Buffer;
        }

        _scratchStatus ??= _device.CreateBuffer<uint>(1, label: "unobserved index status");
        return _scratchStatus.Buffer;
    }

    WideKernelPipeline CreateNd(int width)
    {
        var e = ScatterNd.Variants.Get(new ScatterNd.Key(elem: width switch { 0 => ScatterNd.ElemValue.W16, 2 => ScatterNd.ElemValue.W64, _ => ScatterNd.ElemValue.W32 })).ScatterNdOp;
        return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), NdWords, "scatter nd", _maxBindGroups);
    }

    WideKernelPipeline CreateElements(int width)
    {
        var e = ScatterElements.Variants.Get(new ScatterElements.Key(elem: width switch { 0 => ScatterElements.ElemValue.W16, 2 => ScatterElements.ElemValue.W64, _ => ScatterElements.ElemValue.W32 })).ScatterElementsOp;
        return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), ElementsWords, "scatter elements", _maxBindGroups);
    }
}
