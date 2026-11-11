using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// The ONNX gather operators: <c>Gather</c> (one axis, including the embedding lookup), <c>GatherElements</c> and <c>GatherND</c>. Values are copied
/// bit for bit, so <see cref="GpuKernelMode.Fast"/> and <see cref="GpuKernelMode.Deterministic"/> give the same bits.
/// </summary>
/// <remarks>
/// <para>
/// Payload types are <see cref="float"/>, <see cref="int"/>, <see cref="uint"/> (4 bytes), <see cref="Half"/> (2 bytes; buffers must end on a
/// 32-bit word, see <see cref="GpuKernelBuffers.PaddedHalfLength"/>), <see cref="byte"/> and <see cref="sbyte"/> (1 byte; likewise, see <see cref="GpuKernelBuffers.PaddedBoolLength"/>) and <see cref="long"/> (8 bytes). Indices are <see cref="int"/> or
/// <see cref="long"/>; a negative index counts from the end of its axis, as in ONNX. The output must not overlap the data.
/// </para>
/// <para>
/// An index outside its axis never reads outside the data: that output element is zero and the kernel records the position of the
/// smallest offending index in the <see cref="GpuIndexStatus"/> given to the call (when one is given). After the work has finished
/// the status turns it into a <see cref="GpuKernelIndexException"/>. The host-side <see cref="GpuKernelIndexException.Validate(string, ReadOnlySpan{int}, long)"/> gives the same error before the upload.
/// </para>
/// Not thread-safe.
/// </remarks>
public sealed class GatherKernels : IDisposable
{
    const int GatherWords = 9;
    const int ElementsWords = 26;
    const int NdWords = 22;

    readonly GpuDevice _device;
    readonly WideKernelPipeline?[] _gather = new WideKernelPipeline?[4];
    readonly WideKernelPipeline?[] _elements = new WideKernelPipeline?[4];
    readonly WideKernelPipeline?[] _nd = new WideKernelPipeline?[4];
    readonly int _maxBindGroups;
    GpuBuffer<uint>? _scratchStatus;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public GatherKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>
    /// Records ONNX <c>Gather</c>: the data tensor is <c>[outer, axisLength, inner]</c> around <paramref name="axis"/>, and
    /// <c>output[o, i, n] = data[o, indices[i], n]</c>, so the output shape is the data shape with the axis replaced by the shape of the indices.
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/>, <see cref="long"/>, <see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TIndex"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="data">The data tensor, dense row-major.</param>
    /// <param name="offsetData">Element offset of the tensor in <paramref name="data"/>.</param>
    /// <param name="dataShape">Shape of the data tensor.</param>
    /// <param name="axis">The axis to gather along, in [-rank, rank).</param>
    /// <param name="indices">The indices, in any shape (dense row-major); only their count matters here.</param>
    /// <param name="offsetIndices">Element offset of the first index.</param>
    /// <param name="indexCount">Number of indices.</param>
    /// <param name="output">Receives the gathered tensor, dense row-major.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="status">Receives the position of an invalid index; null to have invalid indices produce zeros silently.</param>
    /// <exception cref="GpuKernelShapeException">A shape, axis, count or range is invalid; names the argument.</exception>
    public void EncodeGather<T, TIndex>(GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, ReadOnlySpan<int> dataShape, int axis, GpuBuffer<TIndex> indices, int offsetIndices, int indexCount, GpuBuffer<T> output, int offsetOutput, GpuIndexStatus? status = null)
        where T : unmanaged
        where TIndex : unmanaged
    {
        const string Name = "Gather";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name, allowBytes: true);
        var wide = IndexSupport.WideIndex<TIndex>(Name);
        var rank = dataShape.Length;
        if (rank == 0)
        {
            throw new GpuKernelShapeException(Name, nameof(dataShape), "a scalar has no axis to gather along");
        }

        var ax = IndexSupport.Axis(Name, axis, rank);
        var dataCount = Check.Product(Name, nameof(dataShape), dataShape);
        if (indexCount < 0)
        {
            throw new GpuKernelShapeException(Name, nameof(indexCount), $"count {indexCount} is negative");
        }

        long outer = 1;
        long inner = 1;
        for (var i = 0; i < ax; i++)
        {
            outer *= dataShape[i];
        }

        for (var i = ax + 1; i < rank; i++)
        {
            inner *= dataShape[i];
        }

        var axisLength = dataShape[ax];
        var total = checked(outer * indexCount * inner);
        if (total > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(indexCount), $"the output holds {total} elements, more than the {Check.MaxElements} the kernels address");
        }

        var elemWidth = width == 2 ? 2L : 1L;
        IndexSupport.Range(Name, nameof(data), width, data.Length, offsetData, dataCount);
        Check.Range(Name, nameof(indices), indices.Length, offsetIndices, indexCount);
        IndexSupport.Range(Name, nameof(output), width, output.Length, offsetOutput, total);
        CheckAddressable(Name, dataCount * elemWidth, total * elemWidth, (long)indexCount * (wide ? 2 : 1));
        if (total == 0)
        {
            return;
        }

        status?.Note(Name);
        Span<uint> w = stackalloc uint[GatherWords];
        var (gx, gy) = Grid.For(_device, (IndexSupport.Threads(width, offsetOutput, total) + 255) / 256);
        w[0] = (uint)total;
        w[1] = (uint)offsetData;
        w[2] = (uint)offsetOutput;
        w[3] = (uint)offsetIndices;
        w[4] = gx;
        w[5] = wide ? 1u : 0u;
        w[6] = (uint)inner;
        w[7] = (uint)indexCount;
        w[8] = (uint)axisLength;
        var pipeline = _gather[width] ??= Create(width, 0, GatherWords);
        pipeline.Record(encoder, w, gx, gy, 1, [data.Buffer, indices.Buffer, output.Buffer, StatusBuffer(status)]);
    }

    /// <summary>
    /// Records ONNX <c>GatherElements</c>: <c>output[i0, .., ik] = data[i0, .., indices[i0, .., ik], .., ik]</c> along <paramref name="axis"/>.
    /// The output has the shape of the indices; the data and the indices have the same rank and agree on every dimension except the axis,
    /// where the indices' dimension may be anything (the data's dimension bounds the index values), and may be smaller on the other dimensions.
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/>, <see cref="long"/>, <see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TIndex"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="data">The data tensor, dense row-major.</param>
    /// <param name="offsetData">Element offset of the tensor.</param>
    /// <param name="dataShape">Shape of the data tensor (rank 1 to 8).</param>
    /// <param name="axis">The axis, in [-rank, rank).</param>
    /// <param name="indices">The indices, dense row-major.</param>
    /// <param name="offsetIndices">Element offset of the first index.</param>
    /// <param name="indicesShape">Shape of the indices.</param>
    /// <param name="output">Receives the result, with the indices' shape.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="status">Receives the position of an invalid index; null for silent zeros.</param>
    /// <exception cref="GpuKernelShapeException">A shape, axis or range is invalid.</exception>
    public void EncodeGatherElements<T, TIndex>(GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, ReadOnlySpan<int> dataShape, int axis, GpuBuffer<TIndex> indices, int offsetIndices, ReadOnlySpan<int> indicesShape, GpuBuffer<T> output, int offsetOutput, GpuIndexStatus? status = null)
        where T : unmanaged
        where TIndex : unmanaged
    {
        const string Name = "GatherElements";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name, allowBytes: true);
        var wide = IndexSupport.WideIndex<TIndex>(Name);
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
        var total = Check.Product(Name, nameof(indicesShape), indicesShape);
        var elemWidth = width == 2 ? 2L : 1L;
        IndexSupport.Range(Name, nameof(data), width, data.Length, offsetData, dataCount);
        Check.Range(Name, nameof(indices), indices.Length, offsetIndices, total);
        IndexSupport.Range(Name, nameof(output), width, output.Length, offsetOutput, total);
        CheckAddressable(Name, dataCount * elemWidth, total * elemWidth, total * (wide ? 2 : 1));
        if (total == 0)
        {
            return;
        }

        status?.Note(Name);
        Span<long> strides = stackalloc long[8];
        IndexSupport.RowMajorStrides(dataShape, strides);
        Span<uint> w = stackalloc uint[ElementsWords];
        var (gx, gy) = Grid.For(_device, (IndexSupport.Threads(width, offsetOutput, total) + 255) / 256);
        w[0] = (uint)total;
        w[1] = (uint)offsetData;
        w[2] = (uint)offsetOutput;
        w[3] = (uint)offsetIndices;
        w[4] = gx;
        w[5] = wide ? 1u : 0u;
        w[6] = (uint)rank;
        w[7] = (uint)(rank - 1 - ax);
        w[8] = (uint)dataShape[ax];
        w[9] = (uint)strides[ax];
        for (var i = 0; i < 8; i++)
        {
            // Innermost first.
            var d = rank - 1 - i;
            w[10 + i] = i < rank ? (uint)indicesShape[d] : 1u;
            w[18 + i] = i < rank ? (uint)strides[d] : 0u;
        }

        var pipeline = _elements[width] ??= Create(width, 1, ElementsWords);
        pipeline.Record(encoder, w, gx, gy, 1, [data.Buffer, indices.Buffer, output.Buffer, StatusBuffer(status)]);
    }

    /// <summary>
    /// Records ONNX <c>GatherND</c>: the last dimension of the indices holds coordinates into the data after <paramref name="batchDims"/> leading
    /// dimensions that data and indices share; the output is <c>indices.shape[:-1] + data.shape[batchDims + last:]</c>.
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/>, <see cref="long"/>, <see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TIndex"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="data">The data tensor, dense row-major.</param>
    /// <param name="offsetData">Element offset of the tensor.</param>
    /// <param name="dataShape">Shape of the data tensor.</param>
    /// <param name="indices">The index tuples, dense row-major.</param>
    /// <param name="offsetIndices">Element offset of the first index.</param>
    /// <param name="indicesShape">Shape of the indices; its last dimension is the tuple length, at most 6 and at most <c>rank(data) - batchDims</c>.</param>
    /// <param name="batchDims">Number of leading dimensions shared by data and indices.</param>
    /// <param name="output">Receives the result.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="status">Receives the position of an invalid index; null for silent zeros.</param>
    /// <exception cref="GpuKernelShapeException">A shape or range is invalid, or the tuple is longer than 6.</exception>
    public void EncodeGatherNd<T, TIndex>(GpuCommandEncoder encoder, GpuBuffer<T> data, int offsetData, ReadOnlySpan<int> dataShape, GpuBuffer<TIndex> indices, int offsetIndices, ReadOnlySpan<int> indicesShape, int batchDims, GpuBuffer<T> output, int offsetOutput, GpuIndexStatus? status = null)
        where T : unmanaged
        where TIndex : unmanaged
    {
        const string Name = "GatherND";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name, allowBytes: true);
        var wide = IndexSupport.WideIndex<TIndex>(Name);
        NdGeometry(Name, dataShape, indicesShape, batchDims, out var last, out var tuples, out var slice, out var perBatch, out var batchStride);
        var dataCount = Check.Product(Name, nameof(dataShape), dataShape);
        var total = checked(tuples * slice);
        if (total > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(indicesShape), $"the output holds {total} elements, more than the {Check.MaxElements} the kernels address");
        }

        var elemWidth = width == 2 ? 2L : 1L;
        var indexCount = checked(tuples * last);
        IndexSupport.Range(Name, nameof(data), width, data.Length, offsetData, dataCount);
        Check.Range(Name, nameof(indices), indices.Length, offsetIndices, indexCount);
        IndexSupport.Range(Name, nameof(output), width, output.Length, offsetOutput, total);
        CheckAddressable(Name, dataCount * elemWidth, total * elemWidth, indexCount * (wide ? 2 : 1));
        if (total == 0)
        {
            return;
        }

        status?.Note(Name);
        Span<uint> w = stackalloc uint[NdWords];
        var (gx, gy) = Grid.For(_device, (IndexSupport.Threads(width, offsetOutput, total) + 255) / 256);
        w[0] = (uint)total;
        w[1] = (uint)offsetData;
        w[2] = (uint)offsetOutput;
        w[3] = (uint)offsetIndices;
        w[4] = gx;
        w[5] = wide ? 1u : 0u;
        w[6] = (uint)last;
        w[7] = (uint)slice;
        w[8] = (uint)perBatch;
        w[9] = (uint)batchStride;
        FillNdDims(w[10..], dataShape, batchDims, last);
        var pipeline = _nd[width] ??= Create(width, 2, NdWords);
        pipeline.Record(encoder, w, gx, gy, 1, [data.Buffer, indices.Buffer, output.Buffer, StatusBuffer(status)]);
    }

    internal int BindGroupsCreated => WideKernelPipeline.Total(_gather, _elements, _nd);

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _gather, _elements, _nd })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.ReleaseBindings();
            }
        }
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        foreach (var set in new[] { _gather, _elements, _nd })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }

        _scratchStatus?.Dispose();
        _scratchStatus = null;
    }

    /// <summary>Shared by GatherND and ScatterND: validates the shapes and derives the tuple count, slice size, tuples per batch and batch stride.</summary>
    internal static void NdGeometry(string name, ReadOnlySpan<int> dataShape, ReadOnlySpan<int> indicesShape, int batchDims, out int last, out long tuples, out long slice, out long perBatch, out long batchStride)
    {
        last = 0;
        tuples = 0;
        slice = 0;
        perBatch = 1;
        batchStride = 0;
        var rank = dataShape.Length;
        if (rank == 0)
        {
            throw new GpuKernelShapeException(name, nameof(dataShape), "the data needs at least one dimension");
        }

        if (indicesShape.Length == 0)
        {
            throw new GpuKernelShapeException(name, nameof(indicesShape), "the indices need at least one dimension");
        }

        if (batchDims < 0 || batchDims >= Math.Min(rank, indicesShape.Length))
        {
            throw new GpuKernelShapeException(name, nameof(batchDims), $"{batchDims} must be in [0, min(rank of data, rank of indices)) = [0, {Math.Min(rank, indicesShape.Length)})");
        }

        for (var i = 0; i < batchDims; i++)
        {
            if (dataShape[i] != indicesShape[i])
            {
                throw new GpuKernelShapeException(name, $"{nameof(indicesShape)}[{i}]", $"batch dimension {indicesShape[i]} differs from the data's {dataShape[i]}");
            }
        }

        last = indicesShape[^1];
        if (last < 1 || last > 6 || last > rank - batchDims)
        {
            throw new GpuKernelShapeException(name, nameof(indicesShape), $"the tuple length {last} must be in [1, min(6, {rank - batchDims}))");
        }

        Check.Product(name, nameof(dataShape), dataShape);
        tuples = Check.Product(name, nameof(indicesShape), indicesShape[..^1]);
        slice = 1;
        for (var i = batchDims + last; i < rank; i++)
        {
            slice *= dataShape[i];
        }

        long batches = 1;
        for (var i = 0; i < batchDims; i++)
        {
            batches *= dataShape[i];
        }

        long perBatchData = 1;
        for (var i = batchDims; i < rank; i++)
        {
            perBatchData *= dataShape[i];
        }

        batchStride = perBatchData;
        perBatch = batches == 0 ? 1 : Math.Max(1, tuples / batches);
    }

    /// <summary>Writes the dimensions and strides of the addressed data dimensions (component order): 12 words.</summary>
    internal static void FillNdDims(Span<uint> w, ReadOnlySpan<int> dataShape, int batchDims, int last)
    {
        Span<long> strides = stackalloc long[Math.Max(dataShape.Length, 1)];
        IndexSupport.RowMajorStrides(dataShape, strides);
        for (var i = 0; i < 6; i++)
        {
            w[i] = i < last ? (uint)dataShape[batchDims + i] : 1u;
            w[6 + i] = i < last ? (uint)strides[batchDims + i] : 0u;
        }
    }

    internal static void CheckAddressable(string name, long dataWords, long outWords, long indexWords)
    {
        const long Limit = uint.MaxValue / 2;
        if (dataWords > Limit || outWords > Limit || indexWords > Limit)
        {
            throw new GpuKernelShapeException(name, "shape", "a tensor is too large for 32-bit word addressing");
        }
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

    WideKernelPipeline Create(int width, int which, int words)
    {
        switch (which)
        {
            case 0:
                {
                    var e = Gather.Variants.Get(new Gather.Key(elem: width switch { 3 => Gather.ElemValue.W8, 0 => Gather.ElemValue.W16, 2 => Gather.ElemValue.W64, _ => Gather.ElemValue.W32 })).GatherOp;
                    return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), words, "gather", _maxBindGroups);
                }

            case 1:
                {
                    var e = GatherElements.Variants.Get(new GatherElements.Key(elem: width switch { 3 => GatherElements.ElemValue.W8, 0 => GatherElements.ElemValue.W16, 2 => GatherElements.ElemValue.W64, _ => GatherElements.ElemValue.W32 })).GatherElementsOp;
                    return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), words, "gather elements", _maxBindGroups);
                }

            default:
                {
                    var e = GatherNd.Variants.Get(new GatherNd.Key(elem: width switch { 3 => GatherNd.ElemValue.W8, 0 => GatherNd.ElemValue.W16, 2 => GatherNd.ElemValue.W64, _ => GatherNd.ElemValue.W32 })).GatherNdOp;
                    return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), words, "gather nd", _maxBindGroups);
                }
        }
    }
}
