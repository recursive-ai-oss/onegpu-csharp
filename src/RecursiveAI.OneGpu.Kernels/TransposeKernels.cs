using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Permutations of dense tensors (ONNX <c>Transpose</c>) and the layout changes the inference kernels need: NCHW to NHWC and back,
/// and splitting or merging attention heads. Values are copied bit for bit.
/// </summary>
/// <remarks>
/// Supported element types are <see cref="float"/>, <see cref="uint"/> and <see cref="int"/> (4 bytes), <see cref="Half"/> (2 bytes) and
/// <see cref="long"/> (8 bytes). Half buffers must end on a 32-bit word (see <see cref="GpuKernelBuffers.PaddedHalfLength"/>). The output must not
/// overlap the input. The copy has no reduction order, so both <see cref="GpuKernelMode"/> values give the same bits. Not thread-safe.
/// </remarks>
public sealed class TransposeKernels : IDisposable
{
    const int Words = 17;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _pipelines = new KernelPipeline?[3];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public TransposeKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Records <c>output = transpose(input, perm)</c>: output dimension <c>i</c> is input dimension <c>perm[i]</c>.</summary>
    /// <typeparam name="T">A 2-, 4- or 8-byte element type: <see cref="Half"/>, <see cref="float"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">Input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor in <paramref name="input"/>.</param>
    /// <param name="shape">Input shape.</param>
    /// <param name="permutation">A permutation of 0..rank-1.</param>
    /// <param name="output">Receives the permuted tensor, dense row-major.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException"><paramref name="permutation"/> is not a permutation of the dimensions, a size is too large, a range reaches past its buffer, or the permutation leaves more than six dimensions that cannot be merged.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, ReadOnlySpan<int> permutation, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Transpose";
        var kind = KernelTypes.Of<T>(Name);
        var elem = kind switch
        {
            ElementKind.F16 => 0,
            ElementKind.I64 => 2,
            _ => 1,
        };
        var half = elem == 0;
        var rank = shape.Length;
        if (permutation.Length != rank)
        {
            throw new GpuKernelShapeException(Name, nameof(permutation), $"has {permutation.Length} entries for a tensor of rank {rank}");
        }

        if (rank > 32)
        {
            throw new GpuKernelShapeException(Name, nameof(shape), $"rank {rank} is above the supported 32");
        }

        var total = Check.Product(Name, nameof(shape), shape);
        Span<bool> seen = stackalloc bool[32];
        for (var i = 0; i < rank; i++)
        {
            var p = permutation[i];
            if ((uint)p >= (uint)rank || seen[p])
            {
                throw new GpuKernelShapeException(Name, $"{nameof(permutation)}[{i}]", $"{p} is not a new dimension in [0, {rank})");
            }

            seen[p] = true;
        }

        Check.Range(Name, nameof(input), input.Length, offsetInput, total, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, total, half);
        if (total == 0)
        {
            return;
        }

        // Input strides, then the output dimensions innermost first with the input stride of each; dimensions of 1 vanish and
        // neighbours that are also neighbours in the input merge into one.
        Span<long> inStride = stackalloc long[32];
        long run = 1;
        for (var i = rank - 1; i >= 0; i--)
        {
            inStride[i] = run;
            run *= shape[i];
        }

        Span<uint> w = stackalloc uint[Words];
        var m = 0;
        Span<long> dims = stackalloc long[32];
        Span<long> strides = stackalloc long[32];
        for (var i = rank - 1; i >= 0; i--)
        {
            var d = shape[permutation[i]];
            if (d == 1)
            {
                continue;
            }

            var s = inStride[permutation[i]];
            if (m > 0 && s == strides[m - 1] * dims[m - 1])
            {
                dims[m - 1] *= d;
                continue;
            }

            dims[m] = d;
            strides[m] = s;
            m++;
        }

        if (m > 6)
        {
            throw new GpuKernelShapeException(Name, nameof(permutation), $"the permutation leaves {m} dimensions that cannot be merged, above the supported 6");
        }

        for (var i = 0; i < 6; i++)
        {
            w[5 + i] = i < m ? (uint)dims[i] : 1u;
            w[11 + i] = i < m ? (uint)strides[i] : 0u;
        }

        w[0] = (uint)total;
        w[1] = (uint)offsetInput;
        w[2] = (uint)offsetOutput;
        w[4] = (uint)m;
        var pipeline = _pipelines[elem] ??= Create(elem);
        var threads = half ? ((offsetOutput + total - 1) >> 1) - (offsetOutput >> 1) + 1 : total;
        var (gx, gy) = Grid.For(_device, (threads + 255) / 256);
        w[3] = gx;
        pipeline.Record(encoder, w, gx, gy, input.Buffer, output.Buffer);
    }

    /// <summary>Records NCHW to NHWC: <c>[n, c, h, w]</c> to <c>[n, h, w, c]</c>.</summary>
    /// <typeparam name="T">An element type <see cref="Encode{T}"/> accepts.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The NCHW tensor.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="n">Batch size.</param>
    /// <param name="c">Channels.</param>
    /// <param name="h">Height.</param>
    /// <param name="w">Width.</param>
    /// <param name="output">Receives the NHWC tensor.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">As for <see cref="Encode{T}"/>.</exception>
    public void EncodeNchwToNhwc<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, int n, int c, int h, int w, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged =>
        Encode(encoder, input, offsetInput, [n, c, h, w], [0, 2, 3, 1], output, offsetOutput);

    /// <summary>Records NHWC to NCHW: <c>[n, h, w, c]</c> to <c>[n, c, h, w]</c>.</summary>
    /// <typeparam name="T">An element type <see cref="Encode{T}"/> accepts.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The NHWC tensor.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="n">Batch size.</param>
    /// <param name="c">Channels.</param>
    /// <param name="h">Height.</param>
    /// <param name="w">Width.</param>
    /// <param name="output">Receives the NCHW tensor.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">As for <see cref="Encode{T}"/>.</exception>
    public void EncodeNhwcToNchw<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, int n, int c, int h, int w, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged =>
        Encode(encoder, input, offsetInput, [n, h, w, c], [0, 3, 1, 2], output, offsetOutput);

    /// <summary>Records the split of attention heads: <c>[batch, sequence, heads * headDim]</c> to <c>[batch, heads, sequence, headDim]</c>.</summary>
    /// <typeparam name="T">An element type <see cref="Encode{T}"/> accepts.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The tensor with heads folded into the last dimension.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="sequence">Sequence length.</param>
    /// <param name="heads">Number of heads.</param>
    /// <param name="headDim">Elements per head.</param>
    /// <param name="output">Receives the head-major tensor.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">As for <see cref="Encode{T}"/>.</exception>
    public void EncodeSplitHeads<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, int batch, int sequence, int heads, int headDim, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged =>
        Encode(encoder, input, offsetInput, [batch, sequence, heads, headDim], [0, 2, 1, 3], output, offsetOutput);

    /// <summary>Records the merge of attention heads: <c>[batch, heads, sequence, headDim]</c> to <c>[batch, sequence, heads * headDim]</c>.</summary>
    /// <typeparam name="T">An element type <see cref="Encode{T}"/> accepts.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The head-major tensor.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="sequence">Sequence length.</param>
    /// <param name="heads">Number of heads.</param>
    /// <param name="headDim">Elements per head.</param>
    /// <param name="output">Receives the tensor with heads folded into the last dimension.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">As for <see cref="Encode{T}"/>.</exception>
    public void EncodeMergeHeads<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, int batch, int sequence, int heads, int headDim, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged =>
        Encode(encoder, input, offsetInput, [batch, heads, sequence, headDim], [0, 2, 1, 3], output, offsetOutput);

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => KernelPipeline.Total(_pipelines);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        for (var i = 0; i < _pipelines.Length; i++)
        {
            _pipelines[i]?.ReleaseBindings();
        }
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        for (var i = 0; i < _pipelines.Length; i++)
        {
            _pipelines[i]?.Dispose();
            _pipelines[i] = null;
        }
    }

    KernelPipeline Create(int elem)
    {
        var v = Transpose.Variants.Get(new Transpose.Key(elem: elem switch { 0 => Transpose.ElemValue.W16, 2 => Transpose.ElemValue.W64, _ => Transpose.ElemValue.W32 }));
        var e = v.TransposeOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, Words, "transpose", _maxBindGroups);
    }
}
