using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>One input of <see cref="LayoutKernels.EncodeConcat{T}"/>: a dense tensor and its length along the concatenation axis.</summary>
/// <typeparam name="T">The element type.</typeparam>
/// <param name="Buffer">The buffer holding the tensor.</param>
/// <param name="Offset">Element offset of the tensor in the buffer.</param>
/// <param name="AxisLength">The tensor's length along the axis; its other dimensions are those of the output.</param>
public readonly record struct GpuConcatInput<T>(GpuBuffer<T> Buffer, int Offset, int AxisLength)
    where T : unmanaged;

/// <summary>
/// The ONNX data-movement operators <c>Slice</c>, <c>Expand</c> and <c>Concat</c>. Values are copied bit for bit, so both
/// <see cref="GpuKernelMode"/> values give the same bits.
/// </summary>
/// <remarks>
/// Payload types are <see cref="float"/>, <see cref="int"/>, <see cref="uint"/> (4 bytes), <see cref="Half"/> (2 bytes; buffers must end on a
/// 32-bit word, see <see cref="GpuKernelBuffers.PaddedHalfLength"/>) and <see cref="long"/> (8 bytes). The output must not overlap an input.
/// Slice and Expand share the strided copy of <see cref="TransposeKernels"/> and need at most six dimensions after neighbours that stay
/// neighbours are merged. Not thread-safe.
/// </remarks>
public sealed class LayoutKernels : IDisposable
{
    const int ConcatWords = 18;
    const int ConcatInputsPerPass = 4;

    readonly GpuDevice _device;
    readonly TransposeKernels _strided;
    readonly WideKernelPipeline?[] _concat = new WideKernelPipeline?[3];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    public LayoutKernels(GpuDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _strided = new TransposeKernels(device);
    }

    /// <summary>
    /// Works out the output shape of ONNX <c>Slice</c> and the per-dimension start and step, following the operator's rules: negative starts and ends
    /// count from the end, they are clamped to the axis, and a negative step walks backwards with its own clamping.
    /// </summary>
    /// <param name="shape">Input shape.</param>
    /// <param name="starts">Start of each sliced axis.</param>
    /// <param name="ends">End (exclusive) of each sliced axis; <c>long.MaxValue</c> and <c>long.MinValue</c> mean "to the end" in either direction.</param>
    /// <param name="axes">The axes the starts and ends apply to; empty means 0..n-1.</param>
    /// <param name="steps">Step of each sliced axis, never 0; empty means 1 for all.</param>
    /// <param name="outputShape">Receives the output shape; its length is <c>shape.Length</c>.</param>
    /// <param name="firstIndex">Receives, per dimension, the input index of the first output element along it.</param>
    /// <param name="step">Receives, per dimension, the step along it (1 for untouched dimensions).</param>
    /// <exception cref="GpuKernelShapeException">The lists disagree in length, an axis is out of range or repeated, or a step is 0.</exception>
    public static void SliceGeometry(ReadOnlySpan<int> shape, ReadOnlySpan<long> starts, ReadOnlySpan<long> ends, ReadOnlySpan<int> axes, ReadOnlySpan<long> steps, Span<int> outputShape, Span<int> firstIndex, Span<int> step) =>
        SliceGeometry("Slice", shape, starts, ends, axes, steps, outputShape, firstIndex, step);

    static void SliceGeometry(string name, ReadOnlySpan<int> shape, ReadOnlySpan<long> starts, ReadOnlySpan<long> ends, ReadOnlySpan<int> axes, ReadOnlySpan<long> steps, Span<int> outputShape, Span<int> firstIndex, Span<int> step)
    {
        var rank = shape.Length;
        if (outputShape.Length < rank || firstIndex.Length < rank || step.Length < rank)
        {
            throw new GpuKernelShapeException(name, nameof(outputShape), "the result spans must hold one entry per dimension");
        }

        if (ends.Length != starts.Length || (axes.Length != 0 && axes.Length != starts.Length) || (steps.Length != 0 && steps.Length != starts.Length))
        {
            throw new GpuKernelShapeException(name, nameof(starts), "starts, ends, axes and steps must have the same length");
        }

        if (starts.Length > rank)
        {
            throw new GpuKernelShapeException(name, nameof(starts), $"{starts.Length} entries for a tensor of rank {rank}");
        }

        for (var i = 0; i < rank; i++)
        {
            outputShape[i] = shape[i];
            firstIndex[i] = 0;
            step[i] = 1;
        }

        Span<bool> seen = stackalloc bool[Math.Max(rank, 1)];
        for (var i = 0; i < starts.Length; i++)
        {
            var axis = axes.Length == 0 ? i : IndexSupport.Axis(name, axes[i], rank);
            if (axes.Length == 0 && i >= rank)
            {
                throw new GpuKernelShapeException(name, nameof(starts), $"entry {i} has no axis");
            }

            if (seen[axis])
            {
                throw new GpuKernelShapeException(name, $"{nameof(axes)}[{i}]", $"axis {axis} appears twice");
            }

            seen[axis] = true;
            long dim = shape[axis];
            var st = steps.Length == 0 ? 1 : steps[i];
            if (st == 0)
            {
                throw new GpuKernelShapeException(name, $"{nameof(steps)}[{i}]", "a step of 0 is not allowed");
            }

            var s = starts[i];
            var e = ends[i];
            if (s < 0)
            {
                s = s < -dim ? long.MinValue : s + dim;
            }

            if (e < 0)
            {
                e = e < -dim ? long.MinValue : e + dim;
            }

            long len;
            if (st > 0)
            {
                s = Math.Clamp(s, 0, dim);
                e = Math.Clamp(e, 0, dim);
                len = e > s ? ((e - s) + st - 1) / st : 0;
            }
            else
            {
                s = Math.Clamp(s, 0, dim - 1);
                e = Math.Clamp(e, -1, dim - 1);
                len = s > e ? ((s - e) + (-st) - 1) / (-st) : 0;
            }

            outputShape[axis] = (int)len;
            firstIndex[axis] = len == 0 ? 0 : (int)s;
            step[axis] = (int)Math.Clamp(st, int.MinValue, int.MaxValue);
        }
    }

    /// <summary>Records ONNX <c>Slice</c>. Use <see cref="SliceGeometry(ReadOnlySpan{int}, ReadOnlySpan{long}, ReadOnlySpan{long}, ReadOnlySpan{int}, ReadOnlySpan{long}, Span{int}, Span{int}, Span{int})"/> to learn the output shape beforehand.</summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="shape">Input shape.</param>
    /// <param name="starts">Start of each sliced axis (negative counts from the end).</param>
    /// <param name="ends">End of each sliced axis, exclusive.</param>
    /// <param name="axes">The axes; empty means 0..n-1.</param>
    /// <param name="steps">The steps; empty means 1.</param>
    /// <param name="output">Receives the slice, dense row-major.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">A list, axis, step or range is invalid.</exception>
    public void EncodeSlice<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, ReadOnlySpan<long> starts, ReadOnlySpan<long> ends, ReadOnlySpan<int> axes, ReadOnlySpan<long> steps, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Slice";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        var half = KernelTypes.Of<T>(Name) == ElementKind.F16;
        var rank = shape.Length;
        if (rank == 0 || rank > 32)
        {
            throw new GpuKernelShapeException(Name, nameof(shape), $"rank {rank} is outside 1..32");
        }

        var inCount = Check.Product(Name, nameof(shape), shape);
        Check.Range(Name, nameof(input), input.Length, offsetInput, inCount, half);
        RequireDistinct(Name, input.Buffer, output.Buffer);
        Span<int> outShape = stackalloc int[32];
        Span<int> first = stackalloc int[32];
        Span<int> stp = stackalloc int[32];
        SliceGeometry(Name, shape, starts, ends, axes, steps, outShape, first, stp);
        var total = Check.Product(Name, "output", outShape[..rank]);
        Span<long> inStride = stackalloc long[32];
        IndexSupport.RowMajorStrides(shape, inStride);
        Span<long> strides = stackalloc long[32];
        long start = offsetInput;
        for (var i = 0; i < rank; i++)
        {
            strides[i] = inStride[i] * stp[i];
            start += (long)first[i] * inStride[i];
        }

        if (total == 0)
        {
            return;
        }

        _strided.EncodeStrided(Name, encoder, input, checked((int)start), outShape[..rank], strides[..rank], output, offsetOutput, total);
    }

    /// <summary>
    /// Records ONNX <c>Expand</c>: broadcasts the input to <paramref name="outputShape"/> (right-aligned, size-1 dimensions repeat).
    /// </summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">The input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor.</param>
    /// <param name="shape">Input shape.</param>
    /// <param name="outputShape">The broadcast result's shape: each input dimension, aligned to the right, is equal to it or 1.</param>
    /// <param name="output">Receives the broadcast tensor.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, or a range is invalid.</exception>
    public void EncodeExpand<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, ReadOnlySpan<int> outputShape, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Expand";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        var half = KernelTypes.Of<T>(Name) == ElementKind.F16;
        var rank = outputShape.Length;
        if (rank > 32 || shape.Length > rank)
        {
            throw new GpuKernelShapeException(Name, nameof(outputShape), $"rank {rank} must be at most 32 and at least the input's {shape.Length}");
        }

        var inCount = Check.Product(Name, nameof(shape), shape);
        var total = Check.Product(Name, nameof(outputShape), outputShape);
        Check.Range(Name, nameof(input), input.Length, offsetInput, inCount, half);
        RequireDistinct(Name, input.Buffer, output.Buffer);
        Span<long> inStride = stackalloc long[32];
        IndexSupport.RowMajorStrides(shape, inStride);
        Span<long> strides = stackalloc long[32];
        var lead = rank - shape.Length;
        for (var i = 0; i < rank; i++)
        {
            if (i < lead)
            {
                strides[i] = 0;
                continue;
            }

            var d = shape[i - lead];
            if (d != outputShape[i] && d != 1)
            {
                throw new GpuKernelShapeException(Name, $"{nameof(shape)}[{i - lead}]", $"dimension {d} cannot broadcast to {outputShape[i]}");
            }

            strides[i] = d == 1 ? 0 : inStride[i - lead];
        }

        if (total == 0)
        {
            return;
        }

        if (inCount == 0)
        {
            throw new GpuKernelShapeException(Name, nameof(shape), "an empty input cannot be expanded to a non-empty output");
        }

        _strided.EncodeStrided(Name, encoder, input, offsetInput, outputShape, strides[..rank], output, offsetOutput, total);
    }

    /// <summary>Records ONNX <c>Concat</c> along an axis: the inputs are laid side by side along it, in order.</summary>
    /// <typeparam name="T">A payload type: <see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="inputs">The inputs, at least one; each is dense row-major with the output's shape except along the axis.</param>
    /// <param name="outputShape">The output shape; its length along <paramref name="axis"/> is the sum of the inputs' axis lengths.</param>
    /// <param name="axis">The axis, in [-rank, rank).</param>
    /// <param name="output">Receives the result, dense row-major.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The lengths do not add up, a range is invalid, or an input is the output buffer.</exception>
    /// <remarks>Up to four inputs are copied per pass; more are copied in further passes.</remarks>
    public void EncodeConcat<T>(GpuCommandEncoder encoder, ReadOnlySpan<GpuConcatInput<T>> inputs, ReadOnlySpan<int> outputShape, int axis, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Concat";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(output);
        var width = IndexSupport.Width<T>(Name);
        var half = width == 0;
        var rank = outputShape.Length;
        if (inputs.Length == 0)
        {
            throw new GpuKernelShapeException(Name, nameof(inputs), "at least one input is needed");
        }

        if (rank == 0)
        {
            throw new GpuKernelShapeException(Name, nameof(outputShape), "a scalar cannot be concatenated");
        }

        var ax = IndexSupport.Axis(Name, axis, rank);
        var total = Check.Product(Name, nameof(outputShape), outputShape);
        long outer = 1;
        long inner = 1;
        for (var i = 0; i < ax; i++)
        {
            outer *= outputShape[i];
        }

        for (var i = ax + 1; i < rank; i++)
        {
            inner *= outputShape[i];
        }

        long sum = 0;
        for (var i = 0; i < inputs.Length; i++)
        {
            var inp = inputs[i];
            ArgumentNullException.ThrowIfNull(inp.Buffer, nameof(inputs));
            if (inp.AxisLength < 0)
            {
                throw new GpuKernelShapeException(Name, $"{nameof(inputs)}[{i}]", $"axis length {inp.AxisLength} is negative");
            }

            sum += inp.AxisLength;
            Check.Range(Name, $"{nameof(inputs)}[{i}]", inp.Buffer.Length, inp.Offset, outer * inp.AxisLength * inner, half);
            RequireDistinct(Name, inp.Buffer.Buffer, output.Buffer);
        }

        if (sum != outputShape[ax])
        {
            throw new GpuKernelShapeException(Name, nameof(outputShape), $"the inputs add up to {sum} along axis {ax} but the output has {outputShape[ax]}");
        }

        Check.Range(Name, nameof(output), output.Length, offsetOutput, total, half);
        if (total == 0)
        {
            return;
        }

        var pipeline = _concat[width] ??= CreateConcat(width);
        var (gx, gy) = Grid.For(_device, (IndexSupport.Threads(width, offsetOutput, total) + 255) / 256);
        Span<uint> w = stackalloc uint[ConcatWords];
        long start = 0;
        for (var first = 0; first < inputs.Length; first += ConcatInputsPerPass)
        {
            var n = Math.Min(ConcatInputsPerPass, inputs.Length - first);
            w.Clear();
            w[0] = (uint)total;
            w[1] = (uint)offsetOutput;
            w[2] = gx;
            w[3] = (uint)inner;
            w[4] = (uint)outputShape[ax];
            w[5] = (uint)n;
            var b0 = inputs[first].Buffer.Buffer;
            GpuBuffer b1 = b0, b2 = b0, b3 = b0;
            for (var k = 0; k < n; k++)
            {
                var inp = inputs[first + k];
                w[6 + k] = (uint)inp.Offset;
                w[10 + k] = (uint)start;
                w[14 + k] = (uint)inp.AxisLength;
                start += inp.AxisLength;
                switch (k)
                {
                    case 1: b1 = inp.Buffer.Buffer; break;
                    case 2: b2 = inp.Buffer.Buffer; break;
                    case 3: b3 = inp.Buffer.Buffer; break;
                }
            }

            pipeline.Record(encoder, w, gx, gy, 1, [b0, b1, b2, b3, output.Buffer]);
        }
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        for (var i = 0; i < _concat.Length; i++)
        {
            _concat[i]?.Dispose();
            _concat[i] = null;
        }

        _strided.Dispose();
    }

    static void RequireDistinct(string name, GpuBuffer input, GpuBuffer output)
    {
        if (ReferenceEquals(input, output))
        {
            throw new GpuKernelShapeException(name, "output", "the output must not be an input buffer");
        }
    }

    WideKernelPipeline CreateConcat(int width)
    {
        var e = Concat.Variants.Get(new Concat.Key(elem: width switch { 0 => Concat.ElemValue.W16, 2 => Concat.ElemValue.W64, _ => Concat.ElemValue.W32 })).ConcatOp;
        return new WideKernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), ConcatWords, "concat");
    }
}
