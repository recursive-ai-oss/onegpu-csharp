using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// ONNX <c>DequantizeLinear</c> for uint8, int8 and int32 tensors, per tensor or per axis, to float32 or <see cref="Half"/>:
/// <c>y = (float)(x - zeroPoint) * scale</c>; and ONNX <c>QuantizeLinear</c> (<see cref="EncodeQuantize{TIn, TOut}"/>), float32 or <see cref="Half"/> to uint8 or int8.
/// </summary>
/// <remarks>
/// <para>
/// The arithmetic is onemodel's CPU kernel's, bit for bit for float32 output: the subtraction wraps in 32 bits, the conversion to float rounds to
/// nearest even, and one float32 multiply follows (no fused multiply-add is possible, there is no add after it). Half output rounds that float32 result to
/// nearest even. Both <see cref="GpuKernelMode"/> values give the same bits because there is no reduction. Byte tensors (uint8 as <see cref="byte"/>, int8 as
/// <see cref="sbyte"/>) are packed four to a 32-bit word, so their buffers must end on a word (<see cref="GpuKernelBuffers.PaddedBoolLength"/>); half
/// outputs likewise (<see cref="GpuKernelBuffers.PaddedHalfLength"/>). Scales are float32. An absent zero point is zero.
/// </para>
/// <para>
/// <c>QuantizeLinear</c> computes <c>y = saturate(round_half_even(x / scale) + zeroPoint)</c> as onemodel's CPU kernel does, bit for bit: the division is the correctly
/// rounded float32 division (done in integer arithmetic, because a GPU's own division is not promised to be correctly rounded, the same routine as
/// <see cref="DynamicQuantizeKernels"/>), the rounding is to nearest even (ONNX operator definition), the zero point is added in float32, the result is clamped to the
/// output type's range and NaN becomes the lowest value. Half input is widened exactly to float32 first. The output type (uint8 or int8) is the zero point's type; without a zero point
/// the caller's <c>TOut</c> decides and the zero point is 0. A scale of 0 or a non-finite value follows float32 arithmetic (x / 0 is an infinity or NaN, which saturates or goes to the lowest value).
/// Both <see cref="GpuKernelMode"/> values give the same bits.
/// </para>
/// Not thread-safe. Bind groups for buffer sets already seen are cached, so a warm call allocates nothing.
/// </remarks>
public sealed class DequantizeKernels : IDisposable
{
    const string Name = "DequantizeLinear";
    const int Words = 9;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _pipelines = new KernelPipeline?[6];
    readonly KernelPipeline?[] _quantize = new KernelPipeline?[4];
    GpuBuffer<uint>? _noZeroPoint;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public DequantizeKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Records <c>DequantizeLinear</c>.</summary>
    /// <typeparam name="TIn"><see cref="byte"/> (uint8), <see cref="sbyte"/> (int8) or <see cref="int"/>.</typeparam>
    /// <typeparam name="TOut"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="x">The quantized tensor, dense row-major.</param>
    /// <param name="offsetX">Element offset of the tensor in <paramref name="x"/>.</param>
    /// <param name="shape">Shape of the tensor.</param>
    /// <param name="axis">The axis a per-axis scale runs along, in [-rank, rank); ignored when <paramref name="perAxis"/> is false.</param>
    /// <param name="perAxis">True when the scale (and zero point) have one value per index of <paramref name="axis"/>; false for one value for the whole tensor.</param>
    /// <param name="scale">The float32 scale, one value or one per index of the axis.</param>
    /// <param name="offsetScale">Element offset of the first scale.</param>
    /// <param name="zeroPoint">The zero point, of the same element type as <paramref name="x"/>, one value or one per index of the axis; null for zero.</param>
    /// <param name="offsetZeroPoint">Element offset of the first zero point.</param>
    /// <param name="output">Receives the result, dense, with the shape of <paramref name="x"/>.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">A shape, axis or range is invalid, or an element type is not supported; names the argument.</exception>
    public void Encode<TIn, TOut>(GpuCommandEncoder encoder, GpuBuffer<TIn> x, int offsetX, ReadOnlySpan<int> shape, int axis, bool perAxis, GpuBuffer<float> scale, int offsetScale, GpuBuffer<TIn>? zeroPoint, int offsetZeroPoint, GpuBuffer<TOut> output, int offsetOutput)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(scale);
        ArgumentNullException.ThrowIfNull(output);
        int kind;
        if (typeof(TIn) == typeof(byte))
        {
            kind = 0;
        }
        else if (typeof(TIn) == typeof(sbyte))
        {
            kind = 1;
        }
        else if (typeof(TIn) == typeof(int))
        {
            kind = 2;
        }
        else
        {
            throw new GpuKernelShapeException(Name, "TIn", $"element type {typeof(TIn).Name} is not supported; use byte (uint8), sbyte (int8) or int");
        }

        var half = KernelTypes.IsHalf<TOut>(Name);
        var rank = shape.Length;
        var count = Check.Product(Name, nameof(shape), shape);
        long outer = 1, axisLen = 1, inner = 1;
        if (perAxis)
        {
            var ax = IndexSupport.Axis(Name, axis, rank);
            for (var i = 0; i < ax; i++)
            {
                outer *= shape[i];
            }

            axisLen = shape[ax];
            for (var i = ax + 1; i < rank; i++)
            {
                inner *= shape[i];
            }
        }

        var parameters = perAxis ? axisLen : 1;
        Check.Range(Name, nameof(scale), scale.Length, offsetScale, parameters);
        if (kind == 2)
        {
            Check.Range(Name, nameof(x), x.Length, offsetX, count);
        }
        else
        {
            Check.RangeBytes(Name, nameof(x), x.Length, offsetX, count);
        }

        if (zeroPoint is not null)
        {
            if (kind == 2)
            {
                Check.Range(Name, nameof(zeroPoint), zeroPoint.Length, offsetZeroPoint, parameters);
            }
            else
            {
                Check.RangeBytes(Name, nameof(zeroPoint), zeroPoint.Length, offsetZeroPoint, parameters);
            }
        }

        Check.Range(Name, nameof(output), output.Length, offsetOutput, count, half);
        if (count == 0)
        {
            return;
        }

        var pipeline = Pipeline(kind, half);
        var zero = zeroPoint is null ? (_noZeroPoint ??= _device.CreateBuffer<uint>(4, label: "dequantize unused zero point")).Buffer : zeroPoint.Buffer;
        Span<uint> w = stackalloc uint[Words];
        w[0] = (uint)count;
        w[1] = (uint)offsetX;
        w[2] = (uint)offsetOutput;
        w[3] = (uint)offsetScale;
        w[4] = (uint)offsetZeroPoint;
        w[6] = (perAxis ? 1u : 0u) | (zeroPoint is null ? 0u : 2u);
        w[7] = (uint)axisLen;
        w[8] = (uint)Math.Max(inner, 1);
        Launch.Record(_device, encoder, pipeline, w, count, offsetOutput, half ? 1 : 0, 5, x.Buffer, scale.Buffer, zero, output.Buffer);
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => KernelPipeline.Total(_pipelines, _quantize);

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        foreach (var p in _pipelines)
        {
            p?.ReleaseBindings();
        }

        foreach (var p in _quantize)
        {
            p?.ReleaseBindings();
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

        for (var i = 0; i < _quantize.Length; i++)
        {
            _quantize[i]?.Dispose();
            _quantize[i] = null;
        }

        _noZeroPoint?.Dispose();
        _noZeroPoint = null;
    }

    /// <summary>Records <c>QuantizeLinear</c> from float32 or <see cref="Half"/> to uint8 or int8; see the remarks of the class for the arithmetic.</summary>
    /// <typeparam name="TIn"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <typeparam name="TOut"><see cref="byte"/> (uint8) or <see cref="sbyte"/> (int8); the zero point, when given, has this type.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="x">The tensor to quantize, dense row-major.</param>
    /// <param name="offsetX">Element offset of the tensor in <paramref name="x"/>.</param>
    /// <param name="shape">Shape of the tensor.</param>
    /// <param name="axis">The axis a per-axis scale runs along, in [-rank, rank); ignored when <paramref name="perAxis"/> is false.</param>
    /// <param name="perAxis">True when the scale (and zero point) have one value per index of <paramref name="axis"/>; false for one value for the whole tensor.</param>
    /// <param name="scale">The float32 scale, one value or one per index of the axis.</param>
    /// <param name="offsetScale">Element offset of the first scale.</param>
    /// <param name="zeroPoint">The zero point, one value or one per index of the axis; null for zero.</param>
    /// <param name="offsetZeroPoint">Element offset of the first zero point.</param>
    /// <param name="output">Receives the result, dense, with the shape of <paramref name="x"/>; the buffer must end on a 32-bit word (<see cref="GpuKernelBuffers.PaddedBoolLength"/>).</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">A shape, axis or range is invalid, or an element type is not supported; names the argument.</exception>
    public void EncodeQuantize<TIn, TOut>(GpuCommandEncoder encoder, GpuBuffer<TIn> x, int offsetX, ReadOnlySpan<int> shape, int axis, bool perAxis, GpuBuffer<float> scale, int offsetScale, GpuBuffer<TOut>? zeroPoint, int offsetZeroPoint, GpuBuffer<TOut> output, int offsetOutput)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        const string QName = "QuantizeLinear";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(scale);
        ArgumentNullException.ThrowIfNull(output);
        bool signed;
        if (typeof(TOut) == typeof(byte))
        {
            signed = false;
        }
        else if (typeof(TOut) == typeof(sbyte))
        {
            signed = true;
        }
        else
        {
            throw new GpuKernelShapeException(QName, "TOut", $"element type {typeof(TOut).Name} is not supported; use byte (uint8) or sbyte (int8)");
        }

        var half = KernelTypes.IsHalf<TIn>(QName);
        var rank = shape.Length;
        var count = Check.Product(QName, nameof(shape), shape);
        long outer = 1, axisLen = 1, inner = 1;
        if (perAxis)
        {
            var ax = IndexSupport.Axis(QName, axis, rank);
            for (var i = 0; i < ax; i++)
            {
                outer *= shape[i];
            }

            axisLen = shape[ax];
            for (var i = ax + 1; i < rank; i++)
            {
                inner *= shape[i];
            }
        }

        var parameters = perAxis ? axisLen : 1;
        Check.Range(QName, nameof(x), x.Length, offsetX, count, half);
        Check.Range(QName, nameof(scale), scale.Length, offsetScale, parameters);
        if (zeroPoint is not null)
        {
            Check.RangeBytes(QName, nameof(zeroPoint), zeroPoint.Length, offsetZeroPoint, parameters);
        }

        Check.RangeBytes(QName, nameof(output), output.Length, offsetOutput, count);
        if (count == 0)
        {
            return;
        }

        var pipeline = QuantizePipeline(signed, half);
        var zero = zeroPoint is null ? (_noZeroPoint ??= _device.CreateBuffer<uint>(4, label: "dequantize unused zero point")).Buffer : zeroPoint.Buffer;
        Span<uint> w = stackalloc uint[Words];
        w[0] = (uint)count;
        w[1] = (uint)offsetX;
        w[2] = (uint)offsetOutput;
        w[3] = (uint)offsetScale;
        w[4] = (uint)offsetZeroPoint;
        w[6] = (perAxis ? 1u : 0u) | (zeroPoint is null ? 0u : 2u);
        w[7] = (uint)axisLen;
        w[8] = (uint)Math.Max(inner, 1);
        Launch.Record(_device, encoder, pipeline, w, count, offsetOutput, 2, 5, x.Buffer, scale.Buffer, zero, output.Buffer);
    }

    KernelPipeline QuantizePipeline(bool signed, bool half)
    {
        var i = (signed ? 2 : 0) + (half ? 1 : 0);
        if (_quantize[i] is { } ready)
        {
            return ready;
        }

        var e = Quantize.Variants.Get(new Quantize.Key(signed: signed, f16: half)).QuantizeOp;
        return _quantize[i] = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 4, Words, "quantize", _maxBindGroups);
    }

    KernelPipeline Pipeline(int kind, bool half)
    {
        var i = kind * 2 + (half ? 1 : 0);
        if (_pipelines[i] is { } ready)
        {
            return ready;
        }

        var k = kind switch { 0 => Dequantize.KindValue.U8, 1 => Dequantize.KindValue.I8, _ => Dequantize.KindValue.I32 };
        var e = Dequantize.Variants.Get(new Dequantize.Key(kind: k, f16: half)).DequantizeOp;
        return _pipelines[i] = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 4, Words, "dequantize", _maxBindGroups);
    }
}
