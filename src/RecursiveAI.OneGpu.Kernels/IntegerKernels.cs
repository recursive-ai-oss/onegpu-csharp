using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Integer and bool tensors: arithmetic, comparison and logic with ONNX broadcasting, plus the generators <c>Range</c> and <c>ConstantOfShape</c>.
/// These are the operations around an attention mask and around <c>Shape</c> arithmetic. Every result is bit-exact against the CPU.
/// </summary>
/// <remarks>
/// <para>Representation. <see cref="int"/> is one 32-bit word per element. <see cref="long"/> is two words (low, then high), handled with
/// 32-bit arithmetic, so the device needs no 64-bit integer support. A bool tensor is a <see cref="GpuBuffer{T}"/> of <see cref="byte"/>
/// holding 0 or 1 per element, packed four to a 32-bit word: allocate <see cref="GpuKernelBuffers.PaddedBoolLength"/> elements. A comparison
/// that writes a bool changes only the bytes it owns; neighbouring bytes of a shared word are kept.</para>
/// <para>Shapes are dense, row-major and broadcast as in ONNX (up to 6 dimensions after merging, inputs up to rank 8). An output may overlap an input only
/// with the same offset and shape. There is one variant for both <see cref="GpuKernelMode"/> values: integer arithmetic has no order to fix. Not thread-safe.</para>
/// </remarks>
public sealed class IntegerKernels : IDisposable
{
    const int IntegerWords = 27;
    const int GenerateWords = 11;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    KernelPipeline? _integer;
    KernelPipeline? _generate;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public IntegerKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Records <c>output = a op b</c> over int32 or int64 tensors with broadcasting.</summary>
    /// <typeparam name="T"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The operation.</param>
    /// <param name="a">First operand.</param>
    /// <param name="offsetA">Element offset of the first operand.</param>
    /// <param name="shapeA">Shape of the first operand; empty for a scalar.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives the broadcast shape's elements, densely.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, an operand reaches past its buffer, or <typeparamref name="T"/> is not supported.</exception>
    public void EncodeBinary<T>(GpuCommandEncoder encoder, GpuIntegerOp op, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "IntegerBinary";
        var kind = Kind<T>(Name, allowBool: false);
        EnumRange(Name, nameof(op), (int)op, (int)GpuIntegerOp.Max);
        Run(Name, encoder, (int)op, kind, kind, a, offsetA, shapeA, b, offsetB, shapeB, output, offsetOutput);
    }

    /// <summary>Records <c>output = a op b</c> as a bool tensor, over int32, int64 or bool operands, with broadcasting.</summary>
    /// <typeparam name="T"><see cref="int"/>, <see cref="long"/> or <see cref="byte"/> (bool; 0 compares below 1).</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The comparison.</param>
    /// <param name="a">First operand.</param>
    /// <param name="offsetA">Element offset of the first operand.</param>
    /// <param name="shapeA">Shape of the first operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives a bool byte per element of the broadcast shape (<see cref="GpuKernelBuffers.PaddedBoolLength"/> elements).</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeCompare<T>(GpuCommandEncoder encoder, GpuCompareOp op, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<byte> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "IntegerCompare";
        var kind = Kind<T>(Name, allowBool: true);
        EnumRange(Name, nameof(op), (int)op, (int)GpuCompareOp.GreaterOrEqual);
        Run(Name, encoder, 8 + (int)op, kind, 2, a, offsetA, shapeA, b, offsetB, shapeB, output, offsetOutput);
    }

    /// <summary>Records <c>output = a op b</c> over bool tensors with broadcasting.</summary>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The operation.</param>
    /// <param name="a">First operand (bool bytes).</param>
    /// <param name="offsetA">Element offset of the first operand.</param>
    /// <param name="shapeA">Shape of the first operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives the result; any nonzero input byte counts as true and the output holds 0 or 1.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeLogic(GpuCommandEncoder encoder, GpuLogicOp op, GpuBuffer<byte> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<byte> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<byte> output, int offsetOutput)
    {
        const string Name = "Logic";
        EnumRange(Name, nameof(op), (int)op, (int)GpuLogicOp.Xor, (int)GpuLogicOp.And);
        Run(Name, encoder, (int)op, 2, 2, a, offsetA, shapeA, b, offsetB, shapeB, output, offsetOutput);
    }

    /// <summary>Records <c>y = f(x)</c> over <paramref name="count"/> consecutive int32 or int64 elements.</summary>
    /// <typeparam name="T"><see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The function.</param>
    /// <param name="x">Input.</param>
    /// <param name="offsetX">Element offset of the first input element.</param>
    /// <param name="y">Output.</param>
    /// <param name="offsetY">Element offset of the first output element.</param>
    /// <param name="count">Number of elements.</param>
    /// <exception cref="GpuKernelShapeException">A range reaches past its buffer, or the count is negative or too large.</exception>
    public void EncodeUnary<T>(GpuCommandEncoder encoder, GpuIntegerUnaryOp op, GpuBuffer<T> x, int offsetX, GpuBuffer<T> y, int offsetY, int count)
        where T : unmanaged
    {
        const string Name = "IntegerUnary";
        var kind = Kind<T>(Name, allowBool: false);
        EnumRange(Name, nameof(op), (int)op, (int)GpuIntegerUnaryOp.Abs, (int)GpuIntegerUnaryOp.Neg);
        Span<int> shape = [count];
        Run(Name, encoder, (int)op, kind, kind, x, offsetX, shape, x, offsetX, shape, y, offsetY);
    }

    /// <summary>Records <c>y = not x</c> over <paramref name="count"/> consecutive bool elements: 1 where the input byte is 0, else 0.</summary>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="x">Input bool bytes.</param>
    /// <param name="offsetX">Element offset of the first input element.</param>
    /// <param name="y">Output bool bytes.</param>
    /// <param name="offsetY">Element offset of the first output element.</param>
    /// <param name="count">Number of elements.</param>
    /// <exception cref="GpuKernelShapeException">A range reaches past its buffer, or the count is negative or too large.</exception>
    public void EncodeNot(GpuCommandEncoder encoder, GpuBuffer<byte> x, int offsetX, GpuBuffer<byte> y, int offsetY, int count)
    {
        Span<int> shape = [count];
        Run("Not", encoder, 18, 2, 2, x, offsetX, shape, x, offsetX, shape, y, offsetY);
    }

    /// <summary>Records a fill: <paramref name="count"/> consecutive elements all set to <paramref name="value"/> (ONNX <c>ConstantOfShape</c>).</summary>
    /// <typeparam name="T"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="output">Receives the values. A <see cref="Half"/> buffer needs <see cref="GpuKernelBuffers.PaddedHalfLength"/> elements.</param>
    /// <param name="offset">Element offset of the first element.</param>
    /// <param name="count">Number of elements.</param>
    /// <param name="value">The value, copied bit for bit.</param>
    /// <exception cref="GpuKernelShapeException">The range reaches past the buffer, or <typeparamref name="T"/> is not supported.</exception>
    public void EncodeFill<T>(GpuCommandEncoder encoder, GpuBuffer<T> output, int offset, int count, T value)
        where T : unmanaged
    {
        const string Name = "Fill";
        var kind = KernelTypes.Of<T>(Name);
        uint v0, v1 = 0;
        int type;
        switch (kind)
        {
            case ElementKind.F32:
                v0 = BitConverter.SingleToUInt32Bits((float)(object)value);
                type = 0;
                break;
            case ElementKind.F16:
                v0 = BitConverter.HalfToUInt16Bits((Half)(object)value);
                type = 2;
                break;
            case ElementKind.I32:
                v0 = (uint)(int)(object)value;
                type = 0;
                break;
            case ElementKind.U32:
                v0 = (uint)(object)value;
                type = 0;
                break;
            case ElementKind.I64:
                var l = (long)(object)value;
                v0 = (uint)l;
                v1 = (uint)(l >> 32);
                type = 1;
                break;
            default:
                throw new GpuKernelShapeException(Name, "T", $"element type {typeof(T).Name} is not supported; use float, Half, int, uint or long");
        }

        Check.Range(Name, "output", output.Length, offset, count, kind == ElementKind.F16);
        WriteGenerated(Name, encoder, output.Buffer, offset, count, 0, type, 0, v0, v1, 0, 0);
    }

    /// <summary>Records a fill of <paramref name="count"/> bool elements.</summary>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="output">Receives the bytes (<see cref="GpuKernelBuffers.PaddedBoolLength"/> elements).</param>
    /// <param name="offset">Element offset of the first element.</param>
    /// <param name="count">Number of elements.</param>
    /// <param name="value">The value; written as 1 or 0.</param>
    /// <exception cref="GpuKernelShapeException">The range reaches past the buffer.</exception>
    public void EncodeFillBool(GpuCommandEncoder encoder, GpuBuffer<byte> output, int offset, int count, bool value)
    {
        const string Name = "Fill";
        Check.RangeBytes(Name, "output", output.Length, offset, count);
        WriteGenerated(Name, encoder, output.Buffer, offset, count, 0, 3, 0, value ? 1u : 0u, 0, 0, 0);
    }

    /// <summary>
    /// The number of elements of ONNX <c>Range(start, limit, delta)</c>: <c>max(ceil((limit - start) / delta), 0)</c>. Integer ranges are exact
    /// for every int32 and int64 value; float ranges are computed in double from the float inputs.
    /// </summary>
    /// <typeparam name="T"><see cref="int"/>, <see cref="long"/> or <see cref="float"/>.</typeparam>
    /// <param name="start">First value.</param>
    /// <param name="limit">The bound (exclusive).</param>
    /// <param name="delta">Step; not 0.</param>
    /// <returns>The element count.</returns>
    /// <exception cref="GpuKernelShapeException">The delta is 0, a float input is not finite, the range is not finite, or the count exceeds 2^31 - 1.</exception>
    public static int RangeLength<T>(T start, T limit, T delta)
        where T : unmanaged
    {
        const string Name = "Range";
        var kind = KernelTypes.Of<T>(Name);
        long n;
        if (kind == ElementKind.F32)
        {
            var s = (float)(object)start;
            var l = (float)(object)limit;
            var d = (float)(object)delta;
            var q = ((double)l - s) / d;
            if (d == 0 || float.IsNaN(l - s) || double.IsInfinity(q) || double.IsNaN(q))
            {
                throw new GpuKernelShapeException(Name, "delta", "delta is 0 or the range is not finite");
            }

            var c = Math.Max(Math.Ceiling(q), 0);
            if (c > Check.MaxElements)
            {
                throw new GpuKernelShapeException(Name, "limit", $"the range holds more than {Check.MaxElements} elements");
            }

            n = (long)c;
        }
        else if (kind is ElementKind.I32 or ElementKind.I64)
        {
            var s = kind == ElementKind.I32 ? (int)(object)start : (long)(object)start;
            var l = kind == ElementKind.I32 ? (int)(object)limit : (long)(object)limit;
            var d = kind == ElementKind.I32 ? (int)(object)delta : (long)(object)delta;
            if (d == 0)
            {
                throw new GpuKernelShapeException(Name, "delta", "delta is 0");
            }

            // ceil(span / |d|) for a span in d's direction, in 128 bits so the extremes of int64 cannot overflow.
            Int128 c = d > 0
                ? (l > s ? (((Int128)l - s) + d - 1) / d : 0)
                : (s > l ? (((Int128)s - l) - d - 1) / -(Int128)d : 0);
            if (c > Check.MaxElements)
            {
                throw new GpuKernelShapeException(Name, "limit", $"the range holds more than {Check.MaxElements} elements");
            }

            n = (long)c;
        }
        else
        {
            throw new GpuKernelShapeException(Name, "T", $"element type {typeof(T).Name} is not supported; use int, long or float");
        }

        return (int)n;
    }

    /// <summary>
    /// Records <c>output[i] = start + i * delta</c> for <paramref name="count"/> elements (ONNX <c>Range</c>; get the count from <see cref="RangeLength{T}"/>).
    /// Integers wrap on overflow; floats compute the product and the sum as two separately rounded float32 operations.
    /// </summary>
    /// <typeparam name="T"><see cref="int"/>, <see cref="long"/> or <see cref="float"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="output">Receives the sequence.</param>
    /// <param name="offset">Element offset of the first element.</param>
    /// <param name="count">Number of elements.</param>
    /// <param name="start">First value.</param>
    /// <param name="delta">Step; may be negative or 0.</param>
    /// <exception cref="GpuKernelShapeException">The range reaches past the buffer, or <typeparamref name="T"/> is not supported.</exception>
    public void EncodeRange<T>(GpuCommandEncoder encoder, GpuBuffer<T> output, int offset, int count, T start, T delta)
        where T : unmanaged
    {
        const string Name = "Range";
        var kind = KernelTypes.Of<T>(Name);
        uint s0, s1 = 0, d0, d1 = 0;
        int type = 0;
        uint isFloat = 0;
        switch (kind)
        {
            case ElementKind.F32:
                s0 = BitConverter.SingleToUInt32Bits((float)(object)start);
                d0 = BitConverter.SingleToUInt32Bits((float)(object)delta);
                isFloat = 1;
                break;
            case ElementKind.I32:
                s0 = (uint)(int)(object)start;
                d0 = (uint)(int)(object)delta;
                break;
            case ElementKind.I64:
                var s = (long)(object)start;
                var d = (long)(object)delta;
                s0 = (uint)s;
                s1 = (uint)(s >> 32);
                d0 = (uint)d;
                d1 = (uint)(d >> 32);
                type = 1;
                break;
            default:
                throw new GpuKernelShapeException(Name, "T", $"element type {typeof(T).Name} is not supported; use int, long or float");
        }

        Check.Range(Name, "output", output.Length, offset, count);
        WriteGenerated(Name, encoder, output.Buffer, offset, count, 1, type, isFloat, s0, s1, d0, d1);
    }

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => (_integer?.Created ?? 0) + (_generate?.Created ?? 0);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        _integer?.ReleaseBindings();
        _generate?.ReleaseBindings();
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _integer?.Dispose();
        _integer = null;
        _generate?.Dispose();
        _generate = null;
    }

    // 0 int32, 1 int64, 2 bool.
    static int Kind<T>(string operation, bool allowBool)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte) && allowBool)
        {
            return 2;
        }

        var k = typeof(T) == typeof(byte) ? ElementKind.U32 : KernelTypes.Of<T>(operation);
        return k switch
        {
            ElementKind.I32 => 0,
            ElementKind.I64 => 1,
            _ => throw new GpuKernelShapeException(operation, "T", $"element type {typeof(T).Name} is not supported; use int or long{(allowBool ? " (or byte for bool)" : string.Empty)}"),
        };
    }

    static void EnumRange(string operation, string name, int value, int max, int min = 0)
    {
        if (value < min || value > max)
        {
            throw new GpuKernelShapeException(operation, name, $"value {value} is not a defined operation");
        }
    }

    void Run<TIn, TOut>(string name, GpuCommandEncoder encoder, int op, int kindIn, int kindOut, GpuBuffer<TIn> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<TIn> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<TOut> output, int offsetOutput)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        Span<uint> w = stackalloc uint[IntegerWords];
        Span<uint> unused = stackalloc uint[Broadcasting.MaxRank];
        var rank = Broadcasting.Plan(name, 2, shapeA, shapeB, default, w.Slice(9, 6), w.Slice(15, 6), w.Slice(21, 6), unused, out var count);
        CheckInput(name, "a", a, offsetA, shapeA, kindIn);
        CheckInput(name, "b", b, offsetB, shapeB, kindIn);
        if (kindOut == 2)
        {
            Check.RangeBytes(name, "output", output.Length, offsetOutput, count);
        }
        else
        {
            Check.Range(name, "output", output.Length, offsetOutput, count);
        }

        w[0] = (uint)count;
        w[1] = (uint)offsetA;
        w[2] = (uint)offsetB;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[5] = (uint)kindIn;
        w[6] = (uint)kindOut;
        w[8] = (uint)rank;
        _integer ??= new KernelPipeline(_device, KernelSources.Make(Integer.IntegerOp.EntryPoint, Integer.IntegerOp.SpirV, Integer.IntegerOp.Msl, Integer.IntegerOp.MslFunctionName, Integer.IntegerOp.Wgsl, Integer.IntegerOp.WgslEntryPoint, Integer.IntegerOp.Layout), 3, IntegerWords, "integer", _maxBindGroups);

        Launch.Record(_device, encoder, _integer, w, count, offsetOutput, kindOut == 2 ? 2 : 0, 7, a.Buffer, b.Buffer, output.Buffer);
    }

    static void CheckInput<T>(string operation, string name, GpuBuffer<T> buffer, int offset, ReadOnlySpan<int> shape, int kind)
        where T : unmanaged
    {
        var n = Check.Product(operation, name + "Shape", shape);
        if (kind == 2)
        {
            Check.RangeBytes(operation, name, buffer.Length, offset, n);
        }
        else
        {
            Check.Range(operation, name, buffer.Length, offset, n);
        }
    }

    void WriteGenerated(string name, GpuCommandEncoder encoder, GpuBuffer output, int offset, int count, int mode, int type, uint isFloat, uint v0, uint v1, uint d0, uint d1)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (count < 0)
        {
            throw new GpuKernelShapeException(name, "count", $"count {count} is negative");
        }

        Span<uint> w = stackalloc uint[GenerateWords];
        w[0] = (uint)count;
        w[1] = (uint)offset;
        w[2] = (uint)mode;
        w[3] = (uint)type;
        w[4] = isFloat;
        w[5] = v0;
        w[6] = v1;
        w[7] = d0;
        w[8] = d1;
        w[10] = uint.MaxValue;
        _generate ??= new KernelPipeline(_device, KernelSources.Make(Shaders.Generate.GenerateOp.EntryPoint, Shaders.Generate.GenerateOp.SpirV, Shaders.Generate.GenerateOp.Msl, Shaders.Generate.GenerateOp.MslFunctionName, Shaders.Generate.GenerateOp.Wgsl, Shaders.Generate.GenerateOp.WgslEntryPoint, Shaders.Generate.GenerateOp.Layout), 1, GenerateWords, "generate", _maxBindGroups);
        Launch.Record(_device, encoder, _generate, w, count, offset, type == 3 ? 2 : (type == 2 ? 1 : 0), 9, output);
    }
}
