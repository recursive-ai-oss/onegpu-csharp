using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Element-wise kernels over float and <see cref="Half"/> tensors: binary arithmetic with ONNX broadcasting (and an optional fused
/// activation), unary functions, comparison and select. Tensors are dense row-major; operands are a buffer, an element offset and a shape.
/// </summary>
/// <remarks>
/// <para>Half tensors are computed in single precision and rounded to nearest even on the way out, exactly like <c>(Half)((float)a + (float)b)</c>.
/// Their buffers must end on a 32-bit word (see <see cref="GpuKernelBuffers.PaddedHalfLength"/>). An output may overlap an input only when
/// both use the same range.</para>
/// <para>Element-wise operations have no reduction order, so <see cref="GpuKernelMode.Deterministic"/> is accepted and gives the same bits as
/// <see cref="GpuKernelMode.Fast"/>. Variants are created on first use; later calls allocate nothing. Not thread-safe.</para>
/// <para>The pipelines are compiled for the platform's shader language by the build, so a Metal build compiles without fast-math
/// assumptions for NaN and infinity only where the kernels test the bits (they do).</para>
/// </remarks>
public sealed class ElementwiseKernels : IDisposable
{
    const int BinaryWords = 28;
    const int UnaryWords = 7;
    const int WhereWords = 31;
    const int CastWords = 6;

    readonly GpuDevice _device;
    readonly KernelPipeline?[] _binary = new KernelPipeline?[2];
    readonly KernelPipeline?[] _compare = new KernelPipeline?[2];
    readonly KernelPipeline?[] _unary = new KernelPipeline?[2];
    readonly KernelPipeline?[] _where = new KernelPipeline?[2];
    KernelPipeline? _cast;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    public ElementwiseKernels(GpuDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
    }

    /// <summary>The broadcast shape of two operands (ONNX multidirectional broadcasting).</summary>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast.</exception>
    public static int[] BroadcastShape(ReadOnlySpan<int> shapeA, ReadOnlySpan<int> shapeB) => Broadcasting.ResultShape("Broadcast", shapeA, shapeB);

    /// <summary>
    /// Records <c>output = activation(a op b)</c> with broadcasting. The output holds the broadcast shape's elements, densely.
    /// </summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The operation.</param>
    /// <param name="a">First operand.</param>
    /// <param name="offsetA">Element offset of the first operand in <paramref name="a"/>.</param>
    /// <param name="shapeA">Shape of the first operand; empty for a scalar.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives the result.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="activation">A function applied to each result, or null.</param>
    /// <param name="mode">Either mode gives the same bits.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeBinary<T>(GpuCommandEncoder encoder, GpuBinaryOp op, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<T> output, int offsetOutput, GpuActivation? activation = null, GpuKernelMode mode = GpuKernelMode.Fast)
        where T : unmanaged
    {
        const string Name = "Binary";
        var half = KernelTypes.IsHalf<T>(Name);
        EnumRange(Name, nameof(op), (int)op, (int)GpuBinaryOp.Max);
        Span<uint> w = stackalloc uint[BinaryWords];
        Span<uint> unused = stackalloc uint[Broadcasting.MaxRank];
        var rank = Broadcasting.Plan(Name, 2, shapeA, shapeB, default, w.Slice(10, 6), w.Slice(16, 6), w.Slice(22, 6), unused, out var count);
        CheckOperand(Name, "a", a, offsetA, shapeA, half);
        CheckOperand(Name, "b", b, offsetB, shapeB, half);
        Check.Range(Name, "output", output.Length, offsetOutput, count, half);
        if (activation is { } act)
        {
            EnumRange(Name, "activation", (int)act.Op, (int)GpuUnaryOp.LeakyRelu);
            w[5] = (uint)act.Op + 1u;
            w[6] = BitConverter.SingleToUInt32Bits(act.P0);
            w[7] = BitConverter.SingleToUInt32Bits(act.P1);
        }
        else
        {
            w[5] = 0;
            w[6] = 0;
            w[7] = 0;
        }

        w[0] = (uint)count;
        w[1] = (uint)offsetA;
        w[2] = (uint)offsetB;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[9] = (uint)rank;
        Dispatch(Name, encoder, Pipeline(_binary, half, mode), w, count, offsetOutput, half, 8, a.Buffer, b.Buffer, output.Buffer);
    }

    /// <summary>Records <c>output = a op b</c> as 32-bit 0 or 1 per element, with broadcasting.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The comparison.</param>
    /// <param name="a">First operand.</param>
    /// <param name="offsetA">Element offset of the first operand.</param>
    /// <param name="shapeA">Shape of the first operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives 0 or 1 per element of the broadcast shape.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeCompare<T>(GpuCommandEncoder encoder, GpuCompareOp op, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<uint> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Compare";
        var half = KernelTypes.IsHalf<T>(Name);
        EnumRange(Name, nameof(op), (int)op, (int)GpuCompareOp.GreaterOrEqual);
        Span<uint> w = stackalloc uint[BinaryWords];
        Span<uint> unused = stackalloc uint[Broadcasting.MaxRank];
        var rank = Broadcasting.Plan(Name, 2, shapeA, shapeB, default, w.Slice(10, 6), w.Slice(16, 6), w.Slice(22, 6), unused, out var count);
        CheckOperand(Name, "a", a, offsetA, shapeA, half);
        CheckOperand(Name, "b", b, offsetB, shapeB, half);
        Check.Range(Name, "output", output.Length, offsetOutput, count);
        w[0] = (uint)count;
        w[1] = (uint)offsetA;
        w[2] = (uint)offsetB;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[5] = 0;
        w[6] = 0;
        w[7] = 0;
        w[9] = (uint)rank;

        // The result is one word per element, so the grid is per element for both input types.
        Dispatch(Name, encoder, Pipeline(_compare, half, GpuKernelMode.Fast), w, count, offsetOutput, false, 8, a.Buffer, b.Buffer, output.Buffer);
    }

    /// <summary>Records <c>y = f(x)</c> over <paramref name="count"/> consecutive elements.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The function.</param>
    /// <param name="x">Input.</param>
    /// <param name="offsetX">Element offset of the first input element.</param>
    /// <param name="y">Output.</param>
    /// <param name="offsetY">Element offset of the first output element.</param>
    /// <param name="count">Number of elements.</param>
    /// <param name="p0">First parameter of <paramref name="op"/> where it has one.</param>
    /// <param name="p1">Second parameter of <paramref name="op"/> where it has one.</param>
    /// <param name="mode">Either mode gives the same bits.</param>
    /// <exception cref="GpuKernelShapeException">A range reaches past its buffer, or the count is negative or above 2^31 - 1.</exception>
    public void EncodeUnary<T>(GpuCommandEncoder encoder, GpuUnaryOp op, GpuBuffer<T> x, int offsetX, GpuBuffer<T> y, int offsetY, int count, float p0 = 0f, float p1 = 0f, GpuKernelMode mode = GpuKernelMode.Fast)
        where T : unmanaged
    {
        const string Name = "Unary";
        var half = KernelTypes.IsHalf<T>(Name);
        EnumRange(Name, nameof(op), (int)op, (int)GpuUnaryOp.LeakyRelu);
        Check.Range(Name, "x", x.Length, offsetX, count, half);
        Check.Range(Name, "y", y.Length, offsetY, count, half);
        Span<uint> w = stackalloc uint[UnaryWords];
        w[0] = (uint)count;
        w[1] = (uint)offsetX;
        w[2] = (uint)offsetY;
        w[3] = (uint)op;
        w[4] = BitConverter.SingleToUInt32Bits(p0);
        w[5] = BitConverter.SingleToUInt32Bits(p1);
        Dispatch(Name, encoder, Pipeline(_unary, half, mode), w, count, offsetY, half, 6, x.Buffer, y.Buffer);
    }

    /// <summary>Records <c>output = condition ? a : b</c> with broadcasting. Values are copied bit for bit.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="condition">Per element 0 (false) or any other value (true), as a 32-bit word.</param>
    /// <param name="offsetCondition">Element offset of the condition.</param>
    /// <param name="shapeCondition">Shape of the condition.</param>
    /// <param name="a">Values where the condition holds.</param>
    /// <param name="offsetA">Element offset of <paramref name="a"/>.</param>
    /// <param name="shapeA">Shape of <paramref name="a"/>.</param>
    /// <param name="b">Values where it does not.</param>
    /// <param name="offsetB">Element offset of <paramref name="b"/>.</param>
    /// <param name="shapeB">Shape of <paramref name="b"/>.</param>
    /// <param name="output">Receives the broadcast shape's elements.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeWhere<T>(GpuCommandEncoder encoder, GpuBuffer<uint> condition, int offsetCondition, ReadOnlySpan<int> shapeCondition, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA,
        GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Where";
        var half = KernelTypes.IsHalf<T>(Name);
        Span<uint> w = stackalloc uint[WhereWords];
        var rank = Broadcasting.Plan(Name, 3, shapeCondition, shapeA, shapeB, w.Slice(7, 6), w.Slice(13, 6), w.Slice(19, 6), w.Slice(25, 6), out var count);
        CheckOperand(Name, "condition", condition, offsetCondition, shapeCondition, false);
        CheckOperand(Name, "a", a, offsetA, shapeA, half);
        CheckOperand(Name, "b", b, offsetB, shapeB, half);
        Check.Range(Name, "output", output.Length, offsetOutput, count, half);
        w[0] = (uint)count;
        w[1] = (uint)offsetCondition;
        w[2] = (uint)offsetA;
        w[3] = (uint)offsetB;
        w[4] = (uint)offsetOutput;
        w[6] = (uint)rank;
        Dispatch(Name, encoder, Pipeline(_where, half, GpuKernelMode.Fast), w, count, offsetOutput, half, 5, condition.Buffer, a.Buffer, b.Buffer, output.Buffer);
    }

    /// <summary>
    /// Records a conversion of <paramref name="count"/> consecutive elements between float, <see cref="Half"/>, int and long. Integer to float
    /// rounds to nearest even; float to integer truncates towards zero, saturates at the destination's range and turns NaN into 0.
    /// Integer width changes sign-extend or truncate. 64-bit integers are handled as pairs of 32-bit words, so the device needs no 64-bit integer support.
    /// </summary>
    /// <typeparam name="TFrom"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <typeparam name="TTo"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="source">Input.</param>
    /// <param name="offsetSource">Element offset of the first input element.</param>
    /// <param name="destination">Output; it must not overlap the input.</param>
    /// <param name="offsetDestination">Element offset of the first output element.</param>
    /// <param name="count">Number of elements.</param>
    /// <exception cref="GpuKernelShapeException">A range reaches past its buffer, or a type is not one of the four.</exception>
    public void EncodeCast<TFrom, TTo>(GpuCommandEncoder encoder, GpuBuffer<TFrom> source, int offsetSource, GpuBuffer<TTo> destination, int offsetDestination, int count)
        where TFrom : unmanaged
        where TTo : unmanaged
    {
        const string Name = "Cast";
        var from = CastKind<TFrom>(Name);
        var to = CastKind<TTo>(Name);
        Check.Range(Name, "source", source.Length, offsetSource, count, from == 1);
        Check.Range(Name, "destination", destination.Length, offsetDestination, count, to == 1);
        if (count == 0)
        {
            return;
        }

        if (_cast is null)
        {
            _cast = new KernelPipeline(_device, KernelSources.Make(Cast.CastOp.EntryPoint, Cast.CastOp.SpirV, Cast.CastOp.Msl, Cast.CastOp.MslFunctionName, Cast.CastOp.Wgsl, Cast.CastOp.WgslEntryPoint, Cast.CastOp.Layout), 2, CastWords, "cast");
        }

        Span<uint> w = stackalloc uint[CastWords];
        w[0] = (uint)count;
        w[1] = (uint)offsetSource;
        w[2] = (uint)offsetDestination;
        w[3] = (uint)from;
        w[4] = (uint)to;
        Dispatch(Name, encoder, _cast, w, count, offsetDestination, to == 1, 5, source.Buffer, destination.Buffer);
    }

    static int CastKind<T>(string operation)
        where T : unmanaged =>
        KernelTypes.Of<T>(operation) switch
        {
            ElementKind.F32 => 0,
            ElementKind.F16 => 1,
            ElementKind.I32 => 2,
            ElementKind.I64 => 3,
            _ => throw new GpuKernelShapeException(operation, "T", $"element type {typeof(T).Name} is not supported; use float, Half, int or long"),
        };

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _cast?.Dispose();
        _cast = null;
        foreach (var set in new[] { _binary, _compare, _unary, _where })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }
    }

    static void EnumRange(string operation, string name, int value, int max)
    {
        if (value < 0 || value > max)
        {
            throw new GpuKernelShapeException(operation, name, $"value {value} is not a defined operation");
        }
    }

    static void CheckOperand<T>(string operation, string name, GpuBuffer<T> buffer, int offset, ReadOnlySpan<int> shape, bool half)
        where T : unmanaged
    {
        Check.Range(operation, name, buffer.Length, offset, Check.Product(operation, name + "Shape", shape), half);
    }

    // Pushes the grid size into the word after the last field of the parameter block, then records the dispatch.
    void Dispatch(string operation, GpuCommandEncoder encoder, KernelPipeline pipeline, Span<uint> words, long count, int offsetOutput, bool half, int gridWord, GpuBuffer b0, GpuBuffer b1, GpuBuffer? b2 = null, GpuBuffer? b3 = null)
    {
        if (count == 0)
        {
            return;
        }

        var threads = half ? ((offsetOutput + count - 1) >> 1) - (offsetOutput >> 1) + 1 : count;
        var (gx, gy) = Grid.For(_device, (threads + 255) / 256);
        words[gridWord] = gx;
        pipeline.Record(encoder, words, gx, gy, b0, b1, b2, b3);
    }

    KernelPipeline Pipeline(KernelPipeline?[] slots, bool half, GpuKernelMode mode)
    {
        _ = mode;
        var i = half ? 1 : 0;
        if (slots[i] is { } ready)
        {
            return ready;
        }

        KernelPipeline created;
        if (ReferenceEquals(slots, _binary))
        {
            var e = Binary.Variants.Get(new Binary.Key(f16: half)).BinaryOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, BinaryWords, "binary");
        }
        else if (ReferenceEquals(slots, _compare))
        {
            var e = Compare.Variants.Get(new Compare.Key(f16: half)).CompareOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, BinaryWords, "compare");
        }
        else if (ReferenceEquals(slots, _unary))
        {
            var e = Unary.Variants.Get(new Unary.Key(f16: half)).UnaryOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, UnaryWords, "unary");
        }
        else
        {
            var e = Where.Variants.Get(new Where.Key(f16: half)).WhereOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 4, WhereWords, "where");
        }

        slots[i] = created;
        return created;
    }
}
