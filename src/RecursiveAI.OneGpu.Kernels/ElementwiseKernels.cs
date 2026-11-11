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
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _binary = new KernelPipeline?[12];
    readonly KernelPipeline?[] _compare = new KernelPipeline?[6];
    readonly KernelPipeline?[] _unary = new KernelPipeline?[2];
    readonly KernelPipeline?[] _where = new KernelPipeline?[2];
    KernelPipeline? _cast;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public ElementwiseKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
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
        Dispatch(Name, encoder, Pipeline(_binary, half, mode, activation is not null, rank), w, count, offsetOutput, half ? 1 : 0, 8, a.Buffer, b.Buffer, output.Buffer);
    }

    /// <summary>Records <c>output = a op b</c> as a bool tensor (one byte, 0 or 1, per element), with broadcasting. NaN compares false for every operation.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The comparison.</param>
    /// <param name="a">First operand.</param>
    /// <param name="offsetA">Element offset of the first operand.</param>
    /// <param name="shapeA">Shape of the first operand.</param>
    /// <param name="b">Second operand.</param>
    /// <param name="offsetB">Element offset of the second operand.</param>
    /// <param name="shapeB">Shape of the second operand.</param>
    /// <param name="output">Receives a bool byte per element of the broadcast shape; allocate <see cref="GpuKernelBuffers.PaddedBoolLength"/> elements.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer.</exception>
    public void EncodeCompare<T>(GpuCommandEncoder encoder, GpuCompareOp op, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA, GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB,
        GpuBuffer<byte> output, int offsetOutput)
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
        Check.RangeBytes(Name, "output", output.Length, offsetOutput, count);
        w[0] = (uint)count;
        w[1] = (uint)offsetA;
        w[2] = (uint)offsetB;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[5] = 0;
        w[6] = 0;
        w[7] = 0;
        w[9] = (uint)rank;

        // The result is four bytes to a word, one thread per word, for both input types.
        Dispatch(Name, encoder, Pipeline(_compare, half, GpuKernelMode.Fast, false, rank), w, count, offsetOutput, 2, 8, a.Buffer, b.Buffer, output.Buffer);
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
        Dispatch(Name, encoder, Pipeline(_unary, half, mode), w, count, offsetY, half ? 1 : 0, 6, x.Buffer, y.Buffer);
    }

    /// <summary>Records <c>output = condition ? a : b</c> with broadcasting. Values are copied bit for bit.</summary>
    /// <typeparam name="T"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="uint"/> or <see cref="long"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="condition">A bool tensor: one byte per element, 0 for false and anything else for true.</param>
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
    /// <exception cref="GpuKernelShapeException">The shapes do not broadcast, are too large, or an operand reaches past its buffer. A <see cref="long"/> tensor is moved as two words per element, which adds a dimension: its shapes may have one fewer dimension than the other types.</exception>
    public void EncodeWhere<T>(GpuCommandEncoder encoder, GpuBuffer<byte> condition, int offsetCondition, ReadOnlySpan<int> shapeCondition, GpuBuffer<T> a, int offsetA, ReadOnlySpan<int> shapeA,
        GpuBuffer<T> b, int offsetB, ReadOnlySpan<int> shapeB, GpuBuffer<T> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Where";
        var kind = KernelTypes.Of<T>(Name);
        if (kind is ElementKind.F32 or ElementKind.F16 or ElementKind.I32 or ElementKind.U32 or ElementKind.I64)
        {
            var half = kind == ElementKind.F16;
            var wide = kind == ElementKind.I64;
            Span<uint> w = stackalloc uint[WhereWords];

            // A 64-bit element is two words: describe the tensors over words, with a trailing dimension of 2 (1 for the condition).
            var rank = Math.Max(shapeCondition.Length, Math.Max(shapeA.Length, shapeB.Length)) + 1;
            Span<int> scratch = stackalloc int[wide ? rank * 3 : 0];
            if (wide)
            {
                Widen(shapeCondition, scratch.Slice(0, rank), 1);
                Widen(shapeA, scratch.Slice(rank, rank), 2);
                Widen(shapeB, scratch.Slice(2 * rank, rank), 2);
            }

            ReadOnlySpan<int> sc = wide ? scratch.Slice(0, rank) : shapeCondition;
            ReadOnlySpan<int> sa = wide ? scratch.Slice(rank, rank) : shapeA;
            ReadOnlySpan<int> sb = wide ? scratch.Slice(2 * rank, rank) : shapeB;
            var planned = Broadcasting.Plan(Name, 3, sc, sa, sb, w.Slice(7, 6), w.Slice(13, 6), w.Slice(19, 6), w.Slice(25, 6), out var count);
            CheckOperand(Name, "condition", condition, offsetCondition, shapeCondition, false, bytes: true);
            CheckOperand(Name, "a", a, offsetA, shapeA, half);
            CheckOperand(Name, "b", b, offsetB, shapeB, half);
            var elements = count / (wide ? 2 : 1);
            Check.Range(Name, "output", output.Length, offsetOutput, elements, half);
            var scale = wide ? 2 : 1;
            w[0] = (uint)count;
            w[1] = (uint)offsetCondition;
            w[2] = (uint)checked(offsetA * scale);
            w[3] = (uint)checked(offsetB * scale);
            w[4] = (uint)checked(offsetOutput * scale);
            w[6] = (uint)planned;
            Dispatch(Name, encoder, Pipeline(_where, half, GpuKernelMode.Fast), w, count, (int)w[4], half ? 1 : 0, 5, condition.Buffer, a.Buffer, b.Buffer, output.Buffer);
            return;
        }

        throw new GpuKernelShapeException(Name, "T", $"element type {typeof(T).Name} is not supported; use float, Half, int, uint or long");
    }

    // Copies a shape right-aligned into a rank-long span and appends the trailing dimension.
    static void Widen(ReadOnlySpan<int> shape, Span<int> into, int trailing)
    {
        into.Fill(1);
        shape.CopyTo(into.Slice(into.Length - 1 - shape.Length, shape.Length));
        into[^1] = trailing;
    }

    /// <summary>
    /// Records a conversion of <paramref name="count"/> consecutive elements between float, <see cref="Half"/>, int, long and bool (a <see cref="byte"/> tensor holding 0 or 1). Integer to float
    /// rounds to nearest even; float to integer truncates towards zero, saturates at the destination's range and turns NaN into 0.
    /// Integer width changes sign-extend or truncate. Anything to bool is "not equal to zero" (NaN included, negative zero not); bool to a number is 0 or 1. 64-bit integers are handled as pairs of 32-bit words, so the device needs no 64-bit integer support.
    /// </summary>
    /// <typeparam name="TFrom"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="long"/> or <see cref="byte"/> (bool).</typeparam>
    /// <typeparam name="TTo"><see cref="float"/>, <see cref="Half"/>, <see cref="int"/>, <see cref="long"/> or <see cref="byte"/> (bool).</typeparam>
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
        CheckCast(Name, "source", source.Length, offsetSource, count, from);
        CheckCast(Name, "destination", destination.Length, offsetDestination, count, to);
        if (count == 0)
        {
            return;
        }

        if (_cast is null)
        {
            _cast = new KernelPipeline(_device, KernelSources.Make(Cast.CastOp.EntryPoint, Cast.CastOp.SpirV, Cast.CastOp.Msl, Cast.CastOp.MslFunctionName, Cast.CastOp.Wgsl, Cast.CastOp.WgslEntryPoint, Cast.CastOp.Layout), 2, CastWords, "cast", _maxBindGroups);
        }

        Span<uint> w = stackalloc uint[CastWords];
        w[0] = (uint)count;
        w[1] = (uint)offsetSource;
        w[2] = (uint)offsetDestination;
        w[3] = (uint)from;
        w[4] = (uint)to;
        Dispatch(Name, encoder, _cast, w, count, offsetDestination, to == 1 ? 1 : (to == 4 ? 2 : 0), 5, source.Buffer, destination.Buffer);
    }

    static void CheckCast(string operation, string name, int length, int offset, int count, int kind)
    {
        if (kind == 4)
        {
            Check.RangeBytes(operation, name, length, offset, count);
        }
        else
        {
            Check.Range(operation, name, length, offset, count, kind == 1);
        }
    }

    static int CastKind<T>(string operation)
        where T : unmanaged =>
        typeof(T) == typeof(byte) ? 4 : KernelTypes.Of<T>(operation) switch
        {
            ElementKind.F32 => 0,
            ElementKind.F16 => 1,
            ElementKind.I32 => 2,
            ElementKind.I64 => 3,
            _ => throw new GpuKernelShapeException(operation, "T", $"element type {typeof(T).Name} is not supported; use float, Half, int, long or byte (bool)"),
        };

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => (_cast?.Created ?? 0) + KernelPipeline.Total(_binary, _compare, _unary, _where);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        _cast?.ReleaseBindings();
        foreach (var set in new[] { _binary, _compare, _unary, _where })
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

    static void CheckOperand<T>(string operation, string name, GpuBuffer<T> buffer, int offset, ReadOnlySpan<int> shape, bool half, bool bytes = false)
        where T : unmanaged
    {
        if (bytes)
        {
            Check.RangeBytes(operation, name, buffer.Length, offset, Check.Product(operation, name + "Shape", shape));
            return;
        }

        Check.Range(operation, name, buffer.Length, offset, Check.Product(operation, name + "Shape", shape), half);
    }

    // Pushes the grid size into the word after the last field of the parameter block, then records the dispatch.
    void Dispatch(string operation, GpuCommandEncoder encoder, KernelPipeline pipeline, Span<uint> words, long count, int offsetOutput, int shift, int gridWord, GpuBuffer b0, GpuBuffer b1, GpuBuffer? b2 = null, GpuBuffer? b3 = null) =>
        Launch.Record(_device, encoder, pipeline, words, count, offsetOutput, shift, gridWord, b0, b1, b2, b3);

    // The fused activation is compiled in only when asked for: its code makes the plain binary kernel needlessly heavy on registers.
    KernelPipeline Pipeline(KernelPipeline?[] slots, bool half, GpuKernelMode mode, bool activation = false, int rank = 0)
    {
        _ = mode;
        // Binary and compare kernels are specialised on the merged rank: one dimension, two, or more.
        var rankClass = rank <= 1 ? 0 : (rank == 2 ? 1 : 2);
        var i = (half ? 1 : 0) + (activation ? 2 : 0) + ((slots.Length > 6 ? 4 : 2) * rankClass);
        if (slots[i] is { } ready)
        {
            return ready;
        }

        KernelPipeline created;
        if (ReferenceEquals(slots, _binary))
        {
            var e = Binary.Variants.Get(new Binary.Key(f16: half, act: activation, rank: rankClass == 0 ? Binary.RankValue.R1 : (rankClass == 1 ? Binary.RankValue.R2 : Binary.RankValue.Rn))).BinaryOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, BinaryWords, "binary", _maxBindGroups);
        }
        else if (ReferenceEquals(slots, _compare))
        {
            var e = Compare.Variants.Get(new Compare.Key(f16: half, rank: rankClass == 0 ? Compare.RankValue.R1 : (rankClass == 1 ? Compare.RankValue.R2 : Compare.RankValue.Rn))).CompareOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, BinaryWords, "compare", _maxBindGroups);
        }
        else if (ReferenceEquals(slots, _unary))
        {
            var e = Unary.Variants.Get(new Unary.Key(f16: half)).UnaryOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, UnaryWords, "unary", _maxBindGroups);
        }
        else
        {
            var e = Where.Variants.Get(new Where.Key(f16: half)).WhereOp;
            created = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 4, WhereWords, "where", _maxBindGroups);
        }

        slots[i] = created;
        return created;
    }
}
