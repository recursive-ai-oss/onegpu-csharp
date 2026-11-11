using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Reductions over a set of axes. The numbers are the shader's operation codes.</summary>
public enum GpuReduceOp
{
    /// <summary>Sum.</summary>
    Sum = 0,
    /// <summary>Mean (sum divided by the count; an empty reduction gives NaN).</summary>
    Mean = 1,
    /// <summary>Largest value (NaN if any is NaN; an empty reduction gives -infinity).</summary>
    Max = 2,
    /// <summary>Smallest value (NaN if any is NaN; an empty reduction gives +infinity).</summary>
    Min = 3,
    /// <summary>Product.</summary>
    Prod = 4,
    /// <summary>Square root of the sum of squares.</summary>
    L2 = 5,
    /// <summary>Sum of squares.</summary>
    SumSquare = 6,
    /// <summary>Sum of absolute values.</summary>
    L1 = 7,
    /// <summary>Natural logarithm of the sum.</summary>
    LogSum = 8,
}

/// <summary>Index reductions.</summary>
public enum GpuArgOp
{
    /// <summary>Index of the largest value.</summary>
    Max = 0,
    /// <summary>Index of the smallest value.</summary>
    Min = 1,
}

/// <summary>
/// Reductions (ONNX <c>Reduce*</c>) and index reductions (<c>ArgMax</c>, <c>ArgMin</c>) over float and <see cref="Half"/> tensors.
/// Accumulation is in single precision. The output holds the kept dimensions densely in order, which is the same data whether the
/// ONNX <c>keepdims</c> attribute is set or not.
/// </summary>
/// <remarks>
/// <para><b>Variants.</b> <see cref="GpuKernelMode.Fast"/> reduces inside subgroups with the device's subgroup arithmetic when the device offers it
/// (<see cref="Variant"/> reports <c>subgroup</c>), else with a shared-memory tree (<c>shared</c>). <see cref="GpuKernelMode.Deterministic"/>
/// always uses the shared-memory tree: every thread sums a fixed strided slice of the reduced elements in index order, and the 256 partial
/// sums are combined by a fixed halving tree, so the bits depend only on the data, not on the device's subgroup size, occupancy or timing.
/// There are no floating-point atomics anywhere.</para>
/// <para>A reduction with fewer than 128 outputs and at least 16384 reduced elements runs in two passes through a scratch buffer: the reduced elements are cut into at most 256 contiguous chunks of at least 4096 (the cut depends only on the shape), each reduced by a workgroup, and the partial results are combined by one workgroup per output in a fixed order. Both modes cut the same way; the deterministic one is bit-identical run to run.</para>
    /// <para>Half outputs are written with integer atomic AND/OR on the 32-bit word that holds two outputs, so outputs of different workgroups never clash.
/// Half buffers must end on a 32-bit word (see <see cref="GpuKernelBuffers.PaddedHalfLength"/>). Not thread-safe.</para>
/// </remarks>
public sealed class ReduceKernels : IDisposable
{
    const int ReduceWords = 28;

    // A reduction with few outputs and many elements is split into this many chunks at most (one workgroup each); a chunk is at least
    // MinChunk elements, and only reductions with fewer than MaxSplitOutputs outputs are split.
    const int MaxSplit = 256;
    const int MinChunk = 4096;
    const int MaxSplitOutputs = 128;
    const int MinSplitElements = 16384;

    /// <summary>Floats of scratch a split reduction needs at most: <c>128 outputs x 256 partial results</c>.</summary>
    public const int ScratchLength = MaxSplitOutputs * MaxSplit;
    const int ArgWords = 16;

    readonly GpuDevice _device;
    readonly bool _subgroups;
    readonly KernelPipeline?[] _block = new KernelPipeline?[8];
    readonly KernelPipeline?[] _thread = new KernelPipeline?[8];
    readonly KernelPipeline?[] _argBlock = new KernelPipeline?[2];
    readonly KernelPipeline?[] _argThread = new KernelPipeline?[2];
    readonly KernelPipeline?[] _split = new KernelPipeline?[8];
    readonly KernelPipeline?[] _combine = new KernelPipeline?[8];
    GpuBuffer<float>? _scratch;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="allowSubgroups">False keeps the fast variant off subgroup operations even where the device has them.</param>
    public ReduceKernels(GpuDevice device, bool allowSubgroups = true)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _subgroups = Variants.UseSubgroups(device, allowSubgroups);
    }

    /// <summary>The block-level combine used by <see cref="GpuKernelMode.Fast"/>: <c>subgroup</c> or <c>shared</c>.</summary>
    public string Variant => _subgroups ? "subgroup" : "shared";

    /// <summary>Records a reduction of <paramref name="shape"/> over <paramref name="axes"/>.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">The reduction.</param>
    /// <param name="input">Input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor in <paramref name="input"/>.</param>
    /// <param name="shape">Input shape.</param>
    /// <param name="axes">Axes to reduce, each in [-rank, rank); empty means all axes.</param>
    /// <param name="output">Receives the product of the kept dimensions' elements.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="mode">Fast or deterministic.</param>
    /// <param name="scratch">Optional scratch for reductions that run in two passes (see the remarks of the class), at least <see cref="ScratchLength"/> floats. When null the family keeps one of its own, created on first use.</param>
    /// <exception cref="GpuKernelShapeException">An axis is out of range or repeated, a size is too large, a range reaches past its buffer, the scratch is too small, or the axes split the tensor into more groups than the kernel handles (four kept and four reduced runs of dimensions).</exception>
    public void EncodeReduce<T>(GpuCommandEncoder encoder, GpuReduceOp op, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, ReadOnlySpan<int> axes, GpuBuffer<T> output, int offsetOutput,
        GpuKernelMode mode = GpuKernelMode.Fast, GpuBuffer<float>? scratch = null)
        where T : unmanaged
    {
        const string Name = "Reduce";
        var half = KernelTypes.IsHalf<T>(Name);
        if ((uint)op > (uint)GpuReduceOp.LogSum)
        {
            throw new GpuKernelShapeException(Name, nameof(op), $"value {(int)op} is not a defined reduction");
        }

        CheckMode(Name, mode);
        var total = Check.Product(Name, nameof(shape), shape);
        Span<uint> w = stackalloc uint[ReduceWords];
        var plan = Plan(Name, shape, axes, w.Slice(8, 4), w.Slice(12, 4), w.Slice(16, 4), w.Slice(20, 4), out var reduced, out var numOut, out var innerContiguous);
        Check.Range(Name, nameof(input), input.Length, offsetInput, total, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, numOut, half);
        w[0] = (uint)numOut;
        w[1] = (uint)reduced;
        w[2] = (uint)offsetInput;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[6] = (uint)(plan >> 8);
        w[7] = (uint)(plan & 0xFF);
        if (numOut == 0)
        {
            return;
        }

        var wave = mode == GpuKernelMode.Fast && _subgroups;
        if (numOut < MaxSplitOutputs && reduced >= MinSplitElements)
        {
            if (scratch is not null && scratch.Length < ScratchLength)
            {
                throw new GpuKernelShapeException(Name, nameof(scratch), $"the scratch holds {scratch.Length} floats, {ScratchLength} are needed");
            }

            EncodeSplit(encoder, w, half, wave, (plan & 0xFF) <= 1, (int)numOut, (int)reduced, input.Buffer, output.Buffer, (scratch ??= _scratch ??= _device.CreateBuffer<float>(ScratchLength, label: "reduce-scratch")).Buffer);
            return;
        }

        var block = reduced >= 256 || (reduced >= 32 && (innerContiguous || numOut < 4096));
        // One run of reduced dimensions (a plain last-axis or all-axes reduction) needs no division to find an element.
        var oneRun = (plan & 0xFF) <= 1;
        var slot = (half ? 1 : 0) + (wave ? 2 : 0) + (oneRun ? 4 : 0);
        var pipeline = block ? Get(_block, slot, half, wave, true) : Get(_thread, slot, half, wave, false);
        var (gx, gy) = block ? Grid.For(_device, numOut) : Grid.For(_device, (numOut + 255) / 256);
        w[5] = gx;
        pipeline.Record(encoder, w, gx, gy, input.Buffer, output.Buffer);
    }

    // Two passes: each output's reduced elements are cut into `split` contiguous chunks, one workgroup per chunk writes its partial
    // result to the scratch; one workgroup per output then combines the partials in index order. The cut depends only on the shape.
    void EncodeSplit(GpuCommandEncoder encoder, Span<uint> w, bool half, bool wave, bool oneRun, int numOut, int reduced, GpuBuffer input, GpuBuffer output, GpuBuffer scratch)
    {
        var split = Math.Min(MaxSplit, (reduced + MinChunk - 1) / MinChunk);
        var chunk = (reduced + split - 1) / split;
        w[24] = (uint)split;
        w[25] = (uint)chunk;
        w[26] = (uint)reduced;
        var slot = (half ? 1 : 0) + (wave ? 2 : 0) + (oneRun ? 4 : 0);
        var first = GetSplit(_split, slot, half, wave, false);
        var (gx, gy) = Grid.For(_device, (long)numOut * split);
        w[5] = gx;
        first.Record(encoder, w, gx, gy, input, scratch);
        var second = GetSplit(_combine, slot, half, wave, true);
        (gx, gy) = Grid.For(_device, numOut);
        w[5] = gx;
        second.Record(encoder, w, gx, gy, scratch, output);
    }

    /// <summary>Records an ArgMax or ArgMin of <paramref name="shape"/> along <paramref name="axis"/>. Ties give the lowest index; NaN counts as the extreme value.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="op">Maximum or minimum.</param>
    /// <param name="input">Input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor in <paramref name="input"/>.</param>
    /// <param name="shape">Input shape.</param>
    /// <param name="axis">The axis in [-rank, rank).</param>
    /// <param name="output">Receives one 64-bit index per kept element.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <exception cref="GpuKernelShapeException">The axis is out of range, the axis has no elements, a size is too large or a range reaches past its buffer.</exception>
    public void EncodeArg<T>(GpuCommandEncoder encoder, GpuArgOp op, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, int axis, GpuBuffer<long> output, int offsetOutput)
        where T : unmanaged
    {
        const string Name = "Arg";
        var half = KernelTypes.IsHalf<T>(Name);
        if ((uint)op > 1u)
        {
            throw new GpuKernelShapeException(Name, nameof(op), $"value {(int)op} is not a defined reduction");
        }

        var total = Check.Product(Name, nameof(shape), shape);
        var rank = shape.Length;
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank)
        {
            throw new GpuKernelShapeException(Name, nameof(axis), $"axis {axis} is outside [{-rank}, {rank})");
        }

        if (shape[a] == 0)
        {
            throw new GpuKernelShapeException(Name, $"shape[{a}]", "the axis has no elements, so there is no index to return");
        }

        long inner = 1;
        for (var i = a + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        long outer = 1;
        for (var i = 0; i < a; i++)
        {
            outer *= shape[i];
        }

        var numOut = outer * inner;
        Check.Range(Name, nameof(input), input.Length, offsetInput, total, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, numOut);
        if (numOut == 0)
        {
            return;
        }

        Span<uint> w = stackalloc uint[ArgWords];
        w[0] = (uint)numOut;
        w[1] = (uint)shape[a];
        w[2] = (uint)offsetInput;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)op;
        w[7] = (uint)inner;

        // Kept dimensions innermost first: the elements after the axis, then those before it.
        var k = 0;
        if (inner > 1)
        {
            w[8 + k] = (uint)inner;
            w[12 + k] = 1;
            k++;
        }

        if (outer > 1)
        {
            w[8 + k] = (uint)outer;
            w[12 + k] = (uint)(inner * shape[a]);
            k++;
        }

        w[6] = (uint)k;
        for (var i = k; i < 4; i++)
        {
            w[8 + i] = 1;
            w[12 + i] = 0;
        }

        var block = shape[a] >= 256 || (shape[a] >= 32 && (inner == 1 || numOut < 4096));
        var h = half ? 1 : 0;
        var slots = block ? _argBlock : _argThread;
        var pipeline = slots[h] ??= CreateArg(half, block);
        var (gx, gy) = block ? Grid.For(_device, numOut) : Grid.For(_device, (numOut + 255) / 256);
        w[5] = gx;
        pipeline.Record(encoder, w, gx, gy, input.Buffer, output.Buffer);
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _scratch?.Dispose();
        _scratch = null;
        foreach (var set in new[] { _block, _thread, _argBlock, _argThread, _split, _combine })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }
    }

    internal static void CheckMode(string operation, GpuKernelMode mode)
    {
        if (mode is not (GpuKernelMode.Fast or GpuKernelMode.Deterministic))
        {
            throw new GpuKernelShapeException(operation, nameof(mode), $"value {(int)mode} is not a defined mode");
        }
    }

    // Returns (kept rank << 8) | reduced rank and fills the dimension and stride words, innermost first.
    static int Plan(string name, ReadOnlySpan<int> shape, ReadOnlySpan<int> axes, Span<uint> kd, Span<uint> ks, Span<uint> rd, Span<uint> rs, out long reduced, out long numOut, out bool innerContiguous)
    {
        var rank = shape.Length;
        if (rank > 32)
        {
            throw new GpuKernelShapeException(name, nameof(shape), $"rank {rank} is above the supported 32");
        }

        Span<bool> isReduced = stackalloc bool[32];
        if (axes.Length == 0)
        {
            for (var i = 0; i < rank; i++)
            {
                isReduced[i] = true;
            }
        }
        else
        {
            foreach (var axis in axes)
            {
                var a = axis < 0 ? axis + rank : axis;
                if (a < 0 || a >= rank)
                {
                    throw new GpuKernelShapeException(name, nameof(axes), $"axis {axis} is outside [{-rank}, {rank})");
                }

                if (isReduced[a])
                {
                    throw new GpuKernelShapeException(name, nameof(axes), $"axis {axis} is listed twice");
                }

                isReduced[a] = true;
            }
        }

        // Walk from the innermost dimension out. Dimensions of 1 vanish; neighbours of the same kind merge (a dense tensor's adjacent
        // dimensions are contiguous, so the merged run has the inner dimension's stride).
        long stride = 1;
        var kn = 0;
        var rn = 0;
        long reducedCount = 1;
        long outCount = 1;
        var lastKind = -1;
        innerContiguous = false;
        var firstRun = true;
        for (var i = rank - 1; i >= 0; i--)
        {
            var dim = shape[i];
            var red = isReduced[i];
            if (dim != 1)
            {
                if (firstRun)
                {
                    innerContiguous = red;
                    firstRun = false;
                }

                var kind = red ? 1 : 0;
                if (kind == lastKind)
                {
                    if (red)
                    {
                        rd[rn - 1] = checked(rd[rn - 1] * (uint)dim);
                    }
                    else
                    {
                        kd[kn - 1] = checked(kd[kn - 1] * (uint)dim);
                    }
                }
                else if (red)
                {
                    if (rn == 4)
                    {
                        throw new GpuKernelShapeException(name, nameof(axes), "the axes split the tensor into more than four reduced runs of dimensions");
                    }

                    rd[rn] = (uint)dim;
                    rs[rn] = (uint)stride;
                    rn++;
                }
                else
                {
                    if (kn == 4)
                    {
                        throw new GpuKernelShapeException(name, nameof(axes), "the axes split the tensor into more than four kept runs of dimensions");
                    }

                    kd[kn] = (uint)dim;
                    ks[kn] = (uint)stride;
                    kn++;
                }

                lastKind = kind;
                if (red)
                {
                    reducedCount *= dim;
                }
                else
                {
                    outCount *= dim;
                }
            }

            stride *= dim;
        }

        for (var i = kn; i < 4; i++)
        {
            kd[i] = 1;
            ks[i] = 0;
        }

        for (var i = rn; i < 4; i++)
        {
            rd[i] = 1;
            rs[i] = 0;
        }

        reduced = reducedCount;
        numOut = outCount;
        return (kn << 8) | rn;
    }

    KernelPipeline Get(KernelPipeline?[] slots, int slot, bool half, bool wave, bool block)
    {
        if (slots[slot] is { } ready)
        {
            return ready;
        }

        var v = Reduce.Variants.Get(new Reduce.Key(f16: half, wave: wave, rrank: slot >= 4 ? Reduce.RrankValue.R1 : Reduce.RrankValue.Rn));
        var created = block
            ? new KernelPipeline(_device, KernelSources.Make(v.ReduceBlock.EntryPoint, v.ReduceBlock.SpirV, v.ReduceBlock.Msl, v.ReduceBlock.MslFunctionName, v.ReduceBlock.Wgsl, v.ReduceBlock.WgslEntryPoint, v.ReduceBlock.Layout), 2, ReduceWords, "reduce-block")
            : new KernelPipeline(_device, KernelSources.Make(v.ReduceThread.EntryPoint, v.ReduceThread.SpirV, v.ReduceThread.Msl, v.ReduceThread.MslFunctionName, v.ReduceThread.Wgsl, v.ReduceThread.WgslEntryPoint, v.ReduceThread.Layout), 2, ReduceWords, "reduce-thread");
        slots[slot] = created;
        return created;
    }

    KernelPipeline GetSplit(KernelPipeline?[] slots, int slot, bool half, bool wave, bool combine)
    {
        if (slots[slot] is { } ready)
        {
            return ready;
        }

        var v = Reduce.Variants.Get(new Reduce.Key(f16: half, wave: wave, rrank: slot >= 4 ? Reduce.RrankValue.R1 : Reduce.RrankValue.Rn));
        var created = combine
            ? new KernelPipeline(_device, KernelSources.Make(v.ReduceCombine.EntryPoint, v.ReduceCombine.SpirV, v.ReduceCombine.Msl, v.ReduceCombine.MslFunctionName, v.ReduceCombine.Wgsl, v.ReduceCombine.WgslEntryPoint, v.ReduceCombine.Layout), 2, ReduceWords, "reduce-combine")
            : new KernelPipeline(_device, KernelSources.Make(v.ReduceSplit.EntryPoint, v.ReduceSplit.SpirV, v.ReduceSplit.Msl, v.ReduceSplit.MslFunctionName, v.ReduceSplit.Wgsl, v.ReduceSplit.WgslEntryPoint, v.ReduceSplit.Layout), 2, ReduceWords, "reduce-split");
        slots[slot] = created;
        return created;
    }

    KernelPipeline CreateArg(bool half, bool block)
    {
        var v = ArgReduce.Variants.Get(new ArgReduce.Key(f16: half));
        return block
            ? new KernelPipeline(_device, KernelSources.Make(v.ArgBlock.EntryPoint, v.ArgBlock.SpirV, v.ArgBlock.Msl, v.ArgBlock.MslFunctionName, v.ArgBlock.Wgsl, v.ArgBlock.WgslEntryPoint, v.ArgBlock.Layout), 2, ArgWords, "arg-block")
            : new KernelPipeline(_device, KernelSources.Make(v.ArgThread.EntryPoint, v.ArgThread.SpirV, v.ArgThread.Msl, v.ArgThread.MslFunctionName, v.ArgThread.Wgsl, v.ArgThread.WgslEntryPoint, v.ArgThread.Layout), 2, ArgWords, "arg-thread");
    }
}
