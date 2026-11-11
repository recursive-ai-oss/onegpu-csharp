using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Softmax and log-softmax along one axis of a float or <see cref="Half"/> tensor, in the numerically stable form (the row maximum is
/// subtracted; the statistics come from the one-pass online normaliser of Milakov and Gimelshein).
/// </summary>
/// <remarks>
/// <para>A row whose entries are all -infinity, or that holds +infinity or NaN, gives NaN for every entry of the row, as the textbook formula does;
/// entries of -infinity in a row with finite entries give 0 (log-softmax: -infinity), so additive attention masks work.</para>
/// <para><b>Variants.</b> Rows shorter than 16 are done by one thread each; rows of 16 to 1024 elements by a power-of-two group of threads (about four
/// elements per thread, so a 256-thread workgroup handles several rows), merged by a fixed halving tree in every mode; longer rows by a whole
/// 256-thread workgroup. In the workgroup form
/// <see cref="GpuKernelMode.Fast"/> merges the partial statistics with subgroup operations where the device has them
/// (<see cref="Variant"/>), <see cref="GpuKernelMode.Deterministic"/> with a fixed halving tree in shared memory: each thread folds a fixed
/// strided slice of the row in order and the tree shape is fixed, so the bits do not depend on the device's subgroup size or on scheduling.
/// No floating-point atomics are used.</para>
/// <para>Half tensors are accumulated in single precision and rounded once on output. They need the softmax axis to be the last one, an even row
/// length and even element offsets, because rows are written as whole 32-bit words. Not thread-safe.</para>
/// </remarks>
public sealed class SoftmaxKernels : IDisposable
{
    const int Words = 8;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly bool _subgroups;
    readonly KernelPipeline?[] _block = new KernelPipeline?[4];
    readonly KernelPipeline?[] _thread = new KernelPipeline?[2];
    readonly KernelPipeline?[] _multi = new KernelPipeline?[2];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="allowSubgroups">False keeps the fast variant off subgroup operations even where the device has them.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public SoftmaxKernels(GpuDevice device, bool allowSubgroups = true, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        _subgroups = Variants.UseSubgroups(device, allowSubgroups);
    }

    /// <summary>How <see cref="GpuKernelMode.Fast"/> merges row statistics: <c>subgroup</c> or <c>shared</c>.</summary>
    public string Variant => _subgroups ? "subgroup" : "shared";

    /// <summary>Records softmax (or log-softmax) of <paramref name="shape"/> along <paramref name="axis"/>.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="input">Input tensor, dense row-major.</param>
    /// <param name="offsetInput">Element offset of the tensor in <paramref name="input"/>.</param>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="axis">The axis in [-rank, rank).</param>
    /// <param name="output">Receives a tensor of the same shape; it may be the input range itself.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="logarithm">True for log-softmax.</param>
    /// <param name="mode">Fast or deterministic.</param>
    /// <exception cref="GpuKernelShapeException">The axis is out of range, a range reaches past its buffer, or a half tensor does not meet the rules above.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, GpuBuffer<T> input, int offsetInput, ReadOnlySpan<int> shape, int axis, GpuBuffer<T> output, int offsetOutput, bool logarithm = false,
        GpuKernelMode mode = GpuKernelMode.Fast)
        where T : unmanaged
    {
        const string Name = "Softmax";
        var half = KernelTypes.IsHalf<T>(Name);
        ReduceKernels.CheckMode(Name, mode);
        var total = Check.Product(Name, nameof(shape), shape);
        var rank = shape.Length;
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank)
        {
            throw new GpuKernelShapeException(Name, nameof(axis), $"axis {axis} is outside [{-rank}, {rank})");
        }

        long inner = 1;
        for (var i = a + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        long len = shape[a];
        var rows = total == 0 ? 0 : total / len;
        Check.Range(Name, nameof(input), input.Length, offsetInput, total, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, total, half);
        if (half && total > 0)
        {
            if (inner != 1 || (len & 1) != 0 || (offsetInput & 1) != 0 || (offsetOutput & 1) != 0)
            {
                throw new GpuKernelShapeException(Name, nameof(shape), "half tensors need the softmax axis last, an even length along it and even element offsets");
            }
        }

        if (total == 0)
        {
            return;
        }

        Span<uint> w = stackalloc uint[Words];
        w[0] = (uint)rows;
        w[1] = (uint)len;
        w[2] = (uint)offsetInput;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)inner;
        w[6] = logarithm ? 1u : 0u;
        var wave = mode == GpuKernelMode.Fast && _subgroups;
        var block = len > 1024;
        var multi = !block && len >= 16;
        KernelPipeline pipeline;
        uint rowsPerGroup = 1;
        if (multi)
        {
            // About four elements per thread, a power of two between 4 and 256 threads.
            var lanes = 4u;
            while (lanes < 256 && lanes * 4 < len)
            {
                lanes <<= 1;
            }

            w[7] = lanes;
            rowsPerGroup = 256 / lanes;
            pipeline = _multi[half ? 1 : 0] ??= Create(half, false, 2);
        }
        else if (block)
        {
            var slot = (half ? 1 : 0) + (wave ? 2 : 0);
            pipeline = _block[slot] ??= Create(half, wave, 1);
        }
        else
        {
            pipeline = _thread[half ? 1 : 0] ??= Create(half, false, 0);
        }

        var (gx, gy) = block ? Grid.For(_device, rows) : Grid.For(_device, (rows + (multi ? rowsPerGroup : 256) - 1) / (multi ? rowsPerGroup : 256));
        w[5] = gx;
        pipeline.Record(encoder, w, gx, gy, input.Buffer, output.Buffer);
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => KernelPipeline.Total(_block, _thread, _multi);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _block, _thread, _multi })
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
        foreach (var set in new[] { _block, _thread, _multi })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }
    }

    // kind: 0 thread per row, 1 workgroup per row, 2 several rows per workgroup.
    KernelPipeline Create(bool half, bool wave, int kind)
    {
        var v = Softmax.Variants.Get(new Softmax.Key(f16: half, wave: wave));
        if (kind == 2)
        {
            return new KernelPipeline(_device, KernelSources.Make(v.SoftmaxMulti.EntryPoint, v.SoftmaxMulti.SpirV, v.SoftmaxMulti.Msl, v.SoftmaxMulti.MslFunctionName, v.SoftmaxMulti.Wgsl, v.SoftmaxMulti.WgslEntryPoint, v.SoftmaxMulti.Layout), 2, Words, "softmax-multi", _maxBindGroups);
        }

        return kind == 1
            ? new KernelPipeline(_device, KernelSources.Make(v.SoftmaxBlock.EntryPoint, v.SoftmaxBlock.SpirV, v.SoftmaxBlock.Msl, v.SoftmaxBlock.MslFunctionName, v.SoftmaxBlock.Wgsl, v.SoftmaxBlock.WgslEntryPoint, v.SoftmaxBlock.Layout), 2, Words, "softmax-block", _maxBindGroups)
            : new KernelPipeline(_device, KernelSources.Make(v.SoftmaxThread.EntryPoint, v.SoftmaxThread.SpirV, v.SoftmaxThread.Msl, v.SoftmaxThread.MslFunctionName, v.SoftmaxThread.Wgsl, v.SoftmaxThread.WgslEntryPoint, v.SoftmaxThread.Layout), 2, Words, "softmax-thread", _maxBindGroups);
    }
}
