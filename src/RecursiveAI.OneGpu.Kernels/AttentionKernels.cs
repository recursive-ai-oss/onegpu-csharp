using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Which attention kernel an <see cref="AttentionKernels"/> instance uses.</summary>
public enum GpuAttentionVariant
{
    /// <summary>Chosen from the device: <see cref="Pairs"/> on Apple GPUs (which have few registers per thread) when shuffles are available, otherwise <see cref="Rows"/>.</summary>
    Auto = 0,

    /// <summary>One thread per query row; needs no subgroup operations.</summary>
    Rows = 1,

    /// <summary>Two neighbouring threads per query row, each holding half of the head dimension; needs subgroup shuffles.</summary>
    Pairs = 2,
}

/// <summary>
/// Fused scaled dot-product attention, <c>O = softmax(Scale * Q Kᵀ + Mask) V</c>, that never writes the score matrix to memory: the keys are
/// visited in tiles with the online softmax normaliser (Dao et al., "FlashAttention", 2022; Milakov and Gimelshein, 2018).
/// </summary>
/// <remarks>
/// <para><b>Inputs.</b> Q, K, V and O are float or <see cref="Half"/> tensors in any layout a <see cref="GpuAttentionLayout"/> can describe
/// (head-first, position-first, slices of a packed projection). Half tensors are accumulated in float32 and rounded once on output; they need
/// an even head dimension and even offsets and strides, because rows are read and written as 32-bit words, and buffers padded with
/// <see cref="GpuKernelBuffers.PaddedHalfLength"/>. The optional additive mask is always float32; it may broadcast over batch, heads and
/// query rows (<see cref="GpuAttentionLayout.ForMask"/>). Optional per-batch key lengths (<c>GpuBuffer&lt;int&gt;</c>) hide the padding keys of
/// a batch; values are clamped to [0, KLen]. <see cref="AttentionProblem.Causal"/> hides later keys.</para>
/// <para><b>Special values.</b> A key hidden by the mask (-infinity), the key lengths or the causal rule gets weight 0. A row without any visible
/// key gives NaN, or zeros with <see cref="GpuAttentionMaskedRows.Zero"/>. A row with a NaN or +infinity score gives NaN. The values V are not
/// inspected: a NaN in a V element of a visible key reaches the output, and one in a hidden key does not.</para>
/// <para><b>Variants.</b> Keys are folded into each query row one at a time with the online normaliser, in increasing key order, from key and value tiles in
/// workgroup memory. With subgroup shuffles (<see cref="Variant"/> <c>pairs</c>) two neighbouring threads share a row and each holds half of the head
/// dimension, which halves the registers per thread (this matters on Apple GPUs); the halves of a dot product are added after one shuffle, and a sum of two
/// numbers does not depend on operand order. Without shuffles (<c>rows</c>) one thread owns a row. The head dimension is rounded up to 32, 64, 128 or 256
/// (<see cref="HeadDimLimit"/>; above 64 the register file spills, so 128 and 256 are correct but slow). When the problem has too few query blocks to occupy
/// a GPU, the keys are cut into up to 16 contiguous ranges of a multiple of 32 keys (the cut depends only on the shape) whose partial results a second pass
/// merges in range order through a scratch buffer (<see cref="ScratchLength"/>). There are no atomics and no subgroup arithmetic, and every order is fixed by
/// the code, so <see cref="GpuKernelMode.Fast"/> and <see cref="GpuKernelMode.Deterministic"/> run the same kernel and give identical bits on one device and
/// driver; <c>pairs</c> and <c>rows</c> may differ from each other in the last bits, and so may devices. A cooperative-matrix variant would plug in beside
/// these, chosen in the constructor from the device capabilities.</para>
/// <para>Not thread-safe. Q, K and V may share one buffer; O must not overlap them.</para>
/// </remarks>
public sealed class AttentionKernels : IDisposable
{
    /// <summary>The largest head dimension supported.</summary>
    public const int HeadDimLimit = 256;

    const int MaxSplit = 16;
    const int SplitGranule = 32;
    const int MinKeysPerSplit = 128;
    const int TargetGroups = 256;
    const int TileWords = 32;
    const int CombineWords = 11;

    readonly GpuDevice _device;
    readonly KernelPipeline?[] _tile = new KernelPipeline?[16];
    readonly bool _pairs;
    readonly KernelPipeline?[] _combine = new KernelPipeline?[2];
    GpuBuffer<float>? _scratch;
    GpuBuffer<float>? _dummyFloat;
    GpuBuffer<float>? _dummyPart;
    GpuBuffer<int>? _dummyInt;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="allowSubgroups">False keeps the kernels off subgroup shuffles even where the device has them.</param>
    /// <param name="variant">The kernel to use; <see cref="GpuAttentionVariant.Auto"/> picks one for the device.</param>
    /// <exception cref="GpuUnsupportedException"><see cref="GpuAttentionVariant.Pairs"/> was requested and the device (or <paramref name="allowSubgroups"/>) rules out subgroup shuffles, or the device offers fewer than 7 storage buffers per stage.</exception>
    public AttentionKernels(GpuDevice device, bool allowSubgroups = true, GpuAttentionVariant variant = GpuAttentionVariant.Auto)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        var c = device.Capabilities.Compute;
        var min = c.SubgroupMinSize != 0 ? c.SubgroupMinSize : c.SubgroupSize;
        var shuffles = allowSubgroups && (c.SubgroupOperations & GpuSubgroupOperations.Basic) != 0 && (c.SubgroupOperations & GpuSubgroupOperations.Shuffle) != 0 && min >= 2;
        if (variant == GpuAttentionVariant.Pairs && !shuffles)
        {
            throw new GpuUnsupportedException("the pairs attention kernel needs subgroup shuffles, which the device does not offer or the caller ruled out");
        }

        // Apple GPUs give a thread about 128 registers before it spills; halving the registers per thread made the kernel 2 to 5 times faster there,
        // and slower on the NVIDIA GPU measured, which has registers to spare.
        const uint AppleVendorId = 0x106B;
        _pairs = variant == GpuAttentionVariant.Pairs || (variant == GpuAttentionVariant.Auto && shuffles && device.Adapter.VendorId == AppleVendorId);
        if (device.Capabilities.Compute.MaxStorageBuffersPerStage is > 0 and < 7)
        {
            throw new GpuUnsupportedException($"attention binds 7 storage buffers per stage; the device allows {device.Capabilities.Compute.MaxStorageBuffersPerStage}");
        }
    }

    /// <summary>
    /// The variant <see cref="GpuKernelMode.Fast"/> and <see cref="GpuKernelMode.Deterministic"/> both use: <c>pairs</c> when the device has subgroup
    /// shuffles (two threads share a query row, each holding half of the head dimension), otherwise <c>rows</c> (one thread per query row).
    /// </summary>
    public string Variant => _pairs ? "pairs" : "rows";

    /// <summary>
    /// Floats of scratch the problem needs: 0 when the keys are not split, otherwise one (maximum, sum, accumulator) record of <c>HeadDim + 2</c>
    /// floats per query row and key range.
    /// </summary>
    /// <param name="problem">The problem; only its sizes are read.</param>
    /// <returns>The length, in floats, of the <c>scratch</c> buffer <c>Encode</c> would use.</returns>
    /// <exception cref="GpuKernelShapeException">A size is negative or too large.</exception>
    public long ScratchLength(in AttentionProblem problem)
    {
        CheckSizes(problem);
        var (splits, _) = Split(problem, _pairs);
        return splits == 1 ? 0 : checked((long)problem.Batch * problem.Heads * problem.QLen * splits * (problem.HeadDim + 2));
    }

    /// <summary>Records the attention described by <paramref name="problem"/>.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>; Q, K, V and O all use it.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="problem">Sizes, layouts and options.</param>
    /// <param name="q">Queries.</param>
    /// <param name="k">Keys.</param>
    /// <param name="v">Values.</param>
    /// <param name="o">Receives the result; every element of the output range is written.</param>
    /// <param name="mask">Optional additive mask, laid out as <see cref="AttentionProblem.Mask"/> says; null for none.</param>
    /// <param name="keyLengths">Optional number of valid keys per batch entry, read at <see cref="AttentionProblem.KeyLengthsOffset"/>; null for all.</param>
    /// <param name="mode">Fast or deterministic (the same kernel; see the remarks of the class).</param>
    /// <param name="scratch">Optional scratch of at least <see cref="ScratchLength"/> floats. When null and one is needed, the family keeps its own and grows it as needed.</param>
    /// <exception cref="GpuKernelShapeException">A size, stride or offset is invalid, a range reaches past its buffer, a half tensor breaks the alignment rules, the head dimension exceeds <see cref="HeadDimLimit"/>, or the scratch is too small.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, in AttentionProblem problem, GpuBuffer<T> q, GpuBuffer<T> k, GpuBuffer<T> v, GpuBuffer<T> o,
        GpuBuffer<float>? mask = null, GpuBuffer<int>? keyLengths = null, GpuKernelMode mode = GpuKernelMode.Fast, GpuBuffer<float>? scratch = null)
        where T : unmanaged
    {
        const string Name = "Attention";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(o);
        var half = KernelTypes.IsHalf<T>(Name);
        ReduceKernels.CheckMode(Name, mode);
        var p = problem;
        CheckSizes(p);
        CheckTensor(Name, "q", q.Length, p.Q, p.Batch, p.Heads, p.QLen, p.HeadDim, half);
        CheckTensor(Name, "k", k.Length, p.K, p.Batch, p.KvHeads, p.KLen, p.HeadDim, half);
        CheckTensor(Name, "v", v.Length, p.V, p.Batch, p.KvHeads, p.KLen, p.HeadDim, half);
        CheckTensor(Name, "o", o.Length, p.O, p.Batch, p.Heads, p.QLen, p.HeadDim, half);
        if (mask is not null)
        {
            CheckTensor(Name, "mask", mask.Length, p.Mask, p.Batch, p.Heads, p.QLen, p.KLen, false);
        }

        if (keyLengths is not null)
        {
            Check.Range(Name, nameof(keyLengths), keyLengths.Length, p.KeyLengthsOffset, p.Batch);
        }

        if (p.Batch == 0 || p.Heads == 0 || p.QLen == 0 || p.HeadDim == 0)
        {
            return;
        }

        var (splits, perSplit) = Split(p, _pairs);
        var blockRows = _pairs ? 32 : 64;
        var groups = checked((long)p.Batch * p.Heads * ((p.QLen + blockRows - 1) / blockRows) * splits);
        // Unused bindings still need a buffer, and a buffer bound both read-only and read-write in one pass is invalid in WebGPU, so each gets its own.
        var unusedMask = _dummyFloat ??= _device.CreateBuffer<float>(4, label: "attention-unused-mask");
        var part = _dummyPart ??= _device.CreateBuffer<float>(4, label: "attention-unused-partials");
        if (splits > 1)
        {
            var need = ScratchLength(p);
            if (scratch is not null)
            {
                if (scratch.Length < need)
                {
                    throw new GpuKernelShapeException(Name, nameof(scratch), $"the scratch holds {scratch.Length} floats, {need} are needed");
                }

                part = scratch;
            }
            else
            {
                if (_scratch is null || _scratch.Length < need)
                {
                    // The old buffer is released once the GPU has finished with commands that still use it.
                    _scratch?.Dispose();
                    _scratch = _device.CreateBuffer<float>((int)Math.Min(int.MaxValue, Math.Max(need, 1 << 16)), label: "attention-scratch");
                }

                part = _scratch;
            }
        }

        var causal = p.Causal;
        var flags = (mask is not null ? 1u : 0u) | (causal ? 2u : 0u) | (keyLengths is not null ? 4u : 0u) | (p.MaskedRows == GpuAttentionMaskedRows.Zero ? 8u : 0u);
        var causalOffset = p.CausalOffset ?? (p.KLen - p.QLen);
        var (gx, gy) = Grid.For(_device, groups);
        Span<uint> w = stackalloc uint[TileWords];
        w[0] = (uint)p.Heads;
        w[1] = (uint)(p.Heads / p.KvHeads);
        w[2] = (uint)p.QLen;
        w[3] = (uint)p.KLen;
        w[4] = (uint)p.HeadDim;
        w[5] = BitConverter.SingleToUInt32Bits(p.Scale);
        w[6] = (uint)causalOffset;
        w[7] = flags;
        Put(w, 8, p.Q);
        Put(w, 11, p.K);
        Put(w, 14, p.V);
        Put(w, 17, p.O);
        w[20] = (uint)p.Q.Offset;
        w[21] = (uint)p.K.Offset;
        w[22] = (uint)p.V.Offset;
        w[23] = (uint)p.O.Offset;
        w[24] = (uint)p.Mask.Offset;
        w[25] = (uint)p.Mask.BatchStride;
        w[26] = (uint)p.Mask.HeadStride;
        w[27] = (uint)p.Mask.RowStride;
        w[28] = (uint)p.KeyLengthsOffset;
        w[29] = (uint)splits;
        w[30] = (uint)perSplit;
        w[31] = gx;

        var dim = p.HeadDim <= 32 ? 0 : (p.HeadDim <= 64 ? 1 : (p.HeadDim <= 128 ? 2 : 3));
        var tile = _tile[dim + (half ? 4 : 0) + (_pairs ? 8 : 0)] ??= CreateTile(half, dim);
        Span<GpuBuffer> bound =
        [
            q.Buffer,
            k.Buffer,
            v.Buffer,
            mask is null ? unusedMask.Buffer : mask.Buffer,
            keyLengths is null ? (_dummyInt ??= _device.CreateBuffer<int>(4, label: "attention-unused")).Buffer : keyLengths.Buffer,
            o.Buffer,
            part.Buffer,
        ];
        tile.Record(encoder, w, gx, gy, bound);
        if (splits == 1)
        {
            return;
        }

        var rows = (long)p.Batch * p.Heads * p.QLen;
        var threads = half ? rows * (p.HeadDim / 2) : rows * p.HeadDim;
        var (cx, cy) = Grid.For(_device, (threads + 63) / 64);
        Span<uint> c = stackalloc uint[CombineWords];
        c[0] = (uint)rows;
        c[1] = (uint)p.Heads;
        c[2] = (uint)p.QLen;
        c[3] = (uint)p.HeadDim;
        c[4] = (uint)splits;
        c[5] = flags;
        c[6] = (uint)p.O.BatchStride;
        c[7] = (uint)p.O.HeadStride;
        c[8] = (uint)p.O.RowStride;
        c[9] = (uint)p.O.Offset;
        c[10] = cx;
        var merge = _combine[half ? 1 : 0] ??= CreateCombine(half);
        merge.Record(encoder, c, cx, cy, part.Buffer, o.Buffer);
    }

    /// <summary>Releases the pipelines, bind groups and the family's own scratch once the GPU has finished with them.</summary>
    public void Dispose()
    {
        foreach (var set in new[] { _tile, _combine })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }

        _scratch?.Dispose();
        _scratch = null;
        _dummyFloat?.Dispose();
        _dummyFloat = null;
        _dummyPart?.Dispose();
        _dummyPart = null;
        _dummyInt?.Dispose();
        _dummyInt = null;
    }

    static void Put(Span<uint> w, int at, GpuAttentionLayout l)
    {
        w[at] = (uint)l.BatchStride;
        w[at + 1] = (uint)l.HeadStride;
        w[at + 2] = (uint)l.RowStride;
    }

    // How many key ranges, and how many keys in each. Depends only on the shape, so both modes and every device cut alike.
    static (int Splits, int PerSplit) Split(in AttentionProblem p, bool pairs)
    {
        if (p.Batch == 0 || p.Heads == 0 || p.QLen == 0 || p.KLen < 2 * MinKeysPerSplit)
        {
            return (1, Math.Max(p.KLen, 1));
        }

        var blockRows = pairs ? 32 : 64;
        var blocks = (long)p.Batch * p.Heads * ((p.QLen + blockRows - 1) / blockRows);
        var want = (int)Math.Min(MaxSplit, (TargetGroups + blocks - 1) / blocks);
        var splits = Math.Min(want, p.KLen / MinKeysPerSplit);
        if (splits <= 1)
        {
            return (1, p.KLen);
        }

        var per = (p.KLen + splits - 1) / splits;
        per = (per + SplitGranule - 1) / SplitGranule * SplitGranule;
        return ((p.KLen + per - 1) / per, per);
    }

    static void CheckSizes(in AttentionProblem p)
    {
        const string Name = "Attention";
        Positive(Name, nameof(p.Batch), p.Batch);
        Positive(Name, nameof(p.Heads), p.Heads);
        Positive(Name, nameof(p.KvHeads), p.KvHeads);
        Positive(Name, nameof(p.QLen), p.QLen);
        Positive(Name, nameof(p.KLen), p.KLen);
        Positive(Name, nameof(p.HeadDim), p.HeadDim);
        if (p.HeadDim > HeadDimLimit)
        {
            throw new GpuKernelShapeException(Name, nameof(p.HeadDim), $"head dimension {p.HeadDim} is above the limit of {HeadDimLimit}");
        }

        if (p.KvHeads == 0 ? p.Heads != 0 : p.Heads % p.KvHeads != 0)
        {
            throw new GpuKernelShapeException(Name, nameof(p.KvHeads), $"{p.Heads} query heads are not a multiple of {p.KvHeads} key/value heads");
        }

        if (!float.IsFinite(p.Scale))
        {
            throw new GpuKernelShapeException(Name, nameof(p.Scale), "the scale must be finite");
        }

        Check.Product(Name, "sizes", [p.Batch, p.Heads, p.QLen, p.HeadDim]);
        Check.Product(Name, "sizes", [p.Batch, p.KvHeads, p.KLen, p.HeadDim]);
        if ((long)p.Batch * p.Heads * p.QLen > int.MaxValue / 64)
        {
            throw new GpuKernelShapeException(Name, "sizes", "too many query rows");
        }
    }

    static void Positive(string name, string arg, int value)
    {
        if (value < 0)
        {
            throw new GpuKernelShapeException(name, arg, $"{value} is negative");
        }
    }

    // The last element a tensor touches must lie inside its buffer; half tensors also need word alignment.
    static void CheckTensor(string name, string arg, int length, in GpuAttentionLayout l, int batch, int heads, int rows, int width, bool half)
    {
        if (l.Offset < 0 || l.BatchStride < 0 || l.HeadStride < 0 || l.RowStride < 0)
        {
            throw new GpuKernelShapeException(name, arg, "offsets and strides must not be negative");
        }

        if (batch == 0 || heads == 0 || rows == 0 || width == 0)
        {
            return;
        }

        long last = l.Offset + ((long)(batch - 1) * l.BatchStride) + ((long)(heads - 1) * l.HeadStride) + ((long)(rows - 1) * l.RowStride) + (width - 1);
        if (last >= length || last > int.MaxValue)
        {
            throw new GpuKernelShapeException(name, arg, $"the tensor reaches element {last}, past the buffer of {length}");
        }

        if (half)
        {
            if ((width & 1) != 0 || (l.Offset & 1) != 0 || (l.BatchStride & 1) != 0 || (l.HeadStride & 1) != 0 || (l.RowStride & 1) != 0)
            {
                throw new GpuKernelShapeException(name, arg, "half tensors need an even head dimension, offset and strides");
            }

            if (((last + 2) >> 1 << 1) > length)
            {
                throw new GpuKernelShapeException(name, arg, $"the buffer of {length} halves ends inside a 32-bit word; allocate GpuKernelBuffers.PaddedHalfLength(n) elements");
            }
        }
    }

    KernelPipeline CreateTile(bool half, int dim)
    {
        var d = dim switch { 0 => Attention.HdValue.D32, 1 => Attention.HdValue.D64, 2 => Attention.HdValue.D128, _ => Attention.HdValue.D256 };
        var wd = dim switch { 0 => AttentionLoops.HdValue.D32, 1 => AttentionLoops.HdValue.D64, 2 => AttentionLoops.HdValue.D128, _ => AttentionLoops.HdValue.D256 };
        if (_device.Backend == GpuBackend.WebGpu)
        {
            var w = AttentionLoops.Variants.Get(new AttentionLoops.Key(f16: half, wave: _pairs, hd: wd)).AttentionTile;
            return new KernelPipeline(_device, KernelSources.Make(w.EntryPoint, default, default, null, w.Wgsl, w.WgslEntryPoint, w.Layout), 7, TileWords, "attention-tile");
        }

        var e = Attention.Variants.Get(new Attention.Key(f16: half, wave: _pairs, hd: d)).AttentionTile;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, default, null, e.Layout), 7, TileWords, "attention-tile");
    }

    KernelPipeline CreateCombine(bool half)
    {
        var e = AttentionCombine.Variants.Get(new AttentionCombine.Key(f16: half)).AttentionMerge;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, CombineWords, "attention-combine");
    }
}
