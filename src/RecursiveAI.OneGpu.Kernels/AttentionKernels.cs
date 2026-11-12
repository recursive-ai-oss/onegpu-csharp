using OneShader.Reflection;
using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Which attention kernel an <see cref="AttentionKernels"/> instance uses.</summary>
public enum GpuAttentionVariant
{
    /// <summary>
    /// Chosen from the device: <see cref="Cooperative"/> for <see cref="Half"/> tensors where the device offers matrix hardware for them; for everything else
    /// <see cref="Pairs"/> on Apple GPUs (which have few registers per thread) when shuffles are available, otherwise <see cref="Rows"/>.
    /// </summary>
    Auto = 0,

    /// <summary>One thread per query row; needs no subgroup operations.</summary>
    Rows = 1,

    /// <summary>Two neighbouring threads per query row, each holding half of the head dimension; needs subgroup shuffles.</summary>
    Pairs = 2,

    /// <summary>
    /// Q Kᵀ and P V on matrix hardware (cooperative matrices on Vulkan, simdgroup matrices on Metal) for <see cref="Half"/> tensors with head dimension up to 128;
    /// float32 tensors and larger heads use the kernel <see cref="Auto"/> would pick without it. Throws <see cref="GpuUnsupportedException"/> where the device lacks it.
    /// </summary>
    Cooperative = 3,
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
/// driver; <c>pairs</c> and <c>rows</c> may differ from each other in the last bits, and so may devices. On matrix hardware (cooperative matrices on Vulkan, simdgroup matrices on Apple GPUs, where
/// float32 too) a matrix kernel replaces them, chosen in the constructor from the device capabilities; its matrix products are fixed functions of their operands and it keeps the same
/// guarantees (see <see cref="Variant"/> and <see cref="FloatVariant"/>).</para>
/// <para>Not thread-safe. Q, K and V may share one buffer; O must not overlap them.</para>
/// </remarks>
public sealed class AttentionKernels : IDisposable
{
    /// <summary>The largest head dimension supported.</summary>
    public const int HeadDimLimit = 256;

    /// <summary>The largest head dimension the matrix-hardware kernel covers; larger heads of half tensors use the scalar kernels.</summary>
    public const int CoopHeadDimLimit = 128;

    /// <summary>The largest head dimension the float32 matrix-hardware kernel covers (Apple GPUs); larger heads of float32 tensors use the scalar kernels.</summary>
    public const int CoopF32HeadDimLimit = 64;

    const int CoopBlockRows = 64;   // query rows per workgroup of the cooperative-matrix kernel on Vulkan
    const int SimdBlockRows = 64;   // ... of the Apple kernel with its Q fragments read from the tensor (AttentionSimd.hlsl, QD)
    const int SimdShortBlockRows = 16;   // ... of the Apple kernel with the Q fragments in registers: one simdgroup
    const int MaxSplit = 16;
    const int SplitGranule = 32;
    const int MinKeysPerSplit = 128;
    const int TargetGroups = 256;
    const int TileWords = 32;
    // A workgroup that walks at least this many keys sums blockwise (see Attention.hlsl): the rounding error of one running sum grows with the key
    // count, while the blockwise form costs registers and with them speed on short key ranges. The value sits above the 512 keys of BERT-class models
    // and below the 768 per range that a Whisper encoder call gets.
    const int BlockwiseSumKeys = 640;
    const double ScalarWorkWeight = 4;    // the scalar kernels run an operation several times slower than the matrix path, which the submission budget counts in matrix-path operations
    const double CoopWorkWeight = 1.5;    // the matrix kernel multiplies Q Kᵀ twice
    const int CombineWords = 11;

    readonly GpuDevice _device;
    readonly KernelPipeline?[] _tile = new KernelPipeline?[32];
    readonly KernelPipeline?[] _coopTile = new KernelPipeline?[18];
    readonly int _maxBindGroups;
    readonly bool _pairs;
    readonly bool _coop;
    readonly bool _coopBf16;
    readonly bool _coopF32;
    readonly ulong _budget;
    readonly KernelPipeline?[] _combine = new KernelPipeline?[3];
    GpuBuffer<float>? _scratch;
    GpuBuffer<float>? _dummyFloat;
    GpuBuffer<float>? _dummyPart;
    GpuBuffer<int>? _dummyInt;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="allowSubgroups">False keeps the kernels off subgroup shuffles even where the device has them.</param>
    /// <param name="variant">The kernel to use; <see cref="GpuAttentionVariant.Auto"/> picks one for the device.</param>
    /// <param name="submissionWorkBudget">Operations after which a long call is cut into dispatches of whole workgroups with the encoder told the work between them (see <see cref="GpuCommandEncoder.ReportWork"/>); null takes the device's <see cref="GpuComputeCapabilities.SubmissionWorkBudget"/>, 0 never cuts. The cut changes no bit.</param>
    /// <param name="allowMatrixHardware">False keeps <see cref="GpuAttentionVariant.Auto"/> off the matrix-hardware kernel even where the device has it.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    /// <exception cref="GpuUnsupportedException"><see cref="GpuAttentionVariant.Pairs"/> was requested and the device (or <paramref name="allowSubgroups"/>) rules out subgroup shuffles; <see cref="GpuAttentionVariant.Cooperative"/> was requested and the device (or <paramref name="allowMatrixHardware"/>) rules out matrix hardware; or the device offers fewer than 7 storage buffers per stage.</exception>
    public AttentionKernels(GpuDevice device, bool allowSubgroups = true, GpuAttentionVariant variant = GpuAttentionVariant.Auto, bool allowMatrixHardware = true, ulong? submissionWorkBudget = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        _budget = submissionWorkBudget ?? device.Capabilities.Compute.SubmissionWorkBudget;
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
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
        var matrices = allowMatrixHardware && (variant is GpuAttentionVariant.Auto or GpuAttentionVariant.Cooperative) && CoopFits(device, bfloat16: false);
        if (variant == GpuAttentionVariant.Cooperative && !matrices)
        {
            throw new GpuUnsupportedException("the cooperative-matrix attention kernel needs 16-bit cooperative matrices with a float32 accumulator, 32-wide full subgroups and the shader features of the kernel, which the device does not offer or the caller ruled out");
        }

        _coop = matrices;
        _coopBf16 = matrices && CoopFits(device, bfloat16: true);
        // Float32 fragments exist on Apple GPUs only. The family wants them whenever the device offers them, whatever 16-bit support it has.
        _coopF32 = allowMatrixHardware && (variant is GpuAttentionVariant.Auto or GpuAttentionVariant.Cooperative) && CoopFitsFloat32(device);
        // The scalar kernels still serve float32 tensors and large heads, so they are chosen even when the matrix kernel is.
        _pairs = variant == GpuAttentionVariant.Pairs || ((variant is GpuAttentionVariant.Auto or GpuAttentionVariant.Cooperative) && shuffles && device.Adapter.VendorId == AppleVendorId);
        if (device.Capabilities.Compute.MaxStorageBuffersPerStage is > 0 and < 7)
        {
            throw new GpuUnsupportedException($"attention binds 7 storage buffers per stage; the device allows {device.Capabilities.Compute.MaxStorageBuffersPerStage}");
        }
    }

    /// <summary>
    /// The variant <see cref="GpuKernelMode.Fast"/> and <see cref="GpuKernelMode.Deterministic"/> both use for <see cref="Half"/> tensors with a head dimension up to 128:
    /// <c>cooperative-matrix</c> when the device has matrix hardware for them, otherwise what <see cref="FloatVariant"/> says.
    /// </summary>
    public string Variant => _coop ? "cooperative-matrix" : ScalarVariant;

    /// <summary>
    /// The variant used for float32 tensors, and for half tensors when <see cref="Variant"/> is not the matrix kernel: <c>pairs</c> when chosen (two threads
    /// share a query row, each holding half of the head dimension; Apple GPUs), otherwise <c>rows</c> (one thread per query row).
    /// </summary>
    public string FloatVariant => _coopF32 ? "cooperative-matrix" : ScalarVariant;

    string ScalarVariant => _pairs ? "pairs" : "rows";

    /// <summary>
    /// Floats of scratch the problem needs: 0 when the keys are not split, otherwise one (maximum, sum, accumulator) record of <c>HeadDim + 2</c>
    /// floats per query row and key range. When the kernels for float32 and half tensors cut the keys differently, the larger need.
    /// </summary>
    /// <param name="problem">The problem; only its sizes are read.</param>
    /// <returns>The length, in floats, of the <c>scratch</c> buffer <c>Encode</c> would use.</returns>
    /// <exception cref="GpuKernelShapeException">A size is negative or too large.</exception>
    public long ScratchLength(in AttentionProblem problem)
    {
        CheckSizes(problem);
        var (a, _) = Split(problem, _pairs ? 32 : 64);
        var (c, _) = (_coop || _coopF32) && problem.HeadDim <= CoopHeadDimLimit ? Split(problem, CoopRows(problem)) : (1, 0);
        var splits = Math.Max(a, c);
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
        where T : unmanaged =>
        EncodeCore(encoder, problem, q, k, v, o, mask, keyLengths, mode, scratch, bfloat16: false);

    /// <summary>
    /// Records the attention described by <paramref name="problem"/> for bfloat16 tensors (the 16-bit pattern in a <see cref="ushort"/>), with float32
    /// accumulation and softmax, on matrix hardware. There is no scalar bfloat16 kernel: the device must offer bfloat16 matrices
    /// (<see cref="BFloat16Available"/>) and the head dimension must not exceed <see cref="CoopHeadDimLimit"/>. The rules for alignment, special values and
    /// ordering are those of <see cref="Encode{T}"/> for <see cref="Half"/>.
    /// </summary>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="problem">Sizes, layouts and options.</param>
    /// <param name="q">Queries.</param>
    /// <param name="k">Keys.</param>
    /// <param name="v">Values.</param>
    /// <param name="o">Receives the result, rounded to nearest even.</param>
    /// <param name="mask">Optional additive float32 mask.</param>
    /// <param name="keyLengths">Optional number of valid keys per batch entry.</param>
    /// <param name="mode">Fast or deterministic (the same kernel).</param>
    /// <param name="scratch">Optional scratch of at least <see cref="ScratchLength"/> floats.</param>
    /// <exception cref="GpuUnsupportedException">The device has no bfloat16 matrices (or the family was created without matrix hardware), or the head dimension is above <see cref="CoopHeadDimLimit"/>.</exception>
    /// <exception cref="GpuKernelShapeException">As for <see cref="Encode{T}"/>.</exception>
    public void EncodeBFloat16(GpuCommandEncoder encoder, in AttentionProblem problem, GpuBuffer<ushort> q, GpuBuffer<ushort> k, GpuBuffer<ushort> v, GpuBuffer<ushort> o,
        GpuBuffer<float>? mask = null, GpuBuffer<int>? keyLengths = null, GpuKernelMode mode = GpuKernelMode.Fast, GpuBuffer<float>? scratch = null)
    {
        if (!_coopBf16)
        {
            throw new GpuUnsupportedException("bfloat16 attention needs bfloat16 cooperative matrices, which the device does not offer or the caller ruled out");
        }

        if (problem.HeadDim > CoopHeadDimLimit)
        {
            throw new GpuUnsupportedException($"bfloat16 attention covers head dimensions up to {CoopHeadDimLimit}; {problem.HeadDim} was asked");
        }

        EncodeCore(encoder, problem, q, k, v, o, mask, keyLengths, mode, scratch, bfloat16: true);
    }

    void EncodeCore<T>(GpuCommandEncoder encoder, in AttentionProblem problem, GpuBuffer<T> q, GpuBuffer<T> k, GpuBuffer<T> v, GpuBuffer<T> o,
        GpuBuffer<float>? mask, GpuBuffer<int>? keyLengths, GpuKernelMode mode, GpuBuffer<float>? scratch, bool bfloat16)
        where T : unmanaged
    {
        const string Name = "Attention";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(o);
        var half = bfloat16 || KernelTypes.IsHalf<T>(Name);
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

        var coop = half ? (bfloat16 ? _coopBf16 : _coop) && p.HeadDim <= CoopHeadDimLimit : _coopF32 && p.HeadDim <= CoopF32HeadDimLimit;
        var pairs = _pairs && !coop;
        var (splits, perSplit) = Split(p, coop ? CoopRows(p) : (pairs ? 32 : 64));
        var blockRows = coop ? CoopRows(p) : (pairs ? 32 : 64);
        var groups = checked((long)p.Batch * p.Heads * ((p.QLen + blockRows - 1) / blockRows) * splits);
        if (groups > int.MaxValue)
        {
            throw new GpuKernelShapeException(Name, "sizes", "too many workgroups");
        }
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

        var dim = p.HeadDim <= 32 ? 0 : (p.HeadDim <= 64 ? 1 : (p.HeadDim <= 128 ? 2 : 3));
        var tile = coop ? _coopTile[dim + (bfloat16 ? 3 : (half ? 0 : 6)) + (QueryDirect(p) ? 9 : 0)] ??= CreateCoopTile(dim, bfloat16, !half, QueryDirect(p)) : _tile[dim + (half ? 4 : 0) + (pairs ? 8 : 0) + (perSplit >= BlockwiseSumKeys ? 16 : 0)] ??= CreateTile(half, dim, perSplit >= BlockwiseSumKeys);
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

        // A submission the platform may stop for running too long is cut between dispatches of whole workgroups; every workgroup computes its rows
        // independently, so the cut changes no bit. The budget counts operations of the matrix path; the scalar kernels are slower per operation.
        var ops = 4.0 * p.Batch * p.Heads * p.QLen * p.KLen * p.HeadDim * (coop ? CoopWorkWeight : ScalarWorkWeight);
        var budget = _budget;
        var perGroup = ops / groups;
        var piece = budget == 0 || ops <= budget ? groups : Math.Max(1, (long)(budget / perGroup));
        // A dispatch covers at most the device's workgroup count along x, so the pieces never overshoot the problem.
        var maxX = _device.Capabilities.Compute.MaxWorkgroupCount;
        piece = Math.Min(piece, maxX == 0 ? 65535 : Math.Min(maxX, 65535u));
        for (var first = 0L; first < groups; first += piece)
        {
            var n = Math.Min(piece, groups - first);
            w[31] = (uint)first;
            tile.Record(encoder, w, (uint)n, 1, bound);
            encoder.ReportWork(perGroup * n);
        }

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
        var merge = _combine[bfloat16 ? 2 : (half ? 1 : 0)] ??= CreateCombine(half, bfloat16);
        merge.Record(encoder, c, cx, cy, part.Buffer, o.Buffer);
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => KernelPipeline.Total(_tile, _coopTile, _combine);

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _tile, _coopTile, _combine })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.ReleaseBindings();
            }
        }
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

        foreach (var pipeline in _coopTile)
        {
            pipeline?.Dispose();
        }

        Array.Clear(_coopTile);
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
    static (int Splits, int PerSplit) Split(in AttentionProblem p, int blockRows)
    {
        if (p.Batch == 0 || p.Heads == 0 || p.QLen == 0 || p.KLen < 2 * MinKeysPerSplit)
        {
            return (1, Math.Max(p.KLen, 1));
        }

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

    // The Apple kernel reads its Q fragments from the tensor when the head dimension fills whole fragments and a block of 16 rows fits inside the queries
    // (it slides the last block back to end at the last row); its workgroup is then four simdgroups. Otherwise it is one simdgroup holding Q in registers.
    bool QueryDirect(in AttentionProblem p) => _device.Backend == GpuBackend.Metal && p.QLen >= 16 && p.HeadDim is 32 or 64 or 128;

    int CoopRows(in AttentionProblem p) => _device.Backend != GpuBackend.Metal ? CoopBlockRows : (QueryDirect(p) ? SimdBlockRows : SimdShortBlockRows);

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

    KernelPipeline CreateTile(bool half, int dim, bool blockwise)
    {
        var d = dim switch { 0 => Attention.HdValue.D32, 1 => Attention.HdValue.D64, 2 => Attention.HdValue.D128, _ => Attention.HdValue.D256 };
        var wd = dim switch { 0 => AttentionLoops.HdValue.D32, 1 => AttentionLoops.HdValue.D64, 2 => AttentionLoops.HdValue.D128, _ => AttentionLoops.HdValue.D256 };
        if (_device.Backend == GpuBackend.WebGpu)
        {
            var w = AttentionLoops.Variants.Get(new AttentionLoops.Key(f16: half, wave: _pairs, acc: blockwise, hd: wd)).AttentionTile;
            return new KernelPipeline(_device, KernelSources.Make(w.EntryPoint, default, default, null, w.Wgsl, w.WgslEntryPoint, w.Layout), 7, TileWords, "attention-tile", _maxBindGroups);
        }

        var e = Attention.Variants.Get(new Attention.Key(f16: half, wave: _pairs, acc: blockwise, hd: d)).AttentionTile;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, default, null, e.Layout), 7, TileWords, "attention-tile", _maxBindGroups);
    }

    // The matrix kernel is written for 32-wide subgroups (four per workgroup) and 16x16x16 fragments of the 16-bit element type with a float32
    // accumulator (8x8x8 simdgroup matrices on Metal, which the shader compiler tiles into 16x16). On Vulkan that means a device whose subgroups are 32
    // wide or can be pinned to 32, that runs pipelines with full subgroups and offers the features the kernel's reflection lists.
    static bool CoopFits(GpuDevice device, bool bfloat16)
    {
        const uint SubgroupSize = 32;
        const uint Invocations = 128;
        const uint WorkgroupBytes = (32 * (64 + 8) * 2) + (4 * 256 * 4) + (4 * 16 * 32 * 2);   // the 64-wide head, the largest of the three shapes
        if (device.Backend is not (GpuBackend.Vulkan or GpuBackend.Metal))
        {
            return false;
        }

        var cc = device.Capabilities.Compute;
        var sizeOk = cc.SubgroupMinSize <= SubgroupSize && SubgroupSize <= cc.SubgroupMaxSize && (cc.SubgroupMinSize == cc.SubgroupMaxSize || cc.SubgroupSizeControl);
        if (!sizeOk || !cc.FullSubgroups || cc.MaxWorkgroupInvocations < Invocations || cc.MaxWorkgroupSizeX < Invocations || cc.MaxWorkgroupMemoryBytes < WorkgroupBytes
            || (cc.SubgroupOperations & GpuSubgroupOperations.Shuffle) == 0)
        {
            return false;
        }

        var required = device.Backend == GpuBackend.Metal
            ? Shaders.AttentionSimd.Variants.Get(new Shaders.AttentionSimd.Key(elem: bfloat16 ? Shaders.AttentionSimd.ElemValue.Bf16 : Shaders.AttentionSimd.ElemValue.F16, hd: Shaders.AttentionSimd.HdValue.D64, qd: false)).AttnSimd.Layout.RequiredFeatures
            : Shaders.AttentionCoop.Variants.Get(new Shaders.AttentionCoop.Key(bf16: bfloat16, hd: Shaders.AttentionCoop.HdValue.D64)).AttnCoop.Layout.RequiredFeatures;
        if ((required & ~cc.ShaderFeatures) != DeviceFeatures.None)
        {
            return false;
        }

        var wanted = bfloat16 ? GpuScalarType.BF16 : GpuScalarType.F16;
        var edge = device.Backend == GpuBackend.Metal ? 8u : 16u;
        foreach (var shape in cc.CooperativeMatrices)
        {
            if (shape is { CType: GpuScalarType.F32, ResultType: GpuScalarType.F32, Saturating: false } && shape.M == edge && shape.N == edge && shape.K == edge
                && shape.AType == wanted && shape.BType == wanted)
            {
                return true;
            }
        }

        return false;
    }

    // Float32 simdgroup matrices (8x8x8) with a float32 accumulator, 32-wide simdgroups; the kernel's tiles are 16 keys, so a head dimension of 64
    // keeps its workgroup memory (about 13 KB) under half of Metal's limit, which shader validation counts twice.
    static bool CoopFitsFloat32(GpuDevice device)
    {
        const uint Invocations = 128;
        const uint WorkgroupBytes = (16 * (64 + 8) * 4) + (4 * 256 * 4) + (4 * 16 * 16 * 4);
        if (device.Backend != GpuBackend.Metal)
        {
            return false;
        }

        var cc = device.Capabilities.Compute;
        if (cc.SubgroupMinSize > 32 || cc.SubgroupMaxSize < 32 || !cc.FullSubgroups || cc.MaxWorkgroupInvocations < Invocations || cc.MaxWorkgroupSizeX < Invocations
            || cc.MaxWorkgroupMemoryBytes < 2 * WorkgroupBytes || (cc.SubgroupOperations & GpuSubgroupOperations.Shuffle) == 0)
        {
            return false;
        }

        if ((Shaders.AttentionSimd.Variants.Get(new Shaders.AttentionSimd.Key(elem: Shaders.AttentionSimd.ElemValue.F32, hd: Shaders.AttentionSimd.HdValue.D64, qd: false)).AttnSimd.Layout.RequiredFeatures & ~cc.ShaderFeatures) != DeviceFeatures.None)
        {
            return false;
        }

        foreach (var shape in cc.CooperativeMatrices)
        {
            if (shape is { AType: GpuScalarType.F32, BType: GpuScalarType.F32, CType: GpuScalarType.F32, ResultType: GpuScalarType.F32, Saturating: false, M: 8, N: 8, K: 8 })
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True when <see cref="EncodeBFloat16"/> can run: the device offers bfloat16 matrices with a float32 accumulator and the family may use matrix hardware.</summary>
    public bool BFloat16Available => _coopBf16;

    KernelPipeline CreateCoopTile(int dim, bool bfloat16, bool float32, bool queryDirect)
    {
        if (_device.Backend == GpuBackend.Metal)
        {
            // Apple GPUs run their own matrix kernel, which reads its fragments straight from the tensors and keeps a reference maximum
            // per row instead of an exact one (AttentionSimd.hlsl).
            var sd = dim switch { 0 => Shaders.AttentionSimd.HdValue.D32, 1 => Shaders.AttentionSimd.HdValue.D64, _ => Shaders.AttentionSimd.HdValue.D128 };
            var se = float32 ? Shaders.AttentionSimd.ElemValue.F32 : (bfloat16 ? Shaders.AttentionSimd.ElemValue.Bf16 : Shaders.AttentionSimd.ElemValue.F16);
            var x = Shaders.AttentionSimd.Variants.Get(new Shaders.AttentionSimd.Key(elem: se, hd: sd, qd: queryDirect)).AttnSimd;
            return new KernelPipeline(_device, KernelSources.Make(x.EntryPoint, default, x.Msl, x.MslFunctionName, default, null, x.Layout), 7, TileWords, "attention-simd-tile", _maxBindGroups, requiredSubgroupSize: 32);
        }

        var d = dim switch { 0 => Shaders.AttentionCoop.HdValue.D32, 1 => Shaders.AttentionCoop.HdValue.D64, _ => Shaders.AttentionCoop.HdValue.D128 };
        var e = Shaders.AttentionCoop.Variants.Get(new Shaders.AttentionCoop.Key(bf16: bfloat16, hd: d)).AttnCoop;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, default, null, default, null, e.Layout), 7, TileWords, "attention-coop-tile", _maxBindGroups, requiredSubgroupSize: 32);
    }

    KernelPipeline CreateCombine(bool half, bool bfloat16)
    {
        var e = AttentionCombine.Variants.Get(new AttentionCombine.Key(f16: half && !bfloat16, bf16: bfloat16)).AttentionMerge;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, CombineWords, "attention-combine", _maxBindGroups);
    }
}
