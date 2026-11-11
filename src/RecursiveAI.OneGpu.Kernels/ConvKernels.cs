using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>How a convolution is computed. <see cref="ConvKernels.ChooseVariant"/> reports it for a shape.</summary>
public enum ConvVariant
{
    /// <summary>One matrix product per image on the input as it is: a 1x1 convolution with unit stride and no padding.</summary>
    MatrixProduct = 0,

    /// <summary>The input windows are unfolded into a matrix (Chellapilla et al., 2006) and multiplied by the weights, per group.</summary>
    Im2ColMatrixProduct = 1,

    /// <summary>Each output element is computed directly: every group sees one input channel (depthwise convolution).</summary>
    Depthwise = 2,
}

/// <summary>
/// Two-dimensional convolution and transposed convolution (ONNX <c>Conv</c> and <c>ConvTranspose</c>) over NCHW tensors for float32 and float16
/// (float32 accumulation), with the bias, a residual addition and an activation fused into the last pass.
/// </summary>
/// <remarks>
/// <para>
/// A convolution with one input channel per group is computed directly, one thread per output element. Every other convolution becomes matrix
/// products through <see cref="Gemm"/> (which uses cooperative matrices where the device has them): a 1x1 convolution with unit stride and no padding multiplies
/// the weights by the input as it lies, anything else first unfolds the windows into a column matrix (<see cref="ConvVariant"/>). A transposed convolution multiplies the
/// transposed weights by the input and then folds the columns into the output, gathering per output element so no atomics are needed. The bias, residual and activation
/// are applied by one element-wise pass after the product, in the order sum, bias, residual, activation, each rounded in float32.
/// </para>
/// <para>
/// The column matrix of the unfold variants and, for float16, the float32 accumulators live in scratch buffers; <see cref="Scratch(in ConvProblem, bool)"/> says what a shape needs
/// on this device (nothing for the variants that gather windows while they multiply). Pass buffers and element offsets to <c>Encode</c> to place the scratch in an arena
/// of the caller's (it must not overlap the other operands); a buffer smaller than the whole batch needs is accepted, and the batch is then processed in chunks of as many
/// images as fit, at least one. Pass null and the instance keeps its own, grown on demand up to 64 MiB. Chunks are independent images, so chunking does not change a bit
/// of the result.
/// </para>
/// <para>
/// <b>Determinism.</b> No kernel here uses an atomic or a subgroup operation: each output is summed in an order fixed by the shape (the matrix product's, see <see cref="Gemm"/>,
/// the window order for the direct kernels, or shape-determined ranges of k added in range order when a small grid splits k). <see cref="GpuKernelMode.Deterministic"/> selects the deterministic
/// matrix product, or on Apple GPUs the float32 simdgroup kernel, whose matrix product is a fixed function of its operands, so results are bit-identical run to run on one device and driver in that mode;
/// <see cref="GpuKernelMode.Fast"/> may use matrix hardware elsewhere, whose summation order is the hardware's.
/// </para>
/// <para>
/// Operands must lie in buffers made by <see cref="GpuDevice.CreateBuffer{T}(int, BufferUsage, string?)"/>; half buffers must end on a 32-bit word
/// (<see cref="GpuKernelBuffers.PaddedHalfLength"/>). Input and weights must not overlap the output; a residual may not either. The instance caches bind groups per set of
/// buffers (<c>maxCachedBindGroups</c>); <see cref="ReleaseBindings"/> drops them. Not thread-safe.
/// </para>
/// </remarks>
public sealed class ConvKernels : IDisposable
{
    const string ConvName = "Conv";
    const string TransposeName = "ConvTranspose";
    const long OwnScratchBytes = 64L << 20;

    // Below this many 64 x 64 workgroups the matrix-hardware kernel uses 64 x 32 tiles (an M2 has ten cores; the 64 x 64 grid of a 7 x 7 layer at batch 1 has 8, a 14 x 14 layer's 16; wider grids measured no better with the narrow tile).
    const long NarrowBelowWorkgroups = 40;

    // A matrix-hardware convolution whose grid still has fewer workgroups than this after the narrow tile cuts k into ranges that run as workgroups of their own and
    // are added in order afterwards (the late layers of a classifier at batch 1: a few tiles, each 4608 deep). The cut depends only on the shape.
    const long SplitBelowWorkgroups = 128;
    const int MaxKSplits = 8;
    const int MinStepsPerSplit = 16;    // 16-wide k steps a range must keep to be worth its own workgroup

    readonly GpuDevice _device;
    readonly GpuKernelMode _mode;
    readonly GemmOptions? _gemmOptions;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _im2col = new KernelPipeline?[2];
    readonly KernelPipeline?[] _epilogue = new KernelPipeline?[3];   // float32 in place, half from accumulators, float32 from split planes
    readonly KernelPipeline?[] _depthwise = new KernelPipeline?[2];
    readonly KernelPipeline?[] _col2im = new KernelPipeline?[2];
    readonly ConvPipeline?[] _convGemm = new ConvPipeline?[3];
    ConvPipeline? _convCoop;
    ConvPipeline? _convCoopNarrow;
    Gemm? _gemm32;
    Gemm? _gemm16;
    Gemm? _probe;   // a fast-mode product, to ask whether the device has simdgroup matrices when the instance is deterministic
    GpuBuffer<float>? _dummy;
    GpuBuffer<float>? _ownColumns32;
    GpuBuffer<Half>? _ownColumns16;
    GpuBuffer<float>? _ownAccumulators;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="mode"><see cref="GpuKernelMode.Deterministic"/> picks the deterministic matrix product; see the remarks.</param>
    /// <param name="gemmOptions">Options of the matrix products (for example to force the portable variant); null for the defaults.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers each pass remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public ConvKernels(GpuDevice device, GpuKernelMode mode = GpuKernelMode.Fast, GemmOptions? gemmOptions = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _mode = mode;
        _gemmOptions = gemmOptions;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Lets tests turn the implicit kernel off to compare it with the column-matrix path.</summary>
    internal bool AllowImplicit { get; set; } = true;

    /// <summary>Lets tests turn the matrix-hardware implicit kernel off to compare it with the column-matrix path.</summary>
    internal bool AllowCoopImplicit { get; set; } = true;

    /// <summary>Lets tests and timing runs turn the split of k off to compare with the whole-k kernel.</summary>
    internal bool AllowKSplit { get; set; } = true;

    /// <summary>The mode the instance was created with.</summary>
    public GpuKernelMode Mode => _mode;

    /// <summary>The matrix-product variant used for float32 (<see cref="Gemm.Variant"/>).</summary>
    public string GemmVariantFloat32 => (_gemm32 ??= new Gemm(_device, GemmElementType.F32, _mode, _gemmOptions, _maxBindGroups)).Variant;

    /// <summary>The matrix-product variant used for float16 inputs (<see cref="Gemm.Variant"/>).</summary>
    public string GemmVariantFloat16 => (_gemm16 ??= new Gemm(_device, GemmElementType.F16, _mode, _gemmOptions, _maxBindGroups)).Variant;

    /// <summary>How <see cref="Encode(GpuCommandEncoder, in ConvProblem, GpuBuffer{float}, GpuBuffer{float}, GpuBuffer{float}?, GpuBuffer{float}?, GpuBuffer{float}, GpuBuffer{float}?, int)"/> computes a shape. Does not validate the shape.</summary>
    /// <param name="problem">The convolution.</param>
    public static ConvVariant ChooseVariant(in ConvProblem problem)
    {
        if (problem.Groups > 0 && problem.InChannels == problem.Groups)
        {
            return ConvVariant.Depthwise;
        }

        return IsPointwise(problem) ? ConvVariant.MatrixProduct : ConvVariant.Im2ColMatrixProduct;
    }

    /// <summary>
    /// Names how this instance runs a convolution of the given shape and element type on its device: the passes, and the matrix product behind them. For reports and for
    /// checking which variant a device took.
    /// </summary>
    /// <param name="problem">The convolution.</param>
    /// <param name="half">True for float16 operands, false for float32.</param>
    public string DescribeVariant(in ConvProblem problem, bool half)
    {
        var shape = ChooseVariant(problem);
        if (shape == ConvVariant.Depthwise)
        {
            return half ? "direct depthwise (halves, float32 sum, fused epilogue)" : "direct depthwise (float32, fused epilogue)";
        }

        var gemm = half ? GemmVariantFloat16 : GemmVariantFloat32;
        if (UsesCoopImplicit(half))
        {
            var (_, splits, _) = CoopPlan(problem);
            return "implicit fused-epilogue simdgroup-matrix product (windows gathered into the matrix tiles; pointwise tiles read the operands directly)" + (splits > 1 ? $", k cut into {splits} ranges added in order by the epilogue pass" : string.Empty);
        }

        if (UsesImplicitGemm(half))
        {
            return (half ? "implicit " : "implicit fused-epilogue ") + gemm + " + " + (half ? "epilogue pass" : "bias/residual/activation in the store");
        }

        return (shape == ConvVariant.MatrixProduct ? "matrix product " : "im2col + matrix product ") + gemm + " + epilogue pass";
    }

    static bool IsPointwise(in ConvProblem p) =>
        p.KernelHeight == 1 && p.KernelWidth == 1 && p.StrideHeight == 1 && p.StrideWidth == 1 && p.PadTop == 0 && p.PadBottom == 0 && p.PadLeft == 0 && p.PadRight == 0;

    /// <summary>
    /// The scratch this instance needs to run a convolution of the given shape and element type on its device: nothing for the variants that gather windows while they
    /// multiply (the portable and the matrix-hardware implicit kernels, depthwise, pointwise float32), the column matrix for the unfold variants, and for halves
    /// float32 accumulators. Sizes are in elements of the scratch buffers' types; a buffer of the per-image size works (the batch then runs in chunks of images,
    /// which changes no bit of the result) and a buffer of the whole-batch size runs it in one go.
    /// </summary>
    /// <param name="problem">The convolution; it must be valid.</param>
    /// <param name="half">True for float16 operands, false for float32.</param>
    public ConvScratch Scratch(in ConvProblem problem, bool half)
    {
        var variant = ChooseVariant(problem);
        long ohw = (long)problem.OutHeight * problem.OutWidth;
        long yImage = problem.OutChannels * ohw;
        long colPerImage = variant == ConvVariant.Im2ColMatrixProduct && !UsesCoopImplicit(half) && !UsesImplicitGemm(half)
            ? (long)problem.InChannels * problem.KernelHeight * problem.KernelWidth * ohw
            : 0;
        if (variant != ConvVariant.Depthwise && UsesCoopImplicit(half))
        {
            var (_, splits, _) = CoopPlan(problem);
            colPerImage = splits > 1 ? splits * yImage : 0;
        }

        var accPerImage = half && variant != ConvVariant.Depthwise ? yImage : 0;
        return new ConvScratch(colPerImage * problem.Batch, accPerImage * problem.Batch, colPerImage, accPerImage);
    }

    /// <summary>The scratch a transposed convolution needs: float32 columns (the matrix product's result), for both element types. See <see cref="Scratch(in ConvProblem, bool)"/>.</summary>
    /// <param name="problem">The transposed convolution; it must be valid.</param>
    public static ConvScratch Scratch(in ConvTransposeProblem problem)
    {
        long perImage = (long)problem.OutChannels * problem.KernelHeight * problem.KernelWidth * problem.Height * problem.Width;
        return new ConvScratch(perImage * problem.Batch, 0, perImage, 0);
    }

    /// <summary>
    /// Elements of the column matrix the whole batch of <paramref name="problem"/> needs (float32 or float16 elements, matching the data type): 0 for variants
    /// that need none. A buffer of at least one image's worth (<c>ColumnElements / Batch</c>) works; the batch is then chunked.
    /// </summary>
    /// <param name="problem">The convolution; it must be valid.</param>
    public static long ColumnElements(in ConvProblem problem) =>
        ChooseVariant(problem) == ConvVariant.Im2ColMatrixProduct
            ? (long)problem.Batch * problem.InChannels * problem.KernelHeight * problem.KernelWidth * problem.OutHeight * problem.OutWidth
            : 0;

    /// <summary>Elements of float32 accumulators the whole batch needs for float16 (the output's element count; float32 convolutions accumulate in the output itself and need none).</summary>
    /// <param name="problem">The convolution; it must be valid.</param>
    public static long AccumulatorElements(in ConvProblem problem) =>
        ChooseVariant(problem) == ConvVariant.Depthwise ? 0 : (long)problem.Batch * problem.OutChannels * problem.OutHeight * problem.OutWidth;

    /// <summary>Elements of float32 scratch the whole batch of a transposed convolution needs (the matrix product's result, one value per output channel, kernel tap and input position).</summary>
    /// <param name="problem">The transposed convolution; it must be valid.</param>
    public static long ColumnElements(in ConvTransposeProblem problem) =>
        (long)problem.Batch * problem.OutChannels * problem.KernelHeight * problem.KernelWidth * problem.Height * problem.Width;

    /// <summary>
    /// Records a float32 convolution <c>y = activation(conv(x, w) + bias + residual)</c>.
    /// </summary>
    /// <param name="encoder">Receives the passes. No other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of the operands.</param>
    /// <param name="x">Input, <c>[Batch, InChannels, Height, Width]</c>.</param>
    /// <param name="w">Weights, <c>[OutChannels, InChannels / Groups, KernelHeight, KernelWidth]</c>.</param>
    /// <param name="bias">Bias, <c>OutChannels</c> elements, or null for none.</param>
    /// <param name="residual">A tensor of the output's shape added after the bias, or null.</param>
    /// <param name="y">Receives the output, <c>[Batch, OutChannels, OutHeight, OutWidth]</c>. Also the accumulators of the matrix product.</param>
    /// <param name="columns">Scratch for the column matrix (see <see cref="Scratch(in ConvProblem, bool)"/> for its size), or null to use the instance's own.</param>
    /// <param name="columnsOffset">Element offset of the scratch in <paramref name="columns"/>; ignored when that is null. May share a buffer with the other operands as long as the ranges do not overlap.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, group count, padding or offset is invalid, a range reaches past its buffer, the scratch is smaller than one image needs, or the scratch overlaps another operand in the same buffer. Names the argument.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void Encode(GpuCommandEncoder encoder, in ConvProblem problem, GpuBuffer<float> x, GpuBuffer<float> w, GpuBuffer<float>? bias, GpuBuffer<float>? residual, GpuBuffer<float> y, GpuBuffer<float>? columns = null, int columnsOffset = 0) =>
        EncodeConv(encoder, problem, x, w, bias, residual, y, columns, columnsOffset, null, 0);

    /// <summary>
    /// Records a float16 convolution with float32 accumulation <c>y = activation(conv(x, w) + bias + residual)</c>; every operand and the output are halves.
    /// </summary>
    /// <param name="encoder">Receives the passes. No other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of the operands.</param>
    /// <param name="x">Input, <c>[Batch, InChannels, Height, Width]</c>.</param>
    /// <param name="w">Weights, <c>[OutChannels, InChannels / Groups, KernelHeight, KernelWidth]</c>.</param>
    /// <param name="bias">Bias, <c>OutChannels</c> elements, or null for none.</param>
    /// <param name="residual">A tensor of the output's shape added after the bias, or null.</param>
    /// <param name="y">Receives the output, <c>[Batch, OutChannels, OutHeight, OutWidth]</c>.</param>
    /// <param name="columns">Scratch of halves for the column matrix (see <see cref="Scratch(in ConvProblem, bool)"/> for its size), or null to use the instance's own.</param>
    /// <param name="accumulators">Float32 scratch for the matrix product's result, or null to use the instance's own.</param>
    /// <param name="columnsOffset">Element offset of the scratch in <paramref name="columns"/>, in halves; ignored when that is null.</param>
    /// <param name="accumulatorsOffset">Element offset of the scratch in <paramref name="accumulators"/>, in floats; ignored when that is null.</param>
    /// <exception cref="GpuKernelShapeException">As for the float32 overload.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void Encode(GpuCommandEncoder encoder, in ConvProblem problem, GpuBuffer<Half> x, GpuBuffer<Half> w, GpuBuffer<Half>? bias, GpuBuffer<Half>? residual, GpuBuffer<Half> y, GpuBuffer<Half>? columns = null, GpuBuffer<float>? accumulators = null, int columnsOffset = 0, int accumulatorsOffset = 0) =>
        EncodeConv(encoder, problem, x, w, bias, residual, y, columns, columnsOffset, accumulators, accumulatorsOffset);

    /// <summary>Records a float32 transposed convolution <c>y = activation(convTranspose(x, w) + bias)</c>.</summary>
    /// <param name="encoder">Receives the passes. No other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of the operands.</param>
    /// <param name="x">Input, <c>[Batch, InChannels, Height, Width]</c>.</param>
    /// <param name="w">Weights, <c>[InChannels, OutChannels / Groups, KernelHeight, KernelWidth]</c>.</param>
    /// <param name="bias">Bias, <c>OutChannels</c> elements, or null for none.</param>
    /// <param name="y">Receives the output, <c>[Batch, OutChannels, OutHeight, OutWidth]</c>.</param>
    /// <param name="columns">Float32 scratch (see <see cref="Scratch(in ConvTransposeProblem)"/> for its size), or null to use the instance's own.</param>
    /// <param name="columnsOffset">Element offset of the scratch in <paramref name="columns"/>; ignored when that is null.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, group count, padding or offset is invalid, a range reaches past its buffer, the scratch is smaller than one image needs, or it overlaps another operand in the same buffer.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void EncodeTranspose(GpuCommandEncoder encoder, in ConvTransposeProblem problem, GpuBuffer<float> x, GpuBuffer<float> w, GpuBuffer<float>? bias, GpuBuffer<float> y, GpuBuffer<float>? columns = null, int columnsOffset = 0) =>
        EncodeTransposeCore(encoder, problem, x, w, bias, y, columns, columnsOffset);

    /// <summary>Records a float16 transposed convolution with float32 accumulation; operands and output are halves.</summary>
    /// <param name="encoder">Receives the passes. No other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of the operands.</param>
    /// <param name="x">Input, <c>[Batch, InChannels, Height, Width]</c>.</param>
    /// <param name="w">Weights, <c>[InChannels, OutChannels / Groups, KernelHeight, KernelWidth]</c>.</param>
    /// <param name="bias">Bias, <c>OutChannels</c> elements, or null for none.</param>
    /// <param name="y">Receives the output, <c>[Batch, OutChannels, OutHeight, OutWidth]</c>.</param>
    /// <param name="columns">Float32 scratch (see <see cref="Scratch(in ConvTransposeProblem)"/> for its size), or null to use the instance's own.</param>
    /// <param name="columnsOffset">Element offset of the scratch in <paramref name="columns"/>, in floats; ignored when that is null.</param>
    /// <exception cref="GpuKernelShapeException">As for the float32 overload.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void EncodeTranspose(GpuCommandEncoder encoder, in ConvTransposeProblem problem, GpuBuffer<Half> x, GpuBuffer<Half> w, GpuBuffer<Half>? bias, GpuBuffer<Half> y, GpuBuffer<float>? columns = null, int columnsOffset = 0) =>
        EncodeTransposeCore(encoder, problem, x, w, bias, y, columns, columnsOffset);

    void EncodeConv<T>(GpuCommandEncoder encoder, in ConvProblem p, GpuBuffer<T> x, GpuBuffer<T> w, GpuBuffer<T>? bias, GpuBuffer<T>? residual, GpuBuffer<T> y, GpuBuffer<T>? columns, int columnsOffset, GpuBuffer<float>? accumulators, int accumulatorsOffset)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(y);
        var half = typeof(T) == typeof(Half);
        long oh = p.OutHeight, ow = p.OutWidth;
        CheckConv(p, oh, ow);
        long ohw = oh * ow;
        var cinG = p.InChannels / p.Groups;
        var coutG = p.OutChannels / p.Groups;
        long kElems = (long)cinG * p.KernelHeight * p.KernelWidth;
        long xImage = (long)p.InChannels * p.Height * p.Width;
        long yImage = p.OutChannels * ohw;
        CheckSize(ConvName, nameof(p.Batch), xImage * p.Batch);
        CheckSize(ConvName, "output", yImage * p.Batch);
        CheckSize(ConvName, "weights", (long)p.OutChannels * kElems);
        Check.Range(ConvName, nameof(x), x.Length, p.OffsetX, xImage * p.Batch, half);
        Check.Range(ConvName, nameof(w), w.Length, p.OffsetW, (long)p.OutChannels * kElems, half);
        Check.Range(ConvName, nameof(y), y.Length, p.OffsetY, yImage * p.Batch, half);
        if (bias is not null)
        {
            Check.Range(ConvName, nameof(bias), bias.Length, p.OffsetBias, p.OutChannels, half);
        }

        if (residual is not null)
        {
            Check.Range(ConvName, nameof(residual), residual.Length, p.OffsetResidual, yImage * p.Batch, half);
        }

        var variant = ChooseVariant(p);
        var (act, p0, p1) = ActivationWords(p.Activation);
        var dummy = Dummy();
        var kind = half ? 1 : 0;
        if (columns is null)
        {
            columnsOffset = 0;
        }

        if (accumulators is null)
        {
            accumulatorsOffset = 0;
        }
        if (variant == ConvVariant.Depthwise)
        {
            var pipe = _depthwise[kind] ??= CreateDepthwise(half);
            Span<uint> dw = stackalloc uint[26];
            dw[0] = (uint)(yImage * p.Batch);
            dw[1] = (uint)p.OffsetX;
            dw[2] = (uint)p.OffsetW;
            dw[3] = (uint)p.OffsetBias;
            dw[4] = (uint)p.OffsetResidual;
            dw[5] = (uint)p.OffsetY;
            dw[7] = (uint)p.InChannels;
            dw[8] = (uint)coutG;
            dw[9] = (uint)p.Height;
            dw[10] = (uint)p.Width;
            dw[11] = (uint)oh;
            dw[12] = (uint)ow;
            dw[13] = (uint)p.KernelHeight;
            dw[14] = (uint)p.KernelWidth;
            dw[15] = (uint)p.StrideHeight;
            dw[16] = (uint)p.StrideWidth;
            dw[17] = (uint)p.DilationHeight;
            dw[18] = (uint)p.DilationWidth;
            dw[19] = (uint)p.PadTop;
            dw[20] = (uint)p.PadLeft;
            dw[21] = bias is null ? 0u : 1u;
            dw[22] = residual is null ? 0u : 1u;
            dw[23] = act;
            dw[24] = p0;
            dw[25] = p1;
            var (gx, gy) = Grid1D(yImage * p.Batch, p.OffsetY, half);
            dw[6] = gx;
            pipe.Record(encoder, dw, gx, gy, [x.Buffer, w.Buffer, (bias?.Buffer ?? dummy.Buffer), (residual?.Buffer ?? dummy.Buffer), y.Buffer]);
            return;
        }

        var pointwise = variant == ConvVariant.MatrixProduct;
        var coop = UsesCoopImplicit(half);
        var implicitGemm = coop || UsesImplicitGemm(half);
        var (narrow, kSplits, kPerSplit) = coop ? CoopPlan(p) : (false, 1, (int)kElems);
        // With float32 operands the implicit kernel applies the bias, residual and the activations it knows as it stores; anything else is left to the epilogue pass.
        // A split convolution leaves all of that to the pass that adds the ranges.
        var fused = implicitGemm && !half && kSplits == 1 && FusableActivation(p.Activation.Op);
        long colPerImage = pointwise || implicitGemm ? 0 : (long)p.InChannels * p.KernelHeight * p.KernelWidth * ohw;
        if (kSplits > 1)
        {
            colPerImage = kSplits * yImage;
        }

        if (colPerImage > 0)
        {
            CheckSize(ConvName, "columns", colPerImage);
        }

        // How many images one round of passes handles: as many as the scratch holds (the caller's buffers, or ours up to a budget).
        long chunk = p.Batch;
        GpuBuffer<T>? cols = null;
        GpuBuffer<float>? acc = null;
        if (colPerImage > 0)
        {
            cols = columns ?? OwnColumns<T>(colPerImage, chunk);
            CheckScratchOffset(ConvName, nameof(columns), columnsOffset, cols.Length);
            var colsAvailable = cols.Length - columnsOffset;
            chunk = Math.Min(chunk, (colsAvailable - (half ? 1 : 0)) / colPerImage);
            if (chunk < 1)
            {
                throw new GpuKernelShapeException(ConvName, nameof(columns), $"the scratch holds {colsAvailable} elements from its offset; one image needs {colPerImage}");
            }

            CheckScratchRange(ConvName, nameof(columns), cols.Buffer, columnsOffset, chunk * colPerImage, [(nameof(x), x.Buffer, p.OffsetX, xImage * p.Batch), (nameof(w), w.Buffer, p.OffsetW, (long)p.OutChannels * kElems), (nameof(y), y.Buffer, p.OffsetY, yImage * p.Batch), (nameof(residual), residual?.Buffer, p.OffsetResidual, yImage * p.Batch), (nameof(bias), bias?.Buffer, p.OffsetBias, p.OutChannels)]);
        }

        if (half)
        {
            acc = accumulators ?? OwnAccumulators(yImage, chunk);
            CheckScratchOffset(ConvName, nameof(accumulators), accumulatorsOffset, acc.Length);
            var fit = (acc.Length - accumulatorsOffset) / yImage;
            if (fit < 1)
            {
                throw new GpuKernelShapeException(ConvName, nameof(accumulators), $"the scratch holds {acc.Length - accumulatorsOffset} elements from its offset; one image needs {yImage}");
            }

            chunk = Math.Min(chunk, fit);
            CheckScratchRange(ConvName, nameof(accumulators), acc.Buffer, accumulatorsOffset, chunk * yImage, [(nameof(columns), cols?.Buffer, columnsOffset, chunk * colPerImage)]);
        }

        var tilesX = ((ohw + (narrow ? 31 : 63)) / (narrow ? 32 : 64));
        var tilesY = (coutG + 63L) / 64;
        if (implicitGemm)
        {
            var limit = _device.Capabilities.Compute.MaxWorkgroupCount;
            limit = limit == 0 ? 65535u : Math.Min(limit, 65535u);
            if (tilesX > limit || tilesY > limit || p.Groups > limit)
            {
                throw new GpuKernelShapeException(ConvName, "grid", $"the grid of {tilesX} x {tilesY} x {p.Groups} workgroups exceeds the device limit of {limit} per axis");
            }

            chunk = Math.Min(chunk, limit / ((long)p.Groups * kSplits));
        }

        var needEpilogue = (half || bias is not null || residual is not null || act != 0 || kSplits > 1) && !fused;
        Span<uint> w17 = stackalloc uint[17];
        Span<uint> e = stackalloc uint[15];
        Span<uint> cg = stackalloc uint[29];
        for (long n0 = 0; n0 < p.Batch; n0 += chunk)
        {
            var cn = (int)Math.Min(chunk, p.Batch - n0);
            if (implicitGemm)
            {
                var pipe = coop ? (narrow ? (_convCoopNarrow ??= CreateConvCoop(true)) : (_convCoop ??= CreateConvCoop(false))) : (_convGemm[kind] ??= CreateConvGemm(half));
                cg[0] = (uint)coutG;
                cg[1] = (uint)ohw;
                cg[2] = (uint)kElems;
                cg[3] = (uint)p.Groups;
                cg[4] = (uint)p.InChannels;
                cg[5] = (uint)p.Height;
                cg[6] = (uint)p.Width;
                cg[7] = (uint)ow;
                cg[8] = (uint)p.KernelHeight;
                cg[9] = (uint)p.KernelWidth;
                cg[10] = (uint)p.StrideHeight;
                cg[11] = (uint)p.StrideWidth;
                cg[12] = (uint)p.DilationHeight;
                cg[13] = (uint)p.DilationWidth;
                cg[14] = (uint)p.PadTop;
                cg[15] = (uint)p.PadLeft;
                cg[16] = (uint)(p.OffsetX + (n0 * xImage));
                cg[17] = (uint)p.OffsetW;
                cg[18] = half ? (uint)accumulatorsOffset : (uint)(p.OffsetY + (n0 * yImage));
                cg[19] = (uint)p.OffsetBias;
                cg[20] = (uint)(p.OffsetResidual + (n0 * yImage));
                cg[21] = (uint)p.OutChannels;
                cg[22] = (fused && bias is not null ? 1u : 0u) | (fused && residual is not null ? 2u : 0u) | (pointwise ? 4u : 0u);
                cg[23] = fused ? act : 0u;
                cg[24] = fused ? p0 : 0u;
                cg[25] = fused ? p1 : 0u;
                var target = half ? acc!.Buffer : y.Buffer;
                var useBias = fused && bias is not null ? bias.Buffer : DummyFloat().Buffer;
                var useRes = fused && residual is not null ? residual.Buffer : DummyFloat().Buffer;
                var groupsZ = (uint)(cn * p.Groups);
                if (coop)
                {
                    cg[26] = 0;
                    cg[27] = 0;
                    cg[28] = 0;
                    if (kSplits > 1)
                    {
                        // The planes of partial sums go to the column scratch; z runs split-major.
                        target = cols!.Buffer;
                        cg[18] = (uint)columnsOffset;
                        cg[26] = (uint)kPerSplit;
                        cg[27] = (uint)(cn * yImage);
                        cg[28] = groupsZ;
                        groupsZ *= (uint)kSplits;
                    }
                }

                pipe.Record(encoder, coop ? cg : cg[..26], (uint)tilesX, (uint)tilesY, groupsZ, x.Buffer, w.Buffer, target, useBias, useRes);
            }
            else
            {
                if (!pointwise)
                {
                    var pipe = _im2col[kind] ??= CreateIm2Col(half);
                    var count = (long)cn * colPerImage;
                    w17[0] = (uint)count;
                    w17[1] = (uint)(p.OffsetX + (n0 * xImage));
                    w17[2] = (uint)columnsOffset;
                    w17[4] = (uint)p.InChannels;
                    w17[5] = (uint)p.Height;
                    w17[6] = (uint)p.Width;
                    w17[7] = (uint)oh;
                    w17[8] = (uint)ow;
                    w17[9] = (uint)p.KernelHeight;
                    w17[10] = (uint)p.KernelWidth;
                    w17[11] = (uint)p.StrideHeight;
                    w17[12] = (uint)p.StrideWidth;
                    w17[13] = (uint)p.DilationHeight;
                    w17[14] = (uint)p.DilationWidth;
                    w17[15] = (uint)p.PadTop;
                    w17[16] = (uint)p.PadLeft;
                    var (gx, gy) = Grid1D(count, 0, half);
                    w17[3] = gx;
                    pipe.Record(encoder, w17, gx, gy, x.Buffer, cols!.Buffer);
                }

                for (var g = 0; g < p.Groups; g++)
                {
                    var gp = new GemmProblem(coutG, (int)ohw, (int)kElems)
                    {
                        Batch = cn,
                        OffsetA = checked(p.OffsetW + (int)(g * coutG * kElems)),
                        StrideA = 0,
                    };
                    if (pointwise)
                    {
                        gp.OffsetB = checked((int)(p.OffsetX + (n0 * xImage) + ((long)g * cinG * ohw)));
                        gp.StrideB = (int)xImage;
                    }
                    else
                    {
                        gp.OffsetB = checked((int)(columnsOffset + ((long)g * kElems * ohw)));
                        gp.StrideB = (int)colPerImage;
                    }

                    gp.StrideC = (int)yImage;
                    gp.OffsetC = half ? checked((int)(accumulatorsOffset + ((long)g * coutG * ohw))) : checked((int)(p.OffsetY + (n0 * yImage) + ((long)g * coutG * ohw)));
                    RunGemm(encoder, gp, w, pointwise ? x : cols!, half ? acc! : (GpuBuffer<float>)(object)y);
                }
            }

            if (needEpilogue)
            {
                var pipe = _epilogue[kSplits > 1 ? 2 : kind] ??= CreateEpilogue(half, kSplits > 1);
                var count = (long)cn * yImage;
                e[0] = (uint)count;
                e[1] = (uint)(p.OffsetY + (n0 * yImage));
                e[2] = (uint)(kSplits > 1 ? columnsOffset : accumulatorsOffset);
                e[3] = (uint)p.OffsetBias;
                e[4] = (uint)(p.OffsetResidual + (n0 * yImage));
                e[5] = (uint)ohw;
                e[6] = (uint)p.OutChannels;
                e[8] = bias is null ? 0u : 1u;
                e[9] = residual is null ? 0u : 1u;
                e[10] = act;
                e[11] = p0;
                e[12] = p1;
                e[13] = (uint)kSplits;
                e[14] = (uint)(cn * yImage);
                var (gx, gy) = Grid1D(count, p.OffsetY + (n0 * yImage), half);
                e[7] = gx;
                if (half)
                {
                    pipe.Record(encoder, e, gx, gy, [y.Buffer, (bias?.Buffer ?? dummy.Buffer), (residual?.Buffer ?? dummy.Buffer), acc!.Buffer]);
                }
                else if (kSplits > 1)
                {
                    pipe.Record(encoder, e, gx, gy, [y.Buffer, (bias?.Buffer ?? dummy.Buffer), (residual?.Buffer ?? dummy.Buffer), cols!.Buffer]);
                }
                else
                {
                    pipe.Record(encoder, e, gx, gy, [y.Buffer, (bias?.Buffer ?? dummy.Buffer), (residual?.Buffer ?? dummy.Buffer)]);
                }
            }
        }
    }

    // The matrix-hardware kernel's tile (narrow below a grid of NarrowBelowWorkgroups, else 64 wide) and how many ranges of k a grid that is still small is cut into, from
    // the shape alone (the whole batch, not the chunk a small scratch buffer allows), so that every device and every chunking adds the same partial sums in the same order.
    (bool Narrow, int Splits, int KPerSplit) CoopPlan(in ConvProblem p)
    {
        long ohw = (long)p.OutHeight * p.OutWidth;
        long coutG = p.OutChannels / p.Groups;
        long kElems = (long)(p.InChannels / p.Groups) * p.KernelHeight * p.KernelWidth;
        long wide = ((ohw + 63) / 64) * ((coutG + 63) / 64) * p.Groups * p.Batch;
        var narrow = wide < NarrowBelowWorkgroups;
        var groups = narrow ? ((ohw + 31) / 32) * ((coutG + 63) / 64) * p.Groups * p.Batch : wide;
        var steps = (kElems + 15) / 16;
        if (!AllowKSplit || groups >= SplitBelowWorkgroups || steps < 2 * MinStepsPerSplit)
        {
            return (narrow, 1, (int)kElems);
        }

        var splits = (int)Math.Min(Math.Min(MaxKSplits, (SplitBelowWorkgroups + groups - 1) / groups), steps / MinStepsPerSplit);
        if (splits < 2)
        {
            return (narrow, 1, (int)kElems);
        }

        var perSplit = (int)((steps + splits - 1) / splits * 16);
        return (narrow, (int)((kElems + perSplit - 1) / perSplit), perSplit);
    }

    // The implicit kernel is the portable tiled product with the window gathering built in. Where the matrix product of the element type runs on matrix
    // hardware the column-matrix path keeps that speed, so the implicit kernel is chosen only next to the portable product.
    bool UsesImplicitGemm(bool half)
    {
        var variant = half ? GemmVariantFloat16 : GemmVariantFloat32;
        return AllowImplicit && variant.StartsWith("tiled", StringComparison.Ordinal);
    }

    // On a device whose float32 matrix product runs on matrix hardware (Metal) the implicit kernel uses it too: the windows are gathered while the tiles are staged,
    // so no column matrix is written and read, and the bias, residual and activation are applied as the result is stored.
    // Deterministic mode takes it too: a simdgroup matrix product is a fixed function of its operands, every output tile is summed by one simdgroup in the order of k
    // (or of the shape-determined ranges of k, added in range order), and nothing here depends on scheduling. The tests compare two runs bit for bit.
    bool UsesCoopImplicit(bool half) =>
        AllowImplicit && AllowCoopImplicit && !half && _device.Backend == GpuBackend.Metal
        && (_mode == GpuKernelMode.Fast ? GemmVariantFloat32 : (_probe ??= new Gemm(_device, GemmElementType.F32, GpuKernelMode.Fast, _gemmOptions, _maxBindGroups)).Variant).StartsWith("simdgroup", StringComparison.Ordinal);

    // Activations the implicit kernel applies itself (see ConvAct in ConvGemm.hlsl).
    static bool FusableActivation(GpuUnaryOp op) =>
        op is GpuUnaryOp.Identity or GpuUnaryOp.Tanh or GpuUnaryOp.Sigmoid or GpuUnaryOp.Relu or GpuUnaryOp.Silu or GpuUnaryOp.HardSigmoid or GpuUnaryOp.HardSwish or GpuUnaryOp.Clip or GpuUnaryOp.LeakyRelu;

    void EncodeTransposeCore<T>(GpuCommandEncoder encoder, in ConvTransposeProblem p, GpuBuffer<T> x, GpuBuffer<T> w, GpuBuffer<T>? bias, GpuBuffer<T> y, GpuBuffer<float>? columns, int columnsOffset)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(y);
        var half = typeof(T) == typeof(Half);
        long oh = p.OutHeight, ow = p.OutWidth;
        CheckTranspose(p, oh, ow);
        var cinG = p.InChannels / p.Groups;
        var coutG = p.OutChannels / p.Groups;
        long hw = (long)p.Height * p.Width;
        long rows = (long)coutG * p.KernelHeight * p.KernelWidth;
        long xImage = p.InChannels * hw;
        long yImage = p.OutChannels * oh * ow;
        long colPerImage = p.Groups * rows * hw;
        CheckSize(TransposeName, nameof(p.Batch), xImage * p.Batch);
        CheckSize(TransposeName, "output", yImage * p.Batch);
        CheckSize(TransposeName, "columns", colPerImage);
        CheckSize(TransposeName, "weights", (long)p.InChannels * rows);
        Check.Range(TransposeName, nameof(x), x.Length, p.OffsetX, xImage * p.Batch, half);
        Check.Range(TransposeName, nameof(w), w.Length, p.OffsetW, (long)p.InChannels * rows, half);
        Check.Range(TransposeName, nameof(y), y.Length, p.OffsetY, yImage * p.Batch, half);
        if (bias is not null)
        {
            Check.Range(TransposeName, nameof(bias), bias.Length, p.OffsetBias, p.OutChannels, half);
        }

        if (columns is null)
        {
            columnsOffset = 0;
        }

        var cols = columns ?? OwnAccumulators(colPerImage, p.Batch);
        CheckScratchOffset(TransposeName, nameof(columns), columnsOffset, cols.Length);
        var chunk = Math.Min(p.Batch, (cols.Length - columnsOffset) / colPerImage);
        if (chunk < 1)
        {
            throw new GpuKernelShapeException(TransposeName, nameof(columns), $"the scratch holds {cols.Length - columnsOffset} elements from its offset; one image needs {colPerImage}");
        }

        CheckScratchRange(TransposeName, nameof(columns), cols.Buffer, columnsOffset, chunk * colPerImage, [(nameof(x), x.Buffer, p.OffsetX, xImage * p.Batch), (nameof(w), w.Buffer, p.OffsetW, (long)p.InChannels * rows), (nameof(y), y.Buffer, p.OffsetY, yImage * p.Batch), (nameof(bias), bias?.Buffer, p.OffsetBias, p.OutChannels)]);

        var (act, p0, p1) = ActivationWords(p.Activation);
        var dummy = Dummy();
        var pipe = _col2im[half ? 1 : 0] ??= CreateCol2Im(half);
        Span<uint> c = stackalloc uint[23];
        for (long n0 = 0; n0 < p.Batch; n0 += chunk)
        {
            var cn = (int)Math.Min(chunk, p.Batch - n0);
            for (var g = 0; g < p.Groups; g++)
            {
                // W is stored [Cin, CoutG * KH * KW]: group g's rows are K x M with M = rows, so the product is W^T * x.
                var gp = new GemmProblem((int)rows, (int)hw, cinG)
                {
                    Batch = cn,
                    TransposeA = true,
                    Lda = (int)rows,
                    OffsetA = checked(p.OffsetW + (int)((long)g * cinG * rows)),
                    StrideA = 0,
                    OffsetB = checked((int)(p.OffsetX + (n0 * xImage) + ((long)g * cinG * hw))),
                    StrideB = (int)xImage,
                    OffsetC = checked((int)(columnsOffset + (g * rows * hw))),
                    StrideC = (int)colPerImage,
                };
                RunGemm(encoder, gp, w, x, cols);
            }

            var count = (long)cn * yImage;
            c[0] = (uint)count;
            c[1] = (uint)columnsOffset;
            c[2] = (uint)p.OffsetBias;
            c[3] = (uint)(p.OffsetY + (n0 * yImage));
            c[5] = (uint)p.OutChannels;
            c[6] = (uint)p.Groups;
            c[7] = (uint)p.Height;
            c[8] = (uint)p.Width;
            c[9] = (uint)oh;
            c[10] = (uint)ow;
            c[11] = (uint)p.KernelHeight;
            c[12] = (uint)p.KernelWidth;
            c[13] = (uint)p.StrideHeight;
            c[14] = (uint)p.StrideWidth;
            c[15] = (uint)p.DilationHeight;
            c[16] = (uint)p.DilationWidth;
            c[17] = (uint)p.PadTop;
            c[18] = (uint)p.PadLeft;
            c[19] = bias is null ? 0u : 1u;
            c[20] = act;
            c[21] = p0;
            c[22] = p1;
            var (gx, gy) = Grid1D(count, p.OffsetY + (n0 * yImage), half);
            c[4] = gx;
            pipe.Record(encoder, c, gx, gy, [cols.Buffer, (bias?.Buffer ?? dummy.Buffer), y.Buffer]);
        }
    }

    // Records one matrix product of A and B into float32 C. T is float or Half.
    void RunGemm<T>(GpuCommandEncoder encoder, in GemmProblem gp, GpuBuffer<T> a, GpuBuffer<T> b, GpuBuffer<float> c)
        where T : unmanaged
    {
        if (typeof(T) == typeof(Half))
        {
            var g = _gemm16 ??= new Gemm(_device, GemmElementType.F16, _mode, _gemmOptions, _maxBindGroups);
            g.Encode(encoder, gp, (GpuBuffer<Half>)(object)a, (GpuBuffer<Half>)(object)b, c);
        }
        else
        {
            var g = _gemm32 ??= new Gemm(_device, GemmElementType.F32, _mode, _gemmOptions, _maxBindGroups);
            g.Encode(encoder, gp, (GpuBuffer<float>)(object)a, (GpuBuffer<float>)(object)b, c);
        }
    }

    static (uint Act, uint P0, uint P1) ActivationWords(in GpuActivation a) =>
        a.Op == GpuUnaryOp.Identity ? (0u, 0u, 0u) : ((uint)a.Op + 1u, BitConverter.SingleToUInt32Bits(a.P0), BitConverter.SingleToUInt32Bits(a.P1));

    (uint X, uint Y) Grid1D(long count, long offsetOutput, bool half)
    {
        var threads = half ? (((offsetOutput + count - 1) >> 1) - (offsetOutput >> 1) + 1) : count;
        return count == 0 ? (0, 0) : Grid.For(_device, (threads + 255) / 256);
    }

    GpuBuffer<float> Dummy() => _dummy ??= _device.CreateBuffer<float>(4);

    GpuBuffer<float> DummyFloat() => Dummy();

    GpuBuffer<T> OwnColumns<T>(long perImage, long batch)
        where T : unmanaged
    {
        var budget = (long)Math.Min((ulong)OwnScratchBytes, Math.Max(1UL, _device.Capabilities.Compute.MaxStorageBufferBindingBytes));
        var elemBytes = typeof(T) == typeof(Half) ? 2 : 4;
        var images = Math.Clamp(budget / (perImage * elemBytes), 1, batch);
        var need = checked((int)(images * perImage));
        if (typeof(T) == typeof(Half))
        {
            need = GpuKernelBuffers.PaddedHalfLength(need + 1);
            if (_ownColumns16 is null || _ownColumns16.Length < need)
            {
                _ownColumns16?.Dispose();
                _ownColumns16 = _device.CreateBuffer<Half>(need);
            }

            return (GpuBuffer<T>)(object)_ownColumns16;
        }

        if (_ownColumns32 is null || _ownColumns32.Length < need)
        {
            _ownColumns32?.Dispose();
            _ownColumns32 = _device.CreateBuffer<float>(need);
        }

        return (GpuBuffer<T>)(object)_ownColumns32;
    }

    GpuBuffer<float> OwnAccumulators(long perImage, long batch)
    {
        var budget = (long)Math.Min((ulong)OwnScratchBytes, Math.Max(1UL, _device.Capabilities.Compute.MaxStorageBufferBindingBytes));
        var images = Math.Clamp(budget / (perImage * 4), 1, batch);
        var need = checked((int)(images * perImage));
        if (_ownAccumulators is null || _ownAccumulators.Length < need)
        {
            _ownAccumulators?.Dispose();
            _ownAccumulators = _device.CreateBuffer<float>(need);
        }

        return _ownAccumulators;
    }

    static void CheckScratchOffset(string name, string argument, int offset, int length)
    {
        if (offset < 0 || offset > length)
        {
            throw new GpuKernelShapeException(name, argument + " offset", $"offset {offset} is outside the scratch buffer of {length} elements");
        }
    }

    // The scratch must not share elements with an operand that lives in the same buffer (the operands are a caller's arena slices); distinct buffers never overlap.
    static void CheckScratchRange(string name, string argument, GpuBuffer scratch, long offset, long length, ReadOnlySpan<(string Name, GpuBuffer? Buffer, long Offset, long Length)> others)
    {
        foreach (var (otherName, buffer, otherOffset, otherLength) in others)
        {
            if (buffer is not null && ReferenceEquals(buffer, scratch) && offset < otherOffset + otherLength && otherOffset < offset + length)
            {
                throw new GpuKernelShapeException(name, argument, $"the scratch range [{offset}, {offset + length}) overlaps {otherName} [{otherOffset}, {otherOffset + otherLength}) in the same buffer");
            }
        }
    }

    static void CheckSize(string name, string argument, long elements)
    {
        if (elements > Check.MaxElements)
        {
            throw new GpuKernelShapeException(name, argument, $"{elements} elements is above the supported {Check.MaxElements}");
        }
    }

    static void CheckWindow(string name, string axis, int input, int kernel, int stride, int dilation, int padBegin, int padEnd)
    {
        if (input < 1)
        {
            throw new GpuKernelShapeException(name, axis + " input", $"extent {input} must be at least 1");
        }

        if (kernel < 1)
        {
            throw new GpuKernelShapeException(name, axis + " kernel", $"extent {kernel} must be at least 1");
        }

        if (stride < 1)
        {
            throw new GpuKernelShapeException(name, axis + " stride", $"stride {stride} must be at least 1");
        }

        if (dilation < 1)
        {
            throw new GpuKernelShapeException(name, axis + " dilation", $"dilation {dilation} must be at least 1");
        }

        if (padBegin < 0 || padEnd < 0)
        {
            throw new GpuKernelShapeException(name, axis + " padding", $"padding {padBegin}, {padEnd} must not be negative");
        }

        if ((long)kernel * dilation > Check.MaxElements || (long)padBegin + padEnd + input > Check.MaxElements)
        {
            throw new GpuKernelShapeException(name, axis, "the window or padding is larger than the supported sizes");
        }
    }

    static void CheckConv(in ConvProblem p, long oh, long ow)
    {
        if (p.Batch < 1)
        {
            throw new GpuKernelShapeException(ConvName, nameof(p.Batch), $"batch {p.Batch} must be at least 1");
        }

        if (p.InChannels < 1 || p.OutChannels < 1)
        {
            throw new GpuKernelShapeException(ConvName, p.InChannels < 1 ? nameof(p.InChannels) : nameof(p.OutChannels), $"channel counts {p.InChannels}, {p.OutChannels} must be at least 1");
        }

        if (p.Groups < 1 || p.InChannels % p.Groups != 0 || p.OutChannels % p.Groups != 0)
        {
            throw new GpuKernelShapeException(ConvName, nameof(p.Groups), $"{p.Groups} groups do not divide {p.InChannels} input and {p.OutChannels} output channels");
        }

        CheckWindow(ConvName, "height", p.Height, p.KernelHeight, p.StrideHeight, p.DilationHeight, p.PadTop, p.PadBottom);
        CheckWindow(ConvName, "width", p.Width, p.KernelWidth, p.StrideWidth, p.DilationWidth, p.PadLeft, p.PadRight);
        if (oh < 1)
        {
            throw new GpuKernelShapeException(ConvName, "height", $"the kernel ({p.KernelHeight} with dilation {p.DilationHeight}) does not fit the padded input ({p.Height} + {p.PadTop} + {p.PadBottom})");
        }

        if (ow < 1)
        {
            throw new GpuKernelShapeException(ConvName, "width", $"the kernel ({p.KernelWidth} with dilation {p.DilationWidth}) does not fit the padded input ({p.Width} + {p.PadLeft} + {p.PadRight})");
        }
    }

    static void CheckTranspose(in ConvTransposeProblem p, long oh, long ow)
    {
        if (p.Batch < 1)
        {
            throw new GpuKernelShapeException(TransposeName, nameof(p.Batch), $"batch {p.Batch} must be at least 1");
        }

        if (p.InChannels < 1 || p.OutChannels < 1)
        {
            throw new GpuKernelShapeException(TransposeName, p.InChannels < 1 ? nameof(p.InChannels) : nameof(p.OutChannels), $"channel counts {p.InChannels}, {p.OutChannels} must be at least 1");
        }

        if (p.Groups < 1 || p.InChannels % p.Groups != 0 || p.OutChannels % p.Groups != 0)
        {
            throw new GpuKernelShapeException(TransposeName, nameof(p.Groups), $"{p.Groups} groups do not divide {p.InChannels} input and {p.OutChannels} output channels");
        }

        CheckWindow(TransposeName, "height", p.Height, p.KernelHeight, p.StrideHeight, p.DilationHeight, p.PadTop, p.PadBottom);
        CheckWindow(TransposeName, "width", p.Width, p.KernelWidth, p.StrideWidth, p.DilationWidth, p.PadLeft, p.PadRight);
        if (p.OutputPaddingHeight < 0 || p.OutputPaddingWidth < 0)
        {
            throw new GpuKernelShapeException(TransposeName, "output padding", "output padding must not be negative");
        }

        if (oh < 1 || ow < 1)
        {
            throw new GpuKernelShapeException(TransposeName, oh < 1 ? "height" : "width", "the output extent is not positive");
        }
    }

    /// <summary>How many bind groups the family has built so far; it stops growing once the buffer sets in use are cached (see <see cref="ReleaseBindings"/>).</summary>
    public int BindGroupsCreated => KernelPipeline.Total(_im2col, _epilogue, _depthwise, _col2im) + ConvPipeline.Total(_convGemm) + (_convCoop?.Created ?? 0) + (_convCoopNarrow?.Created ?? 0) + (_gemm32?.BindGroupsCreated ?? 0) + (_gemm16?.BindGroupsCreated ?? 0) + (_probe?.BindGroupsCreated ?? 0);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _im2col, _epilogue, _depthwise, _col2im })
        {
            foreach (var p in set)
            {
                p?.ReleaseBindings();
            }
        }

        foreach (var p in _convGemm)
        {
            p?.ReleaseBindings();
        }

        _convCoop?.ReleaseBindings();
        _convCoopNarrow?.ReleaseBindings();

        _gemm32?.ReleaseBindings();
        _gemm16?.ReleaseBindings();
        _probe?.ReleaseBindings();
    }

    /// <summary>Releases the pipelines, bind groups and the instance's own scratch once the GPU has finished with them.</summary>
    public void Dispose()
    {
        foreach (var set in new[] { _im2col, _epilogue, _depthwise, _col2im })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }

        for (var i = 0; i < _convGemm.Length; i++)
        {
            _convGemm[i]?.Dispose();
            _convGemm[i] = null;
        }

        _convCoop?.Dispose();
        _convCoop = null;
        _convCoopNarrow?.Dispose();
        _convCoopNarrow = null;
        _gemm32?.Dispose();
        _gemm16?.Dispose();
        _probe?.Dispose();
        _probe = null;
        _dummy?.Dispose();
        _ownColumns32?.Dispose();
        _ownColumns16?.Dispose();
        _ownAccumulators?.Dispose();
        _gemm32 = null;
        _gemm16 = null;
        _dummy = null;
        _ownColumns32 = null;
        _ownColumns16 = null;
        _ownAccumulators = null;
    }

    KernelPipeline CreateIm2Col(bool half)
    {
        var e = Im2Col.Variants.Get(new Im2Col.Key(f16: half)).Im2ColOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, 17, "im2col", _maxBindGroups);
    }

    KernelPipeline CreateEpilogue(bool half, bool split)
    {
        var e = ConvEpilogue.Variants.Get(new ConvEpilogue.Key(f16: half, split: split)).ConvEpilogueOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), half || split ? 4 : 3, 15, "conv-epilogue", _maxBindGroups);
    }

    KernelPipeline CreateDepthwise(bool half)
    {
        var e = DepthwiseConv.Variants.Get(new DepthwiseConv.Key(f16: half)).DepthwiseConvOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 5, 26, "depthwise-conv", _maxBindGroups);
    }

    ConvPipeline CreateConvGemm(bool half)
    {
        var e = ConvGemm.Variants.Get(new ConvGemm.Key(f16: half)).ConvGemmTiled;
        return new ConvPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 26, "conv-gemm", _maxBindGroups);
    }

    ConvPipeline CreateConvCoop(bool narrow)
    {
        if (narrow)
        {
            return new ConvPipeline(_device, new ShaderSource(ShaderStage.Compute, ConvCoopNarrowF32.ConvCoop.EntryPoint) { Msl = ConvCoopNarrowF32.ConvCoop.Msl, MslFunctionName = ConvCoopNarrowF32.ConvCoop.MslFunctionName, Layout = ConvCoopNarrowF32.ConvCoop.Layout }, 29, "conv-coop-narrow", _maxBindGroups, 32);
        }

        return new ConvPipeline(_device, new ShaderSource(ShaderStage.Compute, ConvCoopF32.ConvCoop.EntryPoint) { Msl = ConvCoopF32.ConvCoop.Msl, MslFunctionName = ConvCoopF32.ConvCoop.MslFunctionName, Layout = ConvCoopF32.ConvCoop.Layout }, 29, "conv-coop", _maxBindGroups, 32);
    }

    KernelPipeline CreateCol2Im(bool half)
    {
        var e = Col2Im.Variants.Get(new Col2Im.Key(f16: half)).Col2ImOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, 23, "col2im", _maxBindGroups);
    }
}
