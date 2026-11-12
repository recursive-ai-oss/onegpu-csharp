using System.Runtime.InteropServices;
using OneShader.Reflection;
using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Options for creating a <see cref="Gemm"/>.</summary>
public sealed class GemmOptions
{
    /// <summary>
    /// Allow the matrix-hardware variants (cooperative matrices on Vulkan, simdgroup matrices on Metal) when the device has them. Set to
    /// false to force the portable tiled variant, for example to compare the two or to rule hardware out when chasing a problem.
    /// </summary>
    public bool AllowMatrixHardware { get; init; } = true;

    /// <summary>
    /// Cut a product that is large for the device (more arithmetic than <see cref="GpuComputeCapabilities.SubmissionWorkBudget"/>) into several
    /// dispatches over groups of batches or blocks of rows, and report each to the encoder so it can start a new submission between them. Each output element is still
    /// summed by one thread in the same order, so the bits do not change. Set to false to record every product as one dispatch.
    /// </summary>
    public bool AllowSplitting { get; init; } = true;

    /// <summary>
    /// Arithmetic operations per submission above which a product is split, replacing the device's own
    /// <see cref="GpuComputeCapabilities.SubmissionWorkBudget"/>. Null uses the device's. Meant for tests and for hosts that share the GPU with a display.
    /// </summary>
    public ulong? SubmissionWorkBudget { get; init; }

    /// <summary>Subgroups per workgroup of the matrix-hardware kernel, 4 or 8; 0 picks the best that fits. For tuning in tests.</summary>
    internal int WorkgroupSubgroups { get; init; }
}

/// <summary>
/// Matrix multiplication <c>C = Alpha * op(A) * op(B) + Beta * C</c>, batched, with float32 accumulation and float32 output, for float32,
/// float16 or bfloat16 inputs (<see cref="GemmElementType"/>).
/// </summary>
/// <remarks>
/// <para>
/// The constructor picks a variant from the device's capabilities once and reports it in <see cref="Variant"/>. In
/// <see cref="GpuKernelMode.Fast"/> that is a matrix-hardware variant when the device has one for the element type and an
/// output tile of 64 x 64 is available, otherwise the portable tiled variant. In <see cref="GpuKernelMode.Deterministic"/> it is always
/// the portable tiled variant (or, for float32 products of at most four rows on Vulkan and WebGPU, the few-rows kernel): the order in which each element's terms are summed
/// depends only on the shape, with fused multiply-add, no atomics and no subgroup arithmetic, so a second run on the same device and driver gives the same bits.
/// A small grid over a long k is cut into ranges whose partial sums a second pass adds in range order; that, too, depends only on the shape.
/// </para>
/// <para>
/// Operands must be storage buffers created with <see cref="GpuDevice.CreateBuffer{T}(int, BufferUsage, string?)"/>. A and B may be the
/// same buffer; C must not overlap them. The instance caches bind groups for the buffer triples it has seen (at most <c>maxCachedBindGroups</c>, 1024 by default) so recording a
/// warm call allocates nothing; <see cref="ReleaseBindings"/> or <see cref="Dispose"/> drops them. Not thread-safe.
/// </para>
/// </remarks>
public sealed class Gemm : IDisposable
{
    const string Name = "Gemm";

    readonly GpuDevice _device;
    readonly GpuComputeKernel _kernel;
    readonly int _maxBindGroups;
    readonly GpuComputePipeline _pipeline;
    readonly GpuPipelineLayout _layout;
    readonly uint _tileM;
    readonly uint _tileN;
    readonly ulong _splitBudget;
    GpuBuffer<float>? _noBias;
    KernelPipeline? _skinny;
    KernelPipeline? _splitReduce;
    GpuBuffer<float>? _splitScratch;
    readonly Dictionary<(GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer), GpuBindGroup> _bindGroups = [];

    /// <summary>Creates the kernel for one element type and mode.</summary>
    /// <param name="device">The device the operands live on.</param>
    /// <param name="elementType">Element type of A and B.</param>
    /// <param name="mode">Fast, or the bit-reproducible variant.</param>
    /// <param name="options">Variant options; null means the defaults.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    /// <exception cref="GpuUnsupportedException">The device cannot run the portable variant (workgroup of 256 invocations, 8.5 KB of workgroup memory, three storage buffers).</exception>
    public Gemm(GpuDevice device, GemmElementType elementType, GpuKernelMode mode = GpuKernelMode.Fast, GemmOptions? options = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        ElementType = elementType;
        Mode = mode;
        _tileM = 64;
        _tileN = 64;
        _splitBudget = (options?.AllowSplitting ?? true) ? options?.SubmissionWorkBudget ?? device.Capabilities.Compute.SubmissionWorkBudget : 0;
        // Eight subgroups per workgroup measured faster than four for 16-bit inputs on both APIs (docs/gemm.md), so that is the first choice and
        // four is the fallback for a device whose workgroup limits are smaller. float32 tiles are twice as big: the eight-subgroup shape declares
        // 18944 bytes, which fits Metal's 32768-byte limit, but Metal reports twice the declared size while shader validation is on, so the
        // choice is made from the length the compiled pipeline reports (below) and not from a formula.
        // The float32 kernel reads whole tiles straight from the operands (GEMM_DIRECT); with four subgroups it measured 1.4x the eight-subgroup shape on an Apple M2
        // (1490 against 1065 GFLOP/s on a 1024 x 3072 x 768 product), so float32 asks for four unless a test forces eight.
        var requested = options?.WorkgroupSubgroups ?? 0;
        if (requested == 0 && elementType == GemmElementType.F32)
        {
            requested = 4;
        }

        var wide = requested != 4;
        if (wide && requested == 0 && !MatrixVariantFits(device, elementType, wide: true))
        {
            wide = false;
        }

        if (mode == GpuKernelMode.Fast && (options?.AllowMatrixHardware ?? true) && MatrixVariantFits(device, elementType, wide))
        {
            // The same HLSL kernel runs on both: cooperative matrices on Vulkan, simdgroup matrices on Metal. Its 32-wide subgroup layout
            // is pinned (and the subgroups kept full) so a device that can run other widths still gets the one the kernel is written for.
            _kernel = CreateMatrixKernel(device, elementType, wide);
            if (wide && _kernel.Pipeline.ReportedWorkgroupMemoryBytes > device.Capabilities.Compute.MaxWorkgroupMemoryBytes)
            {
                _kernel.Dispose();
                wide = false;
                _kernel = CreateMatrixKernel(device, elementType, wide);
            }

            _pipeline = _kernel.Pipeline;
            _layout = _kernel.PipelineLayout;
            Variant = (device.Backend == GpuBackend.Metal ? "simdgroup-matrix-16x16x16-tile64x64x16-" : "cooperative-matrix-16x16x16-tile64x64x16-") + (wide ? "8sg" : "4sg");
            return;
        }

        _kernel = elementType switch
        {
            GemmElementType.F32 => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmF32.GemmTiled.EntryPoint) { SpirV = GemmF32.GemmTiled.SpirV, Wgsl = GemmF32.GemmTiled.Wgsl, WgslEntryPoint = GemmF32.GemmTiled.WgslEntryPoint, Msl = GemmF32.GemmTiled.Msl, MslFunctionName = GemmF32.GemmTiled.MslFunctionName, Layout = GemmF32.GemmTiled.Layout }, "gemm f32"),
            GemmElementType.F16 => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmF16.GemmTiled.EntryPoint) { SpirV = GemmF16.GemmTiled.SpirV, Wgsl = GemmF16.GemmTiled.Wgsl, WgslEntryPoint = GemmF16.GemmTiled.WgslEntryPoint, Msl = GemmF16.GemmTiled.Msl, MslFunctionName = GemmF16.GemmTiled.MslFunctionName, Layout = GemmF16.GemmTiled.Layout }, "gemm f16"),
            GemmElementType.BF16 => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmBF16.GemmTiled.EntryPoint) { SpirV = GemmBF16.GemmTiled.SpirV, Wgsl = GemmBF16.GemmTiled.Wgsl, WgslEntryPoint = GemmBF16.GemmTiled.WgslEntryPoint, Msl = GemmBF16.GemmTiled.Msl, MslFunctionName = GemmBF16.GemmTiled.MslFunctionName, Layout = GemmBF16.GemmTiled.Layout }, "gemm bf16"),
            _ => throw new ArgumentOutOfRangeException(nameof(elementType)),
        };
        _pipeline = _kernel.Pipeline;
        _layout = _kernel.PipelineLayout;
        Variant = "tiled-64x64x16";
    }

    static GpuComputeKernel CreateMatrixKernel(GpuDevice device, GemmElementType type, bool wide)
    {
        const string F32 = "gemm f32 matrix hardware", F16 = "gemm f16 matrix hardware", BF16 = "gemm bf16 matrix hardware";
        return (type, wide) switch
        {
            (GemmElementType.F32, false) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopF32.GemmCoop.EntryPoint) { Msl = GemmCoopF32.GemmCoop.Msl, MslFunctionName = GemmCoopF32.GemmCoop.MslFunctionName, Layout = GemmCoopF32.GemmCoop.Layout }, F32, MatrixSubgroupSize),
            (GemmElementType.F32, true) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopWideF32.GemmCoop.EntryPoint) { Msl = GemmCoopWideF32.GemmCoop.Msl, MslFunctionName = GemmCoopWideF32.GemmCoop.MslFunctionName, Layout = GemmCoopWideF32.GemmCoop.Layout }, F32, MatrixSubgroupSize),
            (GemmElementType.F16, false) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopF16.GemmCoop.EntryPoint) { SpirV = GemmCoopF16.GemmCoop.SpirV, Msl = GemmCoopF16.GemmCoop.Msl, MslFunctionName = GemmCoopF16.GemmCoop.MslFunctionName, Layout = GemmCoopF16.GemmCoop.Layout }, F16, MatrixSubgroupSize),
            (GemmElementType.F16, true) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopWideF16.GemmCoop.EntryPoint) { SpirV = GemmCoopWideF16.GemmCoop.SpirV, Msl = GemmCoopWideF16.GemmCoop.Msl, MslFunctionName = GemmCoopWideF16.GemmCoop.MslFunctionName, Layout = GemmCoopWideF16.GemmCoop.Layout }, F16, MatrixSubgroupSize),
            (_, false) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopBF16.GemmCoop.EntryPoint) { SpirV = GemmCoopBF16.GemmCoop.SpirV, Msl = GemmCoopBF16.GemmCoop.Msl, MslFunctionName = GemmCoopBF16.GemmCoop.MslFunctionName, Layout = GemmCoopBF16.GemmCoop.Layout }, BF16, MatrixSubgroupSize),
            (_, true) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopWideBF16.GemmCoop.EntryPoint) { SpirV = GemmCoopWideBF16.GemmCoop.SpirV, Msl = GemmCoopWideBF16.GemmCoop.Msl, MslFunctionName = GemmCoopWideBF16.GemmCoop.MslFunctionName, Layout = GemmCoopWideBF16.GemmCoop.Layout }, BF16, MatrixSubgroupSize),
        };
    }

    static DeviceFeatures MatrixKernelFeatures(GemmElementType type, bool wide) => (type, wide) switch
    {
        (GemmElementType.F32, false) => GemmCoopF32.GemmCoop.Layout.RequiredFeatures,
        (GemmElementType.F32, true) => GemmCoopWideF32.GemmCoop.Layout.RequiredFeatures,
        (GemmElementType.F16, false) => GemmCoopF16.GemmCoop.Layout.RequiredFeatures,
        (GemmElementType.F16, true) => GemmCoopWideF16.GemmCoop.Layout.RequiredFeatures,
        (_, false) => GemmCoopBF16.GemmCoop.Layout.RequiredFeatures,
        (_, true) => GemmCoopWideBF16.GemmCoop.Layout.RequiredFeatures,
    };

    // The matrix kernel is written for 32-wide subgroups (four or eight of them per workgroup) and 16x16x16 fragments of the element type
    // with a float32 accumulator. On Vulkan that means a device whose subgroups are 32 wide or can be pinned to 32, that runs pipelines with
    // full subgroups, and that offers the features the kernel's reflection lists. On Metal the fragments are 2x2 simdgroup tiles of 8x8x8.
    // A device that only has 64-wide subgroups is not eligible and runs the portable kernel (see docs/gemm.md).
    const uint MatrixSubgroupSize = 32;

    internal static bool MatrixVariantFits(GpuDevice device, GemmElementType type, bool wide)
    {
        // float32 fragments exist only as simdgroup matrices; no Vulkan device lists them, so that kernel is built for Metal alone.
        if (device.Backend is not (GpuBackend.Vulkan or GpuBackend.Metal) || (type == GemmElementType.F32 && device.Backend != GpuBackend.Metal))
        {
            return false;
        }

        var cc = device.Capabilities.Compute;
        var sizeOk = cc.SubgroupMinSize <= MatrixSubgroupSize && MatrixSubgroupSize <= cc.SubgroupMaxSize
            && (cc.SubgroupMinSize == cc.SubgroupMaxSize || cc.SubgroupSizeControl);
        var invocations = wide ? 256u : 128u;
        // Operand tiles in the element type (64 x 24 and 16 x 72, padded), plus 1 KB of float32 staging per subgroup for the result.
        var workgroupBytes = ((64u * 24u) + (16u * 72u)) * (type == GemmElementType.F32 ? 4u : 2u) + ((wide ? 8u : 4u) * 1024u);
        if (!sizeOk || !cc.FullSubgroups || cc.MaxWorkgroupInvocations < invocations || cc.MaxWorkgroupSizeX < invocations || cc.MaxWorkgroupMemoryBytes < workgroupBytes)
        {
            return false;
        }

        if ((MatrixKernelFeatures(type, wide) & ~cc.ShaderFeatures) != DeviceFeatures.None)
        {
            return false;
        }

        var wanted = type switch
        {
            GemmElementType.F32 => GpuScalarType.F32,
            GemmElementType.F16 => GpuScalarType.F16,
            _ => GpuScalarType.BF16,
        };
        var edge = device.Backend == GpuBackend.Metal ? 8u : 16u;
        foreach (var shape in cc.CooperativeMatrices)
        {
            if (shape is { CType: GpuScalarType.F32, ResultType: GpuScalarType.F32, Saturating: false } && shape.M == edge && shape.N == edge && shape.K == edge && shape.AType == wanted && shape.BType == wanted)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Element type of A and B.</summary>
    public GemmElementType ElementType { get; }

    /// <summary>The mode this instance was created for.</summary>
    public GpuKernelMode Mode { get; }

    /// <summary>The variant chosen for this device, for logs and diagnostics, for example <c>tiled-64x64x16</c>.</summary>
    public string Variant { get; }

    /// <summary>Records a float32 product into <paramref name="encoder"/>. Operands are checked first (shapes, strides, offsets and buffer ranges, in checked arithmetic), then one compute pass is recorded; nothing is submitted. Allocates nothing once the bind group for these buffers is cached.</summary>
    /// <param name="encoder">Receives the compute pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">The output; also the old C when <see cref="GemmProblem.Beta"/> is not 0.</param>
    /// <param name="epilogue">A column bias and an activation applied to each result element in the same dispatch; the default does nothing. See <see cref="GemmEpilogue"/>.</param>
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void Encode(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<float> a, GpuBuffer<float> b, GpuBuffer<float> c, GemmEpilogue epilogue = default)
    {
        RequireType(GemmElementType.F32);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length, epilogue);
    }

    /// <summary>Records a float16-input product into <paramref name="encoder"/>. Operands are checked first (shapes, strides, offsets and buffer ranges, in checked arithmetic), then one compute pass is recorded; nothing is submitted. Allocates nothing once the bind group for these buffers is cached.</summary>
    /// <param name="encoder">Receives the compute pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">The output; also the old C when <see cref="GemmProblem.Beta"/> is not 0.</param>
    /// <param name="epilogue">A column bias and an activation applied to each result element in the same dispatch; the default does nothing. See <see cref="GemmEpilogue"/>.</param>
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void Encode(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<Half> a, GpuBuffer<Half> b, GpuBuffer<float> c, GemmEpilogue epilogue = default)
    {
        RequireType(GemmElementType.F16);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length, epilogue);
    }

    /// <summary>Records a bfloat16-input product (operands hold the bfloat16 bits as <see cref="ushort"/>) into <paramref name="encoder"/>. Operands are checked first (shapes, strides, offsets and buffer ranges, in checked arithmetic), then one compute pass is recorded; nothing is submitted. Allocates nothing once the bind group for these buffers is cached.</summary>
    /// <param name="encoder">Receives the compute pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">The output; also the old C when <see cref="GemmProblem.Beta"/> is not 0.</param>
    /// <param name="epilogue">A column bias and an activation applied to each result element in the same dispatch; the default does nothing. See <see cref="GemmEpilogue"/>.</param>
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void EncodeBFloat16(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<ushort> a, GpuBuffer<ushort> b, GpuBuffer<float> c, GemmEpilogue epilogue = default)
    {
        RequireType(GemmElementType.BF16);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length, epilogue);
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => _bindGroupsCreated + (_skinny?.Created ?? 0) + (_splitReduce?.Created ?? 0);

    int _bindGroupsCreated;

    /// <summary>Drops the cached bind groups. Call it after disposing operand buffers that will not be used again, so the groups do not keep them reachable.</summary>
    public void ReleaseBindings()
    {
        foreach (var group in _bindGroups.Values)
        {
            group.Dispose();
        }

        _bindGroups.Clear();
        _skinny?.ReleaseBindings();
        _splitReduce?.ReleaseBindings();
    }

    /// <summary>Releases the pipeline and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        ReleaseBindings();
        _noBias?.Dispose();
        _noBias = null;
        _skinny?.Dispose();
        _skinny = null;
        _splitReduce?.Dispose();
        _splitReduce = null;
        _splitScratch?.Dispose();
        _splitScratch = null;
        _kernel.Dispose();

    }

    void RequireType(GemmElementType expected)
    {
        if (ElementType != expected)
        {
            throw new ArgumentException($"this Gemm was created for {ElementType} operands, not {expected}");
        }
    }

    // Buffers whose element is smaller than a 32-bit word are bound as words, so the readable length in elements is the
    // buffer's declared length: the allocation is padded to a whole word.
    void Encode(GpuCommandEncoder encoder, in GemmProblem p, GpuBuffer a, GpuBuffer b, GpuBuffer c, int lengthA, int lengthB, int lengthC, in GemmEpilogue epilogue)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        var (lda, ldb, ldc) = Validate(p, lengthA, lengthB, lengthC);
        if (!Enum.IsDefined(epilogue.Activation))
        {
            throw new GpuKernelShapeException(Name, nameof(epilogue), $"activation {(int)epilogue.Activation} is not a GemmActivation");
        }

        if (!Enum.IsDefined(epilogue.BiasAxis))
        {
            throw new GpuKernelShapeException(Name, nameof(epilogue), $"bias axis {(int)epilogue.BiasAxis} is not a GemmBiasAxis");
        }

        if (epilogue.Bias is { } bias && p.M > 0 && p.N > 0)
        {
            Check.Range(Name, "bias", bias.Length, epilogue.OffsetBias, epilogue.BiasAxis == GemmBiasAxis.Row ? p.M : p.N);
        }

        if (p.M == 0 || p.N == 0)
        {
            return;
        }

        var groupsX = (uint)((p.N + (int)_tileN - 1) / (int)_tileN);
        var groupsY = (uint)((p.M + (int)_tileM - 1) / (int)_tileM);
        var limit = _device.Capabilities.Compute.MaxWorkgroupCount;
        if (limit != 0 && (groupsX > limit || groupsY > limit || (uint)p.Batch > limit))
        {
            throw new GpuKernelShapeException(Name, "M, N or Batch", $"the grid of {groupsX} x {groupsY} x {p.Batch} workgroups exceeds the device limit of {limit} per axis");
        }

        CheckBindable("A", a);
        CheckBindable("B", b);
        CheckBindable("C", c);
        var biasBuffer = epilogue.Bias is { } given ? given.Buffer : (_noBias ??= _device.CreateBuffer<float>(4, label: "gemm unused bias")).Buffer;
        CheckBindable("bias", biasBuffer);

        var key = (a, b, c, biasBuffer);
        if (!_bindGroups.TryGetValue(key, out var group))
        {
            if (_bindGroups.Count >= _maxBindGroups)
            {
                ReleaseBindings();
            }

            group = _device.CreateBindGroup(_layout, 0, [BindGroupEntry.ForBuffer(0, a), BindGroupEntry.ForBuffer(1, b), BindGroupEntry.ForBuffer(2, c), BindGroupEntry.ForBuffer(3, biasBuffer)]);
            _bindGroups.Add(key, group);
            _bindGroupsCreated++;
        }

        var opsAll = 2.0 * p.M * p.N * p.K * p.Batch;
        if (UsesSkinny(p))
        {
            EncodeSkinny(encoder, p, a, b, c, biasBuffer, lda, ldb, ldc, epilogue);
            encoder.ReportWork(opsAll);
            return;
        }

        // The ranges of k come from the whole product, so cutting it into pieces for the submission budget (below) leaves every element's sum as it is.
        var (splits, perSplit) = SplitPlan(p);

        // One dispatch unless the product carries more arithmetic than a submission should (see GemmOptions.AllowSplitting): then whole
        // products are grouped, or a single product is cut into blocks of rows, and the encoder is told about each piece. A piece is a
        // smaller problem over the same operands (offsets moved, M and Batch reduced), so each element's sum is the same one.
        var opsPerProduct = 2.0 * p.M * p.N * p.K;
        var batchesPerPiece = p.Batch;
        var rowsPerPiece = p.M;
        if (_splitBudget != 0 && opsPerProduct * p.Batch > _splitBudget && opsPerProduct > 0)
        {
            if (opsPerProduct <= _splitBudget)
            {
                batchesPerPiece = (int)Math.Max(1, Math.Min(p.Batch, _splitBudget / opsPerProduct));
            }
            else
            {
                batchesPerPiece = 1;
                rowsPerPiece = (int)Math.Min(p.M, Math.Max(_tileM, (long)(_splitBudget / (2.0 * p.N * p.K) / _tileM) * _tileM));
            }
        }

        for (var b0 = 0; b0 < p.Batch; b0 += batchesPerPiece)
        {
            var nb = Math.Min(batchesPerPiece, p.Batch - b0);
            for (var r0 = 0; r0 < p.M; r0 += rowsPerPiece)
            {
                var rows = Math.Min(rowsPerPiece, p.M - r0);
                if (splits > 1)
                {
                    var piece = p;
                    piece.M = rows;
                    piece.Batch = nb;
                    piece.OffsetA = checked((int)(p.OffsetA + ((long)b0 * p.StrideA) + (p.TransposeA ? r0 : (long)r0 * lda)));
                    piece.OffsetB = checked((int)(p.OffsetB + ((long)b0 * p.StrideB)));
                    piece.OffsetC = checked((int)(p.OffsetC + ((long)b0 * p.StrideC) + ((long)r0 * ldc)));
                    EncodeSplit(encoder, piece, a, b, c, biasBuffer, lda, ldb, ldc, epilogue, epilogue.OffsetBias + (epilogue.BiasAxis == GemmBiasAxis.Row ? r0 : 0), splits, perSplit);
                    encoder.ReportWork(2.0 * rows * p.N * p.K * nb);
                    continue;
                }

                var push = new PushBlock
                {
                    M = (uint)rows,
                    N = (uint)p.N,
                    K = (uint)p.K,
                    Lda = (uint)lda,
                    Ldb = (uint)ldb,
                    Ldc = (uint)ldc,
                    OffsetA = (uint)(p.OffsetA + ((long)b0 * p.StrideA) + (p.TransposeA ? r0 : (long)r0 * lda)),
                    OffsetB = (uint)(p.OffsetB + ((long)b0 * p.StrideB)),
                    OffsetC = (uint)(p.OffsetC + ((long)b0 * p.StrideC) + ((long)r0 * ldc)),
                    StrideA = (uint)p.StrideA,
                    StrideB = (uint)p.StrideB,
                    StrideC = (uint)p.StrideC,
                    Flags = (p.TransposeA ? 1u : 0u) | (p.TransposeB ? 2u : 0u) | (epilogue.Bias is null ? 0u : (epilogue.BiasAxis == GemmBiasAxis.Row ? 12u : 4u)),
                    Alpha = p.Alpha,
                    Beta = p.Beta,
                    BiasOffset = (uint)(epilogue.OffsetBias + (epilogue.BiasAxis == GemmBiasAxis.Row ? r0 : 0)),
                    Act = (uint)epilogue.Activation,
                };
                var pass = encoder.BeginComputePass();
                pass.SetPipeline(_pipeline);
                pass.SetBindGroup(0, group);
                pass.SetPushConstants(0, MemoryMarshal.AsBytes(new ReadOnlySpan<PushBlock>(in push)));
                pass.Dispatch((uint)((p.N + (int)_tileN - 1) / (int)_tileN), (uint)((rows + (int)_tileM - 1) / (int)_tileM), (uint)nb);
                pass.EndPass();
                encoder.ReportWork(2.0 * rows * p.N * p.K * nb);
            }
        }
    }

    // Products with at most four rows (decoding): a kernel that streams B once. Not on Metal (the Apple kernels are tuned separately), for float32 and plain A.
    bool UsesSkinny(in GemmProblem p)
    {
        if (ElementType != GemmElementType.F32 || !Variant.StartsWith("tiled", StringComparison.Ordinal) || _device.Backend == GpuBackend.Metal 
            || p.TransposeA || p.M > 4 || p.K == 0)
        {
            return false;
        }

        var limit = _device.Capabilities.Compute.MaxWorkgroupCount;
        var tilesX = (uint)(((long)p.N + (p.TransposeB ? 7 : 15)) / (p.TransposeB ? 8 : 16));
        return limit == 0 || (tilesX <= limit && (uint)p.Batch <= limit);
    }

    void EncodeSkinny(GpuCommandEncoder encoder, in GemmProblem p, GpuBuffer a, GpuBuffer b, GpuBuffer c, GpuBuffer bias, int lda, int ldb, int ldc, in GemmEpilogue epilogue)
    {
        var pipe = _skinny ??= new KernelPipeline(_device, new ShaderSource(ShaderStage.Compute, GemmSkinny.GemmSkinnyOp.EntryPoint) { SpirV = GemmSkinny.GemmSkinnyOp.SpirV, Wgsl = GemmSkinny.GemmSkinnyOp.Wgsl, WgslEntryPoint = GemmSkinny.GemmSkinnyOp.WgslEntryPoint, Layout = GemmSkinny.GemmSkinnyOp.Layout }, 4, 17, "gemm skinny", _maxBindGroups);
        var push = MakePush(p, lda, ldb, ldc, epilogue, p.M, 0);
        var wide = !p.TransposeB && p.N >= SkinnyWideFrom;
        if (wide)
        {
            push.Flags |= 32u;
        }

        var cols = p.TransposeB ? 8 : wide ? 32 : 16;
        var tilesX = (uint)(((long)p.N + cols - 1) / cols);
        pipe.Record(encoder, MemoryMarshal.Cast<PushBlock, uint>(new ReadOnlySpan<PushBlock>(in push)), tilesX, (uint)p.Batch, [a, b, c, bias]);
    }

    // How many ranges of k a product with a small grid is cut into, and how many terms each holds, from the shape alone. A 128 x 768 product at 64 x 64 tiles has 24
    // workgroups on a GPU with dozens of cores, each walking the whole of k; ranges that run as workgroups of their own fill it, and their planes of partial sums are added
    // in range order by a second pass, so the bits depend on the shape and nothing else. A range is a multiple of 32 terms (a block-sum boundary of the kernel).
    (int Splits, int PerSplit) SplitPlan(in GemmProblem p)
    {
        if (!Variant.StartsWith("tiled", StringComparison.Ordinal) || _device.Backend == GpuBackend.Metal || p.M == 0 || p.N == 0)
        {
            return (1, p.K);
        }

        long groups = (((long)p.M + 63) / 64) * (((long)p.N + 63) / 64) * p.Batch;
        var blocks = ((long)p.K + 31) / 32;
        if (groups >= SplitBelow || blocks < 2 * SplitMinBlocks)
        {
            return (1, p.K);
        }

        var perProduct = (long)p.Batch * p.M * p.N;
        var splits = Math.Min(Math.Min(SplitMaxSplits, (SplitBelow + groups - 1) / groups), Math.Min(blocks / SplitMinBlocks, SplitScratchElements / perProduct));
        var limit = _device.Capabilities.Compute.MaxWorkgroupCount;
        if (limit != 0)
        {
            splits = Math.Min(splits, limit / Math.Max(p.Batch, 1));
        }

        if (splits < 2)
        {
            return (1, p.K);
        }

        var per = (int)((blocks + splits - 1) / splits * 32);
        return ((int)(((long)p.K + per - 1) / per), per);
    }

    const long SplitScratchElements = 4L << 20;   // 16 MiB of float32 planes, allocated once on first use
    const long SplitBelow = 384;        // workgroups of 64 x 64
    const long SplitMaxSplits = 16;
    const long SplitMinBlocks = 2;      // 32-term blocks a range must keep
    const long SkinnyWideFrom = 8192;   // columns from which the kernel reads 32 columns of B per warp row

    void EncodeSplit(GpuCommandEncoder encoder, in GemmProblem p, GpuBuffer a, GpuBuffer b, GpuBuffer c, GpuBuffer bias, int lda, int ldb, int ldc, in GemmEpilogue epilogue, int biasOffset, int splits, int perSplit)
    {
        var scratch = _splitScratch ??= _device.CreateBuffer<float>((int)SplitScratchElements, label: "gemm split planes");
        var planeStride = (long)p.Batch * p.M * p.N;
        var unused = (_noBias ??= _device.CreateBuffer<float>(4, label: "gemm unused bias")).Buffer;
        var key = (a, b, scratch.Buffer, unused);
        if (!_bindGroups.TryGetValue(key, out var group))
        {
            if (_bindGroups.Count >= _maxBindGroups)
            {
                ReleaseBindings();
            }

            group = _device.CreateBindGroup(_layout, 0, [BindGroupEntry.ForBuffer(0, a), BindGroupEntry.ForBuffer(1, b), BindGroupEntry.ForBuffer(2, scratch.Buffer), BindGroupEntry.ForBuffer(3, key.Item4)]);
            _bindGroups.Add(key, group);
            _bindGroupsCreated++;
        }

        var push = new PushBlock
        {
            M = (uint)p.M,
            N = (uint)p.N,
            K = (uint)p.K,
            Lda = (uint)lda,
            Ldb = (uint)ldb,
            Ldc = (uint)p.N,
            OffsetA = (uint)p.OffsetA,
            OffsetB = (uint)p.OffsetB,
            OffsetC = 0,
            StrideA = (uint)p.StrideA,
            StrideB = (uint)p.StrideB,
            Flags = (p.TransposeA ? 1u : 0u) | (p.TransposeB ? 2u : 0u) | 16u,
            StrideC = (uint)planeStride,
            BiasOffset = (uint)p.Batch,
            Act = (uint)perSplit,
        };
        var pass = encoder.BeginComputePass();
        pass.SetPipeline(_pipeline);
        pass.SetBindGroup(0, group);
        pass.SetPushConstants(0, MemoryMarshal.AsBytes(new ReadOnlySpan<PushBlock>(in push)));
        pass.Dispatch((uint)((p.N + (int)_tileN - 1) / (int)_tileN), (uint)((p.M + (int)_tileM - 1) / (int)_tileM), (uint)((long)splits * p.Batch));
        pass.EndPass();

        var reduce = _splitReduce ??= new KernelPipeline(_device, new ShaderSource(ShaderStage.Compute, GemmSplitReduce.GemmSplitReduceOp.EntryPoint) { SpirV = GemmSplitReduce.GemmSplitReduceOp.SpirV, Wgsl = GemmSplitReduce.GemmSplitReduceOp.Wgsl, WgslEntryPoint = GemmSplitReduce.GemmSplitReduceOp.WgslEntryPoint, Layout = GemmSplitReduce.GemmSplitReduceOp.Layout }, 3, 14, "gemm split reduce", _maxBindGroups);
        var (gx, gy) = Grid.For(_device, (planeStride + 255) / 256);
        Span<uint> r = stackalloc uint[14];
        r[0] = (uint)planeStride;
        r[1] = (uint)p.N;
        r[2] = (uint)((long)p.M * p.N);
        r[3] = (uint)ldc;
        r[4] = (uint)p.OffsetC;
        r[5] = (uint)p.StrideC;
        r[6] = (uint)splits;
        r[7] = (uint)planeStride;
        r[8] = epilogue.Bias is null ? 0u : (epilogue.BiasAxis == GemmBiasAxis.Row ? 12u : 4u);
        r[9] = BitConverter.SingleToUInt32Bits(p.Alpha);
        r[10] = BitConverter.SingleToUInt32Bits(p.Beta);
        r[11] = (uint)biasOffset;
        r[12] = (uint)epilogue.Activation;
        r[13] = gx;
        reduce.Record(encoder, r, gx, gy, [scratch.Buffer, c, bias]);
    }

    static PushBlock MakePush(in GemmProblem p, int lda, int ldb, int ldc, in GemmEpilogue epilogue, int rows, int r0) => new()
    {
        M = (uint)rows,
        N = (uint)p.N,
        K = (uint)p.K,
        Lda = (uint)lda,
        Ldb = (uint)ldb,
        Ldc = (uint)ldc,
        OffsetA = (uint)(p.OffsetA + (p.TransposeA ? r0 : (long)r0 * lda)),
        OffsetB = (uint)p.OffsetB,
        OffsetC = (uint)(p.OffsetC + ((long)r0 * ldc)),
        StrideA = (uint)p.StrideA,
        StrideB = (uint)p.StrideB,
        StrideC = (uint)p.StrideC,
        Flags = (p.TransposeA ? 1u : 0u) | (p.TransposeB ? 2u : 0u) | (epilogue.Bias is null ? 0u : (epilogue.BiasAxis == GemmBiasAxis.Row ? 12u : 4u)),
        Alpha = p.Alpha,
        Beta = p.Beta,
        BiasOffset = (uint)(epilogue.OffsetBias + (epilogue.BiasAxis == GemmBiasAxis.Row ? r0 : 0)),
        Act = (uint)epilogue.Activation,
    };

    void CheckBindable(string operand, GpuBuffer buffer)
    {
        var maxBinding = _device.Capabilities.Compute.MaxStorageBufferBindingBytes;
        if (maxBinding != 0 && buffer.Size > maxBinding)
        {
            throw new GpuUnsupportedException($"{Name}: operand {operand} is {buffer.Size} bytes, more than one storage binding can cover on {_device.Adapter.Name}", nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes), $"at most {maxBinding} bytes", $"{buffer.Size} bytes");
        }
    }

    // Everything here is checked in long arithmetic, in safe code, before a push constant is built.
    static (int Lda, int Ldb, int Ldc) Validate(in GemmProblem p, int lengthA, int lengthB, int lengthC)
    {
        if (p.M < 0)
        {
            throw new GpuKernelShapeException(Name, "M", $"must not be negative, got {p.M}");
        }

        if (p.N < 0)
        {
            throw new GpuKernelShapeException(Name, "N", $"must not be negative, got {p.N}");
        }

        if (p.K < 0)
        {
            throw new GpuKernelShapeException(Name, "K", $"must not be negative, got {p.K}");
        }

        if (p.Batch < 1)
        {
            throw new GpuKernelShapeException(Name, "Batch", $"must be at least 1, got {p.Batch}");
        }

        CheckNonNegative(nameof(p.Lda), p.Lda);
        CheckNonNegative(nameof(p.Ldb), p.Ldb);
        CheckNonNegative(nameof(p.Ldc), p.Ldc);
        CheckNonNegative(nameof(p.OffsetA), p.OffsetA);
        CheckNonNegative(nameof(p.OffsetB), p.OffsetB);
        CheckNonNegative(nameof(p.OffsetC), p.OffsetC);
        CheckNonNegative(nameof(p.StrideA), p.StrideA);
        CheckNonNegative(nameof(p.StrideB), p.StrideB);
        CheckNonNegative(nameof(p.StrideC), p.StrideC);

        // Rows and columns of the matrices as stored.
        var colsA = p.TransposeA ? p.M : p.K;
        var rowsA = p.TransposeA ? p.K : p.M;
        var colsB = p.TransposeB ? p.K : p.N;
        var rowsB = p.TransposeB ? p.N : p.K;
        var lda = p.Lda == 0 ? Math.Max(colsA, 1) : p.Lda;
        var ldb = p.Ldb == 0 ? Math.Max(colsB, 1) : p.Ldb;
        var ldc = p.Ldc == 0 ? Math.Max(p.N, 1) : p.Ldc;
        if (lda < colsA)
        {
            throw new GpuKernelShapeException(Name, "Lda", $"{lda} is smaller than the stored row length {colsA}");
        }

        if (ldb < colsB)
        {
            throw new GpuKernelShapeException(Name, "Ldb", $"{ldb} is smaller than the stored row length {colsB}");
        }

        if (ldc < p.N)
        {
            throw new GpuKernelShapeException(Name, "Ldc", $"{ldc} is smaller than N = {p.N}");
        }

        if (p.M == 0 || p.N == 0)
        {
            return (lda, ldb, ldc);
        }

        if (p.Batch > 1 && p.StrideC < ((long)(p.M - 1) * ldc) + p.N)
        {
            throw new GpuKernelShapeException(Name, "StrideC", $"{p.StrideC} lets the outputs of a batch overlap; it must be at least {((long)(p.M - 1) * ldc) + p.N}");
        }

        CheckRange("C", p.OffsetC, p.StrideC, p.M, p.N, ldc, p.Batch, lengthC);
        if (p.K > 0)
        {
            CheckRange("A", p.OffsetA, p.StrideA, rowsA, colsA, lda, p.Batch, lengthA);
            CheckRange("B", p.OffsetB, p.StrideB, rowsB, colsB, ldb, p.Batch, lengthB);
        }

        return (lda, ldb, ldc);
    }

    static void CheckNonNegative(string name, int value)
    {
        if (value < 0)
        {
            throw new GpuKernelShapeException(Name, name, $"must not be negative, got {value}");
        }
    }

    static void CheckRange(string operand, int offset, int stride, int rows, int cols, int ld, int batch, int length)
    {
        var last = offset + ((long)(batch - 1) * stride) + ((long)(rows - 1) * ld) + (cols - 1);
        if (last >= length)
        {
            throw new GpuKernelShapeException(Name, "Offset" + operand, $"operand {operand} reaches element {last} but its buffer holds {length}");
        }
    }

    // Mirrors the shader's GemmParams: 17 four-byte members.
    [StructLayout(LayoutKind.Sequential)]
    struct PushBlock
    {
        public uint M;
        public uint N;
        public uint K;
        public uint Lda;
        public uint Ldb;
        public uint Ldc;
        public uint OffsetA;
        public uint OffsetB;
        public uint OffsetC;
        public uint StrideA;
        public uint StrideB;
        public uint StrideC;
        public uint Flags;
        public float Alpha;
        public float Beta;
        public uint BiasOffset;
        public uint Act;
    }
}
