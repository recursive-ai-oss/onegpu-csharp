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
/// the portable tiled variant: each output element is summed by one thread in increasing k order in float32 with fused multiply-add,
/// with no atomics, no split of k and no subgroup arithmetic, so a second run on the same device and driver gives the same bits.
/// </para>
/// <para>
/// Operands must be storage buffers created with <see cref="GpuDevice.CreateBuffer{T}(int, BufferUsage, string?)"/>. A and B may be the
/// same buffer; C must not overlap them. The instance caches bind groups for the buffer triples it has seen (at most 64) so recording a
/// warm call allocates nothing; <see cref="ReleaseBindings"/> or <see cref="Dispose"/> drops them. Not thread-safe.
/// </para>
/// </remarks>
public sealed class Gemm : IDisposable
{
    const string Name = "Gemm";
    const int MaxCachedBindGroups = 64;

    readonly GpuDevice _device;
    readonly GpuComputeKernel _kernel;
    readonly GpuComputePipeline _pipeline;
    readonly GpuPipelineLayout _layout;
    readonly uint _tileM;
    readonly uint _tileN;
    readonly ulong _splitBudget;
    readonly Dictionary<(GpuBuffer, GpuBuffer, GpuBuffer), GpuBindGroup> _bindGroups = [];

    /// <summary>Creates the kernel for one element type and mode.</summary>
    /// <param name="device">The device the operands live on.</param>
    /// <param name="elementType">Element type of A and B.</param>
    /// <param name="mode">Fast, or the bit-reproducible variant.</param>
    /// <param name="options">Variant options; null means the defaults.</param>
    /// <exception cref="GpuUnsupportedException">The device cannot run the portable variant (workgroup of 256 invocations, 8.5 KB of workgroup memory, three storage buffers).</exception>
    public Gemm(GpuDevice device, GemmElementType elementType, GpuKernelMode mode = GpuKernelMode.Fast, GemmOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        ElementType = elementType;
        Mode = mode;
        _tileM = 64;
        _tileN = 64;
        _splitBudget = (options?.AllowSplitting ?? true) ? options?.SubmissionWorkBudget ?? device.Capabilities.Compute.SubmissionWorkBudget : 0;
        // Eight subgroups per workgroup measured faster than four for 16-bit inputs on both APIs (docs/gemm.md), so that is the first choice and
        // four is the fallback for a device whose workgroup limits are smaller. float32 tiles are twice as big, and the Metal compiler counts this
        // kernel's threadgroup memory at twice its declared size (37888 bytes for the 8-subgroup float32 shape against a 32768-byte limit), so
        // float32 keeps four subgroups.
        var wide = elementType != GemmElementType.F32 && (options?.WorkgroupSubgroups ?? 0) != 4;
        if (wide && (options?.WorkgroupSubgroups ?? 0) == 0 && !MatrixVariantFits(device, elementType, wide: true))
        {
            wide = false;
        }

        if (mode == GpuKernelMode.Fast && (options?.AllowMatrixHardware ?? true) && MatrixVariantFits(device, elementType, wide))
        {
            // The same HLSL kernel runs on both: cooperative matrices on Vulkan, simdgroup matrices on Metal. Its 32-wide subgroup layout
            // is pinned (and the subgroups kept full) so a device that can run other widths still gets the one the kernel is written for.
            _kernel = CreateMatrixKernel(device, elementType, wide);
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
            (GemmElementType.F32, _) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopF32.GemmCoop.EntryPoint) { Msl = GemmCoopF32.GemmCoop.Msl, MslFunctionName = GemmCoopF32.GemmCoop.MslFunctionName, Layout = GemmCoopF32.GemmCoop.Layout }, F32, MatrixSubgroupSize),
            (GemmElementType.F16, false) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopF16.GemmCoop.EntryPoint) { SpirV = GemmCoopF16.GemmCoop.SpirV, Msl = GemmCoopF16.GemmCoop.Msl, MslFunctionName = GemmCoopF16.GemmCoop.MslFunctionName, Layout = GemmCoopF16.GemmCoop.Layout }, F16, MatrixSubgroupSize),
            (GemmElementType.F16, true) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopWideF16.GemmCoop.EntryPoint) { SpirV = GemmCoopWideF16.GemmCoop.SpirV, Msl = GemmCoopWideF16.GemmCoop.Msl, MslFunctionName = GemmCoopWideF16.GemmCoop.MslFunctionName, Layout = GemmCoopWideF16.GemmCoop.Layout }, F16, MatrixSubgroupSize),
            (_, false) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopBF16.GemmCoop.EntryPoint) { SpirV = GemmCoopBF16.GemmCoop.SpirV, Msl = GemmCoopBF16.GemmCoop.Msl, MslFunctionName = GemmCoopBF16.GemmCoop.MslFunctionName, Layout = GemmCoopBF16.GemmCoop.Layout }, BF16, MatrixSubgroupSize),
            (_, true) => device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, GemmCoopWideBF16.GemmCoop.EntryPoint) { SpirV = GemmCoopWideBF16.GemmCoop.SpirV, Msl = GemmCoopWideBF16.GemmCoop.Msl, MslFunctionName = GemmCoopWideBF16.GemmCoop.MslFunctionName, Layout = GemmCoopWideBF16.GemmCoop.Layout }, BF16, MatrixSubgroupSize),
        };
    }

    static DeviceFeatures MatrixKernelFeatures(GemmElementType type, bool wide) => (type, wide) switch
    {
        (GemmElementType.F32, _) => GemmCoopF32.GemmCoop.Layout.RequiredFeatures,
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

    static bool MatrixVariantFits(GpuDevice device, GemmElementType type, bool wide)
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
        if (device.Backend == GpuBackend.Metal)
        {
            workgroupBytes *= 2;
        }

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
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void Encode(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<float> a, GpuBuffer<float> b, GpuBuffer<float> c)
    {
        RequireType(GemmElementType.F32);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length);
    }

    /// <summary>Records a float16-input product into <paramref name="encoder"/>. Operands are checked first (shapes, strides, offsets and buffer ranges, in checked arithmetic), then one compute pass is recorded; nothing is submitted. Allocates nothing once the bind group for these buffers is cached.</summary>
    /// <param name="encoder">Receives the compute pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">The output; also the old C when <see cref="GemmProblem.Beta"/> is not 0.</param>
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void Encode(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<Half> a, GpuBuffer<Half> b, GpuBuffer<float> c)
    {
        RequireType(GemmElementType.F16);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length);
    }

    /// <summary>Records a bfloat16-input product (operands hold the bfloat16 bits as <see cref="ushort"/>) into <paramref name="encoder"/>. Operands are checked first (shapes, strides, offsets and buffer ranges, in checked arithmetic), then one compute pass is recorded; nothing is submitted. Allocates nothing once the bind group for these buffers is cached.</summary>
    /// <param name="encoder">Receives the compute pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">The output; also the old C when <see cref="GemmProblem.Beta"/> is not 0.</param>
    /// <exception cref="ArgumentException">This instance was created for another element type.</exception>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset or range is invalid, or the grid exceeds the device's limit; names the dimension.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device (<see cref="GpuComputeCapabilities.MaxStorageBufferBindingBytes"/>).</exception>
    public void EncodeBFloat16(GpuCommandEncoder encoder, in GemmProblem problem, GpuBuffer<ushort> a, GpuBuffer<ushort> b, GpuBuffer<float> c)
    {
        RequireType(GemmElementType.BF16);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        Encode(encoder, problem, a.Buffer, b.Buffer, c.Buffer, a.Length, b.Length, c.Length);
    }

    /// <summary>Drops the cached bind groups. Call it after disposing operand buffers that will not be used again, so the groups do not keep them reachable.</summary>
    public void ReleaseBindings()
    {
        foreach (var group in _bindGroups.Values)
        {
            group.Dispose();
        }

        _bindGroups.Clear();
    }

    /// <summary>Releases the pipeline and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        ReleaseBindings();
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
    void Encode(GpuCommandEncoder encoder, in GemmProblem p, GpuBuffer a, GpuBuffer b, GpuBuffer c, int lengthA, int lengthB, int lengthC)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        var (lda, ldb, ldc) = Validate(p, lengthA, lengthB, lengthC);
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

        var key = (a, b, c);
        if (!_bindGroups.TryGetValue(key, out var group))
        {
            if (_bindGroups.Count >= MaxCachedBindGroups)
            {
                ReleaseBindings();
            }

            group = _device.CreateBindGroup(_layout, 0, [BindGroupEntry.ForBuffer(0, a), BindGroupEntry.ForBuffer(1, b), BindGroupEntry.ForBuffer(2, c)]);
            _bindGroups.Add(key, group);
        }

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
                    Flags = (p.TransposeA ? 1u : 0u) | (p.TransposeB ? 2u : 0u),
                    Alpha = p.Alpha,
                    Beta = p.Beta,
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

    // Mirrors the shader's GemmParams: 15 four-byte members.
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
    }
}
