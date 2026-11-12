using OneShader.Reflection;
using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Options for creating an <see cref="Int8Gemm"/>.</summary>
public sealed class Int8GemmOptions
{
    /// <summary>
    /// Allow the packed 8-bit dot-product variant when the device has one. Set to false to force the portable variant (which also rules out the
    /// matrix-hardware variant of Apple GPUs), for example to compare the variants or to rule the hardware out when chasing a problem.
    /// </summary>
    public bool AllowPackedDot { get; init; } = true;

    /// <summary>Test hook for the packed-dot variant: 0 chooses the tile by problem size, 32 or 64 forces that tile edge.</summary>
    internal int Tile { get; init; }

    /// <summary>Test hook: false keeps the matrix-hardware variant off while leaving the packed-dot variant on.</summary>
    internal bool AllowMatrixHardware { get; init; } = true;

}

/// <summary>A vector or single value operand of an int8 product: a zero point, a scale or a bias.</summary>
/// <typeparam name="T">The element type of the buffer.</typeparam>
/// <param name="Buffer">The buffer holding the values.</param>
/// <param name="Offset">Element offset of the first value.</param>
/// <param name="IsVector">True for one value per row of A (zero point of A) or per column (zero point of B, scale, bias); false for a single value at <paramref name="Offset"/>. A bias is always one per column.</param>
public readonly record struct Int8Operand<T>(GpuBuffer<T> Buffer, int Offset, bool IsVector = false)
    where T : unmanaged;

/// <summary>
/// A caller-owned scratch region for the int32 sums that the float and requantized forms of <see cref="Int8Gemm"/> keep between the product and its
/// last pass. Supplying one means the instance never allocates a buffer of its own, so a bind group built for it stays valid for the whole session.
/// </summary>
/// <param name="Buffer">The buffer; it must not overlap any other operand of the call.</param>
/// <param name="Offset">Element (int32) offset of the region's first element.</param>
public readonly record struct Int8Scratch(GpuBuffer<int> Buffer, int Offset = 0);

/// <summary>
/// The shape of one batch of 8-bit products <c>C[m, n] = sum over k of (A[m, k] - zeroA) * (B[k, n] - zeroB)</c>: A is M x K, B is K x N (or N x K
/// with <see cref="TransposeB"/>), all dense row-major, counted in elements (bytes). The result is dense: <c>[Batch, M, N]</c>.
/// </summary>
public struct Int8GemmProblem
{
    /// <summary>Creates a problem with a single batch and dense leading dimensions.</summary>
    /// <param name="m">Rows of A and of C.</param>
    /// <param name="n">Columns of B and of C.</param>
    /// <param name="k">Columns of A, rows of B.</param>
    public Int8GemmProblem(int m, int n, int k)
    {
        M = m;
        N = n;
        K = k;
        Batch = 1;
    }

    /// <summary>Rows of A and of C. At least 0.</summary>
    public int M { get; set; }

    /// <summary>Columns of B and of C. At least 0.</summary>
    public int N { get; set; }

    /// <summary>Inner dimension. At least 0; with 0 every sum is 0.</summary>
    public int K { get; set; }

    /// <summary>Number of independent products, at least 1.</summary>
    public int Batch { get; set; }

    /// <summary>B is stored N x K instead of K x N.</summary>
    public bool TransposeB { get; set; }

    /// <summary>Row stride of A; 0 means dense (K).</summary>
    public int Lda { get; set; }

    /// <summary>Row stride of the stored B; 0 means dense (N, or K when <see cref="TransposeB"/>).</summary>
    public int Ldb { get; set; }

    /// <summary>Index of A's first byte in its buffer.</summary>
    public int OffsetA { get; set; }

    /// <summary>Index of B's first byte in its buffer.</summary>
    public int OffsetB { get; set; }

    /// <summary>Index of the result's first element in its buffer.</summary>
    public int OffsetC { get; set; }

    /// <summary>Bytes between the A of one product and the next. 0 shares one A across the batch.</summary>
    public int StrideA { get; set; }

    /// <summary>Bytes between the B of one product and the next. 0 shares one B across the batch, as a weight matrix is.</summary>
    public int StrideB { get; set; }

    /// <summary>Multiply-adds of the product counted as two operations: <c>2 * Batch * M * N * K</c>.</summary>
    public readonly double Ops => 2.0 * Batch * M * N * K;
}

/// <summary>
/// 8-bit integer matrix multiplication with 32-bit accumulation, matching ONNX <c>MatMulInteger</c>, <c>QLinearMatMul</c> and the fused
/// <c>MatMulInteger</c>, <c>Cast</c>, <c>Mul</c> chain, bit for bit.
/// </summary>
/// <remarks>
/// <para>
/// A and B are <c>GpuBuffer&lt;byte&gt;</c> (uint8) or <c>GpuBuffer&lt;sbyte&gt;</c> (int8), in any mix. The sums are exact integers
/// (<c>sum (a - zeroA) * (b - zeroB)</c>, wrapping modulo 2^32 like any two's-complement accumulation), so the result does not depend on the
/// device, the variant or <see cref="GpuKernelMode"/>: there is one variant for both modes. A zero point is one value or a vector (one per row of
/// A, one per column of B) of the operand's own type, held in a device buffer so a dynamically quantized tensor's zero point can be used as it was produced.
/// </para>
/// <para>
/// The three forms differ in what follows the sums, exactly as onemodel's CPU does: <see cref="EncodeInt32{TA, TB}"/> returns them;
/// <see cref="EncodeFloat{TA, TB}"/> gives <c>float(sum + intBias) * scale (+ floatBias)</c> with two separate float roundings (the bias is a pass of its own so the target cannot fuse the two);
/// <see cref="EncodeRequantized{TA, TB, TY}"/> gives <c>clamp(round_half_even(float(sum + intBias) * multiplier) + zeroPointY)</c> to uint8 or int8
/// (NaN to the smallest value). The latter two run a second pass over the int32 sums in a scratch buffer the instance keeps.
/// </para>
/// <para>
/// Variants: <see cref="Variant"/> says which was chosen for the device. The portable variant unpacks bytes with integer arithmetic and runs everywhere.
/// The packed-dot variant, chosen when <see cref="GpuComputeCapabilities.Int8DotProduct"/> is true and the options allow it, multiplies four byte pairs per
/// instruction: operands are brought to the signed domain (an unsigned byte <c>u</c> is staged as <c>u - 128</c>, shifting its zero point by the same amount) and
/// the zero points are applied afterwards from byte sums gathered while staging, which gives the same integers modulo 2^32 as the definition.
/// It uses 64x64 tiles, and 32x32 tiles on a discrete GPU when a 64x64 grid would leave much of the device idle. The portable variant multiplies in float
/// (exact, because every partial sum stays below 2^24) and folds the sums into int32 often enough to keep them exact.
/// </para>
/// Bind groups for buffer sets already seen are cached, so a warm call allocates nothing. Not thread-safe.
/// </remarks>
public sealed class Int8Gemm : IDisposable
{
    const string Name = "Int8Gemm";
    const int GemmWords = 19;
    const int EpilogueWords = 11;

    // On a discrete GPU the packed-dot variant runs a product with fewer 64x64 tiles than this on 32x32 tiles, which put four times as many
    // workgroups on the device. The threshold is below one workgroup per compute unit of a mid-range card (an RTX 4070 has 46): there 128 x 384
    // products gain 20 to 30 percent from the small tile, while 128 x 1536 (48 tiles of 64 x 64) already do better on the big one, which reuses its
    // loads more. An integrated GPU has few compute units and the big tile won on every shape measured there, so it keeps the big tile.
    const long SmallTileGroups = 32;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly bool _dot;
    readonly bool _matrix;
    readonly bool _discrete;
    readonly int _tile;
    readonly WideKernelPipeline?[] _gemm = new WideKernelPipeline?[2];
    WideKernelPipeline? _rows;
    readonly WideKernelPipeline _epilogue;
    readonly GpuBuffer<uint> _dummy;
    readonly List<GpuBuffer<int>> _scratch = [];

    /// <summary>Creates the kernel.</summary>
    /// <param name="device">The device the operands live on.</param>
    /// <param name="options">Variant options; null means the defaults.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    /// <exception cref="GpuUnsupportedException">The device cannot run the portable variant (256-invocation workgroups, 8.5 KB of workgroup memory, five storage buffers).</exception>
    public Int8Gemm(GpuDevice device, Int8GemmOptions? options = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        var compute = device.Capabilities.Compute;
        _dot = compute.Int8DotProduct && (compute.ShaderFeatures & DeviceFeatures.IntegerDotProduct) != 0 && (options?.AllowPackedDot ?? true);
        _matrix = (options?.AllowPackedDot ?? true) && (options?.AllowMatrixHardware ?? true) && !_dot && MatrixVariantFits(device);
        _tile = options?.Tile ?? 0;
        _discrete = device.Adapter.Kind == AdapterKind.Discrete;
        Variant = _dot ? "packed-dot-tile64x64x32" : _matrix ? "simdgroup-matrix-half-tile64x64x16" : "portable-tile64x64x16";
        try
        {
            _gemm[0] = CreateGemm(small: false);
            _epilogue = new WideKernelPipeline(device, KernelSources.Make(Int8Epilogue.Int8EpilogueOp.EntryPoint, Int8Epilogue.Int8EpilogueOp.SpirV, Int8Epilogue.Int8EpilogueOp.Msl, Int8Epilogue.Int8EpilogueOp.MslFunctionName, Int8Epilogue.Int8EpilogueOp.Wgsl, Int8Epilogue.Int8EpilogueOp.WgslEntryPoint, Int8Epilogue.Int8EpilogueOp.Layout), EpilogueWords, "int8 epilogue", _maxBindGroups);
        }
        catch
        {
            _gemm[0]?.Dispose();
            _gemm[1]?.Dispose();
            throw;
        }

        _dummy = device.CreateBuffer<uint>(4, label: "int8 gemm unused operand");
    }

    // The matrix variant stages bytes as float16 and runs 16x16x16 float16 fragments with float32 accumulation (Int8GemmCoop.hlsl), so it needs the device to
    // offer exactly what the float16 matrix product needs plus the memory of its tiles; only Apple GPUs list such fragments (the packed dot, where there is one, wins elsewhere).
    static bool MatrixVariantFits(GpuDevice device)
    {
        const uint TileBytes = ((64 * 24) + (16 * 72)) * 2 + (2 * 4 * 1024);
        var cc = device.Capabilities.Compute;
        return device.Backend == GpuBackend.Metal
            && Gemm.MatrixVariantFits(device, GemmElementType.F16, wide: false)
            && (Int8GemmCoop.Int8GemmMatrix.Layout.RequiredFeatures & ~cc.ShaderFeatures) == DeviceFeatures.None
            && cc.MaxWorkgroupMemoryBytes >= TileBytes;
    }

    const int RowsMaxM = 16;
    const int RowsTileN = 32;
    const int RowsTileM = 4;

    WideKernelPipeline CreateRows() =>
        new(_device, new ShaderSource(ShaderStage.Compute, Int8GemmSkinny.Int8GemmRows.EntryPoint) { Msl = Int8GemmSkinny.Int8GemmRows.Msl, MslFunctionName = Int8GemmSkinny.Int8GemmRows.MslFunctionName, Layout = Int8GemmSkinny.Int8GemmRows.Layout }, GemmWords, "int8 gemm few rows", _maxBindGroups);

    WideKernelPipeline CreateGemm(bool small)
    {
        if (_matrix)
        {
            return new WideKernelPipeline(_device, new ShaderSource(ShaderStage.Compute, Int8GemmCoop.Int8GemmMatrix.EntryPoint) { Msl = Int8GemmCoop.Int8GemmMatrix.Msl, MslFunctionName = Int8GemmCoop.Int8GemmMatrix.MslFunctionName, Layout = Int8GemmCoop.Int8GemmMatrix.Layout }, GemmWords, "int8 gemm matrix hardware", _maxBindGroups, 32);
        }

        if (_dot)
        {
            var v = Int8GemmDot.Variants.Get(new Int8GemmDot.Key(small: small)).Int8GemmDotTiled;
            return new WideKernelPipeline(_device, KernelSources.Make(v.EntryPoint, v.SpirV, v.Msl, v.MslFunctionName, v.Wgsl, v.WgslEntryPoint, v.Layout), GemmWords, small ? "int8 dot gemm 32x32" : "int8 dot gemm 64x64", _maxBindGroups);
        }

        return new WideKernelPipeline(_device, KernelSources.Make(Int8GemmKernel.Int8GemmTiled.EntryPoint, Int8GemmKernel.Int8GemmTiled.SpirV, Int8GemmKernel.Int8GemmTiled.Msl, Int8GemmKernel.Int8GemmTiled.MslFunctionName, Int8GemmKernel.Int8GemmTiled.Wgsl, Int8GemmKernel.Int8GemmTiled.WgslEntryPoint, Int8GemmKernel.Int8GemmTiled.Layout), GemmWords, "int8 gemm", _maxBindGroups);
    }

    /// <summary>The variant chosen for this device, for logs and diagnostics: <c>packed-dot-tile64x64x32</c> or <c>portable-tile64x64x16</c> (the packed one uses 32x32 tiles for small products).</summary>
    public string Variant { get; }

    /// <summary>Records the int32 sums <c>C = (A - zeroA) * (B - zeroB)</c>.</summary>
    /// <typeparam name="TA"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TB"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <param name="encoder">Receives the compute pass; no other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="Int8GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">Receives the dense int32 result at <see cref="Int8GemmProblem.OffsetC"/>.</param>
    /// <param name="zeroPointA">Zero point of A (type of A), or null for 0.</param>
    /// <param name="zeroPointB">Zero point of B (type of B), or null for 0.</param>
    /// <param name="bias">An int32 bias per column added to every sum, or null; always one value per column.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset, range or operand type is invalid; names the argument.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void EncodeInt32<TA, TB>(GpuCommandEncoder encoder, in Int8GemmProblem problem, GpuBuffer<TA> a, GpuBuffer<TB> b, GpuBuffer<int> c, Int8Operand<TA>? zeroPointA = null, Int8Operand<TB>? zeroPointB = null, Int8Operand<int>? bias = null)
        where TA : unmanaged
        where TB : unmanaged
    {
        ArgumentNullException.ThrowIfNull(c);
        var (_, _, ok) = Prepare(encoder, problem, a, b, zeroPointA, zeroPointB);
        var count = (long)problem.Batch * problem.M * problem.N;
        Check.Range(Name, nameof(c), c.Length, problem.OffsetC, count);
        CheckBindable("C", c.Buffer);
        Epilogue epi = default;
        if (bias is { } ib)
        {
            CheckVector(nameof(bias), ib.Buffer, ib.Offset, true, problem.N);
            CheckBindable("bias", ib.Buffer.Buffer);
            epi = new Epilogue(2u, ib.Buffer.Buffer, (uint)ib.Offset, null, 0, null, 0);
        }

        if (!ok)
        {
            return;
        }

        Multiply(encoder, problem, a.Buffer, typeof(TA) == typeof(sbyte), b.Buffer, typeof(TB) == typeof(sbyte), c.Buffer, problem.OffsetC, zeroPointA?.Buffer.Buffer, zeroPointA, zeroPointB?.Buffer.Buffer, zeroPointB, in epi);
    }

    /// <summary>Records the float32 result <c>float(sum + intBias) * scale + floatBias</c> (the bias terms optional), the MatMulIntegerToFloat chain in one step.</summary>
    /// <typeparam name="TA"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TB"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <param name="encoder">Receives the compute passes; no other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="Int8GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="c">Receives the dense float32 result at <see cref="Int8GemmProblem.OffsetC"/>.</param>
    /// <param name="scale">One scale, or one per column, in a device buffer (it may have been computed on the GPU).</param>
    /// <param name="zeroPointA">Zero point of A (type of A), or null for 0.</param>
    /// <param name="zeroPointB">Zero point of B (type of B), or null for 0.</param>
    /// <param name="floatBias">A float32 bias per column added after the scaling, or null.</param>
    /// <param name="intBias">An int32 bias per column added to the sums before the conversion, or null.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset, range or operand type is invalid; names the argument.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void EncodeFloat<TA, TB>(GpuCommandEncoder encoder, in Int8GemmProblem problem, GpuBuffer<TA> a, GpuBuffer<TB> b, GpuBuffer<float> c, Int8Operand<float> scale, Int8Operand<TA>? zeroPointA = null, Int8Operand<TB>? zeroPointB = null, Int8Operand<float>? floatBias = null, Int8Operand<int>? intBias = null)
        where TA : unmanaged
        where TB : unmanaged
    {
        ArgumentNullException.ThrowIfNull(c);
        var (_, _, ok) = Prepare(encoder, problem, a, b, zeroPointA, zeroPointB);
        var count = (long)problem.Batch * problem.M * problem.N;
        Check.Range(Name, nameof(c), c.Length, problem.OffsetC, count);
        CheckBindable("C", c.Buffer);
        CheckVector(nameof(scale), scale.Buffer, scale.Offset, scale.IsVector, problem.N);
        if (floatBias is { } fb)
        {
            CheckVector(nameof(floatBias), fb.Buffer, fb.Offset, true, problem.N);
        }

        if (!ok)
        {
            return;
        }

        // The conversion, scaling and both biases happen as each sum is stored, in the same float steps as the separate passes (see the shader).
        var flags = 1u | (scale.IsVector ? 4u : 0u);
        uint offIntBias = 0, offFloatBias = 0;
        GpuBuffer? intBiasBuffer = null, floatBiasBuffer = null;
        if (intBias is { } ib)
        {
            CheckVector(nameof(intBias), ib.Buffer, ib.Offset, true, problem.N);
            CheckBindable("bias", ib.Buffer.Buffer);
            flags |= 2u;
            intBiasBuffer = ib.Buffer.Buffer;
            offIntBias = (uint)ib.Offset;
        }

        if (floatBias is { } fb2)
        {
            CheckBindable("float bias", fb2.Buffer.Buffer);
            flags |= 8u;
            floatBiasBuffer = fb2.Buffer.Buffer;
            offFloatBias = (uint)fb2.Offset;
        }

        CheckBindable("scale", scale.Buffer.Buffer);
        var epi = new Epilogue(flags, intBiasBuffer, offIntBias, scale.Buffer.Buffer, (uint)scale.Offset, floatBiasBuffer, offFloatBias);
        Multiply(encoder, problem, a.Buffer, typeof(TA) == typeof(sbyte), b.Buffer, typeof(TB) == typeof(sbyte), c.Buffer, problem.OffsetC, zeroPointA?.Buffer.Buffer, zeroPointA, zeroPointB?.Buffer.Buffer, zeroPointB, in epi);
    }

    /// <summary>
    /// Records the requantized result of <c>QLinearMatMul</c>: <c>clamp(round_half_even(float(sum + intBias) * multiplier) + zeroPointY)</c> to uint8 or
    /// int8, with NaN going to the smallest value. The multiplier is <c>scaleA * scaleB / scaleY</c> computed in float32 by the caller.
    /// </summary>
    /// <typeparam name="TA"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TB"><see cref="byte"/> or <see cref="sbyte"/>.</typeparam>
    /// <typeparam name="TY"><see cref="byte"/> or <see cref="sbyte"/>: the output type.</typeparam>
    /// <param name="encoder">Receives the compute passes; no other pass may be open on it.</param>
    /// <param name="problem">The shape; see <see cref="Int8GemmProblem"/>.</param>
    /// <param name="a">The A operand.</param>
    /// <param name="b">The B operand.</param>
    /// <param name="y">Receives the dense 8-bit result at byte offset <see cref="Int8GemmProblem.OffsetC"/>.</param>
    /// <param name="multiplier">One multiplier, or one per column.</param>
    /// <param name="zeroPointY">The output zero point, within the output type's range.</param>
    /// <param name="zeroPointA">Zero point of A (type of A), or null for 0.</param>
    /// <param name="zeroPointB">Zero point of B (type of B), or null for 0.</param>
    /// <param name="intBias">An int32 bias per column added to the sums before the conversion, or null.</param>
    /// <param name="scratch">Where to keep the int32 sums between the two passes (see <see cref="ScratchElements"/>); null uses the instance's own buffer, which grows on demand and so may build bind groups when a larger product arrives.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset, range, zero point or operand type is invalid; names the argument.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void EncodeRequantized<TA, TB, TY>(GpuCommandEncoder encoder, in Int8GemmProblem problem, GpuBuffer<TA> a, GpuBuffer<TB> b, GpuBuffer<TY> y, Int8Operand<float> multiplier, int zeroPointY, Int8Operand<TA>? zeroPointA = null, Int8Operand<TB>? zeroPointB = null, Int8Operand<int>? intBias = null, Int8Scratch? scratch = null)
        where TA : unmanaged
        where TB : unmanaged
        where TY : unmanaged
    {
        ArgumentNullException.ThrowIfNull(y);
        var (_, _, ok) = Prepare(encoder, problem, a, b, zeroPointA, zeroPointB);
        RequireInt8<TY>(nameof(y));
        var outSigned = typeof(TY) == typeof(sbyte);
        if (zeroPointY < (outSigned ? -128 : 0) || zeroPointY > (outSigned ? 127 : 255))
        {
            throw new GpuKernelShapeException(Name, nameof(zeroPointY), $"{zeroPointY} is outside the output type's range");
        }

        var count = (long)problem.Batch * problem.M * problem.N;
        Check.Range(Name, nameof(y), y.Length, problem.OffsetC, count);
        CheckBindable("Y", y.Buffer);
        CheckVector(nameof(multiplier), multiplier.Buffer, multiplier.Offset, multiplier.IsVector, problem.N);
        if (!ok)
        {
            return;
        }

        var (sums, at) = Scratch(count, scratch);
        Epilogue none = default;
        Multiply(encoder, problem, a.Buffer, typeof(TA) == typeof(sbyte), b.Buffer, typeof(TB) == typeof(sbyte), sums, at, zeroPointA?.Buffer.Buffer, zeroPointA, zeroPointB?.Buffer.Buffer, zeroPointB, in none);
        Finish(encoder, problem, sums, at, y.Buffer, problem.OffsetC, mode: 2, zeroPointY, intBias, multiplier.Buffer.Buffer, null, multiplier.Offset, (multiplier.IsVector ? 2u : 0u) | (outSigned ? 0u : 8u));
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => WideKernelPipeline.Total(_gemm) + (_rows?.Created ?? 0) + _epilogue.Created;

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        _gemm[0]?.ReleaseBindings();
        _gemm[1]?.ReleaseBindings();
        _rows?.ReleaseBindings();
        _epilogue.ReleaseBindings();
    }

    /// <summary>Releases the pipelines, the scratch buffers and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _gemm[0]?.Dispose();
        _gemm[1]?.Dispose();
        _rows?.Dispose();
        _epilogue.Dispose();
        _dummy.Dispose();
        foreach (var s in _scratch)
        {
            s.Dispose();
        }

        _scratch.Clear();
    }

    static void RequireInt8<T>(string argument)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(sbyte))
        {
            throw new GpuKernelShapeException(Name, argument, $"element type {typeof(T).Name} is not supported; use byte (uint8) or sbyte (int8)");
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

    static void CheckVector<T>(string name, GpuBuffer<T> buffer, int offset, bool isVector, int n)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        Check.Range(Name, name, buffer.Length, offset, isVector ? n : 1);
    }

    // Validates the operands common to the three forms. Returns false when there is nothing to compute.
    (int Lda, int Ldb, bool Ok) Prepare<TA, TB>(GpuCommandEncoder encoder, in Int8GemmProblem p, GpuBuffer<TA> a, GpuBuffer<TB> b, Int8Operand<TA>? zpA, Int8Operand<TB>? zpB)
        where TA : unmanaged
        where TB : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        RequireInt8<TA>(nameof(a));
        RequireInt8<TB>(nameof(b));
        NonNegative("M", p.M);
        NonNegative("N", p.N);
        NonNegative("K", p.K);
        NonNegative("OffsetA", p.OffsetA);
        NonNegative("OffsetB", p.OffsetB);
        NonNegative("OffsetC", p.OffsetC);
        NonNegative("StrideA", p.StrideA);
        NonNegative("StrideB", p.StrideB);
        NonNegative("Lda", p.Lda);
        NonNegative("Ldb", p.Ldb);

        if (p.Batch < 1)
        {
            throw new GpuKernelShapeException(Name, "Batch", $"must be at least 1, got {p.Batch}");
        }

        var lda = p.Lda == 0 ? Math.Max(p.K, 1) : p.Lda;
        var colsB = p.TransposeB ? p.K : p.N;
        var rowsB = p.TransposeB ? p.N : p.K;
        var ldb = p.Ldb == 0 ? Math.Max(colsB, 1) : p.Ldb;
        if (lda < p.K)
        {
            throw new GpuKernelShapeException(Name, "Lda", $"{lda} is smaller than K = {p.K}");
        }

        if (ldb < colsB)
        {
            throw new GpuKernelShapeException(Name, "Ldb", $"{ldb} is smaller than the stored row length {colsB}");
        }

        if (checked((long)p.Batch * p.M * p.N) > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, "Batch", "the result holds more elements than the kernels address");
        }

        if (p.M > 0 && p.N > 0 && p.K > 0)
        {
            CheckRange("A", p.OffsetA, p.StrideA, p.M, p.K, lda, p.Batch, a.Length);
            CheckRange("B", p.OffsetB, p.StrideB, rowsB, colsB, ldb, p.Batch, b.Length);
        }

        if (zpA is { } za)
        {
            CheckVector(nameof(zpA), za.Buffer, za.Offset, za.IsVector, p.M);
            CheckBindable("zero point A", za.Buffer.Buffer);
        }

        if (zpB is { } zb)
        {
            CheckVector(nameof(zpB), zb.Buffer, zb.Offset, zb.IsVector, p.N);
            CheckBindable("zero point B", zb.Buffer.Buffer);
        }

        CheckBindable("A", a.Buffer);
        CheckBindable("B", b.Buffer);
        return (lda, ldb, p.M > 0 && p.N > 0);
    }

    static void NonNegative(string argument, int value)
    {
        if (value < 0)
        {
            throw new GpuKernelShapeException(Name, argument, $"must not be negative, got {value}");
        }
    }

    static void CheckRange(string operand, int offset, int stride, int rows, int cols, int ld, int batch, int length)
    {
        var last = offset + ((long)(batch - 1) * stride) + ((long)(rows - 1) * ld) + (cols - 1);
        if (last >= length)
        {
            throw new GpuKernelShapeException(Name, "Offset" + operand, $"operand {operand} reaches byte {last} but its buffer holds {length}");
        }
    }

    (GpuBuffer Buffer, int Offset) Scratch(long count, Int8Scratch? given)
    {
        if (given is { } g)
        {
            ArgumentNullException.ThrowIfNull(g.Buffer, nameof(given));
            Check.Range(Name, "scratch", g.Buffer.Length, g.Offset, count);
            CheckBindable("scratch", g.Buffer.Buffer);
            return (g.Buffer.Buffer, g.Offset);
        }

        if (_scratch.Count > 0 && _scratch[^1].Length >= count)
        {
            return (_scratch[^1].Buffer, 0);
        }

        // Grow geometrically. A smaller buffer stays alive until Dispose because commands already recorded may still name it.
        var length = (int)Math.Min(Check.MaxElements, Math.Max(count, _scratch.Count > 0 ? (long)_scratch[^1].Length * 2 : 4096));
        var buffer = _device.CreateBuffer<int>(length, label: "int8 gemm sums");
        _scratch.Add(buffer);
        return (buffer.Buffer, 0);
    }

    /// <summary>
    /// The int32 elements of scratch that <see cref="EncodeRequantized{TA, TB, TY}"/> needs for a product: <c>Batch * M * N</c>. The int32 and float forms
    /// apply their epilogue as each sum is stored, so they keep nothing between passes.
    /// </summary>
    /// <param name="problem">The product.</param>
    /// <returns>Elements (4 bytes each), 0 when any of M, N or Batch is not positive.</returns>
    public static long ScratchElements(in Int8GemmProblem problem) => problem.M <= 0 || problem.N <= 0 || problem.Batch <= 0 ? 0 : (long)problem.Batch * problem.M * problem.N;

    /// <summary>
    /// Makes the instance's own scratch at least <paramref name="elements"/> int32 elements long now, so that later calls with no <see cref="Int8Scratch"/> up to that size build no buffer
    /// (and so no bind group) mid-session. A consumer that supplies <see cref="Int8Scratch"/> never needs it.
    /// </summary>
    /// <param name="elements">The largest <see cref="ScratchElements"/> the session will ask for.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="elements"/> is negative or more than the kernels address.</exception>
    public void ReserveScratch(long elements)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(elements, Check.MaxElements);
        if (elements == 0 || (_scratch.Count > 0 && _scratch[^1].Length >= elements))
        {
            return;
        }

        _scratch.Add(_device.CreateBuffer<int>((int)Math.Max(elements, 4096), label: "int8 gemm sums"));
    }

    // What the product's store applies to each sum: bit 0 float result, bit 1 int bias, bit 2 scale per column, bit 3 float bias. Zero stores the raw int32 sum.
    readonly record struct Epilogue(uint Flags, GpuBuffer? IntBias, uint OffIntBias, GpuBuffer? Scale, uint OffScale, GpuBuffer? FloatBias, uint OffFloatBias);

    void Multiply<TA, TB>(GpuCommandEncoder encoder, in Int8GemmProblem p, GpuBuffer a, bool aSigned, GpuBuffer b, bool bSigned, GpuBuffer c, int offsetC, GpuBuffer? zaBuffer, Int8Operand<TA>? za, GpuBuffer? zbBuffer, Int8Operand<TB>? zb, in Epilogue epi)
        where TA : unmanaged
        where TB : unmanaged
    {
        var lda = p.Lda == 0 ? Math.Max(p.K, 1) : p.Lda;
        var ldb = p.Ldb == 0 ? Math.Max(p.TransposeB ? p.K : p.N, 1) : p.Ldb;
        var big = ((long)((p.N + 63) / 64) * ((p.M + 63) / 64)) * p.Batch;
        var small = _dot && (_tile == 32 || (_tile == 0 && _discrete && big < SmallTileGroups));
        // Few rows against weights stored K x N: the weights stream through once, so the row kernel (32 columns by 4 rows per workgroup) beats a 64 x 64 tile.
        var rows = _matrix && !p.TransposeB && p.M <= RowsMaxM && ((p.N + RowsTileN - 1) / RowsTileN) <= (long)(_device.Capabilities.Compute.MaxWorkgroupCount == 0 ? 65535u : _device.Capabilities.Compute.MaxWorkgroupCount);
        var edge = small ? 32 : 64;
        var groupsX = (uint)((p.N + (rows ? RowsTileN : edge) - 1) / (rows ? RowsTileN : edge));
        var groupsY = (uint)((p.M + (rows ? RowsTileM : edge) - 1) / (rows ? RowsTileM : edge));
        var limit = _device.Capabilities.Compute.MaxWorkgroupCount;
        var fold = false;
        if (limit != 0 && (groupsX > limit || groupsY > limit))
        {
            // More tiles along one axis than a grid dimension holds (a very wide or very tall product): number the tiles row-major over x and y.
            (groupsX, groupsY) = Grid.For(_device, (long)groupsX * groupsY);
            fold = true;
        }

        if (limit != 0 && (groupsX > limit || groupsY > limit || (uint)p.Batch > limit))
        {
            throw new GpuKernelShapeException(Name, "M, N or Batch", $"the grid of {groupsX} x {groupsY} x {p.Batch} workgroups exceeds the device limit of {limit} per axis");
        }

        var flags = (aSigned ? 1u : 0u) | (bSigned ? 2u : 0u) | (p.TransposeB ? 4u : 0u);
        uint offZa = 0, offZb = 0;
        if (za is { } a1)
        {
            flags |= 8u | (a1.IsVector ? 16u : 0u);
            offZa = (uint)a1.Offset;
        }

        if (zb is { } b1)
        {
            flags |= 32u | (b1.IsVector ? 64u : 0u);
            offZb = (uint)b1.Offset;
        }

        // Word loads need every row of the operand to start on a word boundary.
        if (((p.OffsetA | p.StrideA | lda) & 3) == 0)
        {
            flags |= 128u;
        }

        if (((p.OffsetB | p.StrideB | ldb) & 3) == 0)
        {
            flags |= 256u;
        }

        if (fold)
        {
            flags |= 512u;
        }

        if (_matrix && !rows)
        {
            flags |= ExactSteps(aSigned && za is null, bSigned && zb is null) << 16;
        }

        Span<uint> w = stackalloc uint[GemmWords];
        w[0] = (uint)p.M;
        w[1] = (uint)p.N;
        w[2] = (uint)p.K;
        w[3] = (uint)p.OffsetA;
        w[4] = (uint)p.OffsetB;
        w[5] = (uint)offsetC;
        w[6] = (uint)p.StrideA;
        w[7] = (uint)p.StrideB;
        w[8] = (uint)((long)p.M * p.N);
        w[9] = (uint)lda;
        w[10] = (uint)ldb;
        w[11] = flags;
        w[12] = offZa;
        w[13] = offZb;
        w[14] = epi.Flags;
        w[15] = epi.OffIntBias;
        w[16] = epi.OffScale;
        w[17] = epi.OffFloatBias;
        w[18] = groupsX;
        var dummy = _dummy.Buffer;
        var pipeline = rows ? _rows ??= CreateRows() : _gemm[small ? 1 : 0] ??= CreateGemm(small);
        pipeline.Record(encoder, w, groupsX, groupsY, (uint)p.Batch, [a, b, c, zaBuffer ?? dummy, zbBuffer ?? dummy, epi.IntBias ?? dummy, epi.Scale ?? dummy, epi.FloatBias ?? dummy]);
    }

    // How many 16-wide steps of k the matrix variant may add in float32 before it must fold into int32: a term is a product of two values that are
    // each at most 255 in magnitude (a byte minus a zero point), or 128 for a plain int8, and every partial sum of integers stays exact while it is at most 2^24.
    static uint ExactSteps(bool aPlainInt8, bool bPlainInt8)
    {
        var bound = (aPlainInt8 ? 128L : 255L) * (bPlainInt8 ? 128L : 255L);
        return (uint)Math.Min(64L, (1L << 24) / bound / 16);
    }

    void Finish(GpuCommandEncoder encoder, in Int8GemmProblem p, GpuBuffer sums, int offsetSums, GpuBuffer output, int offsetOutput, uint mode, int zeroPointY, Int8Operand<int>? intBias, GpuBuffer? scale, Int8Operand<float>? floatBias, int offsetScale, uint scaleFlags)
    {
        var count = (long)p.Batch * p.M * p.N;
        var flags = scaleFlags;
        uint offBias = 0, offFBias = 0;
        if (intBias is { } ib)
        {
            CheckVector(nameof(intBias), ib.Buffer, ib.Offset, true, p.N);
            flags |= 1u;
            offBias = (uint)ib.Offset;
        }

        if (floatBias is { } fb)
        {
            offFBias = (uint)fb.Offset;
        }

        var threads = mode == 2 ? (((long)offsetOutput + count - 1) >> 2) - (offsetOutput >> 2) + 1 : count;
        var (gx, gy) = Grid.For(_device, (threads + 255) / 256);
        Span<uint> w = stackalloc uint[EpilogueWords];
        w[0] = (uint)count;
        w[1] = (uint)offsetSums;
        w[2] = (uint)offsetOutput;
        w[3] = gx;
        w[4] = (uint)p.N;
        w[5] = mode;
        w[6] = flags;
        w[7] = offBias;
        w[8] = (uint)offsetScale;
        w[9] = offFBias;
        w[10] = unchecked((uint)zeroPointY);
        var dummy = _dummy.Buffer;
        _epilogue.Record(encoder, w, gx, gy, 1, [sums, output, intBias?.Buffer.Buffer ?? dummy, scale ?? dummy, floatBias?.Buffer.Buffer ?? dummy]);
    }
}
