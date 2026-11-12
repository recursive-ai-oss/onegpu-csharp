using OneShader.Reflection;
using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Integer convolution, ONNX <c>ConvInteger</c>: <c>y[n, m, oh, ow] = sum over c, kh, kw of (x[...] - xZeroPoint) * (w[m, c, kh, kw] - wZeroPoint[m])</c>
/// in 32-bit integers (wrapping modulo 2^32), bit for bit the result of onemodel's CPU kernel.
/// </summary>
/// <remarks>
/// <para>
/// The input and the weights are <c>GpuBuffer&lt;byte&gt;</c> (uint8) or <c>GpuBuffer&lt;sbyte&gt;</c> (int8), in any mix; the output is int32. The geometry is
/// <see cref="ConvProblem"/>'s: NCHW, OIHW, strides, dilations, asymmetric padding and groups; a one-dimensional convolution is height 1 (<see cref="ConvProblem.Conv1D"/>).
/// <see cref="ConvProblem.Activation"/> must be left at its default and <see cref="ConvProblem.OffsetBias"/> and <see cref="ConvProblem.OffsetResidual"/> are not used.
/// The zero point of <c>x</c> is one value; that of <c>w</c> is one value or one per output channel; both are held in device buffers in their operand's own type, and
/// an absent zero point is 0. A padded position counts as the real value zero, that is as the zero point of <c>x</c>, so it adds nothing.
/// </para>
/// <para>
/// The convolution is lowered to <see cref="Int8Gemm"/> after Chellapilla et al. (2006): one pass unfolds the windows of every image and group into a byte matrix of
/// <c>K = InChannels / Groups * KernelHeight * KernelWidth</c> rows (the matrix's rows are padded to a multiple of four bytes), then the weights of the group are multiplied
/// by it. A convolution with one group makes a single product for the whole batch; with several groups it makes one product per image and group, which is slow for
/// depthwise convolutions with many channels (there is no direct 8-bit depthwise kernel). The sums are exact integers, so the result does not depend on the device, the
/// <see cref="Variant"/> or <see cref="GpuKernelMode"/>: there is one variant for both modes, which uses the packed dot product where the device has one.
/// </para>
/// Bind groups for buffer sets already seen are cached, so a warm call allocates nothing. Not thread-safe.
/// </remarks>
public sealed class ConvIntegerKernels : IDisposable
{
    const string Name = "ConvInteger";
    const int PackWords = 22;

    // Without caller-placed scratch the column matrix of a large batch is limited to this many bytes and the batch runs in chunks of images.
    const long OwnScratchBytes = 64L << 20;

    readonly GpuDevice _device;
    readonly Int8Gemm _gemm;
    readonly KernelPipeline _pack;
    readonly GpuBuffer<uint> _dummy;
    readonly List<IDisposable> _retired = [];
    GpuBuffer<byte>? _ownU8;
    GpuBuffer<sbyte>? _ownI8;

    /// <summary>Creates the kernel.</summary>
    /// <param name="device">The device the operands live on.</param>
    /// <param name="options">Options of the underlying <see cref="Int8Gemm"/>; null means the defaults.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    /// <exception cref="GpuUnsupportedException">The device cannot run the portable variant of <see cref="Int8Gemm"/>.</exception>
    public ConvIntegerKernels(GpuDevice device, Int8GemmOptions? options = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _gemm = new Int8Gemm(device, options, maxCachedBindGroups);
        try
        {
            _pack = new KernelPipeline(device, KernelSources.Make(ConvIntegerPack.ConvIntegerPackOp.EntryPoint, ConvIntegerPack.ConvIntegerPackOp.SpirV, ConvIntegerPack.ConvIntegerPackOp.Msl, ConvIntegerPack.ConvIntegerPackOp.MslFunctionName, ConvIntegerPack.ConvIntegerPackOp.Wgsl, ConvIntegerPack.ConvIntegerPackOp.WgslEntryPoint, ConvIntegerPack.ConvIntegerPackOp.Layout), 3, PackWords, "conv-integer-pack", GpuKernelBindings.Check(maxCachedBindGroups));
        }
        catch
        {
            _gemm.Dispose();
            throw;
        }

        _dummy = device.CreateBuffer<uint>(4, label: "conv integer unused operand");
    }

    /// <summary>The variant of the underlying product chosen for this device; see <see cref="Int8Gemm.Variant"/>.</summary>
    public string Variant => _gemm.Variant;

    /// <summary>How many bind groups this family has built since it was created, for checking that a warm call builds none. Not reset by <see cref="ReleaseBindings"/>.</summary>
    public int BindGroupsCreated => _gemm.BindGroupsCreated + _pack.Created;

    /// <summary>
    /// Bytes of column scratch one image needs: <c>Groups * K * ceil(OutHeight * OutWidth / 4) * 4</c>. A scratch buffer shorter than this cannot run the problem; one
    /// <c>Batch</c> times as long runs the whole batch in one pass, and anything between runs it in chunks of images.
    /// </summary>
    /// <param name="problem">The convolution; its geometry must be valid.</param>
    /// <returns>Bytes (elements of a <c>GpuBuffer&lt;byte&gt;</c>).</returns>
    /// <exception cref="GpuKernelShapeException">The geometry is invalid.</exception>
    public static long ScratchBytesPerImage(in ConvProblem problem)
    {
        Geometry(problem, out _, out _, out var k, out var ldb);
        return checked((long)problem.Groups * k * ldb);
    }

    /// <summary>
    /// Records the convolution.
    /// </summary>
    /// <typeparam name="TX"><see cref="byte"/> or <see cref="sbyte"/>: the input's type.</typeparam>
    /// <typeparam name="TW"><see cref="byte"/> or <see cref="sbyte"/>: the weights' type.</typeparam>
    /// <param name="encoder">Receives the compute passes; no other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of x, w and y (<see cref="ConvProblem.OffsetX"/>, <see cref="ConvProblem.OffsetW"/>, <see cref="ConvProblem.OffsetY"/>, in elements).</param>
    /// <param name="x">The input, <c>[Batch, InChannels, Height, Width]</c>.</param>
    /// <param name="w">The weights, <c>[OutChannels, InChannels / Groups, KernelHeight, KernelWidth]</c>.</param>
    /// <param name="y">Receives the dense int32 result <c>[Batch, OutChannels, OutHeight, OutWidth]</c>.</param>
    /// <param name="xZeroPoint">The single zero point of x (type of x), or null for 0.</param>
    /// <param name="wZeroPoint">The zero point of w (type of w): one value, or one per output channel (<see cref="Int8Operand{T}.IsVector"/>), or null for 0.</param>
    /// <param name="scratch">Where to keep the column matrix (see <see cref="ScratchBytesPerImage"/>), in the input's type; null uses the instance's own buffer, which grows on demand
    /// (to 64 MiB, beyond which the batch runs in chunks) and so may build bind groups when a larger problem arrives. It must not overlap any other operand.</param>
    /// <param name="scratchOffset">Byte offset of the scratch region; a multiple of four.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, offset, range, zero point or operand type is invalid; names the argument.</exception>
    /// <exception cref="GpuUnsupportedException">An operand is larger than one storage-buffer binding can cover on this device.</exception>
    public void Encode<TX, TW>(GpuCommandEncoder encoder, in ConvProblem problem, GpuBuffer<TX> x, GpuBuffer<TW> w, GpuBuffer<int> y, Int8Operand<TX>? xZeroPoint = null, Int8Operand<TW>? wZeroPoint = null, GpuBuffer<TX>? scratch = null, int scratchOffset = 0)
        where TX : unmanaged
        where TW : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(y);
        RequireInt8<TX>(nameof(x));
        RequireInt8<TW>(nameof(w));
        var p = problem;
        Geometry(p, out var oh, out var ow, out var k, out var ldb);
        var nCols = (long)oh * ow;
        if (p.Activation != default)
        {
            throw new GpuKernelShapeException(Name, nameof(p.Activation), "an integer convolution has no fused activation");
        }

        var cinG = p.InChannels / p.Groups;
        var coutG = p.OutChannels / p.Groups;
        var xImage = checked((long)p.InChannels * p.Height * p.Width);
        var yImage = checked(p.OutChannels * nCols);
        Check.Range(Name, nameof(x), x.Length, p.OffsetX, checked(xImage * p.Batch));
        Check.Range(Name, nameof(w), w.Length, p.OffsetW, checked((long)p.OutChannels * k));
        Check.Range(Name, nameof(y), y.Length, p.OffsetY, checked(yImage * p.Batch));
        if (p.OffsetY < 0 || (long)p.OffsetY + (yImage * p.Batch) > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(y), "the result is larger than the supported sizes");
        }

        if (xZeroPoint is { } xz)
        {
            ArgumentNullException.ThrowIfNull(xz.Buffer, nameof(xZeroPoint));
            Check.Range(Name, nameof(xZeroPoint), xz.Buffer.Length, xz.Offset, 1);
            if (xz.IsVector)
            {
                throw new GpuKernelShapeException(Name, nameof(xZeroPoint), "the zero point of x is a single value");
            }
        }

        if (wZeroPoint is { } wz)
        {
            ArgumentNullException.ThrowIfNull(wz.Buffer, nameof(wZeroPoint));
            Check.Range(Name, nameof(wZeroPoint), wz.Buffer.Length, wz.Offset, wz.IsVector ? p.OutChannels : 1);
        }

        var maxBinding = _device.Capabilities.Compute.MaxStorageBufferBindingBytes;
        if (maxBinding != 0 && (x.Buffer.Size > maxBinding || y.Buffer.Size > maxBinding || w.Buffer.Size > maxBinding))
        {
            throw new GpuUnsupportedException($"{Name}: an operand is larger than one storage binding can cover on {_device.Adapter.Name}", nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes), $"at most {maxBinding} bytes", $"{Math.Max(x.Buffer.Size, Math.Max(y.Buffer.Size, w.Buffer.Size))} bytes");
        }

        var perImage = checked((long)p.Groups * k * ldb);
        if (perImage > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, "InChannels", $"the column matrix of one image needs {perImage} bytes, more than the supported sizes");
        }

        GpuBuffer<TX> columns;
        long columnBase;
        long imagesPerChunk;
        if (scratch is not null)
        {
            if (scratchOffset < 0 || (scratchOffset & 3) != 0)
            {
                throw new GpuKernelShapeException(Name, nameof(scratchOffset), $"offset {scratchOffset} must be a non-negative multiple of 4");
            }

            if ((long)scratchOffset + perImage > scratch.Length)
            {
                throw new GpuKernelShapeException(Name, nameof(scratch), $"bytes {scratchOffset}..{(long)scratchOffset + perImage} (one image's columns) reach past the buffer of {scratch.Length}");
            }

            if (maxBinding != 0 && scratch.Buffer.Size > maxBinding)
            {
                throw new GpuUnsupportedException($"{Name}: the scratch is larger than one storage binding can cover on {_device.Adapter.Name}", nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes), $"at most {maxBinding} bytes", $"{scratch.Buffer.Size} bytes");
            }

            columns = scratch;
            columnBase = scratchOffset;
            imagesPerChunk = Math.Min(p.Batch, (scratch.Length - scratchOffset) / perImage);
        }
        else
        {
            imagesPerChunk = Math.Clamp(OwnScratchBytes / perImage, 1, p.Batch);
            columns = Own<TX>(imagesPerChunk * perImage);
            columnBase = 0;
        }

        if (maxBinding != 0 && (ulong)(imagesPerChunk * perImage) > maxBinding)
        {
            imagesPerChunk = Math.Max(1, (long)(maxBinding / (ulong)perImage));
        }

        var zeroBuffer = xZeroPoint is { } z ? z.Buffer.Buffer : _dummy.Buffer;
        var ldb4 = ldb / 4;
        Span<uint> pw = stackalloc uint[PackWords];
        for (var first = 0; first < p.Batch; first += (int)imagesPerChunk)
        {
            var images = (int)Math.Min(imagesPerChunk, p.Batch - first);
            var words = checked((long)images * p.Groups * k * ldb4);
            var (gx, gy) = Grid.For(_device, (words + 255) / 256);
            pw[0] = (uint)words;
            pw[1] = (uint)(p.OffsetX + (first * xImage));
            pw[2] = (uint)(columnBase >> 2);
            pw[3] = gx;
            pw[4] = (uint)cinG;
            pw[5] = (uint)p.InChannels;
            pw[6] = (uint)p.Groups;
            pw[7] = (uint)p.Height;
            pw[8] = (uint)p.Width;
            pw[9] = (uint)oh;
            pw[10] = (uint)ow;
            pw[11] = (uint)p.KernelHeight;
            pw[12] = (uint)p.KernelWidth;
            pw[13] = (uint)p.StrideHeight;
            pw[14] = (uint)p.StrideWidth;
            pw[15] = (uint)p.DilationHeight;
            pw[16] = (uint)p.DilationWidth;
            pw[17] = (uint)p.PadTop;
            pw[18] = (uint)p.PadLeft;
            pw[19] = (uint)ldb4;
            pw[20] = xZeroPoint is null ? 0u : 1u;
            pw[21] = xZeroPoint is { } xo ? (uint)xo.Offset : 0u;
            _pack.Record(encoder, pw, gx, gy, x.Buffer, zeroBuffer, columns.Buffer);

            if (p.Groups == 1)
            {
                var gp = new Int8GemmProblem(p.OutChannels, (int)nCols, (int)k)
                {
                    Batch = images,
                    Ldb = (int)ldb,
                    OffsetA = p.OffsetW,
                    OffsetB = (int)columnBase,
                    OffsetC = (int)(p.OffsetY + (first * yImage)),
                    StrideA = 0,
                    StrideB = (int)(k * ldb),
                };
                _gemm.EncodeInt32(encoder, in gp, w, columns, y, wZeroPoint, xZeroPoint);
                continue;
            }

            for (var img = 0; img < images; img++)
            {
                for (var g = 0; g < p.Groups; g++)
                {
                    var gp = new Int8GemmProblem(coutG, (int)nCols, (int)k)
                    {
                        Ldb = (int)ldb,
                        OffsetA = checked(p.OffsetW + (g * coutG * (int)k)),
                        OffsetB = checked((int)columnBase + (int)(((img * (long)p.Groups) + g) * k * ldb)),
                        OffsetC = checked(p.OffsetY + (int)(((first + img) * yImage) + (g * coutG * nCols))),
                    };
                    Int8Operand<TW>? wzg = wZeroPoint is { } wv && wv.IsVector ? wv with { Offset = wv.Offset + (g * coutG) } : wZeroPoint;
                    _gemm.EncodeInt32(encoder, in gp, w, columns, y, wzg, xZeroPoint);
                }
            }
        }
    }

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        _gemm.ReleaseBindings();
        _pack.ReleaseBindings();
    }

    /// <summary>Releases the pipelines, the scratch buffers and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _gemm.Dispose();
        _pack.Dispose();
        _dummy.Dispose();
        _ownU8?.Dispose();
        _ownI8?.Dispose();
        foreach (var b in _retired)
        {
            b.Dispose();
        }

        _retired.Clear();
    }

    static void RequireInt8<T>(string argument)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(sbyte))
        {
            throw new GpuKernelShapeException(Name, argument, $"element type {typeof(T).Name} is not supported; use byte (uint8) or sbyte (int8)");
        }
    }

    // Validates the geometry and returns the output extent, the rows of the column matrix and the padded length of one of its rows.
    static void Geometry(in ConvProblem p, out int oh, out int ow, out long k, out long ldb)
    {
        oh = p.OutHeight;
        ow = p.OutWidth;
        ConvKernels.CheckConv(in p, oh, ow, Name);
        try
        {
            k = checked((long)(p.InChannels / p.Groups) * p.KernelHeight * p.KernelWidth);
            ldb = (((long)oh * ow) + 3) / 4 * 4;
            _ = checked(p.Batch * (long)p.InChannels * p.Height * p.Width);
        }
        catch (OverflowException)
        {
            throw new GpuKernelShapeException(Name, "shape", "the sizes overflow");
        }

        if (k > Check.MaxElements || ldb > Check.MaxElements || (long)p.Batch * p.InChannels * p.Height * p.Width > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, "shape", "the convolution is larger than the supported sizes");
        }
    }

    // The instance's own column buffer for the element type, grown geometrically; a smaller one stays alive until Dispose because recorded commands may still name it.
    GpuBuffer<T> Own<T>(long bytes)
        where T : unmanaged
    {
        var want = (int)Math.Min(Check.MaxElements, Math.Max(bytes, 4096));
        if (typeof(T) == typeof(byte))
        {
            if (_ownU8 is null || _ownU8.Length < want)
            {
                if (_ownU8 is not null)
                {
                    _retired.Add(_ownU8);
                    want = (int)Math.Min(Check.MaxElements, Math.Max(want, (long)_ownU8.Length * 2));
                }

                _ownU8 = _device.CreateBuffer<byte>(want, label: "conv integer columns");
            }

            return (GpuBuffer<T>)(object)_ownU8;
        }

        if (_ownI8 is null || _ownI8.Length < want)
        {
            if (_ownI8 is not null)
            {
                _retired.Add(_ownI8);
                want = (int)Math.Min(Check.MaxElements, Math.Max(want, (long)_ownI8.Length * 2));
            }

            _ownI8 = _device.CreateBuffer<sbyte>(want, label: "conv integer columns");
        }

        return (GpuBuffer<T>)(object)_ownI8;
    }
}
