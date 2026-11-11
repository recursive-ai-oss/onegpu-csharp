using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Interpolation of <see cref="ResizeKernels"/>.</summary>
public enum GpuResizeMode
{
    /// <summary>The nearest source element, chosen by <see cref="GpuResizeNearest"/>. Copies bits.</summary>
    Nearest = 0,

    /// <summary>Linear along each axis (bilinear in two): two taps per axis.</summary>
    Linear = 1,
}

/// <summary>How an output position maps to an input coordinate (ONNX <c>coordinate_transformation_mode</c>); <c>j</c> is the output index.</summary>
public enum GpuResizeCoordinate
{
    /// <summary><c>(j + 0.5) / scale - 0.5</c>, the ONNX default.</summary>
    HalfPixel = 0,

    /// <summary><c>half_pixel_symmetric</c>: half pixel with the output's centre kept on the input's.</summary>
    HalfPixelSymmetric = 1,

    /// <summary>As <see cref="HalfPixel"/>, but 0 when the output extent is 1.</summary>
    PytorchHalfPixel = 2,

    /// <summary><c>j * (in - 1) / (out - 1)</c>: corners map to corners.</summary>
    AlignCorners = 3,

    /// <summary><c>j / scale</c>.</summary>
    Asymmetric = 4,
}

/// <summary>How a fractional coordinate picks a source element in nearest mode (ONNX <c>nearest_mode</c>).</summary>
public enum GpuResizeNearest
{
    /// <summary>Round to the nearest; ties go down.</summary>
    RoundPreferFloor = 0,

    /// <summary>Round to the nearest; ties go up.</summary>
    RoundPreferCeil = 1,

    /// <summary>Round down.</summary>
    Floor = 2,

    /// <summary>Round up.</summary>
    Ceil = 3,
}

/// <summary>
/// The shape and rules of a resize over the two trailing axes of a tensor, to be turned into a <see cref="ResizePlan"/>. Leading dimensions are not resized and are
/// treated as independent planes: a <c>[N, C, H, W]</c> tensor has <c>N * C</c> planes, a <c>[N, C, L]</c> tensor has <see cref="Height"/> 1 and <see cref="OutHeight"/> 1.
/// </summary>
public struct ResizeProblem
{
    /// <summary>Creates a problem with the ONNX defaults: half-pixel coordinates, round-prefer-floor, scales derived from the extents.</summary>
    /// <param name="planes">Independent planes (batch times channels), at least 1.</param>
    /// <param name="height">Input height, at least 1.</param>
    /// <param name="width">Input width, at least 1.</param>
    /// <param name="outHeight">Output height, at least 1.</param>
    /// <param name="outWidth">Output width, at least 1.</param>
    /// <param name="mode">Interpolation.</param>
    public ResizeProblem(int planes, int height, int width, int outHeight, int outWidth, GpuResizeMode mode)
    {
        Planes = planes;
        Height = height;
        Width = width;
        OutHeight = outHeight;
        OutWidth = outWidth;
        Mode = mode;
    }

    /// <summary>Independent planes, at least 1.</summary>
    public int Planes { get; set; }

    /// <summary>Input height.</summary>
    public int Height { get; set; }

    /// <summary>Input width.</summary>
    public int Width { get; set; }

    /// <summary>Output height.</summary>
    public int OutHeight { get; set; }

    /// <summary>Output width.</summary>
    public int OutWidth { get; set; }

    /// <summary>Interpolation.</summary>
    public GpuResizeMode Mode { get; set; }

    /// <summary>The coordinate transformation.</summary>
    public GpuResizeCoordinate Coordinate { get; set; }

    /// <summary>The rounding of nearest mode.</summary>
    public GpuResizeNearest Nearest { get; set; }

    /// <summary>The vertical scale the model gave (ONNX <c>scales</c>); 0 means output extent / input extent, as when the model gives <c>sizes</c>.</summary>
    public float ScaleHeight { get; set; }

    /// <summary>The horizontal scale the model gave; 0 means output extent / input extent.</summary>
    public float ScaleWidth { get; set; }
}

/// <summary>
/// The source positions and weights of one resize, computed on the host in float32 exactly as the ONNX coordinate rules define them and kept in a small device buffer.
/// Create it once per node (its shapes do not change) and reuse it for every run. Dispose it when the node goes away, after the GPU has finished with it.
/// </summary>
public sealed class ResizePlan : IDisposable
{
    internal ResizePlan(GpuDevice device, ResizeProblem problem, int taps, GpuBuffer<uint> table, uint tableY, uint tableX)
    {
        Device = device;
        Problem = problem;
        Taps = taps;
        Table = table;
        TableY = tableY;
        TableX = tableX;
    }

    internal GpuDevice Device { get; }

    internal GpuBuffer<uint> Table { get; }

    internal int Taps { get; }

    internal uint TableY { get; }

    internal uint TableX { get; }

    /// <summary>The shape and rules the plan was made for.</summary>
    public ResizeProblem Problem { get; }

    /// <summary>Releases the device buffer. The GPU must have finished the commands that use it.</summary>
    public void Dispose() => Table.Dispose();
}

/// <summary>
/// Nearest and linear resizing (ONNX <c>Resize</c>, <c>Upsample</c>) of the two trailing axes for float32 and float16 (float32 arithmetic).
/// </summary>
/// <remarks>
/// <para>
/// The coordinate arithmetic is done once on the host, in float32 with the same operations the ONNX documentation gives, because a GPU's float division is not
/// correctly rounded and a tie in nearest mode must break the same way everywhere. <see cref="CreatePlan"/> makes the table; <see cref="Encode{T}"/> records one thread per output element
/// that sums its taps along columns, then rows, in increasing tap order (taps of weight exactly zero are skipped so an infinity beside an exact hit stays out). Nearest copies
/// bits. Linear output differs from a one-axis-at-a-time evaluation only by rounding. The cubic mode, antialiasing, <c>exclude_outside</c> and <c>tf_crop_and_resize</c> are not implemented.
/// </para>
/// <para>
/// There is no reduction across threads, so both <see cref="GpuKernelMode"/> values give the same bits. Not thread-safe.
/// </para>
/// </remarks>
public sealed class ResizeKernels : IDisposable
{
    const string Name = "Resize";

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _pipelines = new KernelPipeline?[2];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public ResizeKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Computes the source positions and weights of a resize and uploads them.</summary>
    /// <param name="problem">The extents and rules.</param>
    /// <returns>The plan; the caller disposes it.</returns>
    /// <exception cref="GpuKernelShapeException">An extent is below 1, a scale is negative, NaN or infinite, or an enum value is unknown. Names the argument.</exception>
    public ResizePlan CreatePlan(in ResizeProblem problem)
    {
        var p = problem;
        if (p.Planes < 1 || p.Height < 1 || p.Width < 1 || p.OutHeight < 1 || p.OutWidth < 1)
        {
            throw new GpuKernelShapeException(Name, "extents", $"planes {p.Planes}, input {p.Height}x{p.Width} and output {p.OutHeight}x{p.OutWidth} must be at least 1");
        }

        if (p.Mode is not (GpuResizeMode.Nearest or GpuResizeMode.Linear))
        {
            throw new GpuKernelShapeException(Name, nameof(p.Mode), $"unknown mode {(int)p.Mode}");
        }

        if (p.Coordinate is < GpuResizeCoordinate.HalfPixel or > GpuResizeCoordinate.Asymmetric)
        {
            throw new GpuKernelShapeException(Name, nameof(p.Coordinate), $"unknown coordinate transformation {(int)p.Coordinate}");
        }

        if (p.Nearest is < GpuResizeNearest.RoundPreferFloor or > GpuResizeNearest.Ceil)
        {
            throw new GpuKernelShapeException(Name, nameof(p.Nearest), $"unknown nearest mode {(int)p.Nearest}");
        }

        CheckScale(nameof(p.ScaleHeight), p.ScaleHeight);
        CheckScale(nameof(p.ScaleWidth), p.ScaleWidth);
        var taps = p.Mode == GpuResizeMode.Linear ? 2 : 1;
        var scaleY = p.ScaleHeight != 0f ? p.ScaleHeight : (float)p.OutHeight / p.Height;
        var scaleX = p.ScaleWidth != 0f ? p.ScaleWidth : (float)p.OutWidth / p.Width;
        var words = 2L * taps * (p.OutHeight + (long)p.OutWidth);
        if (words > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, "extents", "the output extents are too large");
        }

        var table = new uint[(int)words];
        Fill(table.AsSpan(0, 2 * taps * p.OutHeight), p, p.Height, p.OutHeight, scaleY, taps);
        var tableX = (uint)(2 * taps * p.OutHeight);
        Fill(table.AsSpan((int)tableX), p, p.Width, p.OutWidth, scaleX, taps);
        var buffer = _device.CreateBuffer<uint>(table);
        return new ResizePlan(_device, p, taps, buffer, 0, tableX);
    }

    /// <summary>Records the resize of <paramref name="x"/> into <paramref name="y"/>.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">Receives the pass. No other pass may be open on it.</param>
    /// <param name="plan">The plan made for this shape and these rules.</param>
    /// <param name="x">Input, <c>Planes * Height * Width</c> elements.</param>
    /// <param name="offsetX">Index of the input's first element.</param>
    /// <param name="y">Receives <c>Planes * OutHeight * OutWidth</c> elements.</param>
    /// <param name="offsetY">Index of the output's first element.</param>
    /// <exception cref="GpuKernelShapeException">A range reaches past its buffer, a tensor is too large, or the plan belongs to another device.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, ResizePlan plan, GpuBuffer<T> x, int offsetX, GpuBuffer<T> y, int offsetY)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var half = KernelTypes.IsHalf<T>(Name);
        if (!ReferenceEquals(plan.Device, _device))
        {
            throw new GpuKernelShapeException(Name, nameof(plan), "the plan was made for another device");
        }

        var p = plan.Problem;
        long inCount = (long)p.Planes * p.Height * p.Width, outCount = (long)p.Planes * p.OutHeight * p.OutWidth;
        if (inCount > Check.MaxElements || outCount > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(x), $"the tensors hold more than {Check.MaxElements} elements");
        }

        Check.Range(Name, nameof(x), x.Length, offsetX, inCount, half);
        Check.Range(Name, nameof(y), y.Length, offsetY, outCount, half);
        var pipe = _pipelines[half ? 1 : 0] ??= Create(half);
        Span<uint> w = stackalloc uint[13];
        w[0] = (uint)outCount;
        w[1] = (uint)offsetX;
        w[2] = (uint)offsetY;
        w[4] = (uint)p.Height;
        w[5] = (uint)p.Width;
        w[6] = (uint)p.OutHeight;
        w[7] = (uint)p.OutWidth;
        w[8] = (uint)plan.Taps;
        w[9] = (uint)plan.Taps;
        w[10] = plan.TableY;
        w[11] = plan.TableX;
        w[12] = 0;
        Launch.Record(_device, encoder, pipe, w, outCount, offsetY, half ? 1 : 0, 3, x.Buffer, y.Buffer, plan.Table.Buffer);
    }

    static void CheckScale(string name, float scale)
    {
        if (!float.IsFinite(scale) || scale < 0f)
        {
            throw new GpuKernelShapeException(Name, name, $"scale {scale} must be finite and not negative (0 derives it from the extents)");
        }
    }

    // Per output position j, `taps` pairs (source index, weight bits). The arithmetic is the ONNX Resize definition in float32.
    static void Fill(Span<uint> table, in ResizeProblem p, int inN, int outN, float scale, int taps)
    {
        for (var j = 0; j < outN; j++)
        {
            var xo = Coordinate(p.Coordinate, j, inN, outN, scale);
            var b = j * taps * 2;
            if (p.Mode == GpuResizeMode.Nearest)
            {
                var f = MathF.Floor(xo);
                var pick = p.Nearest switch
                {
                    GpuResizeNearest.Floor => (int)f,
                    GpuResizeNearest.Ceil => (int)MathF.Ceiling(xo),
                    GpuResizeNearest.RoundPreferCeil => xo - f >= 0.5f ? (int)f + 1 : (int)f,
                    _ => xo - f > 0.5f ? (int)f + 1 : (int)f,
                };
                table[b] = (uint)Math.Clamp(pick, 0, inN - 1);
                table[b + 1] = BitConverter.SingleToUInt32Bits(1f);
            }
            else
            {
                var c = Math.Clamp(xo, 0f, inN - 1);
                var f = MathF.Floor(c);
                var t = c - f;
                var i0 = (int)f;
                table[b] = (uint)i0;
                table[b + 1] = BitConverter.SingleToUInt32Bits(1f - t);
                table[b + 2] = (uint)Math.Min(i0 + 1, inN - 1);
                table[b + 3] = BitConverter.SingleToUInt32Bits(t);
            }
        }
    }

    static float Coordinate(GpuResizeCoordinate mode, int j, int inN, int outN, float scale)
    {
        switch (mode)
        {
            case GpuResizeCoordinate.HalfPixel:
                return (((float)j + 0.5f) / scale) - 0.5f;
            case GpuResizeCoordinate.PytorchHalfPixel:
                return outN > 1 ? (((float)j + 0.5f) / scale) - 0.5f : 0f;
            case GpuResizeCoordinate.AlignCorners:
                return outN == 1 ? 0f : (float)j * (inN - 1) / (outN - 1);
            case GpuResizeCoordinate.Asymmetric:
                return (float)j / scale;
            default:
                {
                    var outputWidth = scale * inN;
                    var adjustment = outN / outputWidth;
                    var center = inN / 2f;
                    var offset = center * (1f - adjustment);
                    return offset + ((((float)j + 0.5f) / scale) - 0.5f);
                }
        }
    }

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => KernelPipeline.Total(_pipelines);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        foreach (var p in _pipelines)
        {
            p?.ReleaseBindings();
        }
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        for (var i = 0; i < _pipelines.Length; i++)
        {
            _pipelines[i]?.Dispose();
            _pipelines[i] = null;
        }
    }

    KernelPipeline Create(bool half)
    {
        var e = Resize.Variants.Get(new Resize.Key(f16: half)).ResizeOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, 13, "resize", _maxBindGroups);
    }
}
