using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>What a pooling window computes.</summary>
public enum GpuPoolKind
{
    /// <summary>The maximum of the taps inside the input. NaN taps are skipped unless every tap is NaN; a window with no tap inside the input gives the lowest finite value of the type (-3.4028235e38 for float, -65504 for Half).</summary>
    Max = 0,

    /// <summary>The sum of the taps inside the input divided by their count (ONNX <c>count_include_pad = 0</c>). A window with no tap inside the input gives NaN.</summary>
    Average = 1,

    /// <summary>The sum of the taps inside the input divided by the count of taps inside the padded input (ONNX <c>count_include_pad = 1</c>); taps beyond the padding, which ceil mode can produce, are not counted.</summary>
    AverageIncludePad = 2,
}

/// <summary>
/// The shape of a two-dimensional pooling (ONNX <c>MaxPool</c>, <c>AveragePool</c>) over NCHW tensors, with where its operands sit in their buffers. A one-dimensional pooling
/// has <see cref="Height"/> and <see cref="KernelHeight"/> 1. The output extent is <c>floor((in + padBegin + padEnd - dilation * (kernel - 1) - 1) / stride) + 1</c>, or
/// the ceiling with <see cref="CeilMode"/>.
/// </summary>
public struct PoolProblem
{
    /// <summary>Creates a problem with unit strides and dilations and no padding.</summary>
    /// <param name="batch">Images, at least 1.</param>
    /// <param name="channels">Channels, at least 1.</param>
    /// <param name="height">Input height, at least 1.</param>
    /// <param name="width">Input width, at least 1.</param>
    /// <param name="kernelHeight">Window height, at least 1.</param>
    /// <param name="kernelWidth">Window width, at least 1.</param>
    public PoolProblem(int batch, int channels, int height, int width, int kernelHeight, int kernelWidth)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        StrideHeight = 1;
        StrideWidth = 1;
        DilationHeight = 1;
        DilationWidth = 1;
    }

    /// <summary>Images, at least 1.</summary>
    public int Batch { get; set; }

    /// <summary>Channels, at least 1.</summary>
    public int Channels { get; set; }

    /// <summary>Input height.</summary>
    public int Height { get; set; }

    /// <summary>Input width.</summary>
    public int Width { get; set; }

    /// <summary>Window height.</summary>
    public int KernelHeight { get; set; }

    /// <summary>Window width.</summary>
    public int KernelWidth { get; set; }

    /// <summary>Vertical step, at least 1.</summary>
    public int StrideHeight { get; set; }

    /// <summary>Horizontal step, at least 1.</summary>
    public int StrideWidth { get; set; }

    /// <summary>Vertical spacing of the window's taps, at least 1.</summary>
    public int DilationHeight { get; set; }

    /// <summary>Horizontal spacing of the window's taps, at least 1.</summary>
    public int DilationWidth { get; set; }

    /// <summary>Padding rows above the input.</summary>
    public int PadTop { get; set; }

    /// <summary>Padding columns left of the input.</summary>
    public int PadLeft { get; set; }

    /// <summary>Padding rows below the input.</summary>
    public int PadBottom { get; set; }

    /// <summary>Padding columns right of the input.</summary>
    public int PadRight { get; set; }

    /// <summary>Round the output extent up instead of down (ONNX <c>ceil_mode</c>).</summary>
    public bool CeilMode { get; set; }

    /// <summary>Index of the input's first element in its buffer.</summary>
    public int OffsetX { get; set; }

    /// <summary>Index of the output's first element in its buffer.</summary>
    public int OffsetY { get; set; }

    /// <summary>Output height, or 0 when the window does not fit the padded input.</summary>
    public readonly int OutHeight
    {
        get
        {
            int b = PadTop, e = PadBottom;
            return GpuWindow.OutputExtent(GpuAutoPad.NotSet, Height, KernelHeight, StrideHeight, DilationHeight, CeilMode, ref b, ref e);
        }
    }

    /// <summary>Output width, or 0 when the window does not fit the padded input.</summary>
    public readonly int OutWidth
    {
        get
        {
            int b = PadLeft, e = PadRight;
            return GpuWindow.OutputExtent(GpuAutoPad.NotSet, Width, KernelWidth, StrideWidth, DilationWidth, CeilMode, ref b, ref e);
        }
    }

    /// <summary>Replaces the padding by the one an ONNX <c>auto_pad</c> rule derives for this shape.</summary>
    /// <param name="autoPad">The rule; <see cref="GpuAutoPad.NotSet"/> leaves the padding as it is.</param>
    /// <returns>A copy with the padding set.</returns>
    public readonly PoolProblem WithAutoPad(GpuAutoPad autoPad)
    {
        var p = this;
        int t = PadTop, b = PadBottom, l = PadLeft, r = PadRight;
        GpuWindow.OutputExtent(autoPad, Height, KernelHeight, StrideHeight, DilationHeight, CeilMode, ref t, ref b);
        GpuWindow.OutputExtent(autoPad, Width, KernelWidth, StrideWidth, DilationWidth, CeilMode, ref l, ref r);
        p.PadTop = t;
        p.PadBottom = b;
        p.PadLeft = l;
        p.PadRight = r;
        return p;
    }
}

/// <summary>
/// Max and average pooling and their global forms for float32 and float16 (float32 arithmetic) over NCHW tensors, with ONNX semantics: padding, dilation and ceil mode
/// for the windowed forms, and the NaN rules of <see cref="GpuPoolKind"/>.
/// </summary>
/// <remarks>
/// Windowed pooling gives each output element to one thread, which visits its taps in row-major window order, so the result of every element is fixed by the shape (no
/// atomics, no order that depends on scheduling). The global forms give each (image, channel) plane to one workgroup of 256 threads that folds a fixed strided slice each
/// and combines the 256 partial results by a fixed halving tree in shared memory, so they are deterministic too; both <see cref="GpuKernelMode"/> values give the same bits and are not an
/// argument. Half buffers must end on a 32-bit word (<see cref="GpuKernelBuffers.PaddedHalfLength"/>); the output must not overlap the input. Not thread-safe.
/// </remarks>
public sealed class PoolKernels : IDisposable
{
    const string Name = "Pool";
    const string GlobalName = "GlobalPool";

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _pool = new KernelPipeline?[2];
    readonly KernelPipeline?[] _global = new KernelPipeline?[2];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public PoolKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Records a windowed pooling of <paramref name="x"/> into <paramref name="y"/>.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">Receives the pass. No other pass may be open on it.</param>
    /// <param name="problem">The shape and the offsets of the operands.</param>
    /// <param name="kind">What the window computes.</param>
    /// <param name="x">Input, <c>[Batch, Channels, Height, Width]</c>.</param>
    /// <param name="y">Receives the output, <c>[Batch, Channels, OutHeight, OutWidth]</c>.</param>
    /// <exception cref="GpuKernelShapeException">A dimension, stride, padding or offset is invalid, the window does not fit the padded input, or a range reaches past its buffer. Names the argument.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, in PoolProblem problem, GpuPoolKind kind, GpuBuffer<T> x, GpuBuffer<T> y)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var half = KernelTypes.IsHalf<T>(Name);
        var p = problem;
        if (p.Batch < 1 || p.Channels < 1)
        {
            throw new GpuKernelShapeException(Name, p.Batch < 1 ? nameof(p.Batch) : nameof(p.Channels), $"batch {p.Batch} and channels {p.Channels} must be at least 1");
        }

        if (kind is < GpuPoolKind.Max or > GpuPoolKind.AverageIncludePad)
        {
            throw new GpuKernelShapeException(Name, nameof(kind), $"unknown pooling kind {(int)kind}");
        }

        CheckAxis("height", p.Height, p.KernelHeight, p.StrideHeight, p.DilationHeight, p.PadTop, p.PadBottom);
        CheckAxis("width", p.Width, p.KernelWidth, p.StrideWidth, p.DilationWidth, p.PadLeft, p.PadRight);
        long oh = p.OutHeight, ow = p.OutWidth;
        if (oh < 1 || ow < 1)
        {
            throw new GpuKernelShapeException(Name, oh < 1 ? "height" : "width", "the window does not fit the padded input");
        }

        long planes = (long)p.Batch * p.Channels;
        long inCount = planes * p.Height * p.Width, outCount = planes * oh * ow;
        if (inCount > Check.MaxElements || outCount > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(x), $"the tensors hold more than {Check.MaxElements} elements");
        }

        Check.Range(Name, nameof(x), x.Length, p.OffsetX, inCount, half);
        Check.Range(Name, nameof(y), y.Length, p.OffsetY, outCount, half);
        var pipe = _pool[half ? 1 : 0] ??= Create(half);
        Span<uint> w = stackalloc uint[20];
        w[0] = (uint)outCount;
        w[1] = (uint)p.OffsetX;
        w[2] = (uint)p.OffsetY;
        w[4] = (uint)p.Height;
        w[5] = (uint)p.Width;
        w[6] = (uint)oh;
        w[7] = (uint)ow;
        w[8] = (uint)p.KernelHeight;
        w[9] = (uint)p.KernelWidth;
        w[10] = (uint)p.StrideHeight;
        w[11] = (uint)p.StrideWidth;
        w[12] = (uint)p.DilationHeight;
        w[13] = (uint)p.DilationWidth;
        w[14] = (uint)p.PadTop;
        w[15] = (uint)p.PadLeft;
        w[16] = (uint)p.PadBottom;
        w[17] = (uint)p.PadRight;
        w[18] = kind == GpuPoolKind.Max ? 0u : 1u;
        w[19] = kind == GpuPoolKind.AverageIncludePad ? 1u : 0u;
        Launch.Record(_device, encoder, pipe, w, outCount, p.OffsetY, half ? 1 : 0, 3, x.Buffer, y.Buffer);
    }

    /// <summary>Records a global pooling: one result per plane, the maximum or the mean of all of the plane's elements (ONNX <c>GlobalMaxPool</c>, <c>GlobalAveragePool</c>).</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">Receives the pass. No other pass may be open on it.</param>
    /// <param name="kind"><see cref="GpuPoolKind.Max"/> (NaN elements are ignored; a plane of NaN or negative infinity gives the lowest finite value of the type) or <see cref="GpuPoolKind.Average"/>.</param>
    /// <param name="planes">Number of planes (batch times channels), at least 1.</param>
    /// <param name="planeSize">Elements per plane (height times width), at least 1.</param>
    /// <param name="x">Input, <c>planes * planeSize</c> elements.</param>
    /// <param name="offsetX">Index of the input's first element.</param>
    /// <param name="y">Receives <c>planes</c> results.</param>
    /// <param name="offsetY">Index of the first result.</param>
    /// <exception cref="GpuKernelShapeException">A size or offset is invalid, <paramref name="kind"/> is not Max or Average, or a range reaches past its buffer.</exception>
    public void EncodeGlobal<T>(GpuCommandEncoder encoder, GpuPoolKind kind, int planes, int planeSize, GpuBuffer<T> x, int offsetX, GpuBuffer<T> y, int offsetY)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var half = KernelTypes.IsHalf<T>(GlobalName);
        if (kind is not (GpuPoolKind.Max or GpuPoolKind.Average))
        {
            throw new GpuKernelShapeException(GlobalName, nameof(kind), "a global pooling is Max or Average");
        }

        if (planes < 1 || planeSize < 1)
        {
            throw new GpuKernelShapeException(GlobalName, planes < 1 ? nameof(planes) : nameof(planeSize), $"planes {planes} and plane size {planeSize} must be at least 1");
        }

        var total = (long)planes * planeSize;
        if (total > Check.MaxElements)
        {
            throw new GpuKernelShapeException(GlobalName, nameof(x), $"the tensor holds more than {Check.MaxElements} elements");
        }

        Check.Range(GlobalName, nameof(x), x.Length, offsetX, total, half);
        Check.Range(GlobalName, nameof(y), y.Length, offsetY, planes, half);
        var pipe = _global[half ? 1 : 0] ??= CreateGlobal(half);
        // Half results are written two to a word by one workgroup, so a workgroup per output word.
        var groups = half ? (((long)offsetY + planes - 1) >> 1) - (offsetY >> 1) + 1 : planes;
        var (gx, gy) = Grid.For(_device, groups);
        Span<uint> w = stackalloc uint[6];
        w[0] = (uint)planes;
        w[1] = (uint)planeSize;
        w[2] = (uint)offsetX;
        w[3] = (uint)offsetY;
        w[4] = gx;
        w[5] = kind == GpuPoolKind.Average ? 1u : 0u;
        pipe.Record(encoder, w, gx, gy, x.Buffer, y.Buffer);
    }

    static void CheckAxis(string axis, int input, int kernel, int stride, int dilation, int padBegin, int padEnd)
    {
        if (input < 1 || kernel < 1 || stride < 1 || dilation < 1)
        {
            throw new GpuKernelShapeException(Name, axis, $"input {input}, kernel {kernel}, stride {stride} and dilation {dilation} must be at least 1");
        }

        if (padBegin < 0 || padEnd < 0)
        {
            throw new GpuKernelShapeException(Name, axis + " padding", $"padding {padBegin}, {padEnd} must not be negative");
        }

        if ((long)kernel * dilation > Check.MaxElements || (long)padBegin + padEnd + input > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, axis, "the window or padding is larger than the supported sizes");
        }
    }

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => KernelPipeline.Total(_pool, _global);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        foreach (var p in _pool)
        {
            p?.ReleaseBindings();
        }

        foreach (var p in _global)
        {
            p?.ReleaseBindings();
        }
    }

    /// <summary>Releases the pipelines and bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        for (var i = 0; i < 2; i++)
        {
            _pool[i]?.Dispose();
            _pool[i] = null;
            _global[i]?.Dispose();
            _global[i] = null;
        }
    }

    KernelPipeline Create(bool half)
    {
        var e = Pool.Variants.Get(new Pool.Key(f16: half)).PoolOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, 20, "pool", _maxBindGroups);
    }

    KernelPipeline CreateGlobal(bool half)
    {
        var e = GlobalPool.Variants.Get(new GlobalPool.Key(f16: half)).GlobalPoolOp;
        return new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 2, 6, "global-pool", _maxBindGroups);
    }
}
