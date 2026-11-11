using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// ONNX <c>DynamicQuantizeLinear</c> for float32 input: finds the range of a tensor on the GPU, derives a scale and a uint8 zero point from it
/// and quantizes the tensor with them, so a dynamically quantized product needs no round trip to the host.
/// </summary>
/// <remarks>
/// <para>
/// The arithmetic is onemodel's, bit for bit: <c>lo = min(0, min x)</c>, <c>hi = max(0, max x)</c>, <c>scale = (hi - lo) / 255</c> in float32,
/// <c>zeroPoint = clamp(round_half_even(-(lo / scale)), 0, 255)</c>, <c>y = clamp(round_half_even(x / scale) + zeroPoint, 0, 255)</c> with NaN going to 0,
/// and an all-zero input giving scale 1, zero point 0. The GPU's own float division is not promised to be correctly rounded, so the kernel
/// divides in integer arithmetic (long division of the mantissas, round to nearest even), which is the same on every device. The minimum and maximum do not depend on the order
/// they are found in, so both <see cref="GpuKernelMode"/> values give the same bits.
/// </para>
/// The scale and zero point land in device buffers, ready to feed <see cref="Int8Gemm"/>. Two passes are recorded; the instance keeps a small
/// scratch buffer. Not thread-safe.
/// </remarks>
public sealed class DynamicQuantizeKernels : IDisposable
{
    const int Words = 7;
    const int MaxGroups = 256;
    const long ElementsPerGroup = 1024;

    readonly GpuDevice _device;
    readonly WideKernelPipeline _minMax;
    readonly WideKernelPipeline _quantize;
    readonly int _maxBindGroups;
    readonly GpuBuffer<uint> _partials;
    readonly GpuBuffer<uint> _dummy;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public DynamicQuantizeKernels(GpuDevice device, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        _minMax = new WideKernelPipeline(device, KernelSources.Make(DynamicQuantize.DqMinMax.EntryPoint, DynamicQuantize.DqMinMax.SpirV, DynamicQuantize.DqMinMax.Msl, DynamicQuantize.DqMinMax.MslFunctionName, DynamicQuantize.DqMinMax.Wgsl, DynamicQuantize.DqMinMax.WgslEntryPoint, DynamicQuantize.DqMinMax.Layout), Words, "dynamic quantize range", _maxBindGroups);
        _quantize = new WideKernelPipeline(device, KernelSources.Make(DynamicQuantize.DqFinishQuantize.EntryPoint, DynamicQuantize.DqFinishQuantize.SpirV, DynamicQuantize.DqFinishQuantize.Msl, DynamicQuantize.DqFinishQuantize.MslFunctionName, DynamicQuantize.DqFinishQuantize.Wgsl, DynamicQuantize.DqFinishQuantize.WgslEntryPoint, DynamicQuantize.DqFinishQuantize.Layout), Words, "dynamic quantize", _maxBindGroups);
        _partials = device.CreateBuffer<uint>(2 * MaxGroups, label: "dynamic quantize ranges");
        _dummy = device.CreateBuffer<uint>(4, label: "dynamic quantize unused operand");
    }

    /// <summary>Records the two passes of <c>DynamicQuantizeLinear</c>.</summary>
    /// <param name="encoder">Receives the compute passes; no other pass may be open on it.</param>
    /// <param name="x">The float32 input, dense.</param>
    /// <param name="offsetX">Element offset of the input.</param>
    /// <param name="count">Number of elements, at least 1.</param>
    /// <param name="y">Receives the uint8 result, <paramref name="count"/> bytes.</param>
    /// <param name="offsetY">Byte offset of the result.</param>
    /// <param name="scale">Receives the float32 scale (one element at <paramref name="offsetScale"/>).</param>
    /// <param name="offsetScale">Element offset of the scale.</param>
    /// <param name="zeroPoint">Receives the uint8 zero point (one byte at <paramref name="offsetZeroPoint"/>; the bytes around it are left alone).</param>
    /// <param name="offsetZeroPoint">Byte offset of the zero point.</param>
    /// <exception cref="GpuKernelShapeException">A count or range is invalid; names the argument.</exception>
    public void Encode(GpuCommandEncoder encoder, GpuBuffer<float> x, int offsetX, int count, GpuBuffer<byte> y, int offsetY, GpuBuffer<float> scale, int offsetScale, GpuBuffer<byte> zeroPoint, int offsetZeroPoint)
    {
        const string Name = "DynamicQuantizeLinear";
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(scale);
        ArgumentNullException.ThrowIfNull(zeroPoint);
        if (count < 1)
        {
            throw new GpuKernelShapeException(Name, nameof(count), $"count {count} must be at least 1");
        }

        Check.Range(Name, nameof(x), x.Length, offsetX, count);
        Check.Range(Name, nameof(y), y.Length, offsetY, count);
        Check.Range(Name, nameof(scale), scale.Length, offsetScale, 1);
        Check.Range(Name, nameof(zeroPoint), zeroPoint.Length, offsetZeroPoint, 1);
        var groups = (uint)Math.Clamp((count + ElementsPerGroup - 1) / ElementsPerGroup, 1, MaxGroups);
        var dummy = _dummy.Buffer;
        Span<uint> w = stackalloc uint[Words];
        var words = (((long)offsetY + count - 1) >> 2) - (offsetY >> 2) + 1;
        var (gx, gy) = Grid.For(_device, (words + 255) / 256);
        w[0] = (uint)count;
        w[1] = (uint)offsetX;
        w[2] = (uint)offsetY;
        w[3] = gx;
        w[4] = groups;
        w[5] = (uint)offsetScale;
        w[6] = (uint)offsetZeroPoint;
        _minMax.Record(encoder, w, groups, 1, 1, [x.Buffer, _partials.Buffer, dummy, dummy, dummy]);
        _quantize.Record(encoder, w, gx, gy, 1, [x.Buffer, _partials.Buffer, y.Buffer, scale.Buffer, zeroPoint.Buffer]);
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => _minMax.Created + _quantize.Created;

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        _minMax.ReleaseBindings();
        _quantize.ReleaseBindings();
    }

    /// <summary>Releases the pipelines, the scratch buffers and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        _minMax.Dispose();
        _quantize.Dispose();
        _partials.Dispose();
        _dummy.Dispose();
    }
}
