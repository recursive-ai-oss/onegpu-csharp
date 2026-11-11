using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Layer normalisation (ONNX <c>LayerNormalization</c>) and RMS normalisation (<c>SimplifiedLayerNormalization</c>) over the last axis of a
/// float or <see cref="Half"/> tensor. Mean and variance are computed in single precision, also for half inputs.
/// </summary>
/// <remarks>
/// <para>Layer: <c>y = (x - mean) / sqrt(variance + epsilon) * scale + bias</c> with the biased variance. RMS: <c>y = x / sqrt(mean(x²) + epsilon) * scale</c>.
/// The optional mean and inverse-deviation outputs of the ONNX operator are not produced.</para>
/// <para>Each row is one 256-thread workgroup. <see cref="GpuKernelMode.Fast"/> sums with subgroup operations where the device has them
/// (<see cref="Variant"/>); <see cref="GpuKernelMode.Deterministic"/> sums with the fixed shared-memory tree of the reductions, so the bits
/// do not depend on the device's subgroup size or scheduling.</para>
/// <para>Half tensors need an even row length and even element offsets (rows are written as whole 32-bit words). The scale and bias have the
/// element type of the input. Not thread-safe.</para>
/// </remarks>
public sealed class NormKernels : IDisposable
{
    const int Words = 9;

    readonly GpuDevice _device;
    readonly int _maxBindGroups;
    readonly bool _subgroups;
    readonly KernelPipeline?[] _pipelines = new KernelPipeline?[4];

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="allowSubgroups">False keeps the fast variant off subgroup operations even where the device has them.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public NormKernels(GpuDevice device, bool allowSubgroups = true, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
        _subgroups = Variants.UseSubgroups(device, allowSubgroups);
    }

    /// <summary>How <see cref="GpuKernelMode.Fast"/> sums a row: <c>subgroup</c> or <c>shared</c>.</summary>
    public string Variant => _subgroups ? "subgroup" : "shared";

    /// <summary>Records a normalisation of <paramref name="rows"/> rows of <paramref name="length"/> elements.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into.</param>
    /// <param name="rms">True for RMS normalisation, false for layer normalisation.</param>
    /// <param name="input">Input, rows of <paramref name="length"/> consecutive elements.</param>
    /// <param name="offsetInput">Element offset of the first row.</param>
    /// <param name="rows">Number of rows (the product of all but the last dimension).</param>
    /// <param name="length">Elements per row (the last dimension).</param>
    /// <param name="scale">Per-column multiplier of <paramref name="length"/> elements, or null for 1.</param>
    /// <param name="offsetScale">Element offset of the scale.</param>
    /// <param name="bias">Per-column addend of <paramref name="length"/> elements, or null for 0. Ignored for RMS normalisation.</param>
    /// <param name="offsetBias">Element offset of the bias.</param>
    /// <param name="output">Receives the result; it may be the input range itself.</param>
    /// <param name="offsetOutput">Element offset of the result.</param>
    /// <param name="epsilon">Added to the variance before the square root.</param>
    /// <param name="mode">Fast or deterministic.</param>
    /// <exception cref="GpuKernelShapeException">A size is not positive, a range reaches past its buffer, or a half tensor does not meet the rules above.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, bool rms, GpuBuffer<T> input, int offsetInput, int rows, int length, GpuBuffer<T>? scale, int offsetScale, GpuBuffer<T>? bias, int offsetBias,
        GpuBuffer<T> output, int offsetOutput, float epsilon = 1e-5f, GpuKernelMode mode = GpuKernelMode.Fast)
        where T : unmanaged
    {
        const string Name = "Norm";
        var half = KernelTypes.IsHalf<T>(Name);
        ReduceKernels.CheckMode(Name, mode);
        if (rows < 0)
        {
            throw new GpuKernelShapeException(Name, nameof(rows), $"{rows} is negative");
        }

        if (length <= 0)
        {
            throw new GpuKernelShapeException(Name, nameof(length), $"{length} is not positive");
        }

        var total = (long)rows * length;
        if (total > Check.MaxElements)
        {
            throw new GpuKernelShapeException(Name, nameof(rows), $"the tensor holds more than {Check.MaxElements} elements");
        }

        Check.Range(Name, nameof(input), input.Length, offsetInput, total, half);
        Check.Range(Name, nameof(output), output.Length, offsetOutput, total, half);
        if (scale is not null)
        {
            Check.Range(Name, nameof(scale), scale.Length, offsetScale, length, half);
        }

        if (bias is not null && !rms)
        {
            Check.Range(Name, nameof(bias), bias.Length, offsetBias, length, half);
        }

        if (half && ((length & 1) != 0 || (offsetInput & 1) != 0 || (offsetOutput & 1) != 0))
        {
            throw new GpuKernelShapeException(Name, nameof(length), "half tensors need an even row length and even element offsets");
        }

        if (rows == 0)
        {
            return;
        }

        var useBias = bias is not null && !rms;
        Span<uint> w = stackalloc uint[Words];
        w[0] = (uint)rows;
        w[1] = (uint)length;
        w[2] = (uint)offsetInput;
        w[3] = (uint)offsetOutput;
        w[4] = (uint)offsetScale;
        w[5] = (uint)offsetBias;
        w[6] = (scale is not null ? 1u : 0u) | (useBias ? 2u : 0u) | (rms ? 4u : 0u);
        w[7] = BitConverter.SingleToUInt32Bits(epsilon);
        var wave = mode == GpuKernelMode.Fast && _subgroups;
        var slot = (half ? 1 : 0) + (wave ? 2 : 0);
        var pipeline = _pipelines[slot] ??= Create(half, wave);
        var (gx, gy) = Grid.For(_device, rows);
        w[8] = gx;

        // A shader binding must always be backed by a buffer; the unused scale and bias bind the input, which the shader never reads for them.
        pipeline.Record(encoder, w, gx, gy, input.Buffer, (scale ?? input).Buffer, (useBias ? bias! : scale ?? input).Buffer, output.Buffer);
    }

    /// <summary>How many bind groups the family has built so far; tests use it to see the cache work.</summary>
    internal int BindGroupsCreated => KernelPipeline.Total(_pipelines);

    /// <summary>
    /// Drops every cached bind group. Call it after disposing buffers that will not be used again, so the groups do not keep them alive.
    /// The pipelines stay; the next call with a set of buffers builds its group again.
    /// </summary>
    public void ReleaseBindings()
    {
        for (var i = 0; i < _pipelines.Length; i++)
        {
            _pipelines[i]?.ReleaseBindings();
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

    KernelPipeline Create(bool half, bool wave)
    {
        var v = Norm.Variants.Get(new Norm.Key(f16: half, wave: wave));
        return new KernelPipeline(_device, KernelSources.Make(v.NormBlock.EntryPoint, v.NormBlock.SpirV, v.NormBlock.Msl, v.NormBlock.MslFunctionName, v.NormBlock.Wgsl, v.NormBlock.WgslEntryPoint, v.NormBlock.Layout), 4, Words, "norm", _maxBindGroups);
    }
}
