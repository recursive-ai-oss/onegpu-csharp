namespace RecursiveAI.OneGpu.Kernels;

/// <summary>How a sliding-window operator chooses its padding when the model does not give it (ONNX <c>auto_pad</c>).</summary>
public enum GpuAutoPad
{
    /// <summary>Use the explicit padding of the problem.</summary>
    NotSet = 0,

    /// <summary>The output extent is ceil(input / stride); the odd extra element of padding goes at the end.</summary>
    SameUpper = 1,

    /// <summary>As <see cref="SameUpper"/>, but the extra element goes at the beginning.</summary>
    SameLower = 2,

    /// <summary>No padding.</summary>
    Valid = 3,
}

/// <summary>Output-size and padding arithmetic of sliding windows (convolution, pooling), following the ONNX operator documentation.</summary>
public static class GpuWindow
{
    /// <summary>
    /// The output extent of a window on one axis, and the padding that gives it. With <see cref="GpuAutoPad.SameUpper"/>, <see cref="GpuAutoPad.SameLower"/>
    /// or <see cref="GpuAutoPad.Valid"/> the padding is derived and written to <paramref name="padBegin"/> and <paramref name="padEnd"/>; otherwise their
    /// values are used as given.
    /// </summary>
    /// <param name="autoPad">The padding rule.</param>
    /// <param name="input">Input extent, at least 1.</param>
    /// <param name="kernel">Window extent, at least 1.</param>
    /// <param name="stride">Step, at least 1.</param>
    /// <param name="dilation">Spacing of the window's taps, at least 1.</param>
    /// <param name="ceilMode">Round the extent up instead of down (pooling only); the last window must still start inside the input or its leading padding.</param>
    /// <param name="padBegin">Padding before the input; replaced when <paramref name="autoPad"/> derives it.</param>
    /// <param name="padEnd">Padding after the input; replaced when <paramref name="autoPad"/> derives it.</param>
    /// <returns>The output extent, or 0 when the window does not fit in the padded input.</returns>
    public static int OutputExtent(GpuAutoPad autoPad, int input, int kernel, int stride, int dilation, bool ceilMode, ref int padBegin, ref int padEnd)
    {
        if (input < 1 || kernel < 1 || stride < 1 || dilation < 1)
        {
            return 0;
        }

        long effective = ((long)dilation * (kernel - 1)) + 1;
        if (autoPad is GpuAutoPad.SameUpper or GpuAutoPad.SameLower)
        {
            long same = ((long)input + stride - 1) / stride;
            long total = Math.Max(0, ((same - 1) * stride) + effective - input);
            long small = total / 2, big = total - small;
            padBegin = (int)(autoPad == GpuAutoPad.SameUpper ? small : big);
            padEnd = (int)(autoPad == GpuAutoPad.SameUpper ? big : small);
            return (int)same;
        }

        if (autoPad == GpuAutoPad.Valid)
        {
            padBegin = 0;
            padEnd = 0;
        }

        long span = (long)input + padBegin + padEnd - effective;
        if (span < 0)
        {
            return 0;
        }

        var extent = ceilMode ? ((span + stride - 1) / stride) + 1 : (span / stride) + 1;
        if (ceilMode && (extent - 1) * stride >= (long)input + padBegin)
        {
            extent--;
        }

        return extent > int.MaxValue ? 0 : (int)extent;
    }

    /// <summary>
    /// The output extent of a transposed convolution on one axis: <c>stride * (input - 1) + outputPadding + effectiveKernel - padBegin - padEnd</c>
    /// (ONNX <c>ConvTranspose</c>), or 0 when that is not positive.
    /// </summary>
    /// <param name="input">Input extent, at least 1.</param>
    /// <param name="kernel">Kernel extent, at least 1.</param>
    /// <param name="stride">Upsampling step, at least 1.</param>
    /// <param name="dilation">Kernel dilation, at least 1.</param>
    /// <param name="padBegin">Padding removed from the beginning of the full output.</param>
    /// <param name="padEnd">Padding removed from the end of the full output.</param>
    /// <param name="outputPadding">Extra elements added at the end of the output, below the stride or the dilation.</param>
    public static int TransposeOutputExtent(int input, int kernel, int stride, int dilation, int padBegin, int padEnd, int outputPadding)
    {
        if (input < 1 || kernel < 1 || stride < 1 || dilation < 1)
        {
            return 0;
        }

        var effective = ((long)dilation * (kernel - 1)) + 1;
        var extent = ((long)stride * (input - 1)) + outputPadding + effective - padBegin - padEnd;
        return extent <= 0 || extent > int.MaxValue ? 0 : (int)extent;
    }
}

/// <summary>
/// The shape of a two-dimensional convolution (ONNX <c>Conv</c>) over NCHW tensors, with where its operands sit in their buffers. A one-dimensional
/// convolution is the same with <see cref="Height"/> and <see cref="KernelHeight"/> 1 (see <see cref="Conv1D"/>). All sizes, strides and offsets count elements.
/// </summary>
/// <remarks>
/// Tensors are dense NCHW: <c>x</c> is <c>[Batch, InChannels, Height, Width]</c>, the weights are <c>[OutChannels, InChannels / Groups, KernelHeight, KernelWidth]</c>,
/// the output is <c>[Batch, OutChannels, OutHeight, OutWidth]</c>. The output extent is <c>floor((in + padBegin + padEnd - dilation * (kernel - 1) - 1) / stride) + 1</c>.
/// </remarks>
public struct ConvProblem
{
    /// <summary>Creates a problem with unit strides and dilations, no padding, one group and no activation.</summary>
    /// <param name="batch">Images, at least 1.</param>
    /// <param name="inChannels">Input channels, at least 1.</param>
    /// <param name="height">Input height, at least 1.</param>
    /// <param name="width">Input width, at least 1.</param>
    /// <param name="outChannels">Output channels, at least 1.</param>
    /// <param name="kernelHeight">Kernel height, at least 1.</param>
    /// <param name="kernelWidth">Kernel width, at least 1.</param>
    public ConvProblem(int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth)
    {
        Batch = batch;
        InChannels = inChannels;
        Height = height;
        Width = width;
        OutChannels = outChannels;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        StrideHeight = 1;
        StrideWidth = 1;
        DilationHeight = 1;
        DilationWidth = 1;
        Groups = 1;
    }

    /// <summary>A one-dimensional convolution over <c>[Batch, InChannels, Length]</c> as a two-dimensional one of height 1.</summary>
    /// <param name="batch">Sequences, at least 1.</param>
    /// <param name="inChannels">Input channels, at least 1.</param>
    /// <param name="length">Input length, at least 1.</param>
    /// <param name="outChannels">Output channels, at least 1.</param>
    /// <param name="kernel">Kernel length, at least 1.</param>
    /// <param name="stride">Step, at least 1.</param>
    /// <param name="dilation">Tap spacing, at least 1.</param>
    /// <param name="padBegin">Padding before the sequence.</param>
    /// <param name="padEnd">Padding after the sequence.</param>
    /// <param name="groups">Groups, at least 1.</param>
    public static ConvProblem Conv1D(int batch, int inChannels, int length, int outChannels, int kernel, int stride = 1, int dilation = 1, int padBegin = 0, int padEnd = 0, int groups = 1) =>
        new(batch, inChannels, 1, length, outChannels, 1, kernel)
        {
            StrideWidth = stride,
            DilationWidth = dilation,
            PadLeft = padBegin,
            PadRight = padEnd,
            Groups = groups,
        };

    /// <summary>Images, at least 1.</summary>
    public int Batch { get; set; }

    /// <summary>Input channels; a multiple of <see cref="Groups"/>.</summary>
    public int InChannels { get; set; }

    /// <summary>Input height.</summary>
    public int Height { get; set; }

    /// <summary>Input width.</summary>
    public int Width { get; set; }

    /// <summary>Output channels; a multiple of <see cref="Groups"/>.</summary>
    public int OutChannels { get; set; }

    /// <summary>Kernel height.</summary>
    public int KernelHeight { get; set; }

    /// <summary>Kernel width.</summary>
    public int KernelWidth { get; set; }

    /// <summary>Vertical step, at least 1.</summary>
    public int StrideHeight { get; set; }

    /// <summary>Horizontal step, at least 1.</summary>
    public int StrideWidth { get; set; }

    /// <summary>Vertical spacing of the kernel's taps, at least 1.</summary>
    public int DilationHeight { get; set; }

    /// <summary>Horizontal spacing of the kernel's taps, at least 1.</summary>
    public int DilationWidth { get; set; }

    /// <summary>Zero rows added above the input.</summary>
    public int PadTop { get; set; }

    /// <summary>Zero columns added left of the input.</summary>
    public int PadLeft { get; set; }

    /// <summary>Zero rows added below the input.</summary>
    public int PadBottom { get; set; }

    /// <summary>Zero columns added right of the input.</summary>
    public int PadRight { get; set; }

    /// <summary>Groups, at least 1. <see cref="InChannels"/> / Groups of input channels feed <see cref="OutChannels"/> / Groups output channels; groups equal to the input channels is a depthwise convolution.</summary>
    public int Groups { get; set; }

    /// <summary>A function applied to every output after the bias and the residual; <see cref="GpuUnaryOp.Identity"/> (the default) applies none.</summary>
    public GpuActivation Activation { get; set; }

    /// <summary>Index of the input's first element in its buffer.</summary>
    public int OffsetX { get; set; }

    /// <summary>Index of the weights' first element in their buffer.</summary>
    public int OffsetW { get; set; }

    /// <summary>Index of the bias's first element in its buffer (the bias has <see cref="OutChannels"/> elements).</summary>
    public int OffsetBias { get; set; }

    /// <summary>Index of the residual's first element in its buffer (the residual has the output's shape).</summary>
    public int OffsetResidual { get; set; }

    /// <summary>Index of the output's first element in its buffer.</summary>
    public int OffsetY { get; set; }

    /// <summary>Output height, or 0 when the kernel does not fit the padded input.</summary>
    public readonly int OutHeight
    {
        get
        {
            int b = PadTop, e = PadBottom;
            return GpuWindow.OutputExtent(GpuAutoPad.NotSet, Height, KernelHeight, StrideHeight, DilationHeight, false, ref b, ref e);
        }
    }

    /// <summary>Output width, or 0 when the kernel does not fit the padded input.</summary>
    public readonly int OutWidth
    {
        get
        {
            int b = PadLeft, e = PadRight;
            return GpuWindow.OutputExtent(GpuAutoPad.NotSet, Width, KernelWidth, StrideWidth, DilationWidth, false, ref b, ref e);
        }
    }

    /// <summary>Replaces the padding by the one an ONNX <c>auto_pad</c> rule derives for this shape.</summary>
    /// <param name="autoPad">The rule; <see cref="GpuAutoPad.NotSet"/> leaves the padding as it is.</param>
    /// <returns>A copy with the padding set.</returns>
    public readonly ConvProblem WithAutoPad(GpuAutoPad autoPad)
    {
        var p = this;
        int t = PadTop, b = PadBottom, l = PadLeft, r = PadRight;
        GpuWindow.OutputExtent(autoPad, Height, KernelHeight, StrideHeight, DilationHeight, false, ref t, ref b);
        GpuWindow.OutputExtent(autoPad, Width, KernelWidth, StrideWidth, DilationWidth, false, ref l, ref r);
        p.PadTop = t;
        p.PadBottom = b;
        p.PadLeft = l;
        p.PadRight = r;
        return p;
    }

    /// <summary>Floating-point operations of the convolution, counting a multiply-add as two (the bias, residual and activation are not counted).</summary>
    public readonly double Flops => 2.0 * Batch * OutChannels * (double)OutHeight * OutWidth * (InChannels / Math.Max(Groups, 1)) * KernelHeight * KernelWidth;
}

/// <summary>
/// The shape of a two-dimensional transposed convolution (ONNX <c>ConvTranspose</c>) over NCHW tensors. Each input element adds its weighted kernel
/// into the output: <c>x</c> is <c>[Batch, InChannels, Height, Width]</c>, the weights are <c>[InChannels, OutChannels / Groups, KernelHeight, KernelWidth]</c>
/// (note the order: input channels first), the output is <c>[Batch, OutChannels, OutHeight, OutWidth]</c> with the extent
/// <see cref="GpuWindow.TransposeOutputExtent"/> gives. A one-dimensional transposed convolution has <see cref="Height"/> and <see cref="KernelHeight"/> 1.
/// </summary>
public struct ConvTransposeProblem
{
    /// <summary>Creates a problem with unit strides and dilations, no padding, one group and no activation.</summary>
    /// <param name="batch">Images, at least 1.</param>
    /// <param name="inChannels">Input channels, at least 1.</param>
    /// <param name="height">Input height, at least 1.</param>
    /// <param name="width">Input width, at least 1.</param>
    /// <param name="outChannels">Output channels, at least 1.</param>
    /// <param name="kernelHeight">Kernel height, at least 1.</param>
    /// <param name="kernelWidth">Kernel width, at least 1.</param>
    public ConvTransposeProblem(int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth)
    {
        Batch = batch;
        InChannels = inChannels;
        Height = height;
        Width = width;
        OutChannels = outChannels;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        StrideHeight = 1;
        StrideWidth = 1;
        DilationHeight = 1;
        DilationWidth = 1;
        Groups = 1;
    }

    /// <summary>Images, at least 1.</summary>
    public int Batch { get; set; }

    /// <summary>Input channels; a multiple of <see cref="Groups"/>.</summary>
    public int InChannels { get; set; }

    /// <summary>Input height.</summary>
    public int Height { get; set; }

    /// <summary>Input width.</summary>
    public int Width { get; set; }

    /// <summary>Output channels; a multiple of <see cref="Groups"/>.</summary>
    public int OutChannels { get; set; }

    /// <summary>Kernel height.</summary>
    public int KernelHeight { get; set; }

    /// <summary>Kernel width.</summary>
    public int KernelWidth { get; set; }

    /// <summary>Vertical upsampling step, at least 1.</summary>
    public int StrideHeight { get; set; }

    /// <summary>Horizontal upsampling step, at least 1.</summary>
    public int StrideWidth { get; set; }

    /// <summary>Vertical spacing of the kernel's taps, at least 1.</summary>
    public int DilationHeight { get; set; }

    /// <summary>Horizontal spacing of the kernel's taps, at least 1.</summary>
    public int DilationWidth { get; set; }

    /// <summary>Rows removed from the top of the full output.</summary>
    public int PadTop { get; set; }

    /// <summary>Columns removed from the left of the full output.</summary>
    public int PadLeft { get; set; }

    /// <summary>Rows removed from the bottom of the full output.</summary>
    public int PadBottom { get; set; }

    /// <summary>Columns removed from the right of the full output.</summary>
    public int PadRight { get; set; }

    /// <summary>Rows added at the bottom of the output (ONNX <c>output_padding</c>).</summary>
    public int OutputPaddingHeight { get; set; }

    /// <summary>Columns added at the right of the output (ONNX <c>output_padding</c>).</summary>
    public int OutputPaddingWidth { get; set; }

    /// <summary>Groups, at least 1.</summary>
    public int Groups { get; set; }

    /// <summary>A function applied to every output after the bias; <see cref="GpuUnaryOp.Identity"/> (the default) applies none.</summary>
    public GpuActivation Activation { get; set; }

    /// <summary>Index of the input's first element in its buffer.</summary>
    public int OffsetX { get; set; }

    /// <summary>Index of the weights' first element in their buffer.</summary>
    public int OffsetW { get; set; }

    /// <summary>Index of the bias's first element in its buffer (the bias has <see cref="OutChannels"/> elements).</summary>
    public int OffsetBias { get; set; }

    /// <summary>Index of the output's first element in its buffer.</summary>
    public int OffsetY { get; set; }

    /// <summary>Output height, or 0 when it would not be positive.</summary>
    public readonly int OutHeight => GpuWindow.TransposeOutputExtent(Height, KernelHeight, StrideHeight, DilationHeight, PadTop, PadBottom, OutputPaddingHeight);

    /// <summary>Output width, or 0 when it would not be positive.</summary>
    public readonly int OutWidth => GpuWindow.TransposeOutputExtent(Width, KernelWidth, StrideWidth, DilationWidth, PadLeft, PadRight, OutputPaddingWidth);

    /// <summary>Floating-point operations of the matrix product, counting a multiply-add as two.</summary>
    public readonly double Flops => 2.0 * Batch * InChannels * (double)Height * Width * (OutChannels / Math.Max(Groups, 1)) * KernelHeight * KernelWidth;
}
