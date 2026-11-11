namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Plain C# references for the vision kernels: direct loops, double accumulation, the ONNX operator definitions.</summary>
internal static class ConvReference
{
    public static double Act(GpuActivation a, double v)
    {
        switch (a.Op)
        {
            case GpuUnaryOp.Identity: return v;
            case GpuUnaryOp.Relu: return v < 0 ? 0 : v;
            case GpuUnaryOp.Sigmoid: return 1.0 / (1.0 + Math.Exp(-v));
            case GpuUnaryOp.Silu: return v / (1.0 + Math.Exp(-v));
            case GpuUnaryOp.Tanh: return Math.Tanh(v);
            case GpuUnaryOp.HardSigmoid: return Math.Max(0.0, Math.Min(1.0, (a.P0 * v) + a.P1));
            case GpuUnaryOp.HardSwish: return v * Math.Max(0.0, Math.Min(1.0, (v / 6.0) + 0.5));
            case GpuUnaryOp.Clip: return Math.Min(Math.Max(v, a.P0), a.P1);
            case GpuUnaryOp.LeakyRelu: return v >= 0 ? v : a.P0 * v;
            default: throw new NotSupportedException(a.Op.ToString());
        }
    }

    /// <summary>NCHW convolution with bias, residual and activation. Operands are dense (no offsets).</summary>
    public static double[] Conv(in ConvProblem p, float[] x, float[] w, float[]? bias, float[]? residual)
    {
        int oh = p.OutHeight, ow = p.OutWidth;
        var cinG = p.InChannels / p.Groups;
        var coutG = p.OutChannels / p.Groups;
        var y = new double[p.Batch * p.OutChannels * oh * ow];
        var pp = p;
        Parallel.For(0, p.Batch * p.OutChannels, plane =>
        {
            var n = plane / pp.OutChannels;
            var co = plane % pp.OutChannels;
            var g = co / coutG;
            for (var i = 0; i < oh; i++)
            {
                for (var j = 0; j < ow; j++)
                {
                    double acc = 0;
                    for (var ci = 0; ci < cinG; ci++)
                    {
                        var c = (g * cinG) + ci;
                        for (var kh = 0; kh < pp.KernelHeight; kh++)
                        {
                            var ih = (i * pp.StrideHeight) - pp.PadTop + (kh * pp.DilationHeight);
                            if ((uint)ih >= (uint)pp.Height)
                            {
                                continue;
                            }

                            for (var kw = 0; kw < pp.KernelWidth; kw++)
                            {
                                var iw = (j * pp.StrideWidth) - pp.PadLeft + (kw * pp.DilationWidth);
                                if ((uint)iw >= (uint)pp.Width)
                                {
                                    continue;
                                }

                                acc += (double)x[((((n * pp.InChannels) + c) * pp.Height) + ih) * pp.Width + iw] *
                                       w[((((co * cinG) + ci) * pp.KernelHeight) + kh) * pp.KernelWidth + kw];
                            }
                        }
                    }

                    var at = ((((n * pp.OutChannels) + co) * oh) + i) * ow + j;
                    if (bias is not null)
                    {
                        acc += bias[co];
                    }

                    if (residual is not null)
                    {
                        acc += residual[at];
                    }

                    y[at] = Act(pp.Activation, acc);
                }
            }
        });
        return y;
    }

    /// <summary>NCHW transposed convolution, scattering every input element's weighted kernel into the output, as the ONNX definition reads.</summary>
    public static double[] ConvTranspose(in ConvTransposeProblem p, float[] x, float[] w, float[]? bias)
    {
        int oh = p.OutHeight, ow = p.OutWidth;
        var cinG = p.InChannels / p.Groups;
        var coutG = p.OutChannels / p.Groups;
        var y = new double[p.Batch * p.OutChannels * oh * ow];
        for (var n = 0; n < p.Batch; n++)
        {
            for (var ci = 0; ci < p.InChannels; ci++)
            {
                var g = ci / cinG;
                for (var ih = 0; ih < p.Height; ih++)
                {
                    for (var iw = 0; iw < p.Width; iw++)
                    {
                        double v = x[((((n * p.InChannels) + ci) * p.Height) + ih) * p.Width + iw];
                        for (var cl = 0; cl < coutG; cl++)
                        {
                            var co = (g * coutG) + cl;
                            for (var kh = 0; kh < p.KernelHeight; kh++)
                            {
                                var oy = (ih * p.StrideHeight) + (kh * p.DilationHeight) - p.PadTop;
                                if ((uint)oy >= (uint)oh)
                                {
                                    continue;
                                }

                                for (var kw = 0; kw < p.KernelWidth; kw++)
                                {
                                    var ox = (iw * p.StrideWidth) + (kw * p.DilationWidth) - p.PadLeft;
                                    if ((uint)ox >= (uint)ow)
                                    {
                                        continue;
                                    }

                                    y[((((n * p.OutChannels) + co) * oh) + oy) * ow + ox] += v * w[((((ci * coutG) + cl) * p.KernelHeight) + kh) * p.KernelWidth + kw];
                                }
                            }
                        }
                    }
                }
            }
        }

        for (var i = 0; i < y.Length; i++)
        {
            var co = (i / (oh * ow)) % p.OutChannels;
            y[i] = Act(p.Activation, y[i] + (bias is null ? 0 : bias[co]));
        }

        return y;
    }

    /// <summary>Pooling with ONNX semantics; Max follows maxNum (NaN skipped unless all NaN), Average sums in double.</summary>
    public static double[] Pool(in PoolProblem p, GpuPoolKind kind, float[] x, double lowest = -3.4028234663852886e38)
    {
        int oh = p.OutHeight, ow = p.OutWidth;
        var y = new double[p.Batch * p.Channels * oh * ow];
        for (var plane = 0; plane < p.Batch * p.Channels; plane++)
        {
            for (var i = 0; i < oh; i++)
            {
                for (var j = 0; j < ow; j++)
                {
                    double best = lowest, sum = 0, nan = 0;
                    var any = false;
                    var sawNaN = false;
                    var count = 0;
                    for (var kh = 0; kh < p.KernelHeight; kh++)
                    {
                        var ih = (i * p.StrideHeight) - p.PadTop + (kh * p.DilationHeight);
                        for (var kw = 0; kw < p.KernelWidth; kw++)
                        {
                            var iw = (j * p.StrideWidth) - p.PadLeft + (kw * p.DilationWidth);
                            var inside = (uint)ih < (uint)p.Height && (uint)iw < (uint)p.Width;
                            if (inside)
                            {
                                double v = x[(((plane * p.Height) + ih) * p.Width) + iw];
                                if (double.IsNaN(v))
                                {
                                    if (!sawNaN)
                                    {
                                        sawNaN = true;
                                        nan = v;
                                    }
                                }
                                else if (!any || v > best)
                                {
                                    best = v;
                                    any = true;
                                }

                                sum += v;
                                count++;
                            }
                            else if (kind == GpuPoolKind.AverageIncludePad && ih >= -p.PadTop && ih < p.Height + p.PadBottom && iw >= -p.PadLeft && iw < p.Width + p.PadRight)
                            {
                                count++;
                            }
                        }
                    }

                    y[(((plane * oh) + i) * ow) + j] = kind == GpuPoolKind.Max ? (!any && sawNaN ? nan : best) : (count == 0 ? double.NaN : sum / count);
                }
            }
        }

        return y;
    }

    /// <summary>Global pooling: maximum (NaN ignored, lowest finite float for an empty result) or mean per plane.</summary>
    public static double[] GlobalPool(GpuPoolKind kind, int planes, int size, float[] x, double lowest = -3.4028234663852886e38)
    {
        var y = new double[planes];
        for (var p = 0; p < planes; p++)
        {
            double best = lowest, sum = 0;
            for (var i = 0; i < size; i++)
            {
                double v = x[(p * size) + i];
                if (v > best)
                {
                    best = v;
                }

                sum += v;
            }

            y[p] = kind == GpuPoolKind.Max ? best : sum / size;
        }

        return y;
    }

    /// <summary>
    /// Resize of the two trailing axes with the ONNX definition: per axis the source position of each output index in float32, the pick (nearest) or the two
    /// weights (linear); separable evaluation in double.
    /// </summary>
    public static double[] Resize(in ResizeProblem p, float[] x)
    {
        var rows = Axis(p, p.Height, p.OutHeight, p.ScaleHeight);
        var cols = Axis(p, p.Width, p.OutWidth, p.ScaleWidth);
        var y = new double[p.Planes * p.OutHeight * p.OutWidth];
        for (var plane = 0; plane < p.Planes; plane++)
        {
            for (var i = 0; i < p.OutHeight; i++)
            {
                for (var j = 0; j < p.OutWidth; j++)
                {
                    double acc = 0;
                    foreach (var (iy, wy) in rows[i])
                    {
                        foreach (var (ix, wx) in cols[j])
                        {
                            acc += wy * wx * x[(((plane * p.Height) + iy) * p.Width) + ix];
                        }
                    }

                    y[(((plane * p.OutHeight) + i) * p.OutWidth) + j] = acc;
                }
            }
        }

        return y;
    }

    static List<(int, double)>[] Axis(in ResizeProblem p, int inN, int outN, float scaleIn)
    {
        var scale = scaleIn != 0f ? scaleIn : (float)outN / inN;
        var r = new List<(int, double)>[outN];
        for (var j = 0; j < outN; j++)
        {
            float xo;
            switch (p.Coordinate)
            {
                case GpuResizeCoordinate.HalfPixel: xo = (((float)j + 0.5f) / scale) - 0.5f; break;
                case GpuResizeCoordinate.PytorchHalfPixel: xo = outN > 1 ? (((float)j + 0.5f) / scale) - 0.5f : 0f; break;
                case GpuResizeCoordinate.AlignCorners: xo = outN == 1 ? 0f : (float)j * (inN - 1) / (outN - 1); break;
                case GpuResizeCoordinate.Asymmetric: xo = (float)j / scale; break;
                default:
                    {
                        var adjustment = outN / (scale * inN);
                        xo = (inN / 2f * (1f - adjustment)) + ((((float)j + 0.5f) / scale) - 0.5f);
                        break;
                    }
            }

            if (p.Mode == GpuResizeMode.Nearest)
            {
                var f = Math.Floor((double)xo);
                var frac = xo - f;
                var pick = p.Nearest switch
                {
                    GpuResizeNearest.Floor => (int)f,
                    GpuResizeNearest.Ceil => (int)Math.Ceiling((double)xo),
                    GpuResizeNearest.RoundPreferCeil => frac >= 0.5 ? (int)f + 1 : (int)f,
                    _ => frac > 0.5 ? (int)f + 1 : (int)f,
                };
                r[j] = [(Math.Clamp(pick, 0, inN - 1), 1.0)];
            }
            else
            {
                var c = Math.Clamp(xo, 0f, inN - 1);
                var fl = Math.Floor((double)c);
                var t = c - fl;
                r[j] = [((int)fl, 1.0 - t), (Math.Min((int)fl + 1, inN - 1), t)];
            }
        }

        return r;
    }
}
