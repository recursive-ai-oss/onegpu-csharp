namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// A plain C# reference for ONNX RNN, GRU and LSTM in double precision, written from the operator definitions (gate order, clip, peepholes,
/// input_forget, linear_before_reset, sequence_lens, layouts). It shares no code with the kernels.
/// </summary>
internal static class RecurrentReference
{
    public sealed class Input
    {
        public required double[] X { get; init; }
        public required double[] W { get; init; }
        public required double[] R { get; init; }
        public double[]? B { get; init; }
        public double[]? P { get; init; }
        public int[]? Lens { get; init; }
        public double[]? H0 { get; init; }
        public double[]? C0 { get; init; }
    }

    public static double Act(GpuRecurrentActivation a, double x)
    {
        switch (a.Kind)
        {
            case GpuRecurrentActivationKind.Sigmoid: return 1.0 / (1.0 + Math.Exp(-x));
            case GpuRecurrentActivationKind.Tanh: return Math.Tanh(x);
            case GpuRecurrentActivationKind.Relu: return x > 0 || double.IsNaN(x) ? x : 0;
            case GpuRecurrentActivationKind.Affine: return (double)a.Alpha * x + a.Beta;
            case GpuRecurrentActivationKind.LeakyRelu: return x < 0 ? (double)a.Alpha * x : x;
            case GpuRecurrentActivationKind.ThresholdedRelu: return x > a.Alpha || double.IsNaN(x) ? x : 0;
            case GpuRecurrentActivationKind.ScaledTanh: return (double)a.Alpha * Math.Tanh((double)a.Beta * x);
            case GpuRecurrentActivationKind.HardSigmoid:
                var v = (double)a.Alpha * x + a.Beta;
                return v > 1 ? 1 : (v < 0 ? 0 : v);
            case GpuRecurrentActivationKind.Elu: return x >= 0 || double.IsNaN(x) ? x : (double)a.Alpha * (Math.Exp(x) - 1);
            case GpuRecurrentActivationKind.Softsign: return x / (1 + Math.Abs(x));
            default: return x > 20 || double.IsNaN(x) ? x : Math.Log(1 + Math.Exp(x));
        }
    }

    public static (double[] Y, double[] Yh, double[] Yc) Run(RecurrentProblem p, Input inp)
    {
        int dirs = p.Directions, S = p.Seq, B = p.Batch, I = p.Input, H = p.Hidden, G = p.Gates;
        var y = new double[S * dirs * B * H];
        var yh = new double[dirs * B * H];
        var yc = new double[dirs * B * H];
        var fwd = p.Activations ?? GpuRecurrentActivations.Default(p.Cell);
        var rev = p.ReverseActivations ?? fwd;
        double Clip(double v) => p.Clip > 0 ? (v > p.Clip ? p.Clip : (v < -p.Clip ? -p.Clip : v)) : v;
        int XIndex(int t, int b) => p.Layout1 ? b * S + t : t * B + b;
        int StateIndex(int d, int b) => p.Layout1 ? (b * dirs + d) * H : (d * B + b) * H;
        int YIndex(int t, int d, int b) => p.Layout1 ? ((b * S + t) * dirs + d) * H : ((t * dirs + d) * B + b) * H;
        for (var d = 0; d < dirs; d++)
        {
            var acts = d == 0 ? fwd : rev;
            var reverse = p.Direction == GpuRecurrentDirection.Reverse || d == 1;
            for (var b = 0; b < B; b++)
            {
                var len = inp.Lens is null ? S : inp.Lens[b];
                var h = new double[H];
                var c = new double[H];
                if (inp.H0 is not null)
                {
                    Array.Copy(inp.H0, StateIndex(d, b), h, 0, H);
                }

                if (inp.C0 is not null)
                {
                    Array.Copy(inp.C0, StateIndex(d, b), c, 0, H);
                }

                for (var s = 0; s < len; s++)
                {
                    var t = reverse ? len - 1 - s : s;
                    var xo = XIndex(t, b) * I;
                    double Gate(int g, int j, double[] vec)
                    {
                        // x Wg^T + vec Rg^T + Wb + Rb for row g*H + j.
                        var row = d * G * H + g * H + j;
                        double acc = 0;
                        for (var k = 0; k < I; k++)
                        {
                            acc += inp.X[xo + k] * inp.W[row * I + k];
                        }

                        for (var k = 0; k < H; k++)
                        {
                            acc += vec[k] * inp.R[row * H + k];
                        }

                        if (inp.B is not null)
                        {
                            acc += inp.B[d * 2 * G * H + g * H + j] + inp.B[d * 2 * G * H + G * H + g * H + j];
                        }

                        return acc;
                    }

                    var hNew = new double[H];
                    if (p.Cell == GpuRecurrentCell.Lstm)
                    {
                        var cNew = new double[H];
                        var o = new double[H];
                        var cOld = (double[])c.Clone();
                        for (var j = 0; j < H; j++)
                        {
                            var pi = Gate(0, j, h);
                            var pf = Gate(2, j, h);
                            var pc = Gate(3, j, h);
                            if (inp.P is not null)
                            {
                                pi += inp.P[d * 3 * H + j] * cOld[j];
                                pf += inp.P[d * 3 * H + 2 * H + j] * cOld[j];
                            }

                            var gi = Act(acts.F, Clip(pi));
                            var gf = p.InputForget ? 1 - gi : Act(acts.F, Clip(pf));
                            var gc = Act(acts.G, Clip(pc));
                            cNew[j] = gf * cOld[j] + gi * gc;
                        }

                        for (var j = 0; j < H; j++)
                        {
                            var po = Gate(1, j, h);
                            if (inp.P is not null)
                            {
                                po += inp.P[d * 3 * H + H + j] * cNew[j];
                            }

                            o[j] = Act(acts.F, Clip(po));
                            hNew[j] = o[j] * Act(acts.H, cNew[j]);
                        }

                        c = cNew;
                    }
                    else if (p.Cell == GpuRecurrentCell.Gru)
                    {
                        var z = new double[H];
                        var r = new double[H];
                        for (var j = 0; j < H; j++)
                        {
                            z[j] = Act(acts.F, Clip(Gate(0, j, h)));
                            r[j] = Act(acts.F, Clip(Gate(1, j, h)));
                        }

                        var rh = new double[H];
                        for (var j = 0; j < H; j++)
                        {
                            rh[j] = r[j] * h[j];
                        }

                        for (var j = 0; j < H; j++)
                        {
                            var row = d * G * H + 2 * H + j;
                            double pre;
                            if (p.LinearBeforeReset)
                            {
                                double xs = 0, hs = 0;
                                for (var k = 0; k < I; k++)
                                {
                                    xs += inp.X[xo + k] * inp.W[row * I + k];
                                }

                                for (var k = 0; k < H; k++)
                                {
                                    hs += h[k] * inp.R[row * H + k];
                                }

                                var wb = inp.B?[d * 2 * G * H + 2 * H + j] ?? 0;
                                var rb = inp.B?[d * 2 * G * H + G * H + 2 * H + j] ?? 0;
                                pre = xs + r[j] * (hs + rb) + wb;
                            }
                            else
                            {
                                pre = Gate(2, j, rh);
                            }

                            var cand = Act(acts.G, Clip(pre));
                            hNew[j] = (1 - z[j]) * cand + z[j] * h[j];
                        }
                    }
                    else
                    {
                        for (var j = 0; j < H; j++)
                        {
                            hNew[j] = Act(acts.F, Clip(Gate(0, j, h)));
                        }
                    }

                    h = hNew;
                    Array.Copy(h, 0, y, YIndex(t, d, b) + 0, H);
                }

                var zeroFinal = inp.Lens is not null && S > 0 && len == 0;
                if (!zeroFinal)
                {
                    Array.Copy(h, 0, yh, StateIndex(d, b), H);
                    Array.Copy(c, 0, yc, StateIndex(d, b), H);
                }
            }
        }

        return (y, yh, yc);
    }
}
