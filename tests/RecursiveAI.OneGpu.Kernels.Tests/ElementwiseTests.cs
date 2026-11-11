namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class ElementwiseTests
{
    static float RefBinary(GpuBinaryOp op, float a, float b) => op switch
    {
        GpuBinaryOp.Add => a + b,
        GpuBinaryOp.Sub => a - b,
        GpuBinaryOp.Mul => a * b,
        GpuBinaryOp.Div => a / b,
        GpuBinaryOp.Pow => (float)Math.Pow(a, b),
        GpuBinaryOp.Min => float.IsNaN(a) || float.IsNaN(b) ? float.NaN : Math.Min(a, b),
        GpuBinaryOp.Max => float.IsNaN(a) || float.IsNaN(b) ? float.NaN : Math.Max(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    internal static double RefUnary(GpuUnaryOp op, double x, double p0, double p1) => op switch
    {
        GpuUnaryOp.Identity => x,
        GpuUnaryOp.Neg => -x,
        GpuUnaryOp.Abs => Math.Abs(x),
        GpuUnaryOp.Sqrt => Math.Sqrt(x),
        GpuUnaryOp.Reciprocal => 1 / x,
        GpuUnaryOp.Exp => Math.Exp(x),
        GpuUnaryOp.Log => Math.Log(x),
        GpuUnaryOp.Tanh => Math.Tanh(x),
        GpuUnaryOp.Erf => Erf(x),
        GpuUnaryOp.Sigmoid => 1 / (1 + Math.Exp(-x)),
        GpuUnaryOp.Relu => double.IsNaN(x) ? x : Math.Max(x, 0),
        GpuUnaryOp.GeluErf => 0.5 * x * (1 + Erf(x / Math.Sqrt(2))),
        GpuUnaryOp.GeluTanh => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
        GpuUnaryOp.Silu => x / (1 + Math.Exp(-x)),
        GpuUnaryOp.HardSigmoid => Math.Max(0, Math.Min(1, p0 * x + p1)),
        GpuUnaryOp.HardSwish => x * Math.Max(0, Math.Min(1, x / 6 + 0.5)),
        GpuUnaryOp.Clip => Math.Min(Math.Max(x, p0), p1),
        GpuUnaryOp.LeakyRelu => x < 0 ? p0 * x : x,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    // Erf to double precision: series for small |x|, continued fraction for the tail (Numerical Recipes style, written from the formulas).
    internal static double Erf(double x)
    {
        var a = Math.Abs(x);
        double r;
        if (a < 2.5)
        {
            double sum = a, term = a;
            for (var n = 1; n < 200; n++)
            {
                term *= -a * a / n;
                var add = term / (2 * n + 1);
                sum += add;
                if (Math.Abs(add) < 1e-17)
                {
                    break;
                }
            }

            r = 2 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            // erfc(a) = exp(-a^2) / (a sqrt(pi)) * 1 / (1 + 1/(2a^2) / (1 + 2/(2a^2) / (1 + ...)))
            double f = 0;
            for (var k = 60; k >= 1; k--)
            {
                f = k / 2.0 / (a + f);
            }

            r = 1 - Math.Exp(-a * a) / Math.Sqrt(Math.PI) / (a + f);
        }

        return x < 0 ? -r : r;
    }

    static (float[] A, float[] B) Operands(GpuBinaryOp op, int na, int nb, int seed)
    {
        // Pow gets positive bases and small exponents (negative bases are covered separately); everything else a mixed-sign range.
        var a = op == GpuBinaryOp.Pow ? Rig.Random(na, seed, 0.1f, 4f) : Rig.Random(na, seed, -8f, 8f);
        var b = op == GpuBinaryOp.Pow ? Rig.Random(nb, seed + 1, -3f, 3f) : Rig.Random(nb, seed + 1, op == GpuBinaryOp.Div ? 0.25f : -8f, 8f);
        return (a, b);
    }

    static int Count(int[] shape) => shape.Aggregate(1, (x, y) => x * y);

    /// <summary>Broadcast pairs of every rank combination up to six, with ones placed at assorted positions.</summary>
    internal static IEnumerable<(int[] A, int[] B)> BroadcastCases()
    {
        var sizes = new[] { 3, 2, 4, 2, 3, 2 };
        for (var ra = 0; ra <= 6; ra++)
        {
            for (var rb = 0; rb <= 6; rb++)
            {
                var rank = Math.Max(ra, rb);
                var full = sizes.Take(rank).ToArray();
                for (var pattern = 0; pattern < 4; pattern++)
                {
                    int[] Make(int r, int salt)
                    {
                        var s = new int[r];
                        for (var i = 0; i < r; i++)
                        {
                            var dim = rank - r + i;
                            var squash = ((pattern * 7 + salt * 3 + dim * 5 + i) % 3) == 0;
                            s[i] = squash ? 1 : full[dim];
                        }

                        return s;
                    }

                    yield return (Make(ra, 0), Make(rb, 1));
                }
            }
        }
    }

    [Fact]
    public void BinaryBroadcastMatchesTheCpuBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        foreach (var half in new[] { false, true })
        {
            var cases = 0;
            var bad = 0;
            foreach (var op in new[] { GpuBinaryOp.Add, GpuBinaryOp.Sub, GpuBinaryOp.Mul, GpuBinaryOp.Min, GpuBinaryOp.Max })
            {
                foreach (var (sa, sb) in BroadcastCases())
                {
                    var rs = ElementwiseKernels.BroadcastShape(sa, sb);
                    var (a, b) = Operands(op, Count(sa), Count(sb), cases + 1);
                    var n = Count(rs);
                    if (half)
                    {
                        bad += RunBinary<Half>(d, k, op, Rig.ToHalf(a), sa, Rig.ToHalf(b), sb, rs, n, Rig.ToHalf(a).Select(x => (float)x).ToArray(), Rig.ToHalf(b).Select(x => (float)x).ToArray());
                    }
                    else
                    {
                        bad += RunBinary<float>(d, k, op, a, sa, b, sb, rs, n, a, b);
                    }

                    cases++;
                }
            }

            Rig.Row(d, "elementwise", half ? "f16" : "f32", $"binary add/sub/mul/min/max, {cases} broadcast cases (every rank pair 0..6)", $"bit mismatches {bad}");
            Assert.Equal(0, bad);
        }
    }

    // Offsets are nonzero and odd on purpose: the half kernels write inside shared 32-bit words.
    static int RunBinary<T>(GpuDevice d, ElementwiseKernels k, GpuBinaryOp op, T[] a, int[] sa, T[] b, int[] sb, int[] rs, int n, float[] af, float[] bf)
        where T : unmanaged
    {
        const int OffA = 3, OffB = 1, OffO = 5;
        var pa = new T[a.Length + OffA + 1];
        var pb = new T[b.Length + OffB + 1];
        a.CopyTo(pa, OffA);
        b.CopyTo(pb, OffB);
        var outLen = typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n + OffO + 2) : n + OffO + 2;
        if (typeof(T) == typeof(Half))
        {
            pa = Pad(pa);
            pb = Pad(pb);
        }

        using var ba = Rig.Upload(d, pa);
        using var bb = Rig.Upload(d, pb);
        using var bo = d.CreateBuffer<T>(outLen);
        var fill = new T[outLen];
        if (typeof(T) == typeof(float))
        {
            Array.Fill((float[])(object)fill, -777f);
        }
        else
        {
            Array.Fill((Half[])(object)fill, (Half)(-77f));
        }

        bo.Write(fill);
        Rig.Run(d, e => k.EncodeBinary(e, op, ba, OffA, sa, bb, OffB, sb, bo, OffO));
        var got = Rig.Download(bo);
        var bad = 0;
        var idx = new int[rs.Length];
        for (var i = 0; i < n; i++)
        {
            var ia = 0;
            var ib = 0;
            var rem = i;
            for (var dim = rs.Length - 1; dim >= 0; dim--)
            {
                var c = rem % rs[dim];
                rem /= rs[dim];
                idx[dim] = c;
            }

            ia = Index(sa, idx, rs.Length);
            ib = Index(sb, idx, rs.Length);
            var v = RefBinary(op, af[ia], bf[ib]);
            if (typeof(T) == typeof(float))
            {
                bad += Rig.BitMismatches(new[] { ((float[])(object)got)[OffO + i] }, new[] { v });
            }
            else
            {
                bad += Rig.BitMismatches(new[] { ((Half[])(object)got)[OffO + i] }, new[] { (Half)v });
            }
        }

        // The neighbours of the written range must be untouched.
        for (var i = 0; i < OffO; i++)
        {
            bad += typeof(T) == typeof(float) ? (((float[])(object)got)[i] == -777f ? 0 : 1) : (((Half[])(object)got)[i] == (Half)(-77f) ? 0 : 1);
        }

        for (var i = OffO + n; i < outLen; i++)
        {
            bad += typeof(T) == typeof(float) ? (((float[])(object)got)[i] == -777f ? 0 : 1) : (((Half[])(object)got)[i] == (Half)(-77f) ? 0 : 1);
        }

        return bad;
    }

    static T[] Pad<T>(T[] v)
        where T : unmanaged
    {
        var r = new T[GpuKernelBuffers.PaddedHalfLength(v.Length)];
        v.CopyTo(r, 0);
        return r;
    }

    static int Index(int[] shape, int[] outIdx, int outRank)
    {
        var idx = 0;
        var stride = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            var o = outIdx[outRank - shape.Length + i];
            idx += (shape[i] == 1 ? 0 : o) * stride;
            stride *= shape[i];
        }

        return idx;
    }
}
