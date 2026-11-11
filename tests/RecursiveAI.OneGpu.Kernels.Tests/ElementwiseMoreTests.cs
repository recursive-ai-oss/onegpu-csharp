namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class ElementwiseMoreTests
{
    static readonly GpuUnaryOp[] AllUnary = Enum.GetValues<GpuUnaryOp>();

    static float[] Range(GpuUnaryOp op, int n, int seed) => op switch
    {
        GpuUnaryOp.Sqrt or GpuUnaryOp.Log => Rig.Random(n, seed, 0.01f, 100f),
        GpuUnaryOp.Reciprocal => Rig.Random(n, seed, 0.25f, 8f).Select((v, i) => i % 2 == 0 ? v : -v).ToArray(),
        GpuUnaryOp.Exp => Rig.Random(n, seed, -10f, 10f),
        _ => Rig.Random(n, seed, -8f, 8f),
    };

    static (float P0, float P1) Params(GpuUnaryOp op) => op switch
    {
        GpuUnaryOp.HardSigmoid => (0.2f, 0.5f),
        GpuUnaryOp.Clip => (-1.5f, 2.25f),
        GpuUnaryOp.LeakyRelu => (0.01f, 0f),
        _ => (0f, 0f),
    };

    // Runs a unary op over x with odd offsets; returns the outputs as doubles.
    static double[] RunUnary(GpuDevice d, ElementwiseKernels k, GpuUnaryOp op, float[] x, bool half, int offX = 1, int offY = 3)
    {
        var (p0, p1) = Params(op);
        var n = x.Length;
        if (half)
        {
            var hx = new Half[GpuKernelBuffers.PaddedHalfLength(n + offX)];
            for (var i = 0; i < n; i++)
            {
                hx[offX + i] = (Half)x[i];
            }

            using var bx = Rig.Upload(d, hx);
            using var by = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(n + offY));
            Rig.Run(d, e => k.EncodeUnary(e, op, bx, offX, by, offY, n, p0, p1));
            var got = Rig.Download(by);
            return Rig.D(got.AsSpan(offY, n).ToArray(), n);
        }

        var fx = new float[n + offX];
        x.CopyTo(fx, offX);
        using var gx = Rig.Upload(d, fx);
        using var gy = d.CreateBuffer<float>(n + offY);
        Rig.Run(d, e => k.EncodeUnary(e, op, gx, offX, gy, offY, n, p0, p1));
        var r = Rig.Download(gy);
        return Rig.D(r.AsSpan(offY, n).ToArray(), n);
    }

    static double[] Reference(GpuUnaryOp op, float[] x, bool half)
    {
        var (p0, p1) = Params(op);
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            var xi = half ? (double)(Half)x[i] : x[i];
            var v = EDouble(op, xi, p0, p1);
            r[i] = half ? (double)(Half)v : v;
        }

        return r;
    }

    static double EDouble(GpuUnaryOp op, double x, float p0, float p1) => ElementwiseTests.RefUnary(op, x, p0, p1);

    [Fact]
    public void UnaryFunctionsMatchTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        foreach (var half in new[] { false, true })
        {
            foreach (var op in AllUnary)
            {
                double worstScaled = 0, worstRel = 0;
                foreach (var n in new[] { 1, 7, 255, 256, 257, 1000, 70001 })
                {
                    var x = Range(op, n, (int)op * 31 + n);

                    // The error is scaled by the largest result, so every input holds one value of ordinary size.
                    x[^1] = 0.75f;
                    var got = RunUnary(d, k, op, x, half);
                    var want = Reference(op, x, half);
                    var (scaled, rel) = Rig.Errors(got, want);
                    worstScaled = Math.Max(worstScaled, scaled);
                    worstRel = Math.Max(worstRel, rel);
                }

                var exact = op is GpuUnaryOp.Identity or GpuUnaryOp.Neg or GpuUnaryOp.Abs or GpuUnaryOp.Relu;
                var limit = exact ? 0 : half ? 2e-3 : 1e-5;
                Rig.Row(d, "elementwise", half ? "f16" : "f32", $"unary {op}", $"max scaled error {Rig.Sci(worstScaled)}, max relative {Rig.Sci(worstRel)} (limit {Rig.Sci(limit)})");
                Assert.True(worstScaled <= limit, $"{op} {(half ? "f16" : "f32")}: scaled error {worstScaled}");
            }
        }
    }

    [Fact]
    public void UnaryFunctionsFollowIeeeForSpecialValues()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        foreach (var half in new[] { false, true })
        {
            var special = half
                ? new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, 1f, -1f, 3f, -3f }
                : new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, 1f, -1f, 1e30f, -1e30f, -0f, 3f };
            var bad = 0;
            foreach (var op in AllUnary)
            {
                var got = RunUnary(d, k, op, special, half);
                var want = Reference(op, special, half);
                var (scaled, _) = Rig.Errors(got, want);
                if (!(scaled <= (half ? 2e-3 : 1e-5)))
                {
                    bad++;
                    Console.WriteLine($"special values: {op} {(half ? "f16" : "f32")}: scaled {scaled}; got [{string.Join(", ", got)}] want [{string.Join(", ", want)}]");
                }
            }

            Rig.Row(d, "elementwise", half ? "f16" : "f32", "unary NaN/inf/zero/huge inputs, all functions", $"ops disagreeing with IEEE reference: {bad}");
            Assert.Equal(0, bad);
        }
    }

    [Fact]
    public void BinaryDivisionAndPowerStayWithinTolerance()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        foreach (var half in new[] { false, true })
        {
            foreach (var op in new[] { GpuBinaryOp.Div, GpuBinaryOp.Pow })
            {
                var n = 5000;
                var a = op == GpuBinaryOp.Pow ? Rig.Random(n, 5, 0.1f, 4f) : Rig.Random(n, 5, -8f, 8f);
                var b = op == GpuBinaryOp.Pow ? Rig.Random(n, 6, -3f, 3f) : Rig.Random(n, 6, 0.25f, 8f);
                if (op == GpuBinaryOp.Pow)
                {
                    // Negative bases with integral exponents, a zero exponent and a non-integral exponent (NaN).
                    var neg = new[] { (-2f, 2f), (-2f, 3f), (-2f, -1f), (-2f, 0f), (-0.5f, 4f), (-2f, 0.5f), (0f, 0f), (0f, 2f), (3f, 0f) };
                    for (var i = 0; i < neg.Length; i++)
                    {
                        a[i] = neg[i].Item1;
                        b[i] = neg[i].Item2;
                    }
                }

                double[] got;
                var want = new double[n];
                if (half)
                {
                    var ha = Rig.ToHalf(a);
                    var hb = Rig.ToHalf(b);
                    using var ba = Rig.Upload(d, ha);
                    using var bb = Rig.Upload(d, hb);
                    using var bo = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(n));
                    Rig.Run(d, e => k.EncodeBinary(e, op, ba, 0, [n], bb, 0, [n], bo, 0));
                    got = Rig.D(Rig.Download(bo), n);
                    for (var i = 0; i < n; i++)
                    {
                        want[i] = (double)(Half)RefD(op, (double)ha[i], (double)hb[i]);
                    }
                }
                else
                {
                    using var ba = Rig.Upload(d, a);
                    using var bb = Rig.Upload(d, b);
                    using var bo = d.CreateBuffer<float>(n);
                    Rig.Run(d, e => k.EncodeBinary(e, op, ba, 0, [n], bb, 0, [n], bo, 0));
                    got = Rig.D(Rig.Download(bo), n);
                    for (var i = 0; i < n; i++)
                    {
                        want[i] = RefD(op, a[i], b[i]);
                    }
                }

                var (scaled, rel) = Rig.Errors(got, want);
                var limit = half ? 2e-3 : 1e-5;
                Rig.Row(d, "elementwise", half ? "f16" : "f32", $"binary {op}", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)} (limit {Rig.Sci(limit)})");
                Assert.True(scaled <= limit, $"{op}: {scaled}");
            }
        }
    }

    static double RefD(GpuBinaryOp op, double a, double b) => op == GpuBinaryOp.Div ? a / b : Math.Pow(a, b);

    [Fact]
    public void BinaryOpsFollowIeeeForNanAndInfinity()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        var v = new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, 1f, -2f };
        var a = new List<float>();
        var b = new List<float>();
        foreach (var x in v)
        {
            foreach (var y in v)
            {
                a.Add(x);
                b.Add(y);
            }
        }

        var n = a.Count;
        using var ba = Rig.Upload(d, a.ToArray());
        using var bb = Rig.Upload(d, b.ToArray());
        using var bo = d.CreateBuffer<float>(n);
        var bad = 0;
        foreach (var op in new[] { GpuBinaryOp.Add, GpuBinaryOp.Sub, GpuBinaryOp.Mul, GpuBinaryOp.Div, GpuBinaryOp.Min, GpuBinaryOp.Max })
        {
            Rig.Run(d, e => k.EncodeBinary(e, op, ba, 0, [n], bb, 0, [n], bo, 0));
            var got = Rig.Download(bo);
            for (var i = 0; i < n; i++)
            {
                float want = op switch
                {
                    GpuBinaryOp.Add => a[i] + b[i],
                    GpuBinaryOp.Sub => a[i] - b[i],
                    GpuBinaryOp.Mul => a[i] * b[i],
                    GpuBinaryOp.Div => a[i] / b[i],
                    GpuBinaryOp.Min => float.IsNaN(a[i]) || float.IsNaN(b[i]) ? float.NaN : Math.Min(a[i], b[i]),
                    _ => float.IsNaN(a[i]) || float.IsNaN(b[i]) ? float.NaN : Math.Max(a[i], b[i]),
                };
                bad += Rig.BitMismatches([got[i]], [want]) != 0 && !(want == 0 && got[i] == 0) ? 1 : 0;
            }
        }

        Rig.Row(d, "elementwise", "f32", "binary add/sub/mul/div/min/max on NaN, ±inf, 0, 1, -2 (36 pairs each)", $"mismatches {bad} (zero signs not compared)");
        Assert.Equal(0, bad);
    }

    [Fact]
    public void FusedActivationMatchesTheTwoStepResult()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        const int Rows = 37, Cols = 384;
        var x = Rig.Random(Rows * Cols, 1, -4f, 4f);
        var bias = Rig.Random(Cols, 2, -1f, 1f);
        using var bx = Rig.Upload(d, x);
        using var bbias = Rig.Upload(d, bias);
        using var bo = d.CreateBuffer<float>(Rows * Cols);
        foreach (var op in new[] { GpuUnaryOp.GeluErf, GpuUnaryOp.GeluTanh, GpuUnaryOp.Silu, GpuUnaryOp.Relu, GpuUnaryOp.Clip, GpuUnaryOp.HardSwish })
        {
            var (p0, p1) = Params(op);
            Rig.Run(d, e => k.EncodeBinary(e, GpuBinaryOp.Add, bx, 0, [Rows, Cols], bbias, 0, [Cols], bo, 0, new GpuActivation(op, p0, p1)));
            var got = Rig.Download(bo);
            var want = new double[Rows * Cols];
            for (var i = 0; i < want.Length; i++)
            {
                want[i] = ElementwiseTests.RefUnary(op, x[i] + bias[i % Cols], p0, p1);
            }

            var (scaled, rel) = Rig.Errors(Rig.D(got, got.Length), want);
            Rig.Row(d, "elementwise", "f32", $"fused bias add + {op}, [{Rows},{Cols}]", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
            Assert.True(scaled <= 1e-5, $"{op}: {scaled}");
        }
    }

    [Fact]
    public void CompareAndWhereMatchTheCpuExactly()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        foreach (var half in new[] { false, true })
        {
            var bad = 0;
            var cases = 0;
            foreach (var (sa, sb) in ElementwiseTests.BroadcastCases().Where((_, i) => i % 3 == 0))
            {
                var rs = ElementwiseKernels.BroadcastShape(sa, sb);
                var na = sa.Aggregate(1, (x, y) => x * y);
                var nb = sb.Aggregate(1, (x, y) => x * y);
                var n = rs.Aggregate(1, (x, y) => x * y);
                // Few distinct values so that equality is common, with a NaN in each operand.
                var a = Rig.Random(na, cases, 0, 4).Select(v => MathF.Floor(v)).ToArray();
                var b = Rig.Random(nb, cases + 1, 0, 4).Select(v => MathF.Floor(v)).ToArray();
                a[0] = float.NaN;
                b[^1] = float.NaN;
                var cond = Rig.Random(n, cases + 2, 0, 2).Select(v => (uint)v).ToArray();
                var condShape = rs;
                var ha = Rig.ToHalf(a);
                var hb = Rig.ToHalf(b);
                using var bcond = Rig.Upload(d, cond);
                using var bout = d.CreateBuffer<uint>(Math.Max(1, n));
                foreach (var op in Enum.GetValues<GpuCompareOp>())
                {
                    if (half)
                    {
                        using var ba = Rig.Upload(d, ha);
                        using var bb = Rig.Upload(d, hb);
                        Rig.Run(d, e => k.EncodeCompare(e, op, ba, 0, sa, bb, 0, sb, bout, 0));
                    }
                    else
                    {
                        using var ba = Rig.Upload(d, a);
                        using var bb = Rig.Upload(d, b);
                        Rig.Run(d, e => k.EncodeCompare(e, op, ba, 0, sa, bb, 0, sb, bout, 0));
                    }

                    var got = Rig.Download(bout, n);
                    for (var i = 0; i < n; i++)
                    {
                        var (x, y) = (Pick(a, sa, rs, i), Pick(b, sb, rs, i));
                        var want = op switch { GpuCompareOp.Equal => x == y, GpuCompareOp.Less => x < y, GpuCompareOp.Greater => x > y, GpuCompareOp.LessOrEqual => x <= y, _ => x >= y };
                        bad += got[i] == (want ? 1u : 0u) ? 0 : 1;
                    }
                }

                // Where: values are copied bit for bit.
                if (half)
                {
                    using var ba = Rig.Upload(d, ha);
                    using var bb = Rig.Upload(d, hb);
                    using var bo = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(Math.Max(1, n)));
                    Rig.Run(d, e => k.EncodeWhere(e, bcond, 0, condShape, ba, 0, sa, bb, 0, sb, bo, 0));
                    var got = Rig.Download(bo, n);
                    for (var i = 0; i < n; i++)
                    {
                        var want = cond[i] != 0 ? (Half)Pick(a, sa, rs, i) : (Half)Pick(b, sb, rs, i);
                        bad += Rig.BitMismatches([got[i]], [want]);
                    }
                }
                else
                {
                    using var ba = Rig.Upload(d, a);
                    using var bb = Rig.Upload(d, b);
                    using var bo = d.CreateBuffer<float>(Math.Max(1, n));
                    Rig.Run(d, e => k.EncodeWhere(e, bcond, 0, condShape, ba, 0, sa, bb, 0, sb, bo, 0));
                    var got = Rig.Download(bo, n);
                    for (var i = 0; i < n; i++)
                    {
                        var want = cond[i] != 0 ? Pick(a, sa, rs, i) : Pick(b, sb, rs, i);
                        bad += Rig.BitMismatches([got[i]], [want]);
                    }
                }

                cases++;
            }

            Rig.Row(d, "elementwise", half ? "f16" : "f32", $"compare (5 ops) and where, {cases} broadcast cases with NaN", $"mismatches {bad}");
            Assert.Equal(0, bad);
        }
    }

    static float Pick(float[] v, int[] shape, int[] rs, int flat)
    {
        var idx = new int[rs.Length];
        var rem = flat;
        for (var dim = rs.Length - 1; dim >= 0; dim--)
        {
            idx[dim] = rem % rs[dim];
            rem /= rs[dim];
        }

        var at = 0;
        var stride = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            at += (shape[i] == 1 ? 0 : idx[rs.Length - shape.Length + i]) * stride;
            stride *= shape[i];
        }

        return v[at];
    }

    [Fact]
    public void BadShapesAreRejectedWithTheOperationAndDimension()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        using var a = d.CreateBuffer<float>(12);
        using var b = d.CreateBuffer<float>(12);
        using var o = d.CreateBuffer<float>(12);
        var enc = d.BeginCommands();
        var e1 = Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuBinaryOp.Add, a, 0, [3, 4], b, 0, [2, 4], o, 0));
        Assert.Equal("Binary", e1.Operation);
        Assert.Equal("shape[0]", e1.Argument);
        var e2 = Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuBinaryOp.Add, a, 1, [3, 4], b, 0, [3, 4], o, 0));
        Assert.Equal("a", e2.Argument);
        var e3 = Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuBinaryOp.Add, a, 0, [int.MaxValue, 4], b, 0, [3, 4], o, 0));
        Assert.Contains("shape", e3.Argument, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuBinaryOp.Add, a, 0, [-1], b, 0, [3, 4], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeUnary(enc, GpuUnaryOp.Exp, a, 0, o, 0, 13));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeUnary(enc, (GpuUnaryOp)99, a, 0, o, 0, 3));
        d.Submit(enc);
    }
}
