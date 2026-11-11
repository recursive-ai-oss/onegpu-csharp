namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class SoftmaxTests
{
    /// <summary>Shapes and axes: attention scores, classifier rows, vocabulary rows, long and short rows, and axes that are not the last.</summary>
    internal static readonly (int[] Shape, int Axis)[] Cases =
    [
        ([1], 0), ([3, 1], 1), ([5, 2], 1), ([7, 8], -1), ([4, 63], 1), ([4, 64], 1), ([2, 65], 1), ([1, 384], -1), ([1, 12, 1, 1], -1), ([1, 12, 128, 128], -1), ([1, 12, 512, 512], -1),
        ([2, 12, 7, 7], -1), ([1, 51865], -1), ([8, 51865], 1), ([1, 32769], 1), ([1, 131072], 1), ([2, 3000], 1),
        ([3, 5, 7], 0), ([3, 5, 7], 1), ([2, 12, 64], 1), ([4, 1000, 3], 1), ([6], 0),
        ([4, 15], 1), ([4, 16], 1), ([5, 100], 1), ([3, 1024], 1), ([3, 1025], 1), ([2, 12, 17, 17], -1), ([9, 200, 5], 1),
    ];

    internal static double[] Reference(double[] x, int[] shape, int axis, bool log)
    {
        var rank = shape.Length;
        var a = axis < 0 ? axis + rank : axis;
        var len = shape[a];
        var inner = 1;
        for (var i = a + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        var outer = x.Length / Math.Max(1, len * inner);
        var y = new double[x.Length];
        for (var o = 0; o < outer; o++)
        {
            for (var q = 0; q < inner; q++)
            {
                double m = double.NegativeInfinity;
                var nan = false;
                for (var r = 0; r < len; r++)
                {
                    var v = x[(o * len + r) * inner + q];
                    nan |= double.IsNaN(v);
                    m = Math.Max(m, v);
                }

                double sum = 0;
                for (var r = 0; r < len; r++)
                {
                    sum += Math.Exp(x[(o * len + r) * inner + q] - m);
                }

                for (var r = 0; r < len; r++)
                {
                    var i = (o * len + r) * inner + q;
                    y[i] = nan ? double.NaN : log ? x[i] - m - Math.Log(sum) : Math.Exp(x[i] - m) / sum;
                }
            }
        }

        return y;
    }

    static async Task<(double[] Out, string Hash)> Run(GpuDevice d, SoftmaxKernels k, float[] x, int[] shape, int axis, bool log, bool half, GpuKernelMode mode)
    {
        const int OffIn = 2, OffOut = 4;
        var n = x.Length;
        if (half)
        {
            var hx = new Half[GpuKernelBuffers.PaddedHalfLength(n + OffIn)];
            for (var i = 0; i < n; i++)
            {
                hx[OffIn + i] = (Half)x[i];
            }

            using var bi = d.CreateBuffer<Half>(hx);
            using var bo = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(n + OffOut + 2));
            Rig.Run(d, e => k.Encode(e, bi, OffIn, shape, axis, bo, OffOut, log, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, n).ToArray();
            return (Rig.D(got, n), Rig.Hash<Half>(got));
        }
        else
        {
            var fx = new float[n + OffIn];
            x.CopyTo(fx, OffIn);
            using var bi = d.CreateBuffer<float>(fx);
            using var bo = d.CreateBuffer<float>(n + OffOut + 2);
            Rig.Run(d, e => k.Encode(e, bi, OffIn, shape, axis, bo, OffOut, log, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, n).ToArray();
            return (Rig.D(got, n), Rig.Hash<float>(got));
        }
    }

    static bool HalfOk((int[] Shape, int Axis) c)
    {
        var rank = c.Shape.Length;
        var a = c.Axis < 0 ? c.Axis + rank : c.Axis;
        return a == rank - 1 && c.Shape[a] % 2 == 0;
    }

    [Fact]
    public async Task SoftmaxMatchesTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var withSub = new SoftmaxKernels(d);
        using var noSub = new SoftmaxKernels(d, allowSubgroups: false);
        Rig.Row(d, "softmax", "-", "variant chosen for Fast", $"{withSub.Variant} (forced-off instance: {noSub.Variant})");
        foreach (var half in new[] { false, true })
        {
            foreach (var log in new[] { false, true })
            {
                var worst = new Dictionary<string, (double, double)>();
                foreach (var c in Cases.Where(c => !half || HalfOk(c)))
                {
                    var n = c.Shape.Aggregate(1, (p, q) => p * q);
                    var x = Rig.Random(n, n + c.Axis, -10f, 10f);
                    var xr = half ? x.Select(v => (double)(Half)v).ToArray() : x.Select(v => (double)v).ToArray();
                    var want = Reference(xr, c.Shape, c.Axis, log);
                    if (half)
                    {
                        want = want.Select(v => (double)(Half)v).ToArray();
                    }

                    foreach (var (name, kernels, mode) in new[] { ("fast", withSub, GpuKernelMode.Fast), ("fast-no-subgroups", noSub, GpuKernelMode.Fast), ("deterministic", withSub, GpuKernelMode.Deterministic) })
                    {
                        var (got, _) = await Run(d, kernels, x, c.Shape, c.Axis, log, half, mode);
                        var (scaled, rel) = Rig.Errors(got, want);
                        var limit = half ? 2e-3 : 1e-5;
                        Assert.True(scaled <= limit, $"{(log ? "log" : "")}softmax {(half ? "f16" : "f32")} {name} [{string.Join(",", c.Shape)}] axis {c.Axis}: {scaled}");
                        worst[name] = worst.TryGetValue(name, out var w) ? (Math.Max(w.Item1, scaled), Math.Max(w.Item2, rel)) : (scaled, rel);
                    }
                }

                foreach (var (name, (scaled, rel)) in worst)
                {
                    Rig.Row(d, "softmax", half ? "f16" : "f32", $"{(log ? "log-softmax" : "softmax")}, {name}, {Cases.Count(c => !half || HalfOk(c))} shapes (rows 1..131072 long)", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
                }
            }
        }
    }

    [Fact]
    public async Task LargeOffsetsAndMasksStayStable()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new SoftmaxKernels(d);
        var bad = 0;
        foreach (var len in new[] { 1, 5, 8, 64, 300, 40000 })
        {
            foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
            {
                // Row 0: huge common offset. Row 1: additive mask of -inf on three quarters. Row 2: mask of -1e9. Row 3: single finite entry.
                var rows = 4;
                var x = Rig.Random(rows * len, len, -5f, 5f);
                for (var i = 0; i < len; i++)
                {
                    x[i] += 1000f;
                    if (i % 4 != 0)
                    {
                        x[len + i] = float.NegativeInfinity;
                    }

                    if (i % 4 != 1)
                    {
                        x[2 * len + i] = -1e9f;
                    }

                    x[3 * len + i] = float.NegativeInfinity;
                }

                x[3 * len + len / 2] = 3f;
                var want = Reference(x.Select(v => (double)v).ToArray(), [rows, len], 1, false);
                var (got, _) = await Run(d, k, x, [rows, len], 1, false, false, mode);
                var (scaled, _) = Rig.Errors(got, want);
                if (!(scaled <= 1e-5))
                {
                    bad++;
                    Console.WriteLine($"mask len {len} {mode}: scaled {scaled}");
                }
            }
        }

        Rig.Row(d, "softmax", "f32", "offset +1000, additive -inf mask, -1e9 mask, single finite entry; row lengths 1, 5, 8, 64, 300, 40000", $"failures {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task SpecialRowsGiveNan()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new SoftmaxKernels(d);
        if (Rig.SkipInBrowser(d, "softmax", "rows of all -inf, with NaN and with +inf"))
        {
            return;
        }

        var bad = 0;
        foreach (var len in new[] { 6, 300 })
        {
            // Row 0 all -inf, row 1 holds a NaN, row 2 holds +inf, row 3 is ordinary.
            var x = Rig.Random(4 * len, 1, -2f, 2f);
            for (var i = 0; i < len; i++)
            {
                x[i] = float.NegativeInfinity;
            }

            x[len + len / 2] = float.NaN;
            x[2 * len + 1] = float.PositiveInfinity;
            foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
            {
                var (got, _) = await Run(d, k, x, [4, len], 1, false, false, mode);
                for (var i = 0; i < 3 * len; i++)
                {
                    bad += double.IsNaN(got[i]) ? 0 : 1;
                }

                var want = Reference(x.Skip(3 * len).Select(v => (double)v).ToArray(), [1, len], 1, false);
                var (scaled, _) = Rig.Errors(got.Skip(3 * len).ToArray(), want);
                bad += scaled <= 1e-5 ? 0 : 1;
            }
        }

        Rig.Row(d, "softmax", "f32", "rows of all -inf, with NaN, with +inf give NaN; a clean row beside them is right", $"failures {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task DeterministicSoftmaxIsBitIdenticalAcrossRuns()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new SoftmaxKernels(d);
        foreach (var half in new[] { false, true })
        {
            var same = 0;
            var runs = 0;
            string example = "";
            foreach (var c in Cases.Where(c => !half || HalfOk(c)))
            {
                var n = c.Shape.Aggregate(1, (p, q) => p * q);
                var x = Rig.Random(n, n + 3, -10f, 10f);
                foreach (var log in new[] { false, true })
                {
                    var (_, h1) = await Run(d, k, x, c.Shape, c.Axis, log, half, GpuKernelMode.Deterministic);
                    var (_, h2) = await Run(d, k, x, c.Shape, c.Axis, log, half, GpuKernelMode.Deterministic);
                    runs++;
                    same += h1 == h2 ? 1 : 0;
                    example = $"[{string.Join(",", c.Shape)}] {h1}/{h2}";
                }
            }

            Rig.Row(d, "softmax", half ? "f16" : "f32", $"deterministic softmax and log-softmax run twice on {runs / 2} shapes", $"{same}/{runs} identical; last {example}");
            Assert.Equal(runs, same);
        }
    }

    [Fact]
    public async Task BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new SoftmaxKernels(d);
        using var f = d.CreateBuffer<float>(24);
        using var o = d.CreateBuffer<float>(24);
        using var h = d.CreateBuffer<Half>(24);
        using var ho = d.CreateBuffer<Half>(24);
        var enc = d.BeginCommands();
        Assert.Equal("axis", Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, f, 0, [4, 6], 2, o, 0)).Argument);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, f, 1, [4, 6], 1, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, h, 0, [4, 6], 0, ho, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, h, 1, [4, 6], 1, ho, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, h, 0, [8, 3], 1, ho, 0));
        d.Submit(enc);
    }
}
