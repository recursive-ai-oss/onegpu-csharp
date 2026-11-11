namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class ReduceTests
{
    /// <summary>Shapes and axes of the transformer reference models plus edge cases.</summary>
    internal static readonly (int[] Shape, int[] Axes)[] Cases =
    [
        ([1, 384], [-1]), ([1, 1, 384], [2]), ([4, 128, 384], [2]), ([2, 512, 768], [2]), ([1, 7, 768], [2]), ([1, 3072], [-1]), ([12, 64, 1536], [2]),
        ([2, 512, 768], [1]), ([2, 12, 128, 64], [2]), ([2, 512, 768], [0, 2]), ([2, 12, 33, 64], [1, 3]), ([2, 512, 768], []),
        ([1, 51865], [1]), ([8, 51865], [-1]), ([300000], [0]),
        ([1], [0]), ([1, 1], [0, 1]), ([5, 1, 7], [1]), ([3, 5, 7], [0]), ([3, 5, 7], [1]), ([3, 5, 7], [2]), ([3, 5, 7], [0, 1]), ([3, 5, 7], [0, 2]), ([2, 3, 4, 5, 6], [1, 3]),
        ([16384], [0]), ([16385], [0]), ([3, 100000], [1]), ([100000, 3], [0]), ([127, 16384], [1]), ([128, 16384], [1]), ([2, 5, 40000], [0, 2]),
        ([3, 0, 4], [1]), ([255], [0]), ([256], [0]), ([257], [0]), ([33, 31], [1]), ([100, 200], [0]),
    ];

    static float[] Data(GpuReduceOp op, int n, int seed) => op switch
    {
        GpuReduceOp.Prod => Rig.Random(n, seed, 0.999f, 1.001f),
        GpuReduceOp.LogSum => Rig.Random(n, seed, 0.1f, 2f),
        // Mostly positive, so that sums are well conditioned: the achievable error of any float sum is its rounding error times sum|x| / |sum|.
        _ => Rig.Random(n, seed, -0.5f, 1f),
    };

    internal static double[] Reference(GpuReduceOp op, double[] x, int[] shape, int[] axes, out int count)
    {
        var rank = shape.Length;
        var red = new bool[rank];
        foreach (var a in axes.Length == 0 ? Enumerable.Range(0, rank).ToArray() : axes)
        {
            red[a < 0 ? a + rank : a] = true;
        }

        var outShape = shape.Where((_, i) => !red[i]).ToArray();
        var numOut = outShape.Aggregate(1, (p, q) => p * q);
        var r = shape.Where((_, i) => red[i]).Aggregate(1, (p, q) => p * q);
        count = numOut;
        var acc = new double[numOut];
        Array.Fill(acc, op switch { GpuReduceOp.Max => double.NegativeInfinity, GpuReduceOp.Min => double.PositiveInfinity, GpuReduceOp.Prod => 1.0, _ => 0.0 });
        var idx = new int[rank];
        for (var i = 0; i < x.Length; i++)
        {
            var rem = i;
            for (var d = rank - 1; d >= 0; d--)
            {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }

            var o = 0;
            for (var d = 0; d < rank; d++)
            {
                if (!red[d])
                {
                    o = o * shape[d] + idx[d];
                }
            }

            var v = x[i];
            acc[o] = op switch
            {
                GpuReduceOp.Sum or GpuReduceOp.Mean or GpuReduceOp.LogSum => acc[o] + v,
                GpuReduceOp.Max => double.IsNaN(acc[o]) || double.IsNaN(v) ? double.NaN : Math.Max(acc[o], v),
                GpuReduceOp.Min => double.IsNaN(acc[o]) || double.IsNaN(v) ? double.NaN : Math.Min(acc[o], v),
                GpuReduceOp.Prod => acc[o] * v,
                GpuReduceOp.L2 or GpuReduceOp.SumSquare => acc[o] + v * v,
                _ => acc[o] + Math.Abs(v),
            };
        }

        for (var o = 0; o < numOut; o++)
        {
            acc[o] = op switch { GpuReduceOp.Mean => acc[o] / r, GpuReduceOp.L2 => Math.Sqrt(acc[o]), GpuReduceOp.LogSum => Math.Log(acc[o]), _ => acc[o] };
        }

        return acc;
    }

    // Runs one reduction; returns the output as doubles and a hash of its raw bits.
    internal static async Task<(double[] Out, string Hash)> Run(GpuDevice d, ReduceKernels k, GpuReduceOp op, float[] x, int[] shape, int[] axes, int numOut, bool half, GpuKernelMode mode)
    {
        const int OffIn = 1, OffOut = 3;
        if (half)
        {
            var hx = new Half[GpuKernelBuffers.PaddedHalfLength(x.Length + OffIn)];
            for (var i = 0; i < x.Length; i++)
            {
                hx[OffIn + i] = (Half)x[i];
            }

            using var bi = d.CreateBuffer<Half>(hx);
            using var bo = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(numOut + OffOut + 1));
            Rig.Run(d, e => k.EncodeReduce(e, op, bi, OffIn, shape, axes, bo, OffOut, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, numOut).ToArray();
            return (Rig.D(got, numOut), Rig.Hash<Half>(got));
        }
        else
        {
            var fx = new float[x.Length + OffIn];
            x.CopyTo(fx, OffIn);
            using var bi = d.CreateBuffer<float>(fx);
            using var bo = d.CreateBuffer<float>(numOut + OffOut + 1);
            Rig.Run(d, e => k.EncodeReduce(e, op, bi, OffIn, shape, axes, bo, OffOut, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, numOut).ToArray();
            return (Rig.D(got, numOut), Rig.Hash<float>(got));
        }
    }

    [Fact]
    public async Task ReductionsMatchTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var withSub = new ReduceKernels(d);
        using var noSub = new ReduceKernels(d, allowSubgroups: false);
        Rig.Row(d, "reduce", "-", "variant chosen for Fast", $"{withSub.Variant} (forced-off instance: {noSub.Variant}); subgroup size {d.Capabilities.Compute.SubgroupSize} in [{d.Capabilities.Compute.SubgroupMinSize}, {d.Capabilities.Compute.SubgroupMaxSize}]");
        foreach (var half in new[] { false, true })
        {
            foreach (var op in Enum.GetValues<GpuReduceOp>())
            {
                var worst = new Dictionary<string, (double Scaled, double Rel)>();
                foreach (var (shape, axes) in Cases)
                {
                    var n = shape.Aggregate(1, (p, q) => p * q);
                    if (Rig.Browser && shape.Contains(0) && op is GpuReduceOp.Mean or GpuReduceOp.Max or GpuReduceOp.Min or GpuReduceOp.LogSum)
                    {
                        // The result of an empty reduction is NaN or an infinity, which WGSL does not guarantee.
                        continue;
                    }

                    var x = Data(op, n, n + (int)op);
                    var xr = half ? x.Select(v => (double)(Half)v).ToArray() : x.Select(v => (double)v).ToArray();
                    var want = Reference(op, xr, shape, axes, out var numOut);
                    if (half)
                    {
                        want = want.Select(v => (double)(Half)v).ToArray();
                    }

                    foreach (var (name, kernels, mode) in new[] { ("fast", withSub, GpuKernelMode.Fast), ("fast-no-subgroups", noSub, GpuKernelMode.Fast), ("deterministic", withSub, GpuKernelMode.Deterministic) })
                    {
                        var (got, _) = await Run(d, kernels, op, x, shape, axes, numOut, half, mode);
                        var (scaled, rel) = Rig.Errors(got, want);
                        var limit = half ? 2e-3 : 1e-5;
                        // Max and Min select an input, so they are exact.
                        if (op is GpuReduceOp.Max or GpuReduceOp.Min)
                        {
                            limit = 0;
                        }

                        // A product of hundreds of thousands of factors accumulates a rounding error per multiplication, whatever the order.
                        if (op == GpuReduceOp.Prod && !half)
                        {
                            limit = Math.Max(1e-5, 1e-6 * Math.Sqrt(n));
                        }

                        Assert.True(scaled <= limit, $"{op} {(half ? "f16" : "f32")} {name} shape [{string.Join(",", shape)}] axes [{string.Join(",", axes)}]: scaled error {scaled} > {limit}");
                        worst[name] = worst.TryGetValue(name, out var w) ? (Math.Max(w.Scaled, scaled), Math.Max(w.Rel, rel)) : (scaled, rel);
                    }
                }

                foreach (var (name, (scaled, rel)) in worst)
                {
                    Rig.Row(d, "reduce", half ? "f16" : "f32", $"{op}, {name}, {Cases.Length} shapes/axis sets", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
                }
            }
        }
    }

    [Fact]
    public async Task ReductionsPropagateNanAndInfinity()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ReduceKernels(d);
        if (Rig.SkipInBrowser(d, "reduce", "reductions with NaN and infinity inside a row"))
        {
            return;
        }

        var bad = 0;
        foreach (var length in new[] { 5, 300 })
        {
            foreach (var (label, poison) in new (string, float)[] { ("NaN", float.NaN), ("+inf", float.PositiveInfinity), ("-inf", float.NegativeInfinity) })
            {
                foreach (var op in new[] { GpuReduceOp.Sum, GpuReduceOp.Mean, GpuReduceOp.Max, GpuReduceOp.Min, GpuReduceOp.L2, GpuReduceOp.SumSquare, GpuReduceOp.L1 })
                {
                    // Row 0 is clean, row 1 holds the special value in the middle.
                    var x = Rig.Random(2 * length, 1, -1f, 1f);
                    x[length + length / 2] = poison;
                    var want = Reference(op, x.Select(v => (double)v).ToArray(), [2, length], [1], out _);
                    foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
                    {
                        var (got, _) = await Run(d, k, op, x, [2, length], [1], 2, false, mode);
                        var (scaled, _) = Rig.Errors(got, want);
                        if (!(scaled <= 1e-5))
                        {
                            bad++;
                            Console.WriteLine($"{op} {label} length {length} {mode}: got [{string.Join(", ", got)}] want [{string.Join(", ", want)}]");
                        }
                    }
                }
            }
        }

        Rig.Row(d, "reduce", "f32", "NaN, +inf, -inf inside a row (sum, mean, max, min, l2, sumsq, l1; lengths 5 and 300; both modes)", $"disagreements with IEEE reference {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task DeterministicReductionsAreBitIdenticalAcrossRuns()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ReduceKernels(d);
        foreach (var half in new[] { false, true })
        {
            var same = 0;
            var runs = 0;
            string first = "", last = "";
            foreach (var (shape, axes) in Cases)
            {
                var n = shape.Aggregate(1, (p, q) => p * q);
                foreach (var op in new[] { GpuReduceOp.Sum, GpuReduceOp.Mean, GpuReduceOp.L2 })
                {
                    var x = Data(op, n, n + 7);
                    var numOut = shape.Select((dim, i) => axes.Length != 0 && !axes.Select(a => a < 0 ? a + shape.Length : a).Contains(i) ? dim : 1).Aggregate(1, (p, q) => p * q);
                    var (_, h1) = await Run(d, k, op, x, shape, axes, numOut, half, GpuKernelMode.Deterministic);
                    var (_, h2) = await Run(d, k, op, x, shape, axes, numOut, half, GpuKernelMode.Deterministic);
                    runs++;
                    same += h1 == h2 ? 1 : 0;
                    if (h1 != h2)
                    {
                        Console.WriteLine($"NOT IDENTICAL {op} half={half} shape [{string.Join(",", shape)}] axes [{string.Join(",", axes)}] numOut {numOut}: {h1} vs {h2}");
                    }

                    if (runs == 1)
                    {
                        first = $"[{string.Join(",", shape)}] {h1}/{h2}";
                    }

                    last = $"[{string.Join(",", shape)}] {h1}/{h2}";
                }
            }

            Rig.Row(d, "reduce", half ? "f16" : "f32", $"deterministic Sum/Mean/L2 run twice on {Cases.Length} shapes", $"{same}/{runs} identical; e.g. {first}, {last}");
            Assert.Equal(runs, same);
        }
    }

    [Fact]
    public async Task ArgMaxAndArgMinMatchTheCpu()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ReduceKernels(d);
        int[][] shapes = [[1], [7], [1, 384], [3, 51865], [4, 5, 6], [2, 300, 9], [257], [1000, 3], [2, 0 + 1, 33], [16, 12, 64]];
        foreach (var half in new[] { false, true })
        {
            var bad = 0;
            var runs = 0;
            foreach (var shape in shapes)
            {
                var n = shape.Aggregate(1, (p, q) => p * q);
                for (var axis = 0; axis < shape.Length; axis++)
                {
                    foreach (var op in new[] { GpuArgOp.Max, GpuArgOp.Min })
                    {
                        // Quantised values force ties; a NaN is planted in some rows.
                        var x = Rig.Random(n, n + axis, -3f, 3f).Select(v => MathF.Round(v * 4f) / 4f).ToArray();
                        if (n > 10)
                        {
                            x[n / 2] = float.NaN;
                        }

                        var xd = half ? x.Select(v => (double)(Half)v).ToArray() : x.Select(v => (double)v).ToArray();
                        var numOut = n / shape[axis];
                        using var bo = d.CreateBuffer<long>(new long[numOut + 1]);
                        if (half)
                        {
                            using var bi = d.CreateBuffer<Half>(Rig.ToHalf(x));
                            Rig.Run(d, e => k.EncodeArg(e, op, bi, 0, shape, axis, bo, 1));
                        }
                        else
                        {
                            using var bi = d.CreateBuffer<float>(x);
                            Rig.Run(d, e => k.EncodeArg(e, op, bi, 0, shape, axis, bo, 1));
                        }

                        var got = await Rig.DownloadAsync(bo);
                        var inner = 1;
                        for (var i = axis + 1; i < shape.Length; i++)
                        {
                            inner *= shape[i];
                        }

                        var outer = n / (shape[axis] * inner);
                        for (var o = 0; o < outer; o++)
                        {
                            for (var q = 0; q < inner; q++)
                            {
                                var best = 0;
                                for (var r = 1; r < shape[axis]; r++)
                                {
                                    var cur = xd[(o * shape[axis] + best) * inner + q];
                                    var v = xd[(o * shape[axis] + r) * inner + q];
                                    var take = !double.IsNaN(cur) && (double.IsNaN(v) || (op == GpuArgOp.Max ? v > cur : v < cur));
                                    if (take)
                                    {
                                        best = r;
                                    }
                                }

                                if (got[1 + o * inner + q] != best)
                                {
                                    if (bad < 6)
                                    {
                                        Console.WriteLine($"arg {op} half={half} shape [{string.Join(",", shape)}] axis {axis} out ({o},{q}): gpu {got[1 + o * inner + q]} cpu {best}");
                                    }

                                    bad++;
                                }
                            }
                        }

                        bad += got[0] == 0 ? 0 : 1;
                        runs++;
                    }
                }
            }

            Rig.Row(d, "argreduce", half ? "f16" : "f32", $"ArgMax/ArgMin over every axis of {shapes.Length} shapes with ties and NaN ({runs} runs)", $"wrong indices {bad}");
            Assert.Equal(0, bad);
        }
    }

    [Fact]
    public async Task BadAxesAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ReduceKernels(d);
        using var a = d.CreateBuffer<float>(24);
        using var o = d.CreateBuffer<float>(24);
        var enc = d.BeginCommands();
        var e1 = Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, GpuReduceOp.Sum, a, 0, [2, 3, 4], [3], o, 0));
        Assert.Equal("axes", e1.Argument);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, GpuReduceOp.Sum, a, 0, [2, 3, 4], [1, -2], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, GpuReduceOp.Sum, a, 1, [2, 3, 4], [1], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, GpuReduceOp.Sum, a, 0, [2, 3, 4], [1], o, 23));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, (GpuReduceOp)42, a, 0, [2, 3, 4], [1], o, 0));
        // Alternating kept and reduced dimensions beyond what the kernel addresses.
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeReduce(enc, GpuReduceOp.Sum, a, 0, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 4, 6, 8], o, 0));
        d.Submit(enc);
    }
}
