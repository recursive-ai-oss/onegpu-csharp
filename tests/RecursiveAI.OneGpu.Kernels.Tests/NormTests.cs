namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class NormTests
{
    internal static readonly (int Rows, int Len)[] Cases = [(1, 1), (3, 5), (7, 31), (2, 257), (1, 384), (128, 384), (512, 768), (13, 768), (4, 1536), (4, 3072), (3, 4096), (1, 20000)];

    static double[] Reference(double[] x, int rows, int len, double[]? scale, double[]? bias, bool rms, double eps)
    {
        var y = new double[x.Length];
        for (var r = 0; r < rows; r++)
        {
            double mean = 0;
            if (!rms)
            {
                for (var c = 0; c < len; c++)
                {
                    mean += x[r * len + c];
                }

                mean /= len;
            }

            double v = 0;
            for (var c = 0; c < len; c++)
            {
                var dlt = x[r * len + c] - mean;
                v += dlt * dlt;
            }

            v /= len;
            var inv = 1 / Math.Sqrt(v + eps);
            for (var c = 0; c < len; c++)
            {
                var o = (x[r * len + c] - mean) * inv;
                if (scale is not null)
                {
                    o *= scale[c];
                }

                if (bias is not null && !rms)
                {
                    o += bias[c];
                }

                y[r * len + c] = o;
            }
        }

        return y;
    }

    static async Task<(double[] Out, string Hash)> Run(GpuDevice d, NormKernels k, bool rms, float[] x, int rows, int len, float[]? scale, float[]? bias, float eps, bool half, GpuKernelMode mode)
    {
        const int OffIn = 2, OffOut = 4, OffS = 2, OffB = 6;
        var n = rows * len;
        if (half)
        {
            Half[] H(float[] v, int off)
            {
                var h = new Half[GpuKernelBuffers.PaddedHalfLength(v.Length + off)];
                for (var i = 0; i < v.Length; i++)
                {
                    h[off + i] = (Half)v[i];
                }

                return h;
            }

            using var bi = d.CreateBuffer<Half>(H(x, OffIn));
            using var bs = scale is null ? null : d.CreateBuffer<Half>(H(scale, OffS));
            using var bb = bias is null ? null : d.CreateBuffer<Half>(H(bias, OffB));
            using var bo = d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(n + OffOut + 2));
            Rig.Run(d, e => k.Encode(e, rms, bi, OffIn, rows, len, bs, OffS, bb, OffB, bo, OffOut, eps, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, n).ToArray();
            return (Rig.D(got, n), Rig.Hash<Half>(got));
        }
        else
        {
            float[] F(float[] v, int off)
            {
                var f = new float[v.Length + off];
                v.CopyTo(f, off);
                return f;
            }

            using var bi = d.CreateBuffer<float>(F(x, OffIn));
            using var bs = scale is null ? null : d.CreateBuffer<float>(F(scale, OffS));
            using var bb = bias is null ? null : d.CreateBuffer<float>(F(bias, OffB));
            using var bo = d.CreateBuffer<float>(n + OffOut + 2);
            Rig.Run(d, e => k.Encode(e, rms, bi, OffIn, rows, len, bs, OffS, bb, OffB, bo, OffOut, eps, mode));
            var got = (await Rig.DownloadAsync(bo)).AsSpan(OffOut, n).ToArray();
            return (Rig.D(got, n), Rig.Hash<float>(got));
        }
    }

    [Fact]
    public async Task NormalisationMatchesTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var withSub = new NormKernels(d);
        using var noSub = new NormKernels(d, allowSubgroups: false);
        Rig.Row(d, "norm", "-", "variant chosen for Fast", $"{withSub.Variant} (forced-off instance: {noSub.Variant})");
        foreach (var half in new[] { false, true })
        {
            foreach (var rms in new[] { false, true })
            {
                var worst = new Dictionary<string, (double, double)>();
                var count = 0;
                foreach (var (rows, len) in Cases.Where(c => !half || c.Len % 2 == 0))
                {
                    foreach (var affine in new[] { 0, 1, 2 })
                    {
                        var eps = affine == 2 ? 1e-12f : 1e-5f;
                        var x = Rig.Random(rows * len, rows + len, -2f, 4f);
                        var scale = affine >= 1 ? Rig.Random(len, 3, 0.5f, 1.5f) : null;
                        var bias = affine == 2 ? Rig.Random(len, 4, -1f, 1f) : null;
                        double[] R(float[] v) => half ? v.Select(q => (double)(Half)q).ToArray() : v.Select(q => (double)q).ToArray();
                        var want = Reference(R(x), rows, len, scale is null ? null : R(scale), bias is null ? null : R(bias), rms, eps);
                        if (half)
                        {
                            want = want.Select(v => (double)(Half)v).ToArray();
                        }

                        foreach (var (name, kernels, mode) in new[] { ("fast", withSub, GpuKernelMode.Fast), ("fast-no-subgroups", noSub, GpuKernelMode.Fast), ("deterministic", withSub, GpuKernelMode.Deterministic) })
                        {
                            var (got, _) = await Run(d, kernels, rms, x, rows, len, scale, bias, eps, half, mode);
                            var (scaled, rel) = Rig.Errors(got, want);
                            var limit = half ? 2e-3 : 1e-5;
                            Assert.True(scaled <= limit, $"{(rms ? "rms" : "layer")} {(half ? "f16" : "f32")} {name} [{rows},{len}] affine {affine}: {scaled}");
                            worst[name] = worst.TryGetValue(name, out var w) ? (Math.Max(w.Item1, scaled), Math.Max(w.Item2, rel)) : (scaled, rel);
                        }

                        count++;
                    }
                }

                foreach (var (name, (scaled, rel)) in worst)
                {
                    Rig.Row(d, "norm", half ? "f16" : "f32", $"{(rms ? "rms" : "layer")}, {name}, {count} shape/scale/bias combinations (rows to 20000 long)", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
                }
            }
        }
    }

    [Fact]
    public async Task DeterministicNormalisationIsBitIdenticalAcrossRuns()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new NormKernels(d);
        foreach (var half in new[] { false, true })
        {
            var same = 0;
            var runs = 0;
            foreach (var (rows, len) in Cases.Where(c => !half || c.Len % 2 == 0))
            {
                var x = Rig.Random(rows * len, 11, -2f, 4f);
                var scale = Rig.Random(len, 3, 0.5f, 1.5f);
                var bias = Rig.Random(len, 4, -1f, 1f);
                foreach (var rms in new[] { false, true })
                {
                    var (_, h1) = await Run(d, k, rms, x, rows, len, scale, bias, 1e-5f, half, GpuKernelMode.Deterministic);
                    var (_, h2) = await Run(d, k, rms, x, rows, len, scale, bias, 1e-5f, half, GpuKernelMode.Deterministic);
                    runs++;
                    same += h1 == h2 ? 1 : 0;
                }
            }

            Rig.Row(d, "norm", half ? "f16" : "f32", "deterministic layer and RMS norm run twice per shape", $"{same}/{runs} identical");
            Assert.Equal(runs, same);
        }
    }

    [Fact]
    public async Task BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new NormKernels(d);
        using var x = d.CreateBuffer<float>(24);
        using var s = d.CreateBuffer<float>(4);
        using var o = d.CreateBuffer<float>(24);
        using var h = d.CreateBuffer<Half>(24);
        using var ho = d.CreateBuffer<Half>(24);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, false, x, 0, 4, 0, null, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, false, x, 0, 4, 6, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, false, x, 0, 5, 6, null, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, false, h, 0, 8, 3, null, 0, null, 0, ho, 0));
        d.Submit(enc);
    }
}
