using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Time of the convolution layers of the vision models, from GPU timestamps (the passes of one layer summed, median over runs) and from the host
/// clock (a command buffer of repeated runs between submit and idle). Runs only with ONEGPU_CONV_BENCH=1 (ONEGPU_CONV_BENCH_MODE=deterministic for the deterministic mode); the numbers are indicative.
/// A model's total counts each distinct shape as often as the model uses it.
/// </summary>
public sealed class ConvThroughput
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_CONV_BENCH=1 to time the vision layers");
        }

        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var verbose = Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_LAYERS") == "1";
        var deterministic = Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_MODE") == "deterministic";
        var noHardware = Environment.GetEnvironmentVariable("ONEGPU_GEMM_NO_HARDWARE") == "1";
        var batches = Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_BATCHES") is { Length: > 0 } b ? b.Split(',').Select(int.Parse).ToArray() : [1, 8];
        foreach (var half in new[] { false, true })
        {
            using var k = ConvTests.Make(d, deterministic ? GpuKernelMode.Deterministic : GpuKernelMode.Fast, !noHardware);
            k.AllowKSplit = Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_NOSPLIT") != "1";
            k.AllowCoopImplicit = Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_NOCOOP") != "1";
            foreach (var batch in batches)
            {
                foreach (var (model, layers) in new (string, List<ConvTests.Layer>)[] { ("ResNet-50", ConvTests.ResNet50(batch, 224)), ("MobileNetV3-large", ConvTests.MobileNetV3(batch, 224)), ("DBNet 512", ConvTests.DbNet(batch, 512)) })
                {
                    double gpuTotal = 0, wallTotal = 0, flops = 0;
                    var byVariant = new Dictionary<ConvVariant, double>();
                    foreach (var l in layers)
                    {
                        if (Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH_ONLY") is { Length: > 0 } only && !l.Name.Contains(only, StringComparison.Ordinal))
                        {
                            continue;
                        }

                        var p = l.P;
                        p.Activation = l.Act;
                        var (g, w) = half ? Time<Half>(d, k, p, l.Bias) : Time<float>(d, k, p, l.Bias);
                        gpuTotal += g * l.Count;
                        wallTotal += w * l.Count;
                        flops += p.Flops * l.Count;
                        var v = ConvKernels.ChooseVariant(p);
                        byVariant[v] = byVariant.GetValueOrDefault(v) + g * l.Count;
                        if (verbose)
                        {
                            Rig.Row(d, "conv-bench-layer", half ? "f16" : "f32", $"{model} b{batch} {l.Name} {p.InChannels}->{p.OutChannels} {p.Height}x{p.Width} k{p.KernelHeight} s{p.StrideHeight} g{p.Groups} x{l.Count} [{v}]", $"{g / 1e3:F1} us gpu, {w / 1e3:F1} us wall, {p.Flops / g:F0} GFLOP/s");
                        }
                    }

                    var split = string.Join(", ", byVariant.Select(v => $"{v.Key} {v.Value / 1e6:F2} ms"));
                    Rig.Row(d, "conv-bench-sum", half ? "f16" : "f32", $"{model} {(deterministic ? "deterministic" : noHardware ? "portable" : "fast")} b{batch}", $"{gpuTotal / 1e6:F2} ms gpu, {wallTotal / 1e6:F2} ms wall, {flops / gpuTotal / 1e3:F2} TFLOP/s gpu, {flops / wallTotal / 1e3:F2} TFLOP/s wall; {split}");
                }
            }
        }
    }

    // Returns (GPU ns per call, host ns per call). Without timestamp queries the GPU figure is the host one.
    static (double Gpu, double Wall) Time<T>(GpuDevice d, ConvKernels k, ConvProblem p, bool bias)
        where T : unmanaged
    {
        var timestamps = d.Capabilities.Timestamps;
        var cinG = p.InChannels / p.Groups;
        var nx = p.Batch * p.InChannels * p.Height * p.Width;
        var nw = p.OutChannels * cinG * p.KernelHeight * p.KernelWidth;
        var ny = p.Batch * p.OutChannels * p.OutHeight * p.OutWidth;
        var half = typeof(T) == typeof(Half);
        using var x = ConvTests.Upload<T>(d, Rig.Random(Math.Min(nx, 1 << 20), 1, -1f, 1f).Concat(new float[Math.Max(0, nx - (1 << 20))]).ToArray(), 0, 0f);
        using var w = ConvTests.Upload<T>(d, Rig.Random(nw, 2, -0.1f, 0.1f), 0, 0f);
        using var bb = ConvTests.Upload<T>(d, new float[p.OutChannels], 0, 0f);
        using var y = ConvTests.Upload<T>(d, new float[ny], 0, 0f);
        const int Warm = 2;
        const int Iterations = 10;
        const int MaxPasses = 8;
        void Record(GpuCommandEncoder e)
        {
            if (half)
            {
                k.Encode(e, p, (GpuBuffer<Half>)(object)x, (GpuBuffer<Half>)(object)w, bias ? (GpuBuffer<Half>)(object)bb : null, null, (GpuBuffer<Half>)(object)y);
            }
            else
            {
                k.Encode(e, p, (GpuBuffer<float>)(object)x, (GpuBuffer<float>)(object)w, bias ? (GpuBuffer<float>)(object)bb : null, null, (GpuBuffer<float>)(object)y);
            }
        }

        for (var i = 0; i < Warm; i++)
        {
            var e = d.BeginCommands();
            Record(e);
            d.Submit(e);
        }

        d.WaitIdle();
        // One command buffer of Iterations calls, timed on the host; a second, with pass timestamps, for the GPU figure.
        var enc = d.BeginCommands();
        for (var i = 0; i < Iterations; i++)
        {
            Record(enc);
        }

        var sw = Stopwatch.StartNew();
        d.Submit(enc);
        d.WaitIdle();
        var wall = sw.Elapsed.TotalMilliseconds * 1e6 / Iterations;
        if (!timestamps)
        {
            return (wall, wall);
        }

        using var set = d.CreateTimestampSet(2 * Iterations * MaxPasses);
        var enc2 = d.BeginCommands();
        enc2.BeginPassTiming(new PassTiming(set, 0, Iterations * MaxPasses));
        for (var i = 0; i < Iterations; i++)
        {
            Record(enc2);
        }

        var passes = (int)enc2.EndPassTiming();
        d.Submit(enc2);
        d.WaitIdle();
        var t = new ulong[2 * passes];
        Assert.True(d.TryReadTimestamps(set, 0, t));
        var per = passes / Iterations;
        var sums = new double[Iterations];
        for (var i = 0; i < Iterations; i++)
        {
            for (var j = 0; j < per; j++)
            {
                var q = (i * per) + j;
                sums[i] += t[(2 * q) + 1] - t[2 * q];
            }
        }

        Array.Sort(sums);
        return (sums[Iterations / 2], wall);
    }
}
