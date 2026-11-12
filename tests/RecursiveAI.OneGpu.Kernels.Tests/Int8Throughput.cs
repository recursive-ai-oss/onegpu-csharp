using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Throughput of int8 GEMM (tera-operations per second, a multiply-add counting as two) and of embedding gathers (GB/s of table rows moved),
/// from GPU timestamps (median per dispatch) and from the host clock. Runs only with ONEGPU_GEMM_BENCH=1; the numbers it prints are indicative.
/// </summary>
public sealed class Int8Throughput
{
    static readonly (string Name, int M, int N, int K)[] Shapes =
    [
        ("bert ffn-up 8x128", 1024, 3072, 768),
        ("bert ffn-down 8x128", 1024, 768, 3072),
        ("bert proj 8x128", 1024, 768, 768),
        ("minilm proj 1x128", 128, 384, 384),
        ("minilm ffn-up 1x128", 128, 1536, 384),
        ("minilm ffn-down 1x128", 128, 384, 1536),
        ("minilm ffn-up 8x128", 1024, 1536, 384),
        ("whisper enc proj 1500", 1500, 768, 768),
        ("whisper enc ffn-up 1500", 1500, 3072, 768),
        ("whisper dec proj 1", 1, 768, 768),
        ("whisper dec ffn-up 1", 1, 3072, 768),
        ("whisper dec ffn-down 1", 1, 768, 3072),
        ("whisper dec logits 1", 1, 51865, 768),
        ("whisper dec proj 4", 4, 768, 768),
        ("square 2048", 2048, 2048, 2048),
    ];

    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_GEMM_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_GEMM_BENCH=1 to time int8 GEMM and gathers");
        }

        using var f = new GemmFixture();
        var d = f.Device;
        // Each variant the device can run, with the tile chosen by size (auto) and, for the packed-dot variant, with each tile forced.
        var gemms = new List<(string Tile, Int8Gemm Gemm)>();
        var hasDot = d.Capabilities.Compute.Int8DotProduct;
        foreach (var allowDot in new[] { true, false })
        {
            foreach (var tile in allowDot && hasDot ? new[] { 0, 32, 64 } : new[] { 0 })
            {
                var g = new Int8Gemm(d, new Int8GemmOptions { AllowPackedDot = allowDot, Tile = tile });
                if (gemms.Any(x => x.Gemm.Variant == g.Variant && tile == 0))
                {
                    g.Dispose();
                    continue;
                }

                gemms.Add((tile == 0 ? "auto" : $"tile{tile}", g));
            }
        }

        foreach (var (tile, gemm) in gemms)
        {
            if (Environment.GetEnvironmentVariable("ONEGPU_INT8_VARIANT") is { Length: > 0 } variant && !gemm.Variant.Contains(variant, StringComparison.Ordinal))
            {
                continue;
            }

            foreach (var (name, m, n, k) in Shapes)
            {
                if (Environment.GetEnvironmentVariable("ONEGPU_INT8_ONLY") is { Length: > 0 } only && !name.Contains(only, StringComparison.Ordinal))
                {
                    continue;
                }

                var p = new Int8GemmProblem(m, n, k) { Lda = k, Ldb = n };
                using var a = d.CreateBuffer<sbyte>(m * k);
                using var b = d.CreateBuffer<sbyte>(k * n);
                using var c = d.CreateBuffer<int>(m * n);
                var (gpu, wall) = Time(d, enc => gemm.EncodeInt32(enc, p, a, b, c));
                Report($"INT8\t{d.Adapter.Name}\t{gemm.Variant} {tile}\ts8 x s8\t{name}\t{gpu / 1000:F1} us gpu\t{wall / 1000:F1} us wall\t{2.0 * m * n * k / gpu / 1000:F2} TOPS gpu\t{2.0 * m * n * k / wall / 1000:F2} TOPS wall");

                // What a quantized model runs: uint8 activations with a zero point, int8 weights stored N x K.
                var pt = new Int8GemmProblem(m, n, k) { Lda = k, Ldb = k, TransposeB = true };
                using var au = d.CreateBuffer<byte>(m * k);
                using var zp = d.CreateBuffer<byte>(4);
                var (gpu2, wall2) = Time(d, enc => gemm.EncodeInt32(enc, pt, au, b, c, new Int8Operand<byte>(zp, 0)));
                Report($"INT8\t{d.Adapter.Name}\t{gemm.Variant} {tile}\tu8+zp x s8 T\t{name}\t{gpu2 / 1000:F1} us gpu\t{wall2 / 1000:F1} us wall\t{2.0 * m * n * k / gpu2 / 1000:F2} TOPS gpu\t{2.0 * m * n * k / wall2 / 1000:F2} TOPS wall");
            }
        }

        foreach (var (_, g) in gemms)
        {
            g.Dispose();
        }

        foreach (var (vocab, hidden, rows) in new[] { (30522, 768, 8 * 512), (250002, 384, 8 * 512) })
        {
            using var gather = new GatherKernels(d);
            var table = (long)vocab * hidden * 4;
            if (d.Capabilities.Compute.MaxStorageBufferBindingBytes != 0 && table > (long)d.Capabilities.Compute.MaxStorageBufferBindingBytes)
            {
                continue;
            }

            var ids = new int[rows];
            for (var i = 0; i < rows; i++)
            {
                ids[i] = (int)((i * 7919L) % vocab);
            }

            using var data = d.CreateBuffer<float>(vocab * hidden);
            using var idx = d.CreateBuffer<int>(ids);
            using var output = d.CreateBuffer<float>(rows * hidden);
            var (gpu, wall) = Time(d, enc => gather.EncodeGather(enc, data, 0, [vocab, hidden], 0, idx, 0, rows, output, 0));
            var bytes = 2.0 * rows * hidden * 4;     // each row is read once and written once
            Report($"GATHER\t{d.Adapter.Name}\t{vocab}x{hidden} f32 {rows} rows\t{gpu / 1000:F1} us gpu\t{wall / 1000:F1} us wall\t{bytes / gpu:F0} GB/s gpu\t{bytes / wall:F0} GB/s wall");
        }
    }

    static void Report(string line)
    {
        Console.WriteLine(line);
        var path = Environment.GetEnvironmentVariable("ONEGPU_REPORT");
        if (!string.IsNullOrEmpty(path))
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    // Returns (median GPU ns per dispatch, host ns per dispatch); without timestamp queries both are the host figure.
    static (double Gpu, double Wall) Time(GpuDevice d, Action<GpuCommandEncoder> record)
    {
        const int Warm = 3;
        const int Iterations = 30;
        using var set = d.Capabilities.Timestamps ? d.CreateTimestampSet(2 * Iterations) : null;
        for (var i = 0; i < Warm; i++)
        {
            var w = d.BeginCommands();
            record(w);
            d.Submit(w);
        }

        d.WaitIdle();
        var enc = d.BeginCommands();
        if (set is not null)
        {
            enc.BeginPassTiming(new PassTiming(set, 0, Iterations));
        }

        for (var i = 0; i < Iterations; i++)
        {
            record(enc);
        }

        if (set is not null)
        {
            Assert.Equal((uint)Iterations, enc.EndPassTiming());
        }

        var sw = Stopwatch.StartNew();
        d.Submit(enc);
        d.WaitIdle();
        var wall = sw.Elapsed.TotalMilliseconds * 1e6 / Iterations;
        if (set is null)
        {
            return (wall, wall);
        }

        var t = new ulong[2 * Iterations];
        Assert.True(d.TryReadTimestamps(set, 0, t));
        var durations = new double[Iterations];
        for (var i = 0; i < Iterations; i++)
        {
            durations[i] = t[(2 * i) + 1] - t[2 * i];
        }

        Array.Sort(durations);
        return (durations[Iterations / 2], wall);
    }
}
