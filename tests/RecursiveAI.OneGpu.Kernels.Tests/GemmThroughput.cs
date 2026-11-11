using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Throughput of the large model shapes, from GPU timestamps (median per dispatch) and from the host clock (a whole batch of dispatches
/// between submit and idle). Runs only with ONEGPU_GEMM_BENCH=1; the numbers it prints are indicative.
/// </summary>
public sealed class GemmThroughput
{
    static readonly (string Name, int M, int N, int K, int Batch, bool TransB)[] Shapes =
    [
        ("bert ffn-up 8x128", 1024, 3072, 768, 1, false),
        ("bert ffn-down 8x128", 1024, 768, 3072, 1, false),
        ("whisper enc ffn-up 1500", 1500, 3072, 768, 1, false),
        ("whisper enc proj 1500", 1500, 768, 768, 1, false),
        ("bert scores 8x12x128", 128, 128, 64, 96, true),
        ("whisper dec logits 1", 1, 51865, 768, 1, true),
        ("square 4096", 4096, 4096, 4096, 1, false),
    ];

    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_GEMM_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_GEMM_BENCH=1 to time the large shapes");
        }

        using var f = new GemmFixture();
        var d = f.Device;
        var timestamps = d.Capabilities.Timestamps;
        var noHardware = Environment.GetEnvironmentVariable("ONEGPU_GEMM_NO_HARDWARE") == "1";
        foreach (var type in new[] { GemmElementType.F32, GemmElementType.F16, GemmElementType.BF16 })
        {
            using var gemm = new Gemm(d, type, GpuKernelMode.Fast, new GemmOptions { AllowMatrixHardware = !noHardware });
            foreach (var (name, m, n, k, batch, transB) in Shapes)
            {
                var p = new GemmProblem(m, n, k) { Batch = batch, TransposeB = transB, StrideA = m * k, StrideB = batch > 1 ? k * n : 0, StrideC = m * n };
                p.Lda = k;
                p.Ldb = transB ? k : n;
                p.Ldc = n;
                var lenA = batch * m * k;
                var lenB = (batch > 1 ? batch : 1) * k * n;
                var lenC = batch * m * n;
                if ((long)lenC * 4 > (long)d.Capabilities.Compute.MaxStorageBufferBindingBytes && d.Capabilities.Compute.MaxStorageBufferBindingBytes != 0)
                {
                    continue;
                }

                // Zeros are enough: the kernels do the same work whatever the data (power draw aside).
                using var cb = d.CreateBuffer<float>(lenC);
                var (median, wall) = type switch
                {
                    GemmElementType.F32 => TimeIt(d, timestamps, gemm, p, d.CreateBuffer<float>(lenA), d.CreateBuffer<float>(lenB), cb, batch),
                    GemmElementType.F16 => TimeIt(d, timestamps, gemm, p, d.CreateBuffer<Half>(lenA), d.CreateBuffer<Half>(lenB), cb, batch),
                    _ => TimeIt(d, timestamps, gemm, p, d.CreateBuffer<ushort>(lenA), d.CreateBuffer<ushort>(lenB), cb, batch),
                };
                var gflopsGpu = p.Flops / median;      // flop per ns = GFLOP/s
                var gflopsWall = p.Flops / wall;
                Console.WriteLine($"THROUGHPUT\t{d.Adapter.Name}\t{gemm.Variant}\t{type}\t{name}\t{median / 1000:F1} us gpu\t{wall / 1000:F1} us wall\t{gflopsGpu:F0} GFLOP/s gpu\t{gflopsWall:F0} GFLOP/s wall");
                AppendReport($"THROUGHPUT\t{d.Adapter.Name}\t{gemm.Variant}\t{type}\t{name}\t{median:F0}\t{wall:F0}\t{gflopsGpu:F0}\t{gflopsWall:F0}");
            }
        }
    }

    static void AppendReport(string line)
    {
        var path = Environment.GetEnvironmentVariable("ONEGPU_REPORT");
        if (!string.IsNullOrEmpty(path))
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    // Returns (median GPU ns per dispatch, host ns per dispatch). Without timestamp queries the GPU figure is the host one.
    static (double Gpu, double Wall) TimeIt<T>(GpuDevice d, bool timestamps, Gemm gemm, GemmProblem p, GpuBuffer<T> a, GpuBuffer<T> b, GpuBuffer<float> c, int batch)
        where T : unmanaged
    {
        using (a)
        using (b)
        {
            const int Warm = 3;
            const int Iterations = 30;
            using var set = timestamps ? d.CreateTimestampSet(2 * Iterations) : null;
            void Record(GpuCommandEncoder enc)
            {
                switch (a)
                {
                    case GpuBuffer<float> fa: gemm.Encode(enc, p, fa, (GpuBuffer<float>)(object)b, c); break;
                    case GpuBuffer<Half> ha: gemm.Encode(enc, p, ha, (GpuBuffer<Half>)(object)b, c); break;
                    default: gemm.EncodeBFloat16(enc, p, (GpuBuffer<ushort>)(object)a, (GpuBuffer<ushort>)(object)b, c); break;
                }
            }

            for (var i = 0; i < Warm; i++)
            {
                var w = d.BeginCommands();
                Record(w);
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
                Record(enc);
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
}
