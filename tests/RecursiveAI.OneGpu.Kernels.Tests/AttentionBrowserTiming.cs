using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Wall-clock timing of fused attention against GEMM + softmax + GEMM at the shapes of the reference models, for browsers (which cannot read GPU
/// timestamps without blocking) and any real GPU there. Each trial records a batch of calls, submits it and waits for a one-element readback, which
/// finishes only after the GPU has. Rows go to the console as <c>ROW</c> lines; the numbers are indicative.
/// </summary>
public sealed class AttentionBrowserTiming
{
    [Fact]
    public async Task FusedAgainstUnfusedWallTime()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        if (!Rig.Browser || d.Adapter.Kind == AdapterKind.Software)
        {
            Assert.Skip("wall-clock timing runs in a browser on a hardware adapter; the native throughput test uses GPU timestamps");
        }

        using var fused = new AttentionKernels(d);
        using var gemm = new Gemm(d, GemmElementType.F32);
        using var soft = new SoftmaxKernels(d);
        foreach (var (name, b, h, sq, sk, dd, causal, calls) in new[]
        {
            ("minilm B1 S128", 1, 12, 128, 128, 32, false, 20), ("minilm B8 S128", 8, 12, 128, 128, 32, false, 10), ("bert B1 S128", 1, 12, 128, 128, 64, false, 20),
            ("bert B1 S512", 1, 12, 512, 512, 64, false, 10), ("bert B8 S128", 8, 12, 128, 128, 64, false, 10), ("bert B8 S512", 8, 12, 512, 512, 64, false, 3),
            ("whisper encoder S1500", 1, 12, 1500, 1500, 64, false, 3), ("whisper decoder cache448", 1, 12, 1, 448, 64, true, 20), ("whisper cross 1 query", 1, 12, 1, 1500, 64, false, 20),
        })
        {
            var bh = b * h;
            using var q = d.CreateBuffer<float>(Rig.Random(bh * sq * dd, 1, -2f, 2f));
            using var k = d.CreateBuffer<float>(Rig.Random(bh * sk * dd, 2, -2f, 2f));
            using var v = d.CreateBuffer<float>(Rig.Random(bh * sk * dd, 3, -1f, 1f));
            using var scores = d.CreateBuffer<float>(bh * sq * sk);
            using var probs = d.CreateBuffer<float>(bh * sq * sk);
            using var o1 = d.CreateBuffer<float>(bh * sq * dd);
            using var o2 = d.CreateBuffer<float>(bh * sq * dd);
            var s = new GemmProblem(sq, sk, dd) { Batch = bh, TransposeB = true, Alpha = 1f / MathF.Sqrt(dd), Lda = dd, Ldb = dd, Ldc = sk, StrideA = sq * dd, StrideB = sk * dd, StrideC = sq * sk };
            var pv = new GemmProblem(sq, dd, sk) { Batch = bh, Lda = sk, Ldb = dd, Ldc = dd, StrideA = sq * sk, StrideB = sk * dd, StrideC = sq * dd };
            var prob = new AttentionProblem(b, h, sq, sk, dd) { Causal = causal };
            var one = new float[1];

            async Task<double> Trial(Action<GpuCommandEncoder> record, GpuBuffer<float> o)
            {
                var times = new double[3];
                for (var t = -1; t < times.Length; t++)
                {
                    var enc = d.BeginCommands();
                    for (var i = 0; i < calls; i++)
                    {
                        record(enc);
                    }

                    var sw = Stopwatch.StartNew();
                    d.Submit(enc);
                    await o.ReadAsync(one);
                    if (t >= 0)
                    {
                        times[t] = sw.Elapsed.TotalMilliseconds / calls;
                    }
                }

                Array.Sort(times);
                return times[1];
            }

            var unfused = await Trial(e =>
            {
                gemm.Encode(e, s, q, k, scores);
                soft.Encode(e, scores, 0, [b, h, sq, sk], 3, probs, 0);
                gemm.Encode(e, pv, probs, v, o2);
            }, o2);
            var fusedMs = await Trial(e => fused.Encode(e, prob, q, k, v, o1), o1);
            Rig.Row(d, "attention", "f32", $"wall time {name} [{fused.Variant}]", $"fused {fusedMs:F2} ms, unfused {unfused:F2} ms, fused/unfused {fusedMs / unfused:F2}");
        }
    }
}
