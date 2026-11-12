using static RecursiveAI.OneGpu.Kernels.Tests.AttentionReference;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Fused float32 attention against the unfused GEMM + softmax + GEMM path, both measured against a double-precision reference, at long key
/// lengths and with scores that spread over tens of units (trained models' scores do; the uniform inputs of the shape sweeps barely do).
/// </summary>
public sealed class AttentionAccuracyTests
{
    // Query and key elements are uniform in [-a, a]; with a head dimension of 64 and the usual 1/sqrt(d) scale the scores have a standard
    // deviation of about a^2 / 3 (1.3, 5.3 and 12). Beyond that, rounding the float32 scores themselves limits both paths to about 1e-5.
    static IEnumerable<(string Name, int B, int H, int Sq, int Sk, float Amp)> Cases(bool software)
    {
        foreach (var amp in new[] { 2f, 4f, 6f })
        {
            yield return ($"whisper encoder B1 1500 keys, score sd {amp * amp / 3:F0}", 1, 8, software ? 128 : 1500, 1500, amp);
            // 1024 query rows keep the unfused score matrix under the 256 MB a browser allows for one buffer.
            yield return ($"whisper encoder B4 1500 keys, score sd {amp * amp / 3:F0}", 4, 8, software ? 64 : 1024, 1500, amp);
            yield return ($"4096 keys, score sd {amp * amp / 3:F0}", 1, 8, software ? 32 : 256, 4096, amp);
        }
    }

    // The fused kernel keeps a running maximum and rescales its output once per key tile, so it rounds about as often as the unfused path
    // does per key but not in the same places; the unfused path's products sum in blocks (error a few times 1e-7 at these lengths), so
    // "no worse than the unfused path" is not a property to hold it to. The gate is the float32 bound plus a factor of four over unfused.
    [Fact]
    public async Task FusedStaysCloseToUnfusedAtLongKeyLengths()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        // Software rasterizers get fewer query rows; the key length is what the test is about.
        var software = d.Adapter.Name.Contains("swiftshader", StringComparison.OrdinalIgnoreCase) || d.Adapter.Name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase);
        using var fused = new AttentionKernels(d);
        using var gemm = new Gemm(d, GemmElementType.F32);
        using var soft = new SoftmaxKernels(d);
        const int D = 64;
        var worstFused = 0.0;
        var worstUnfused = 0.0;
        var failures = new List<string>();
        foreach (var (name, b, h, sq, sk, amp) in Cases(software))
        {
            var c = new Case(name, b, h, h, sq, sk, D);
            var q = Rig.Random(b * h * sq * D, sq + 11, -amp, amp);
            var k = Rig.Random(b * h * sk * D, sk + 23, -amp, amp);
            var v = Rig.Random(b * h * sk * D, sk + 37, -1f, 1f);
            var scale = 1f / MathF.Sqrt(D);
            var (rows, want) = Compute(c, scale, q, k, v, [], null);
            var bh = b * h;
            using var bq = d.CreateBuffer<float>(q);
            using var bk = d.CreateBuffer<float>(k);
            using var bv = d.CreateBuffer<float>(v);
            using var scores = d.CreateBuffer<float>(bh * sq * sk);
            using var probs = d.CreateBuffer<float>(bh * sq * sk);
            using var oRef = d.CreateBuffer<float>(bh * sq * D);
            using var oFused = d.CreateBuffer<float>(bh * sq * D);
            var s = new GemmProblem(sq, sk, D) { Batch = bh, TransposeB = true, Alpha = scale, Lda = D, Ldb = D, Ldc = sk, StrideA = sq * D, StrideB = sk * D, StrideC = sq * sk };
            var pv = new GemmProblem(sq, D, sk) { Batch = bh, Lda = sk, Ldb = D, Ldc = D, StrideA = sq * sk, StrideB = sk * D, StrideC = sq * D };
            var prob = new AttentionProblem(b, h, sq, sk, D) { Scale = scale };
            Rig.Run(d, e =>
            {
                gemm.Encode(e, s, bq, bk, scores);
                soft.Encode(e, scores, 0, [b, h, sq, sk], 3, probs, 0);
                gemm.Encode(e, pv, probs, bv, oRef);
                fused.Encode(e, prob, bq, bk, bv, oFused);
            });
            var fusedOut = await Rig.DownloadAsync(oFused);
            var unfusedOut = await Rig.DownloadAsync(oRef);
            // Run the fused kernel again: the bits must not change.
            Rig.Run(d, e => fused.Encode(e, prob, bq, bk, bv, oFused, mode: GpuKernelMode.Deterministic));
            var again = await Rig.DownloadAsync(oFused);
            Assert.Equal(Rig.Hash<float>(fusedOut), Rig.Hash<float>(again));

            double[] Pick(float[] all)
            {
                var r = new double[rows.Length * D];
                for (var i = 0; i < rows.Length; i++)
                {
                    for (var x = 0; x < D; x++)
                    {
                        r[(i * D) + x] = all[(rows[i] * D) + x];
                    }
                }

                return r;
            }

            var (fs, fr) = Rig.Errors(Pick(fusedOut), want);
            var (us, ur) = Rig.Errors(Pick(unfusedOut), want);
            Rig.Row(d, "attention", "f32", $"accuracy vs double reference, {name}", $"fused scaled {Rig.Sci(fs)} (relative {Rig.Sci(fr)}), unfused scaled {Rig.Sci(us)} (relative {Rig.Sci(ur)})");
            worstFused = Math.Max(worstFused, fs);
            worstUnfused = Math.Max(worstUnfused, us);
            if (fs > 1e-5 || fs > (us * 4) + 1e-6)
            {
                failures.Add($"{name}: fused scaled error {fs}, unfused {us}");
            }
        }

        Rig.Row(d, "attention", "f32", "accuracy, worst over the long-key cases", $"fused {Rig.Sci(worstFused)}, unfused {Rig.Sci(worstUnfused)}");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }
}
