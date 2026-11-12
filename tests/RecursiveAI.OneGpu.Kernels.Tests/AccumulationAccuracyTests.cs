namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Float32 products and convolutions against a double-precision sum, at the depths where a single chain over k loses accuracy
/// (Higham, Accuracy and Stability of Numerical Algorithms, chapter 4). The scaled error is max|gpu - ref| / max|ref|. Two data kinds:
/// zero-mean values, and non-negative activations against mixed-sign weights, whose partial sums drift the way post-activation features do.
/// Each case also runs twice in both modes: the bits must repeat.
/// </summary>
public sealed class AccumulationAccuracyTests
{
    // The bound of the kernel tolerance table for float32 products and convolutions.
    const double Bound = 1e-5;

    static (float[] A, float[] B) Data(int na, int nb, int seed, bool biased)
    {
        var a = biased ? Rig.Random(na, seed, 0f, 1f) : Rig.Random(na, seed, -1f, 1f);
        var b = biased ? Rig.Random(nb, seed + 1, -0.2f, 1f) : Rig.Random(nb, seed + 1, -1f, 1f);
        return (a, b);
    }

    static double[] Reference(int m, int n, int k, float[] a, float[] b)
    {
        var bt = new float[n * k];
        for (var kk = 0; kk < k; kk++)
        {
            for (var nn = 0; nn < n; nn++)
            {
                bt[(nn * k) + kk] = b[(kk * n) + nn];
            }
        }

        var r = new double[m * n];
        Parallel.For(0, m, i =>
        {
            for (var j = 0; j < n; j++)
            {
                double acc = 0;
                for (var kk = 0; kk < k; kk++)
                {
                    acc += (double)a[(i * k) + kk] * bt[(j * k) + kk];
                }

                r[(i * n) + j] = acc;
            }
        });
        return r;
    }

    public static IEnumerable<TheoryDataRow<int, int, int>> GemmShapes()
    {
        yield return new(64, 64, 768);
        yield return new(64, 64, 3072);
        yield return new(64, 64, 4096);
        yield return new(1500, 768, 3072); // Whisper-small encoder fc2
    }

    [Theory]
    [MemberData(nameof(GemmShapes))]
    public async Task Float32GemmErrorAgainstDouble(int m, int n, int k)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        foreach (var biased in new[] { false, true })
        {
            var (a, b) = Data(m * k, k * n, m + n + k, biased);
            var want = Reference(m, n, k, a, b);
            using var ba = d.CreateBuffer<float>(a);
            using var bb = d.CreateBuffer<float>(b);
            using var bc = d.CreateBuffer<float>(m * n);
            var p = new GemmProblem(m, n, k) { Lda = k, Ldb = n, Ldc = n };
            foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
            {
                using var gemm = new Gemm(d, GemmElementType.F32, mode);
                Rig.Run(d, e => gemm.Encode(e, p, ba, bb, bc));
                var first = await Rig.DownloadAsync(bc);
                Rig.Run(d, e => gemm.Encode(e, p, ba, bb, bc));
                var second = await Rig.DownloadAsync(bc);
                Assert.Equal(Rig.Hash<float>(first), Rig.Hash<float>(second));
                var (s, r) = Rig.Errors(first.Select(x => (double)x).ToArray(), want);
                Rig.Row(d, "gemm", "f32", $"accumulation vs double, {m}x{n}x{k} {(biased ? "activations" : "zero-mean")} {mode} [{gemm.Variant}]", $"scaled {Rig.Sci(s)} (relative {Rig.Sci(r)}), reruns identical");
                Assert.True(s <= Bound, $"scaled error {s}");
            }
        }
    }

    public static IEnumerable<TheoryDataRow<string, int, int, int, int, int, int, int>> ConvShapes()
    {
        // name, in channels, height, width, out channels, kernel h, kernel w, stride (width); pad is "same" for the kernel size.
        yield return new("whisper stem 1 (80 -> 384, k3, 3000 frames)", 80, 1, 3000, 384, 1, 3, 1);
        yield return new("whisper stem 2 (384 -> 384, k3, stride 2)", 384, 1, 3000, 384, 1, 3, 2);
        yield return new("resnet-50 stage 1 3x3 (64 -> 64, 56x56)", 64, 56, 56, 64, 3, 3, 1);
        yield return new("resnet-50 stage 4 3x3 (512 -> 512, 7x7)", 512, 7, 7, 512, 3, 3, 1);
    }

    [Theory]
    [MemberData(nameof(ConvShapes))]
    public async Task Float32ConvolutionErrorAgainstDouble(string name, int cin, int h, int w, int cout, int kh, int kw, int stride)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var p = new ConvProblem(1, cin, h, w, cout, kh, kw)
        {
            StrideWidth = stride,
            PadLeft = kw / 2,
            PadRight = kw / 2,
            PadTop = kh / 2,
            PadBottom = kh / 2,
        };
        foreach (var biased in new[] { false, true })
        {
            var (x, wt) = Data(cin * h * w, cout * cin * kh * kw, cin + h, biased);
            var want = ConvReference.Conv(p, x, wt, null, null);
            using var bx = d.CreateBuffer<float>(x);
            using var bw = d.CreateBuffer<float>(wt);
            using var by = d.CreateBuffer<float>(want.Length);
            var colCount = ConvKernels.ColumnElements(p);
            using var cols = colCount > 0 ? d.CreateBuffer<float>((int)colCount) : null;
            foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
            {
                using var k = new ConvKernels(d, mode);
                Rig.Run(d, e => k.Encode(e, p, bx, bw, null, null, by, cols));
                var first = await Rig.DownloadAsync(by);
                Rig.Run(d, e => k.Encode(e, p, bx, bw, null, null, by, cols));
                var second = await Rig.DownloadAsync(by);
                Assert.Equal(Rig.Hash<float>(first), Rig.Hash<float>(second));
                var (s, r) = Rig.Errors(first.Select(v => (double)v).ToArray(), want);
                Rig.Row(d, "conv", "f32", $"accumulation vs double, {name} {(biased ? "activations" : "zero-mean")} {mode} [{ConvKernels.ChooseVariant(p)}]", $"scaled {Rig.Sci(s)} (relative {Rig.Sci(r)}), reruns identical");
                Assert.True(s <= Bound, $"scaled error {s}");
            }
        }
    }
}
