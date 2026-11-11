namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>The tanh form of GELU against a double reference over the whole range models use, through the element-wise kernel and the matrix-product epilogue.</summary>
public sealed class GeluTanhTests
{
    static double Reference(double x) => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + (0.044715 * x * x * x))));

    static float[] Inputs()
    {
        // A dense sweep of the range models see, plus a coarse sweep of the tails.
        var list = new List<float>();
        for (var i = 0; i <= 40000; i++)
        {
            list.Add(-6f + (12f * i / 40000f));
        }

        for (var i = 0; i < 4000; i++)
        {
            list.Add(-30f + (60f * i / 4000f));
        }

        list.Add(0f);
        list.Add(-0f);
        return [.. list];
    }

    [Fact]
    public async Task ElementwiseMatchesTheDoubleReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        var x = Inputs();
        using var bx = Rig.Upload(d, x);
        using var by = d.CreateBuffer<float>(x.Length);
        Rig.Run(d, e => k.EncodeUnary(e, GpuUnaryOp.GeluTanh, bx, 0, by, 0, x.Length));
        var got = await Rig.DownloadAsync(by);
        var want = x.Select(v => Reference(v)).ToArray();
        var (scaled, rel) = Rig.Errors(Rig.D(got, got.Length), want);
        Rig.Row(d, "elementwise", "f32", $"GeluTanh over [-30, 30], {x.Length} points", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
        Assert.True(scaled <= 1e-5, $"scaled error {scaled}");
    }

    [Fact]
    public async Task MatrixProductEpilogueMatchesTheDoubleReferenceAndRepeats()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var x = Inputs();
        var n = x.Length;
        var one = new float[] { 1f };
        using var a = Rig.Upload(d, one);
        using var b = Rig.Upload(d, x);
        foreach (var mode in new[] { GpuKernelMode.Fast, GpuKernelMode.Deterministic })
        {
            using var gemm = new Gemm(d, GemmElementType.F32, mode);
            var epilogue = new GemmEpilogue(null, 0, GemmActivation.GeluTanh);
            string? firstHash = null;
            for (var run = 0; run < 2; run++)
            {
                using var c = d.CreateBuffer<float>(n);
                Rig.Run(d, e => gemm.Encode(e, new GemmProblem(1, n, 1), a, b, c, epilogue));
                var got = await Rig.DownloadAsync(c);
                var hash = Rig.Hash<float>(got);
                firstHash ??= hash;
                Assert.Equal(firstHash, hash);
                if (run == 0)
                {
                    var (scaled, rel) = Rig.Errors(Rig.D(got, got.Length), x.Select(v => Reference(v)).ToArray());
                    Rig.Row(d, "gemm epilogue", "f32", $"GeluTanh over [-30, 30], {n} points, {mode}", $"max scaled error {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}, two runs identical");
                    Assert.True(scaled <= 1e-5, $"{mode}: scaled error {scaled}");
                }
            }
        }
    }
}
