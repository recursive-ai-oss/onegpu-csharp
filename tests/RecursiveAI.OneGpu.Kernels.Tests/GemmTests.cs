using System.Text;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Matrix multiplication against a double-precision CPU reference, on whatever device the environment selects.</summary>
public sealed class GemmTests
{
    static void Report(string line)
    {
        Console.WriteLine(line);
        var path = Environment.GetEnvironmentVariable("ONEGPU_REPORT");
        if (!string.IsNullOrEmpty(path))
        {
            File.AppendAllText(path, line + "\n");
        }
    }

    static IEnumerable<GemmCase> AllCases() => GemmReference.EdgeCases().Concat(GemmReference.ModelCases(GemmFixture.FullSet));

    static (bool Hardware, string Tag) Setup() => (Environment.GetEnvironmentVariable("ONEGPU_GEMM_NO_HARDWARE") != "1", Environment.GetEnvironmentVariable("ONEGPU_GEMM_NO_HARDWARE") == "1" ? "forced-portable" : "auto");

    static void Parity(GemmElementType type, GpuKernelMode mode)
    {
        using var f = new GemmFixture();
        var (hw, tag) = Setup();
        using var gemm = new Gemm(f.Device, type, mode, new GemmOptions { AllowMatrixHardware = hw });
        var tol = GemmFixture.Tolerance(type);
        var failures = new StringBuilder();
        uint seed = 1;
        foreach (var c in AllCases())
        {
            (float[] Gpu, double[] Reference, bool[] Written, string Variant) run;
            try
            {
                run = f.Run(gemm, c, seed++);
            }
            catch (GpuUnsupportedException e) when (e.Capability == nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes))
            {
                Report($"PARITY\t{f.Device.Adapter.Name}\t{f.Device.Backend}\t{tag}\t{gemm.Variant}\t{type}\t{mode}\t{c.Name}\t-\t-\tskipped: {e.Message}");
                continue;
            }

            var (gpu, reference, written, variant) = run;
            var (scaled, rel) = GemmReference.Compare(gpu, reference, written);
            var ok = scaled <= tol;
            Report($"PARITY\t{f.Device.Adapter.Name}\t{f.Device.Backend}\t{tag}\t{variant}\t{type}\t{mode}\t{c.Name}\t{GemmReference.Fmt(scaled)}\t{GemmReference.Fmt(rel)}\t{(ok ? "ok" : "FAIL")}");
            if (!ok)
            {
                failures.AppendLine(Describe(c.Name, scaled));
            }
        }

        Assert.True(failures.Length == 0, failures.ToString());
    }

    static string Describe(string name, double scaled) => $"{name}: scaled error {scaled:e2}";

    [Fact]
    public void ReportsDeviceCapabilities()
    {
        using var f = new GemmFixture();
        var cc = f.Device.Capabilities.Compute;
        Report($"CAPS\t{f.Device.Adapter.Name}\t{f.Device.Backend}\t{cc.Describe().ReplaceLineEndings(" | ")}");
        foreach (var shape in cc.CooperativeMatrices)
        {
            Report($"CAPS\t{f.Device.Adapter.Name}\tcooperative matrix\t{shape}");
        }
    }

    [Fact] public void F32Fast() => Parity(GemmElementType.F32, GpuKernelMode.Fast);

    [Fact] public void F16Fast() => Parity(GemmElementType.F16, GpuKernelMode.Fast);

    [Fact] public void BF16Fast() => Parity(GemmElementType.BF16, GpuKernelMode.Fast);

    [Fact] public void F32Deterministic() => Parity(GemmElementType.F32, GpuKernelMode.Deterministic);

    [Fact] public void F16Deterministic() => Parity(GemmElementType.F16, GpuKernelMode.Deterministic);

    [Theory]
    [InlineData(GemmElementType.F32)]
    [InlineData(GemmElementType.F16)]
    public void DeterministicRunsAreBitIdentical(GemmElementType type)
    {
        using var f = new GemmFixture();
        using var gemm = new Gemm(f.Device, type, GpuKernelMode.Deterministic);
        uint seed = 100;
        foreach (var c in AllCases())
        {
            var s = seed++;
            float[] first, second;
            try
            {
                first = f.Run(gemm, c, s).Gpu;
                second = f.Run(gemm, c, s).Gpu;
            }
            catch (GpuUnsupportedException e) when (e.Capability == nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes))
            {
                Report($"DETERMINISM\t{f.Device.Adapter.Name}\t{f.Device.Backend}\t{gemm.Variant}\t{type}\t{c.Name}\t-\t-\tskipped: operand over the binding limit");
                continue;
            }

            var h1 = GemmReference.Hash(first);
            var h2 = GemmReference.Hash(second);
            Report($"DETERMINISM\t{f.Device.Adapter.Name}\t{f.Device.Backend}\t{gemm.Variant}\t{type}\t{c.Name}\t{h1}\t{h2}\t{(h1 == h2 ? "equal" : "DIFFERENT")}");
            Assert.Equal(h1, h2);
        }
    }

    [Fact]
    public void BadShapesThrowTypedExceptionsNamingTheDimension()
    {
        using var f = new GemmFixture();
        using var gemm = new Gemm(f.Device, GemmElementType.F32);
        using var a = f.Device.CreateBuffer<float>(100);
        using var b = f.Device.CreateBuffer<float>(100);
        using var c = f.Device.CreateBuffer<float>(100);
        var enc = f.Device.BeginCommands();
        var ex = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, new GemmProblem(-1, 4, 4), a, b, c));
        Assert.Equal("M", ex.Dimension);
        ex = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, new GemmProblem(10, 10, 11), a, b, c));
        Assert.Equal("OffsetA", ex.Dimension);
        ex = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, new GemmProblem(4, 4, 4) { Lda = 2 }, a, b, c));
        Assert.Equal("Lda", ex.Dimension);
        ex = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, new GemmProblem(4, 4, 4) { Batch = 3, StrideC = 8 }, a, b, c));
        Assert.Equal("StrideC", ex.Dimension);
        ex = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, new GemmProblem(int.MaxValue, 4, 4), a, b, c));
        Assert.Equal("OffsetC", ex.Dimension);
        Assert.Throws<ArgumentException>(() => gemm.Encode(enc, new GemmProblem(4, 4, 4), f.Device.CreateBuffer<Half>(16), f.Device.CreateBuffer<Half>(16), c));
        f.Device.Submit(enc);
    }
}
