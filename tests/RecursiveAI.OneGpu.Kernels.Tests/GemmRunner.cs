namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Owns the device for a test and runs one case through <see cref="Gemm"/>.</summary>
internal sealed class GemmFixture : IDisposable
{
    public GpuDevice Device { get; }

    public GemmFixture()
    {
        try
        {
            Device = GpuBackends.CreateDevice(new GpuDeviceOptions { Validation = true, RequireValidation = true });
        }
        catch (GpuUnsupportedException e) when (e.Capability is null && Environment.GetEnvironmentVariable("ONEGPU_REQUIRE_DEVICE") != "1")
        {
            Assert.Skip("no GPU device available: " + e.Message);
            throw;
        }
    }

    public void Dispose()
    {
        Device.WaitIdle();
        var messages = Device.TakeValidationMessages();
        Device.Dispose();
        Assert.True(messages.Length == 0, "validation layer reported:\n" + string.Join("\n", messages));
    }

    public static bool FullSet => Environment.GetEnvironmentVariable("ONEGPU_GEMM_FULL") == "1";

    public static double Tolerance(GemmElementType t) => t switch
    {
        GemmElementType.F32 => 1e-5,
        GemmElementType.F16 => 2e-3,
        _ => 8e-3,
    };

    /// <summary>Runs the case once and returns C as read back, plus the inputs' reference and masks.</summary>
    public (float[] Gpu, double[] Reference, bool[] Written, string Variant) Run(Gemm gemm, GemmCase c, uint seed, float[]? previousOut = null)
    {
        var p = c.Problem;
        var storedColsA = p.TransposeA ? p.M : p.K;
        var storedRowsA = p.TransposeA ? p.K : p.M;
        var storedColsB = p.TransposeB ? p.K : p.N;
        var storedRowsB = p.TransposeB ? p.N : p.K;
        p.Lda = storedColsA + c.PadA;
        p.Ldb = storedColsB + c.PadB;
        p.Ldc = p.N + c.PadC;
        var lenA = p.OffsetA + (Math.Max(p.Batch - 1, 0) * p.StrideA) + (storedRowsA * p.Lda);
        var lenB = p.OffsetB + (Math.Max(p.Batch - 1, 0) * p.StrideB) + (storedRowsB * p.Ldb);
        var lenC = p.OffsetC + (Math.Max(p.Batch - 1, 0) * p.StrideC) + (p.M * p.Ldc);
        var a = GemmReference.RandomFloats(Math.Max(lenA, 1), seed);
        var b = GemmReference.RandomFloats(Math.Max(lenB, 1), seed + 1000);
        var cOld = GemmReference.RandomFloats(lenC, seed + 2000);
        var written = GemmReference.WrittenMask(p, lenC, p.Ldc);
        if (p.Beta == 0f && c.Name.Contains("nan", StringComparison.Ordinal))
        {
            for (var i = 0; i < lenC; i++)
            {
                if (written[i])
                {
                    cOld[i] = float.NaN;
                }
            }
        }

        // Round the inputs to the element type first, so the reference sees the values the GPU sees.
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = Round(gemm.ElementType, a[i]);
        }

        for (var i = 0; i < b.Length; i++)
        {
            b[i] = Round(gemm.ElementType, b[i]);
        }

        var reference = GemmReference.Multiply(p, a, b, cOld, p.Lda, p.Ldb, p.Ldc);
        using var cb = Device.CreateBuffer<float>(cOld);
        var enc = Device.BeginCommands();
        switch (gemm.ElementType)
        {
            case GemmElementType.F32:
                {
                    using var ab = Device.CreateBuffer<float>(a);
                    using var bb = Device.CreateBuffer<float>(b);
                    gemm.Encode(enc, p, ab, bb, cb);
                    Device.Submit(enc);
                    Device.WaitIdle();
                    break;
                }

            case GemmElementType.F16:
                {
                    using var ab = Device.CreateBuffer<Half>(a.Select(x => (Half)x).ToArray());
                    using var bb = Device.CreateBuffer<Half>(b.Select(x => (Half)x).ToArray());
                    gemm.Encode(enc, p, ab, bb, cb);
                    Device.Submit(enc);
                    Device.WaitIdle();
                    break;
                }

            default:
                {
                    using var ab = Device.CreateBuffer<ushort>(a.Select(GemmReference.ToBFloat16).ToArray());
                    using var bb = Device.CreateBuffer<ushort>(b.Select(GemmReference.ToBFloat16).ToArray());
                    gemm.EncodeBFloat16(enc, p, ab, bb, cb);
                    Device.Submit(enc);
                    Device.WaitIdle();
                    break;
                }
        }

        var gpu = new float[lenC];
        cb.Read(gpu);
        gemm.ReleaseBindings();
        return (gpu, reference, written, gemm.Variant);
    }

    static float Round(GemmElementType t, float x) => t switch
    {
        GemmElementType.F32 => x,
        GemmElementType.F16 => (float)(Half)x,
        _ => GemmReference.FromBFloat16(GemmReference.ToBFloat16(x)),
    };
}
