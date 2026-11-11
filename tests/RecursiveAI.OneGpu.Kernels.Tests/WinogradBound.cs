using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// A lower bound for Winograd F(2x2,3x3) (Lavin and Gray, 2016) on the stride-1 3x3 layers of ResNet-50: the sixteen batched matrix products it needs, timed with the
/// real matrix kernel, plus a plain copy that moves as many bytes as the input, filter-domain and output transforms must. A real implementation cannot be faster than the sum, so
/// if the sum is not below the measured layer time the variant cannot pay. Runs only with ONEGPU_CONV_BENCH=1.
/// </summary>
public sealed class WinogradBound
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_CONV_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_CONV_BENCH=1 to time the Winograd bound");
        }

        using var gpu = new GpuFixture();
        var d = gpu.Device;
        foreach (var half in new[] { false, true })
        {
            double directSum = 0, boundSum = 0;
            foreach (var l in ConvTests.ResNet50(8, 224))
            {
                var p = l.P;
                if (p.KernelHeight != 3 || p.KernelWidth != 3 || p.StrideHeight != 1 || p.Groups != 1)
                {
                    continue;
                }

                var direct = Time(d, p, half, winograd: false);
                var bound = Time(d, p, half, winograd: true);
                directSum += direct * l.Count;
                boundSum += bound * l.Count;
                Rig.Row(d, "winograd-bound", half ? "f16" : "f32", $"{l.Name} {p.InChannels}->{p.OutChannels} {p.Height}x{p.Width} b{p.Batch} x{l.Count}", $"direct {direct / 1e3:F1} us, Winograd lower bound {bound / 1e3:F1} us");
            }

            Rig.Row(d, "winograd-bound", half ? "f16" : "f32", "ResNet-50 stride-1 3x3 layers, b8, summed", $"direct {directSum / 1e3:F0} us, Winograd lower bound {boundSum / 1e3:F0} us");
        }
    }

    // Host-clock nanoseconds per repetition: the conv itself, or the sixteen products plus the transforms' traffic.
    static double Time(GpuDevice d, ConvProblem p, bool half, bool winograd)
    {
        var tiles = p.Batch * ((p.OutHeight + 1) / 2) * ((p.OutWidth + 1) / 2);
        var cin = p.InChannels;
        var cout = p.OutChannels;
        long inElems = (long)p.Batch * cin * p.Height * p.Width;
        long outElems = (long)p.Batch * cout * p.OutHeight * p.OutWidth;
        long traffic = inElems + (16L * cin * tiles) + (16L * cout * tiles) + outElems;
        var copyCount = (int)(half ? traffic / 4 : traffic / 2);
        using var k = new ConvKernels(d);
        using var gemm = new Gemm(d, half ? GemmElementType.F16 : GemmElementType.F32);
        using var ew = new ElementwiseKernels(d);
        var gp = new GemmProblem(cout, tiles, cin) { Batch = 16, StrideA = cout * cin, StrideB = cin * tiles, StrideC = cout * tiles };
        using var af = d.CreateBuffer<float>(half ? 4 : 16 * cout * cin);
        using var bf = d.CreateBuffer<float>(half ? 4 : 16 * cin * tiles);
        using var ah = d.CreateBuffer<Half>(half ? GpuKernelBuffers.PaddedHalfLength(16 * cout * cin) : 2);
        using var bh = d.CreateBuffer<Half>(half ? GpuKernelBuffers.PaddedHalfLength(16 * cin * tiles) : 2);
        using var c = d.CreateBuffer<float>(16 * cout * tiles);
        using var copyIn = d.CreateBuffer<float>(Math.Max(copyCount, 1));
        using var copyOut = d.CreateBuffer<float>(Math.Max(copyCount, 1));
        var nx = (int)inElems;
        var nw = cout * cin * 9;
        var ny = (int)outElems;
        using var x = ConvTests.Upload<float>(d, new float[nx], 0, 0f);
        using var w = ConvTests.Upload<float>(d, new float[nw], 0, 0f);
        using var y = ConvTests.Upload<float>(d, new float[ny], 0, 0f);
        using var xh = ConvTests.Upload<Half>(d, new float[nx], 0, 0f);
        using var wh = ConvTests.Upload<Half>(d, new float[nw], 0, 0f);
        using var yh = ConvTests.Upload<Half>(d, new float[ny], 0, 0f);
        void Record(GpuCommandEncoder e)
        {
            if (!winograd)
            {
                if (half)
                {
                    k.Encode(e, p, xh, wh, null, null, yh);
                }
                else
                {
                    k.Encode(e, p, x, w, null, null, y);
                }

                return;
            }

            if (half)
            {
                gemm.Encode(e, gp, ah, bh, c);
            }
            else
            {
                gemm.Encode(e, gp, af, bf, c);
            }

            ew.EncodeUnary(e, GpuUnaryOp.Identity, copyIn, 0, copyOut, 0, copyCount);
        }

        for (var i = 0; i < 3; i++)
        {
            var e = d.BeginCommands();
            Record(e);
            d.Submit(e);
        }

        d.WaitIdle();
        const int Reps = 20;
        var best = double.MaxValue;
        for (var round = 0; round < 3; round++)
        {
            var enc = d.BeginCommands();
            for (var i = 0; i < Reps; i++)
            {
                Record(enc);
            }

            var sw = Stopwatch.StartNew();
            d.Submit(enc);
            d.WaitIdle();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / Reps);
        }

        return best;
    }
}
