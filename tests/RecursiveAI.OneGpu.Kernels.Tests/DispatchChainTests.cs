using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Passes recorded one after another in one command encoder are ordered: a pass sees every write of the passes before it, and a later write to a buffer wins.
/// On Metal consecutive compute passes share one compute encoder, so this is the test that the sharing keeps that promise. The timing test (opt-in) measures what a
/// dependent dispatch costs.
/// </summary>
public sealed class DispatchChainTests
{
    [Fact]
    public async Task ReadAfterWriteAcrossWorkgroupsSeesEveryEarlierWrite()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new TransposeKernels(d);

        // Transposing a square matrix back and forth: every element of a pass is read from a different workgroup's output of the pass before,
        // so an overlap between consecutive passes corrupts the result. After an odd number of transposes the matrix is transposed.
        const int N = 1024;
        const int Passes = 65;
        var data = new float[N * N];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = i;
        }

        using var a = d.CreateBuffer<float>(data);
        using var b = d.CreateBuffer<float>(N * N);
        int[] shape = [N, N];
        int[] perm = [1, 0];
        Rig.Run(d, e =>
        {
            for (var p = 0; p < Passes; p++)
            {
                if ((p & 1) == 0)
                {
                    k.Encode(e, a, 0, shape, perm, b, 0);
                }
                else
                {
                    k.Encode(e, b, 0, shape, perm, a, 0);
                }
            }
        });
        var got = await Rig.DownloadAsync(b);
        var bad = 0;
        for (var r = 0; r < N; r++)
        {
            for (var c = 0; c < N; c++)
            {
                bad += got[(r * N) + c] == data[(c * N) + r] ? 0 : 1;
            }
        }

        Rig.Row(d, "dispatch chain", "f32", $"{Passes} dependent transposes of {N}x{N} in one encoder", $"{bad} wrong elements");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task WriteAfterWriteLeavesTheLaterValue()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        const int N = 1 << 20;
        const int Rounds = 100;
        var x = new float[N];
        var y = new float[N];
        for (var i = 0; i < N; i++)
        {
            x[i] = i + 1;
            y[i] = -(i + 3);
        }

        using var bx = d.CreateBuffer<float>(x);
        using var by = d.CreateBuffer<float>(y);
        using var target = d.CreateBuffer<float>(N);
        Rig.Run(d, e =>
        {
            // Each round writes the whole target twice; the second write must be the one that stays, and the next round's first write must not land
            // before this round's second one. The last write of the encoder is neg(y).
            for (var r = 0; r < Rounds; r++)
            {
                k.EncodeUnary(e, GpuUnaryOp.Neg, bx, 0, target, 0, N);
                k.EncodeUnary(e, GpuUnaryOp.Neg, by, 0, target, 0, N);
            }
        });
        var got = await Rig.DownloadAsync(target);
        var bad = 0;
        for (var i = 0; i < N; i++)
        {
            bad += got[i] == -y[i] ? 0 : 1;
        }

        Rig.Row(d, "dispatch chain", "f32", $"{Rounds} rounds of two full writes of {N} elements to one buffer", $"{bad} elements not the later write");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task CopiesAndPassesInterleaveInOrder()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        const int N = 4096;
        var x = new float[N];
        for (var i = 0; i < N; i++)
        {
            x[i] = i;
        }

        using var a = d.CreateBuffer<float>(x, BufferUsage.Storage | BufferUsage.CopySource | BufferUsage.CopyDestination);
        using var b = d.CreateBuffer<float>(N, BufferUsage.Storage | BufferUsage.CopySource | BufferUsage.CopyDestination);
        Rig.Run(d, e =>
        {
            // pass, copy, pass, copy, pass: the copies sit between compute passes and close the shared compute encoder.
            k.EncodeUnary(e, GpuUnaryOp.Neg, a, 0, b, 0, N);
            e.CopyBuffer(b.Buffer, 0, a.Buffer, 0, N * 4ul);
            k.EncodeUnary(e, GpuUnaryOp.Neg, a, 0, b, 0, N);
            e.CopyBuffer(b.Buffer, 0, a.Buffer, 0, N * 4ul);
            k.EncodeUnary(e, GpuUnaryOp.Neg, a, 0, b, 0, N);
        });
        var got = await Rig.DownloadAsync(b);
        var bad = 0;
        for (var i = 0; i < N; i++)
        {
            bad += got[i] == -x[i] ? 0 : 1;
        }

        Assert.Equal(0, bad);
    }

    [Fact]
    public void DependentDispatchCost()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_DISPATCH_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_DISPATCH_BENCH=1 to time chains of dependent dispatches");
        }

        var validate = Environment.GetEnvironmentVariable("ONEGPU_BENCH_VALIDATION") != "0";
        using var d = GpuBackends.CreateDevice(new GpuDeviceOptions { Validation = validate, RequireValidation = validate });
        using var ew = new ElementwiseKernels(d);
        using var a = d.CreateBuffer<float>(1 << 20);
        using var b = d.CreateBuffer<float>(1 << 20);

        // Returns the median microseconds to record the chain and the median from recording to idle.
        (double Record, double Total) Wall(int passes, int count)
        {
            var times = new List<double>();
            var records = new List<double>();
            for (var rep = 0; rep < 12; rep++)
            {
                var sw = Stopwatch.StartNew();
                var enc = d.BeginCommands();
                for (var s = 0; s < passes; s++)
                {
                    if ((s & 1) == 0)
                    {
                        ew.EncodeUnary(enc, GpuUnaryOp.Neg, a, 0, b, 0, count);
                    }
                    else
                    {
                        ew.EncodeUnary(enc, GpuUnaryOp.Neg, b, 0, a, 0, count);
                    }
                }

                var recorded = sw.Elapsed.TotalMilliseconds * 1000;
                d.Submit(enc);
                d.WaitIdle();
                if (rep >= 2)
                {
                    times.Add(sw.Elapsed.TotalMilliseconds * 1000);
                    records.Add(recorded);
                }
            }

            times.Sort();
            records.Sort();
            return (records[records.Count / 2], times[times.Count / 2]);
        }

        _ = Wall(50, 256);
        const int Few = 100;
        const int Many = 1100;
        foreach (var count in new[] { 256, 256 * 64, 256 * 1024 })
        {
            var few = Wall(Few, count);
            var many = Wall(Many, count);
            var perRecord = (many.Record - few.Record) / (Many - Few);
            var perTotal = (many.Total - few.Total) / (Many - Few);
            Console.WriteLine($"DISPATCH\t{d.Adapter.Name}\t{count / 256} workgroups: record {perRecord:F2} us, record to idle {perTotal:F2} us per dependent dispatch");
            Rig.Row(d, "dependent dispatch cost (indicative)", "f32", $"{Few} and {Many} dependent dispatches of {count / 256} workgroups of 256 threads", $"{perRecord:F2} us per dispatch to record, {perTotal:F2} us per dispatch from recording to idle ({few.Total:F0} and {many.Total:F0} us total)");
        }
    }
}
