using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>What one dependent dispatch costs: chains of passes that each read the previous pass's output, timed from recording to idle. Runs only with ONEGPU_DISPATCH_BENCH=1; the numbers it prints are indicative.</summary>
public sealed class DispatchCostThroughput
{
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
