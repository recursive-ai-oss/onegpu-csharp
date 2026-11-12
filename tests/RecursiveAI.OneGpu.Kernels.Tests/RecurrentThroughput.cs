using System.Diagnostics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Time per call and per step of a bidirectional LSTM on the shapes of a text-line recognizer, from GPU timestamps (one pair per pass) and from the host clock,
/// next to the cost of a chain of dependent one-workgroup passes (what any per-step design pays before it does a single multiply). Runs only with
/// ONEGPU_RECURRENT_BENCH=1; the numbers it prints are indicative.
/// </summary>
public sealed class RecurrentThroughput
{
    const int Calls = 15;

    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_RECURRENT_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_RECURRENT_BENCH=1 to time the recognizer shapes");
        }

        // ONEGPU_BENCH_VALIDATION=0 times without the validation layers, which slow Metal by an order of magnitude and add host time everywhere.
        var validate = Environment.GetEnvironmentVariable("ONEGPU_BENCH_VALIDATION") != "0";
        using var d = GpuBackends.CreateDevice(new GpuDeviceOptions { Validation = validate, RequireValidation = validate });
        using var k = new RecurrentKernels(d);
        foreach (var (seq, batch, input, hidden) in new[] { (80, 1, 256, 256), (80, 8, 256, 256), (160, 1, 256, 256), (80, 1, 512, 512), (48, 8, 192, 96) })
        {
            Time<float>(d, k, GpuRecurrentCell.Lstm, seq, batch, input, hidden);
            Time<Half>(d, k, GpuRecurrentCell.Lstm, seq, batch, input, hidden);
        }

        Time<float>(d, k, GpuRecurrentCell.Gru, 80, 1, 256, 256);

        Floor(d, 80);
        d.WaitIdle();
    }

    static GpuBuffer<T> Buffer<T>(GpuDevice d, int n)
        where T : unmanaged => d.CreateBuffer<T>(new T[typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n) : n]);

    static void Time<T>(GpuDevice d, RecurrentKernels k, GpuRecurrentCell cell, int seq, int batch, int input, int hidden)
        where T : unmanaged
    {
        var p = new RecurrentProblem(cell, seq, batch, input, hidden) { Direction = GpuRecurrentDirection.Bidirectional };
        var gh = p.Gates * hidden;
        using var x = Buffer<T>(d, seq * batch * input);
        using var w = Buffer<T>(d, 2 * gh * input);
        using var r = Buffer<T>(d, 2 * gh * hidden);
        using var b = Buffer<T>(d, 2 * 2 * gh);
        using var y = Buffer<T>(d, seq * 2 * batch * hidden);
        using var yh = Buffer<T>(d, 2 * batch * hidden);
        var o = new RecurrentOperands<T> { X = new(x), W = new(w), R = new(r), B = new(b), Y = new(y), YH = new(yh) };
        var timestamps = d.Capabilities.Timestamps;
        var maxPasses = (uint)((2 * seq) + 16);
        using var set = timestamps ? d.CreateTimestampSet(2 * maxPasses) : null;
        var outputs = 2;
        for (var i = 0; i < 3; i++)
        {
            Rig.Run(d, e => k.Encode(e, p, o));
        }

        d.WaitIdle();
        // Wall time with no pass timing: on Metal every timed pass gets an encoder of its own, so the timed runs below carry an encoder per pass.
        var plain = new List<double>();
        var recordTimes = new List<double>();
        var submitTimes = new List<double>();
        for (var i = 0; i < Calls; i++)
        {
            var sw0 = Stopwatch.StartNew();
            var enc = d.BeginCommands();
            k.Encode(enc, p, o);
            recordTimes.Add(sw0.Elapsed.TotalMilliseconds * 1e6);
            var sw = Stopwatch.StartNew();
            d.Submit(enc);
            submitTimes.Add(sw.Elapsed.TotalMilliseconds * 1e6);
            d.WaitIdle();
            plain.Add(sw.Elapsed.TotalMilliseconds * 1e6);
        }

        plain.Sort();
        recordTimes.Sort();
        submitTimes.Sort();
        var spans = new List<double>();
        var steps = new List<double>();
        var walls = new List<double>();
        for (var i = 0; i < Calls; i++)
        {
            var enc = d.BeginCommands();
            if (set is not null)
            {
                enc.BeginPassTiming(new PassTiming(set, 0, maxPasses));
            }

            k.Encode(enc, p, o);
            var n = set is null ? 0u : enc.EndPassTiming();
            var sw = Stopwatch.StartNew();
            d.Submit(enc);
            d.WaitIdle();
            walls.Add(sw.Elapsed.TotalMilliseconds * 1e6);
            if (set is not null)
            {
                var t = new ulong[2 * n];
                Assert.True(d.TryReadTimestamps(set, 0, t));
                spans.Add(t[2 * n - 1] - t[0]);
                var split = cell == GpuRecurrentCell.Gru ? 2 : 1;
                var first = (int)n - outputs - (seq * split);
                var sum = 0.0;
                for (var s = first; s < first + (seq * split); s++)
                {
                    sum += t[2 * s + 1] - t[2 * s];
                }

                steps.Add(sum / seq);
            }
        }

        double Median(List<double> v)
        {
            v.Sort();
            return v.Count == 0 ? double.NaN : v[v.Count / 2];
        }

        var gpu = Median(spans);
        var wall = Median(walls);
        var step = Median(steps);
        var line = $"{d.Adapter.Name}\t{cell} bidirectional {typeof(T).Name}\tS{seq} B{batch} I{input} H{hidden}\t{gpu / 1000:F0} us gpu per call\t{wall / 1000:F0} us wall per call\t{step / 1000:F1} us per step (gpu passes)\t{wall / seq / 1000:F1} us wall per step\t{plain[plain.Count / 2] / 1000:F0} us wall per call untimed";
        Console.WriteLine("RECURRENT\t" + line);
        Rig.Row(d, "recurrent timing (indicative)", typeof(T).Name, $"{cell} bidirectional S{seq} B{batch} I{input} H{hidden}", $"{gpu / 1000:F0} us gpu per call, {wall / 1000:F0} us wall per call, {step / 1000:F1} us per step (sum of the step passes), {wall / seq / 1000:F1} us wall per step, {plain[plain.Count / 2] / 1000:F0} us wall per call with no pass timing (record {recordTimes[recordTimes.Count / 2] / 1000:F0} us before it, submit call {submitTimes[submitTimes.Count / 2] / 1000:F0} us of it)");
    }

    // A chain of dependent passes of one small workgroup each: the floor of any one-pass-per-step design.
    static void Floor(GpuDevice d, int passes)
    {
        using var ew = new ElementwiseKernels(d);
        using var a = d.CreateBuffer<float>(1024);
        using var b = d.CreateBuffer<float>(1024);
        var timestamps = d.Capabilities.Timestamps;
        using var set = timestamps ? d.CreateTimestampSet(2 * (uint)passes) : null;
        var spans = new List<double>();
        var walls = new List<double>();
        var each = new List<double>();
        for (var i = 0; i < Calls + 3; i++)
        {
            var enc = d.BeginCommands();
            if (set is not null)
            {
                enc.BeginPassTiming(new PassTiming(set, 0, (uint)passes));
            }

            for (var s = 0; s < passes; s++)
            {
                if ((s & 1) == 0)
                {
                    ew.EncodeUnary(enc, GpuUnaryOp.Neg, a, 0, b, 0, 256);
                }
                else
                {
                    ew.EncodeUnary(enc, GpuUnaryOp.Neg, b, 0, a, 0, 256);
                }
            }

            var n = set is null ? 0u : enc.EndPassTiming();
            var sw = Stopwatch.StartNew();
            d.Submit(enc);
            d.WaitIdle();
            if (i < 3)
            {
                continue;
            }

            walls.Add(sw.Elapsed.TotalMilliseconds * 1e6);
            if (set is not null)
            {
                var t = new ulong[2 * n];
                Assert.True(d.TryReadTimestamps(set, 0, t));
                spans.Add(t[2 * n - 1] - t[0]);
                var sum = 0.0;
                for (var s = 0; s < n; s++)
                {
                    sum += t[2 * s + 1] - t[2 * s];
                }

                each.Add(sum / n);
            }
        }

        double Median(List<double> v)
        {
            v.Sort();
            return v.Count == 0 ? double.NaN : v[v.Count / 2];
        }

        var line = $"{passes} dependent one-workgroup passes: {Median(spans) / passes / 1000:F2} us gpu per pass (start to end of the chain), {Median(each) / 1000:F2} us inside each pass, {Median(walls) / passes / 1000:F2} us wall per pass";
        Console.WriteLine("RECURRENT\t" + d.Adapter.Name + "\tfloor\t" + line);
        Rig.Row(d, "recurrent timing (indicative)", "-", "floor of a per-step design", line);
    }
}
