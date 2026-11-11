using OneShader.Layouts;
using RecursiveAI.OneGpu.Vulkan.Tests.Shaders;

namespace RecursiveAI.OneGpu.Vulkan.Tests;

/// <summary>
/// Several threads create, use and dispose devices at the same time, as a host with parallel sessions or a parallel test runner does. Creation
/// and teardown share the process-wide instance and the validation router, so they must be safe to overlap.
/// </summary>
public class ConcurrentDeviceTests
{
    const int Threads = 8;
    const int Iterations = 50;

    [Fact]
    public async Task DevicesCanBeCreatedUsedAndDisposedFromManyThreads()
    {
        var options = new GpuDeviceOptions { Validation = true, RequireValidation = true };
        // Fail early and clearly when there is no device, before eight threads race to the same error.
        try
        {
            GpuBackends.CreateDevice(options).Dispose();
        }
        catch (GpuUnsupportedException e) when (e.Capability is null && Environment.GetEnvironmentVariable("ONEGPU_REQUIRE_DEVICE") != "1")
        {
            Assert.Skip("no GPU device available: " + e.Message);
            throw;
        }

        var failures = new System.Collections.Concurrent.ConcurrentQueue<string>();
        var start = new Barrier(Threads);
        var tasks = Enumerable.Range(0, Threads).Select(t => Task.Factory.StartNew(() =>
        {
            start.SignalAndWait();
            for (var i = 0; i < Iterations; i++)
            {
                try
                {
                    Cycle(options, t, i, failures);
                }
                catch (Exception e)
                {
                    failures.Enqueue($"thread {t} iteration {i}: {e.GetType().Name}: {e.Message}");
                    return;
                }
            }
        }, TaskCreationOptions.LongRunning)).ToArray();
        await Task.WhenAll(tasks);
        Assert.True(failures.IsEmpty, $"{failures.Count} failures, first:\n" + string.Join("\n", failures.Take(5)));
        Console.WriteLine($"concurrent device cycles completed: {Threads} threads x {Iterations}");
    }

    static void Cycle(GpuDeviceOptions options, int thread, int iteration, System.Collections.Concurrent.ConcurrentQueue<string> failures)
    {
        var device = GpuBackends.CreateDevice(options);
        string[] messages;
        using (device)
        {
            // A little real work: a kernel, bind group, dispatch and read back, so the teardown has live objects to retire.
            using var kernel = device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, VectorAdd.AddUint.EntryPoint) { SpirV = VectorAdd.AddUint.SpirV, Wgsl = VectorAdd.AddUint.Wgsl, WgslEntryPoint = VectorAdd.AddUint.WgslEntryPoint, Msl = VectorAdd.AddUint.Msl, MslFunctionName = VectorAdd.AddUint.MslFunctionName, Layout = VectorAdd.AddUint.Layout });
            const int N = 300;
            var a = new uint[N];
            var b = new uint[N];
            for (var i = 0; i < N; i++)
            {
                a[i] = (uint)(i + thread);
                b[i] = (uint)(i * 3 + iteration);
            }

            using var ba = device.CreateBuffer<uint>(a);
            using var bb = device.CreateBuffer<uint>(b);
            using var bs = device.CreateBuffer<uint>(N);
            using var group = kernel.CreateBindGroup(device, 0, [BindGroupEntry.ForBuffer(0, ba.Buffer), BindGroupEntry.ForBuffer(1, bb.Buffer), BindGroupEntry.ForBuffer(2, bs.Buffer)]);
            var enc = device.BeginCommands();
            var pass = enc.BeginComputePass();
            pass.SetPipeline(kernel.Pipeline);
            pass.SetBindGroup(0, group);
            pass.SetPushConstants(0, BitConverter.GetBytes((uint)N));
            var (x, y, z) = kernel.GroupsFor(N);
            pass.Dispatch(x, y, z);
            pass.EndPass();
            device.Submit(enc);
            var sum = new uint[N];
            bs.Read(sum);
            for (var i = 0; i < N; i++)
            {
                if (sum[i] != a[i] + b[i])
                {
                    failures.Enqueue($"thread {thread} iteration {iteration}: wrong sum at {i}");
                    break;
                }
            }

            device.WaitIdle();
            messages = device.TakeValidationMessages();
        }

        messages = [.. messages, .. device.TakeValidationMessages()];
        if (messages.Length != 0)
        {
            failures.Enqueue($"thread {thread} iteration {iteration}: validation layer reported:\n" + string.Join("\n", messages));
        }
    }
}
