namespace RecursiveAI.OneGpu.Vulkan.Tests;

/// <summary>
/// A long-lived tool or test process creates and destroys devices again and again. Some drivers refuse new instances after a few dozen
/// create/destroy pairs in one process, so the backend must not need a fresh instance per device.
/// </summary>
public class DeviceCycleTests
{
    /// <summary>Cycles one process must survive; the NVIDIA 595 driver gave up near 88 when each device had its own instance.</summary>
    const int Cycles = 500;

    [Fact]
    public void ManyCreateDisposeCyclesWorkInOneProcess()
    {
        Assert.SkipUnless(GpuBackends.Selected == GpuBackend.Vulkan, "instance reuse is a Vulkan matter");
        var options = new GpuDeviceOptions { Validation = true, RequireValidation = true };
        for (var i = 0; i < Cycles; i++)
        {
            GpuDevice device;
            try
            {
                device = GpuBackends.CreateDevice(options);
            }
            catch (GpuUnsupportedException e) when (i == 0 && e.Capability is null && Environment.GetEnvironmentVariable("ONEGPU_REQUIRE_DEVICE") != "1")
            {
                Assert.Skip("no GPU device available: " + e.Message);
                throw;
            }
            catch (GpuUnsupportedException e)
            {
                throw new InvalidOperationException($"device creation failed at cycle {i} of {Cycles}: {e.Message}", e);
            }

            using (device)
            {
                // A little real work, so each cycle exercises memory, a submission and teardown with work retired.
                using var buffer = device.CreateBuffer<uint>(256);
                device.WaitIdle();
                var messages = device.TakeValidationMessages();
                Assert.True(messages.Length == 0, $"cycle {i}: validation layer reported:\n" + string.Join("\n", messages));
            }
        }

        Console.WriteLine($"device cycles completed: {Cycles}");
    }
}
