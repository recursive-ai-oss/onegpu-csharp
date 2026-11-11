namespace RecursiveAI.OneGpu.Vulkan.Tests;

/// <summary>
/// Devices in one process share a Vulkan instance and its one message callback, so each message has to be sent to the device it concerns,
/// including what the layer says while a device is destroyed.
/// </summary>
public sealed class ValidationRoutingTests
{
    static GpuDevice Create()
    {
        try
        {
            return GpuBackends.CreateDevice(new GpuDeviceOptions { Validation = true, RequireValidation = true });
        }
        catch (GpuUnsupportedException e) when (e.Capability is null && Environment.GetEnvironmentVariable("ONEGPU_REQUIRE_DEVICE") != "1")
        {
            Assert.Skip("no GPU device available: " + e.Message);
            throw;
        }
    }

    [Fact]
    public void ALeakedBufferIsReportedToItsOwnDeviceOnly()
    {
        var a = Create();
        var b = Create();
        try
        {
            // A buffer that is never disposed: the layer lists it when its device is destroyed.
            var leaked = a.CreateBuffer<uint>(64);
            using var fine = b.CreateBuffer<uint>(64);
            a.WaitIdle();
            b.WaitIdle();
            Assert.Empty(a.TakeValidationMessages());
            Assert.Empty(b.TakeValidationMessages());

            a.Dispose();
            var report = a.TakeValidationMessages();
            Assert.NotEmpty(report);
            Console.WriteLine("leak report: " + string.Join(" | ", report).Replace("\n", " "));
            Assert.Contains(report, m => m.Contains("Buffer", StringComparison.OrdinalIgnoreCase) || m.Contains("leak", StringComparison.OrdinalIgnoreCase));

            // B was alive while A was destroyed and must not have been told about it.
            Assert.Empty(b.TakeValidationMessages());
            GC.KeepAlive(leaked);

            // And the other way round: B's own clean teardown reports nothing, to anyone.
            b.WaitIdle();
        }
        finally
        {
            a.Dispose();
        }

        // Disposing B without its buffer outliving it is clean.
        b.Dispose();
        Assert.Empty(b.TakeValidationMessages());
    }

    [Fact]
    public void TwoLiveDevicesEachReportTheirOwnLeak()
    {
        var a = Create();
        var b = Create();
        var leakedA = a.CreateBuffer<uint>(64);
        var leakedB = b.CreateBuffer<uint>(128);
        b.Dispose();
        var fromB = b.TakeValidationMessages();
        a.Dispose();
        var fromA = a.TakeValidationMessages();
        GC.KeepAlive(leakedA);
        GC.KeepAlive(leakedB);
        Assert.NotEmpty(fromA);
        Assert.NotEmpty(fromB);

        // Each report names its own device's leak only: one buffer each.
        Assert.DoesNotContain(fromA, m => fromB.Contains(m));
    }
}
