namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Creates a device with the validation layer required, and fails the test if the layer reported anything.</summary>
internal sealed class GpuFixture : IDisposable
{
    public GpuDevice Device { get; }

    public GpuFixture()
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
}
