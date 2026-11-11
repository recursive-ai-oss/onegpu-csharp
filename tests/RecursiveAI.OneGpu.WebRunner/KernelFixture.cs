namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>The browser's counterpart of the kernel tests' fixture: the shared WebGPU device, and a check that the test left no validation messages.</summary>
internal sealed class GpuFixture : IDisposable
{
    public GpuDevice Device { get; } = RecursiveAI.OneGpu.Vulkan.Tests.GpuFixture.Shared ?? throw new InvalidOperationException("No WebGPU device.");

    public void Dispose()
    {
        var messages = Device.TakeValidationMessages();
        Assert.True(messages.Length == 0, "WebGPU reported:\n" + string.Join("\n", messages));
    }
}
