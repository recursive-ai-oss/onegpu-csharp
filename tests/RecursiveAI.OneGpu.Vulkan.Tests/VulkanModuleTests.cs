using RecursiveAI.OneGpu.Vulkan;

namespace RecursiveAI.OneGpu.Vulkan.Tests;

public class VulkanModuleTests
{
    [Fact]
    public void ModuleIsNamedAfterItsAssembly() => Assert.Equal("RecursiveAI.OneGpu.Vulkan", VulkanModule.Name);
}
