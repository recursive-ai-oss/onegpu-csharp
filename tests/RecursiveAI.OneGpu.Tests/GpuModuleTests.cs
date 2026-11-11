using RecursiveAI.OneGpu;

namespace RecursiveAI.OneGpu.Tests;

public class GpuModuleTests
{
    [Fact]
    public void ModuleIsNamedAfterItsAssembly() => Assert.Equal("RecursiveAI.OneGpu", GpuModule.Name);
}
