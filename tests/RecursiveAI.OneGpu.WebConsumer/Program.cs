using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.WebGpu;
using WebConsumer.Shaders;

// The page calls Run.Go() after the runtime starts; there is nothing to do at start-up.
return;

#pragma warning disable CA1050, CA1852
[SupportedOSPlatform("browser")]
public static partial class Run
{
    /// <summary>Adds one to 1000 integers on the GPU and reports the adapter and whether every sum is right.</summary>
    [JSExport]
    public static async Task<string> Go()
    {
        using var device = await WebGpu.CreateDeviceAsync();
        using var kernel = device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, AddOne.Increment.EntryPoint) { Wgsl = AddOne.Increment.Wgsl, WgslEntryPoint = AddOne.Increment.WgslEntryPoint, Layout = AddOne.Increment.Layout });
        const int n = 1000;
        var input = new uint[n];
        for (var i = 0; i < n; i++)
        {
            input[i] = (uint)(i * 7);
        }

        using var a = device.CreateBuffer<uint>(input);
        using var b = device.CreateBuffer<uint>(n);
        using var group = kernel.CreateBindGroup(device, 0, [BindGroupEntry.ForBuffer(0, a.Buffer), BindGroupEntry.ForBuffer(1, b.Buffer)]);
        var enc = device.BeginCommands();
        var pass = enc.BeginComputePass();
        pass.SetPipeline(kernel.Pipeline);
        pass.SetBindGroup(0, group);
        pass.SetPushConstants(0, BitConverter.GetBytes((uint)n));
        var (x, y, z) = kernel.GroupsFor(n);
        pass.Dispatch(x, y, z);
        pass.EndPass();
        device.Submit(enc);
        var sums = new uint[n];
        await b.ReadAsync(sums);
        var wrong = 0;
        for (var i = 0; i < n; i++)
        {
            if (sums[i] != input[i] + 1)
            {
                wrong++;
            }
        }

        var messages = device.TakeValidationMessages();
        return $"{{\"adapter\":\"{device.Adapter.Name}\",\"count\":{n},\"wrong\":{wrong},\"messages\":{messages.Length}}}";
    }
}
