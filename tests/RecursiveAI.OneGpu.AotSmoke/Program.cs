using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.AotSmoke.Shaders;

// Creates a device, prints the adapter and its compute capabilities, adds two arrays on the GPU and compares with the CPU bit for bit.
// Exit code 0 means the device works; 1 means the result differed; 2 means no device. With ONEGPU_VALIDATION=1 the API's validation
// layer is on, and any message it reports fails the run.
using var device = GpuBackends.CreateDevice(new GpuDeviceOptions { ApplicationName = "onegpu-smoke" });
Console.WriteLine($"backend: {device.Backend}");
Console.WriteLine($"adapter: {device.Adapter.Name} ({device.Adapter.Kind}, vendor 0x{device.Adapter.VendorId:x4}, device 0x{device.Adapter.DeviceId:x4}, API {device.Adapter.ApiVersion})");
Console.WriteLine(device.Capabilities.Compute.Describe());

const int Count = 100_000;
var a = new uint[Count];
var b = new uint[Count];
var expected = new uint[Count];
for (var i = 0; i < Count; i++)
{
    a[i] = (uint)((ulong)i * 2654435761UL);
    b[i] = (uint)((ulong)i * 40503UL) + 7;
    expected[i] = unchecked(a[i] + b[i]);
}

using var kernel = device.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, VectorAdd.AddUint.EntryPoint)
{
    SpirV = VectorAdd.AddUint.SpirV,
    Msl = VectorAdd.AddUint.Msl,
    MslFunctionName = VectorAdd.AddUint.MslFunctionName,
    Wgsl = VectorAdd.AddUint.Wgsl,
    WgslEntryPoint = VectorAdd.AddUint.WgslEntryPoint,
    Layout = VectorAdd.AddUint.Layout,
});
using var ba = device.CreateBuffer<uint>(a);
using var bb = device.CreateBuffer<uint>(b);
using var bs = device.CreateBuffer<uint>(Count);
using var group = kernel.CreateBindGroup(device, 0, [BindGroupEntry.ForBuffer(0, ba.Buffer), BindGroupEntry.ForBuffer(1, bb.Buffer), BindGroupEntry.ForBuffer(2, bs.Buffer)]);

var commands = device.BeginCommands();
var pass = commands.BeginComputePass();
pass.SetPipeline(kernel.Pipeline);
pass.SetBindGroup(0, group);
pass.SetPushConstants(0, BitConverter.GetBytes((uint)Count));
var (x, y, z) = kernel.GroupsFor(Count);
pass.Dispatch(x, y, z);
pass.EndPass();
device.Submit(commands);

var actual = new uint[Count];
await bs.ReadAsync(actual);

var wrong = 0;
for (var i = 0; i < Count; i++)
{
    if (actual[i] != expected[i])
    {
        wrong++;
    }
}

device.WaitIdle();
var messages = device.TakeValidationMessages();
foreach (var m in messages)
{
    Console.WriteLine("validation: " + m);
}

Console.WriteLine($"vector add of {Count} elements: {wrong} wrong, {messages.Length} validation messages");
return wrong == 0 && messages.Length == 0 ? 0 : 1;
