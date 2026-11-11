using System.Runtime.InteropServices;
using OneShader.Layouts;
using OneShader.Reflection;
using RecursiveAI.OneGpu.Vulkan.Tests.Shaders;

namespace RecursiveAI.OneGpu.Vulkan.Tests;

/// <summary>The compute path end to end: kernels built from reflection, typed buffers, push constants, async readback and capability checks.</summary>
public sealed class ComputeTests
{
    static GpuComputeKernel AddUintKernel(GpuDevice d) =>
        d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, VectorAdd.AddUint.EntryPoint) { SpirV = VectorAdd.AddUint.SpirV, Wgsl = VectorAdd.AddUint.Wgsl, WgslEntryPoint = VectorAdd.AddUint.WgslEntryPoint, Msl = VectorAdd.AddUint.Msl, MslFunctionName = VectorAdd.AddUint.MslFunctionName, Layout = VectorAdd.AddUint.Layout });

    static GpuComputeKernel AddFloatKernel(GpuDevice d) =>
        d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, VectorAdd.AddFloat.EntryPoint) { SpirV = VectorAdd.AddFloat.SpirV, Wgsl = VectorAdd.AddFloat.Wgsl, WgslEntryPoint = VectorAdd.AddFloat.WgslEntryPoint, Msl = VectorAdd.AddFloat.Msl, MslFunctionName = VectorAdd.AddFloat.MslFunctionName, Layout = VectorAdd.AddFloat.Layout });

    static GpuComputeKernel FillKernel(GpuDevice d) =>
        d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, BufferFill.CSMain.EntryPoint) { SpirV = BufferFill.CSMain.SpirV, Wgsl = BufferFill.CSMain.Wgsl, WgslEntryPoint = BufferFill.CSMain.WgslEntryPoint, Msl = BufferFill.CSMain.Msl, MslFunctionName = BufferFill.CSMain.MslFunctionName, Layout = BufferFill.CSMain.Layout });

    static void Dispatch(GpuDevice d, GpuComputeKernel kernel, GpuBindGroup group, ReadOnlySpan<byte> push, uint threads)
    {
        var enc = d.BeginCommands();
        var pass = enc.BeginComputePass();
        pass.SetPipeline(kernel.Pipeline);
        pass.SetBindGroup(0, group);
        pass.SetPushConstants(0, push);
        var (x, y, z) = kernel.GroupsFor(threads);
        pass.Dispatch(x, y, z);
        pass.EndPass();
        d.Submit(enc);
    }

    static uint[] RandomUints(int n, uint seed)
    {
        var r = new uint[n];
        var s = seed;
        for (var i = 0; i < n; i++)
        {
            s = (s * 1664525u) + 1013904223u;
            r[i] = s;
        }

        return r;
    }

    [Fact]
    public async Task VectorAddOfIntegersMatchesTheCpuBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernel = AddUintKernel(d);
        Assert.Equal(64u, kernel.WorkgroupSizeX);
        foreach (var n in new[] { 1, 31, 63, 64, 65, 1000, 4097 })
        {
            var a = RandomUints(n, 1);
            var b = RandomUints(n, 2);
            using var ba = d.CreateBuffer<uint>(a);
            using var bb = d.CreateBuffer<uint>(b);
            using var bs = d.CreateBuffer<uint>(n);
            using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, ba.Buffer), BindGroupEntry.ForBuffer(1, bb.Buffer), BindGroupEntry.ForBuffer(2, bs.Buffer)]);
            Dispatch(d, kernel, group, BitConverter.GetBytes((uint)n), (uint)n);
            var sum = new uint[n];
            await bs.ReadAsync(sum);
            for (var i = 0; i < n; i++)
            {
                Assert.Equal(unchecked(a[i] + b[i]), sum[i]);
            }
        }
    }

    [Fact]
    public async Task VectorAddOfFloatsMatchesTheCpuBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernel = AddFloatKernel(d);
        foreach (var n in new[] { 1, 33, 64, 129, 2048, 5000 })
        {
            // Values in [-1024, 1024) with 10 fraction bits: sums are exact in single precision, so any IEEE adder agrees.
            var a = RandomUints(n, 3).Select(u => ((int)(u >> 11) - (1 << 20)) / 1024f).ToArray();
            var b = RandomUints(n, 4).Select(u => ((int)(u >> 11) - (1 << 20)) / 1024f).ToArray();
            using var ba = d.CreateBuffer<float>(a);
            using var bb = d.CreateBuffer<float>(b);
            using var bs = d.CreateBuffer<float>(n);
            using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(3, ba.Buffer), BindGroupEntry.ForBuffer(4, bb.Buffer), BindGroupEntry.ForBuffer(5, bs.Buffer)]);
            Dispatch(d, kernel, group, BitConverter.GetBytes((uint)n), (uint)n);
            var sum = new float[n];
            await bs.ReadAsync(sum);
            for (var i = 0; i < n; i++)
            {
                Assert.Equal(BitConverter.SingleToUInt32Bits(a[i] + b[i]), BitConverter.SingleToUInt32Bits(sum[i]));
            }
        }
    }

    /// <summary>
    /// GPU time of a compute pass is in nanoseconds and within the wall time of the run that contained it. The work grows until the run is long
    /// enough (tens of milliseconds) that submission overhead is small next to it, so a wrong tick rate shows as a wrong ratio.
    /// </summary>
    [Fact]
    public void ComputePassTimingIsOrderedAndMatchesTheWallClock()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        if (!d.Capabilities.Timestamps)
        {
            return;
        }

        using var kernel = d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, Spin.Burn.EntryPoint) { SpirV = Spin.Burn.SpirV, Wgsl = Spin.Burn.Wgsl, WgslEntryPoint = Spin.Burn.WgslEntryPoint, Msl = Spin.Burn.Msl, MslFunctionName = Spin.Burn.MslFunctionName, Layout = Spin.Burn.Layout });
        const uint Threads = 1 << 16;
        using var result = d.CreateBuffer<float>((int)Threads);
        using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, result.Buffer)]);
        using var set = d.CreateTimestampSet(4);
        var (gx, gy, gz) = kernel.GroupsFor(Threads);

        void Record(GpuCommandEncoder enc, uint iterations)
        {
            var pass = enc.BeginComputePass();
            pass.SetPipeline(kernel.Pipeline);
            pass.SetBindGroup(0, group);
            pass.SetPushConstants(0, BitConverter.GetBytes(iterations));
            pass.Dispatch(gx, gy, gz);
            pass.EndPass();
        }

        var t = new ulong[4];

        // One timed submission: GPU time of the two passes and the wall time of submit-and-wait around them.
        (double First, double Second, double Wall) Measure(uint iterations)
        {
            var enc = d.BeginCommands();
            enc.BeginPassTiming(new PassTiming(set, 0, 2));
            Record(enc, iterations);
            Record(enc, iterations * 2);
            Assert.Equal(2u, enc.EndPassTiming());
            d.WaitIdle();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            d.Submit(enc);
            d.WaitIdle();
            var wall = clock.Elapsed.TotalMilliseconds * 1e6;
            Assert.True(d.TryReadTimestamps(set, 0, t), "the timestamps of a finished submission are readable");
            Assert.True(t[1] > t[0] && t[3] > t[2], "each pass ends after it begins");
            Assert.True(t[2] >= t[1], "passes run in order");
            return ((double)(t[1] - t[0]), (double)(t[3] - t[2]), wall);
        }

        // Grow the work until the GPU time is long enough (tens of milliseconds) for submission overhead to be small beside it.
        uint iterations = 1 << 10;
        var run = Measure(iterations);
        for (var attempt = 0; attempt < 14 && run.First + run.Second < 40e6; attempt++)
        {
            iterations *= 2;
            run = Measure(iterations);
        }

        // Other tests may share the GPU, and time-slicing adds to the wall time but not to the GPU time of the passes; the least disturbed of
        // a few repeats is the one that shows whether the tick rate is right.
        var best = run;
        for (var repeat = 0; repeat < 6; repeat++)
        {
            run = Measure(iterations);
            if (run.Wall < best.Wall)
            {
                best = run;
            }
        }

        var total = best.First + best.Second;

        // The pair of passes fits inside the wall time of the submit-and-wait, and (the wall time being mostly GPU work) fills most of it.
        Assert.True(total <= best.Wall * 1.02, $"GPU time {total / 1e6:F2} ms exceeds wall time {best.Wall / 1e6:F2} ms");
        Assert.True(total >= best.Wall * 0.3, $"GPU time {total / 1e6:F2} ms is far below wall time {best.Wall / 1e6:F2} ms");

        // The second pass does twice the work of the first.
        Assert.InRange(best.Second / best.First, 1.4, 2.8);
        Console.WriteLine($"compute pass timing on {d.Adapter.Name}: {best.First / 1e6:F3} ms and {best.Second / 1e6:F3} ms of {best.Wall / 1e6:F3} ms wall (iterations {iterations})");
    }

    /// <summary>Many timed passes in one command buffer give timestamps that never go backwards.</summary>
    [Fact]
    public void ManyTimedPassesStayInOrder()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        if (!d.Capabilities.Timestamps)
        {
            return;
        }

        using var kernel = d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, Spin.Burn.EntryPoint) { SpirV = Spin.Burn.SpirV, Wgsl = Spin.Burn.Wgsl, WgslEntryPoint = Spin.Burn.WgslEntryPoint, Msl = Spin.Burn.Msl, MslFunctionName = Spin.Burn.MslFunctionName, Layout = Spin.Burn.Layout });
        using var result = d.CreateBuffer<float>(64);
        using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, result.Buffer)]);
        const uint Passes = 256;
        using var set = d.CreateTimestampSet(Passes * 2);
        var enc = d.BeginCommands();
        enc.BeginPassTiming(new PassTiming(set, 0, Passes));
        for (var i = 0; i < Passes; i++)
        {
            var pass = enc.BeginComputePass();
            pass.SetPipeline(kernel.Pipeline);
            pass.SetBindGroup(0, group);
            pass.SetPushConstants(0, BitConverter.GetBytes(1u));
            pass.Dispatch(1, 1, 1);
            pass.EndPass();
        }

        Assert.Equal(Passes, enc.EndPassTiming());
        d.Submit(enc);
        d.WaitIdle();
        var t = new ulong[Passes * 2];
        Assert.True(d.TryReadTimestamps(set, 0, t));
        for (var i = 1; i < t.Length; i++)
        {
            Assert.True(t[i] >= t[i - 1], $"timestamp {i} is earlier than timestamp {i - 1}");
        }

        Assert.True(t[^1] > t[0], "the passes took time");
        Console.WriteLine($"{Passes} one-group passes on {d.Adapter.Name} spanned {t[^1] - t[0]} ns");
    }

    [Fact]
    public async Task FillWritesOnlyTheRequestedRange()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernel = FillKernel(d);
        const int Length = 300;
        var sentinel = new uint[Length];
        Array.Fill(sentinel, 0xDEADBEEFu);
        using var data = d.CreateBuffer<uint>(sentinel);
        using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, data.Buffer)]);
        var push = new uint[] { 1000, 100, 50 };
        Dispatch(d, kernel, group, MemoryMarshal.AsBytes<uint>(push), 100);
        var result = new uint[Length];
        await data.ReadAsync(result);
        for (var i = 0; i < Length; i++)
        {
            Assert.Equal(i is >= 50 and < 150 ? 1000u + (uint)(i - 50) : 0xDEADBEEFu, result[i]);
        }

        // A partial read from an element offset returns the same bytes.
        var part = new uint[10];
        await data.ReadAsync(part, 45);
        Assert.Equal(result[45..55], part);
    }

    [Fact]
    public async Task ReadbackOfEarlierWorkDoesNotWaitForLaterWork()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernel = FillKernel(d);
        using var first = d.CreateBuffer<uint>(256);
        using var second = d.CreateBuffer<uint>(256);
        using var g1 = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, first.Buffer)]);
        using var g2 = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, second.Buffer)]);
        Dispatch(d, kernel, g1, MemoryMarshal.AsBytes<uint>(new uint[] { 10, 256, 0 }), 256);
        var out1 = new uint[256];
        var pending1 = first.ReadAsync(out1);

        // Work submitted while the first readback is pending, with its own readback behind it.
        Dispatch(d, kernel, g2, MemoryMarshal.AsBytes<uint>(new uint[] { 20, 256, 0 }), 256);
        var out2 = new uint[256];
        var pending2 = second.ReadAsync(out2);

        await pending2;
        await pending1;
        for (uint i = 0; i < 256; i++)
        {
            Assert.Equal(10 + i, out1[i]);
            Assert.Equal(20 + i, out2[i]);
        }
    }

    [Fact]
    public async Task UntypedAsyncReadbackFillsAReusedBuffer()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernel = FillKernel(d);
        using var data = d.CreateBuffer<uint>(64);
        using var group = kernel.CreateBindGroup(d, 0, [BindGroupEntry.ForBuffer(0, data.Buffer)]);
        var bytes = new byte[64 * 4];
        for (uint round = 0; round < 3; round++)
        {
            Dispatch(d, kernel, group, MemoryMarshal.AsBytes<uint>(new uint[] { round * 100, 64, 0 }), 64);
            await d.ReadBufferAsync(data.Buffer, 0, bytes);
            Assert.Equal(round * 100, BitConverter.ToUInt32(bytes, 0));
            Assert.Equal((round * 100) + 63, BitConverter.ToUInt32(bytes, 63 * 4));
        }
    }

    [Fact]
    public void CapabilitiesAreReportedFromTheDevice()
    {
        using var gpu = new GpuFixture();
        var c = gpu.Device.Capabilities.Compute;

        // The sample kernels need 64 invocations and a push constant, which every backend offers.
        Assert.True(c.MaxWorkgroupInvocations >= 64, c.Describe());
        Assert.True(c.MaxWorkgroupSizeX >= 64 && c.MaxWorkgroupSizeY >= 1 && c.MaxWorkgroupSizeZ >= 1, c.Describe());
        Assert.True(c.MaxWorkgroupMemoryBytes >= 16384, c.Describe());
        Assert.True(c.MaxStorageBufferBindingBytes >= 128u << 20, c.Describe());
        Assert.True(c.MaxStorageBuffersPerStage >= 4 && c.MaxBindGroups >= 4, c.Describe());
        if (c.SubgroupMinSize != 0)
        {
            Assert.True(c.SubgroupMinSize <= c.SubgroupMaxSize && c.SubgroupMaxSize <= 128, c.Describe());
            Assert.True(c.SubgroupSize == 0 || (c.SubgroupSize >= c.SubgroupMinSize && c.SubgroupSize <= c.SubgroupMaxSize), c.Describe());
        }

        foreach (var m in c.CooperativeMatrices)
        {
            Assert.True(m.M > 0 && m.N > 0 && m.K > 0, m.ToString());
        }

        Assert.Equal(c.CooperativeMatrices.Count > 0, c.CooperativeMatrix);
    }

    [Fact]
    public void KernelsThatDoNotFitTheDeviceAreRejectedWithTheReason()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;

        // Layouts built by hand: only the checks that run before any shader code is looked at are exercised.
        var tooBig = new EntryLayout("Big", OneShader.ShaderStage.Compute, [], workgroupSize: (4096, 1, 1));
        var e1 = Assert.Throws<GpuUnsupportedException>(() => d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, "Big") { Layout = tooBig }));
        Assert.Equal("MaxWorkgroupSize", e1.Capability);
        Assert.Contains("4096", e1.Message, StringComparison.Ordinal);

        var tooManyInvocations = new EntryLayout("Cube", OneShader.ShaderStage.Compute, [], workgroupSize: (16, 16, 16));
        if (d.Capabilities.Compute.MaxWorkgroupInvocations < 4096)
        {
            var e2 = Assert.Throws<GpuUnsupportedException>(() => d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, "Cube") { Layout = tooManyInvocations }));
            Assert.Equal(nameof(GpuComputeCapabilities.MaxWorkgroupInvocations), e2.Capability);
        }

        // 64-bit atomics are not enabled by any backend here, so a kernel that needs them says so.
        var atomics = new EntryLayout("Atomics", OneShader.ShaderStage.Compute, [], workgroupSize: (1, 1, 1), requiredFeatures: DeviceFeatures.Int64Atomics);
        var e3 = Assert.Throws<GpuUnsupportedException>(() => d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, "Atomics") { Layout = atomics }));
        Assert.Equal(nameof(GpuComputeCapabilities.ShaderFeatures), e3.Capability);
        Assert.Contains("Int64Atomics", e3.Expected, StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => d.CreateComputeKernel(new ShaderSource(ShaderStage.Compute, "NoLayout")));
        Assert.Throws<ArgumentException>(() => d.CreateComputeKernel(new ShaderSource(ShaderStage.Vertex, "Vs") { Layout = tooBig }));
    }

    [Fact]
    public async Task BufferBoundsAreChecked()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var buffer = d.CreateBuffer<uint>(8);
        Assert.Throws<ArgumentOutOfRangeException>(() => buffer.Write(new uint[9]));
        Assert.Throws<ArgumentOutOfRangeException>(() => buffer.Write(new uint[2], 7));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await buffer.ReadAsync(new uint[1], 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => d.CreateBuffer<uint>(0));
        Assert.Equal(32ul, buffer.SizeInBytes);

        // One element more than the device's largest buffer, when that fits an int-sized element count.
        var max = d.Capabilities.Compute.MaxBufferBytes;
        if (max != 0 && max / sizeof(ulong) < int.MaxValue - 1)
        {
            var huge = Assert.Throws<GpuUnsupportedException>(() => d.CreateBuffer<ulong>((int)(max / sizeof(ulong)) + 1));
            Assert.Equal(nameof(GpuComputeCapabilities.MaxBufferBytes), huge.Capability);
        }
    }
}
