namespace RecursiveAI.OneGpu.Tests;

/// <summary>The asynchronous readback machinery without a device: completion order, results and failures.</summary>
public sealed class ReadbackTests
{
    sealed class FakeReadback(byte[] source, Action<string> log, string name, bool fail = false) : ReadbackOperation
    {
        public Task<bool> Gate = Task.FromResult(true);
        public void Start(Memory<byte> destination) => Begin(destination);

        protected override void WaitForGpu()
        {
            Gate.GetAwaiter().GetResult();
            if (fail)
            {
                throw new GpuDeviceLostException("lost", "fake");
            }
        }

        protected override void CopyOut(Span<byte> destination) => source.CopyTo(destination);

        protected override void Release() => log("release " + name);
    }

    [Fact]
    public async Task OperationsCompleteInQueueOrderAndFillTheirDestinations()
    {
        var log = new List<string>();
        using var waiter = new ReadbackWaiter();
        var gate = new TaskCompletionSource<bool>();
        var first = new FakeReadback([1, 2, 3, 4], log.Add, "first") { Gate = gate.Task };
        var second = new FakeReadback([9, 8], log.Add, "second");
        var a = new byte[4];
        var b = new byte[2];
        first.Start(a);
        second.Start(b);
        waiter.Enqueue(first);
        waiter.Enqueue(second);

        // The second operation is behind the first one; nothing completes until the first is released.
        var t1 = first.AsValueTask().AsTask();
        var t2 = second.AsValueTask().AsTask();
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.False(t1.IsCompleted || t2.IsCompleted);
        gate.SetResult(true);
        await Task.WhenAll(t1, t2);
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, a);
        Assert.Equal(new byte[] { 9, 8 }, b);
        Assert.Equal(["release first", "release second"], log);
    }

    [Fact]
    public async Task AFailureReachesTheAwaiterAndStillReleasesTheMemory()
    {
        var log = new List<string>();
        using var waiter = new ReadbackWaiter();
        var op = new FakeReadback([1], log.Add, "lost", fail: true);
        op.Start(new byte[1]);
        waiter.Enqueue(op);
        var e = await Assert.ThrowsAsync<GpuDeviceLostException>(async () => await op.AsValueTask());
        Assert.Equal("fake", e.Operation);
        Assert.Equal(["release lost"], log);
    }

    [Fact]
    public async Task TypedMemoryExposesTheElementsAsBytes()
    {
        var values = new uint[3];
        using var owner = new TypedMemory<uint>(values);
        var bytes = owner.Memory;
        Assert.Equal(12, bytes.Length);
        var op = new FakeReadback([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0], _ => { }, "typed");
        using var waiter = new ReadbackWaiter();
        op.Start(bytes);
        waiter.Enqueue(op);
        await op.AsValueTask();
        Assert.Equal(new uint[] { 1, 2, 3 }, values);
    }

    [Fact]
    public void ErrorsCarryWhatFailedAndWhatWasExpected()
    {
        var unsupported = new GpuUnsupportedException("no f16", "Float16", "shader float16 arithmetic", "none");
        Assert.Equal("Float16", unsupported.Capability);
        Assert.Equal("shader float16 arithmetic", unsupported.Expected);
        Assert.Equal("none", unsupported.Actual);
        Assert.Null(new GpuUnsupportedException("plain").Capability);
        Assert.Equal(1024ul, new GpuOutOfMemoryException("oom", 1024).RequestedBytes);
        Assert.Equal("vkQueueSubmit2", new GpuDeviceLostException("lost", "vkQueueSubmit2").Operation);
    }

    [Fact]
    public void CapabilityDescriptionNamesTheMatrixShapesAndLimits()
    {
        var caps = new GpuComputeCapabilities
        {
            SubgroupSize = 32,
            SubgroupMinSize = 32,
            SubgroupMaxSize = 32,
            CooperativeMatrices = [new GpuCooperativeMatrixShape(16, 16, 16, GpuScalarType.F16, GpuScalarType.F16, GpuScalarType.F32, GpuScalarType.F32)],
            MaxWorkgroupInvocations = 1024,
        };
        Assert.True(caps.CooperativeMatrix);
        var text = caps.Describe();
        Assert.Contains("16x16x16 F16*F16+F32->F32", text, StringComparison.Ordinal);
        Assert.Contains("invocations 1024", text, StringComparison.Ordinal);
        Assert.False(GpuComputeCapabilities.None.CooperativeMatrix);
    }
}
