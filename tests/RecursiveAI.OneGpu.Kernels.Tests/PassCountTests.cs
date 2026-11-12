namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// The encoder counts the compute passes and dispatches it records, without timestamps. On Metal consecutive passes share one native
/// compute encoder, so the count has to follow the passes the caller began, not the native encoders.
/// </summary>
public sealed class PassCountTests
{
    [Fact]
    public async Task EncoderCountsPassesAndDispatchesItRecorded()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        const int N = 4096;
        using var x = d.CreateBuffer<float>(Enumerable.Range(0, N).Select(i => (float)i).ToArray());
        using var y = d.CreateBuffer<float>(N);
        using var copy = d.CreateBuffer<float>(N);

        var e = d.BeginCommands();
        Assert.Equal(0u, e.ComputePassCount);
        Assert.Equal(0u, e.DispatchCount);

        const int Rounds = 7;
        for (var i = 0; i < Rounds; i++)
        {
            k.EncodeUnary(e, GpuUnaryOp.Neg, x, 0, y, 0, N);
        }

        // Copies are not compute passes.
        e.CopyBuffer(y.Buffer, 0, copy.Buffer, 0, N * sizeof(float));
        var passes = e.ComputePassCount;
        var dispatches = e.DispatchCount;
        d.Submit(e);
        Rig.Row(d, "pass count", "f32", $"{Rounds} unary kernels and a copy in one encoder", $"{passes} passes, {dispatches} dispatches");
        Assert.Equal(Rounds, (int)passes);
        Assert.Equal(Rounds, (int)dispatches);

        // A recycled encoder starts again from zero.
        var again = d.BeginCommands();
        Assert.Equal(0u, again.ComputePassCount);
        Assert.Equal(0u, again.DispatchCount);
        k.EncodeUnary(again, GpuUnaryOp.Neg, x, 0, y, 0, N);
        Assert.Equal(1u, again.ComputePassCount);
        Assert.Equal(1u, again.DispatchCount);
        d.Submit(again);

        var got = await Rig.DownloadAsync(copy);
        Assert.Equal(-1f, got[1]);
    }
}
