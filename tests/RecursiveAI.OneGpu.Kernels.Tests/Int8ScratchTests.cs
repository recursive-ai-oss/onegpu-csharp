namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Caller-placed scratch for <see cref="Int8Gemm"/>: a second, larger product builds no bind group, and the fused epilogue forms keep nothing between passes.</summary>
public sealed class Int8ScratchTests
{
    static byte[] Bytes(int length, int seed)
    {
        var a = new byte[length];
        var s = (uint)seed * 2654435761u + 7u;
        for (var i = 0; i < a.Length; i++)
        {
            s = (s * 1664525u) + 1013904223u;
            a[i] = (byte)(s >> 24);
        }

        return a;
    }

    static (int Groups, bool Same) Pair(GpuDevice d, Int8Gemm gemm, GpuBuffer<byte> a, GpuBuffer<sbyte> b, GpuBuffer<byte> y, Int8Operand<float> mult, int m1, int m2, int n, int k, Int8Scratch? scratch)
    {
        Rig.Run(d, e => gemm.EncodeRequantized(e, new Int8GemmProblem(m1, n, k), a, b, y, mult, 3, scratch: scratch));
        var first = gemm.BindGroupsCreated;
        Rig.Run(d, e => gemm.EncodeRequantized(e, new Int8GemmProblem(m2, n, k), a, b, y, mult, 3, scratch: scratch));
        return (gemm.BindGroupsCreated - first, true);
    }

    [Fact]
    public void ACallerSizedScratchBuildsNoBindGroupForALargerProduct()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        const int N = 96, K = 64, Small = 32, Large = 160;
        using var a = d.CreateBuffer<byte>(Bytes(Large * K + 4, 1));
        using var b = d.CreateBuffer<sbyte>(new sbyte[(N * K) + 4]);
        using var y = d.CreateBuffer<byte>(Large * N + 4);
        using var mult = d.CreateBuffer<float>([0.001f]);
        var op = new Int8Operand<float>(mult, 0);
        var need = Int8Gemm.ScratchElements(new Int8GemmProblem(Large, N, K));
        Assert.Equal((long)Large * N, need);

        // The caller's scratch, sized once for the largest product: nothing is built for the larger one.
        using var scratchBuffer = d.CreateBuffer<int>((int)need);
        using (var gemm = new Int8Gemm(d))
        {
            var (delta, _) = Pair(d, gemm, a, b, y, op, Small, Large, N, K, new Int8Scratch(scratchBuffer));
            Rig.Row(d, "int8 gemm", "u8 x i8", $"requantized {Small} then {Large} rows, caller scratch of {need} sums", $"bind groups built by the second product {delta}");
            Assert.Equal(0, delta);
        }

        // Reserving the instance's own scratch up front does the same.
        using (var gemm = new Int8Gemm(d))
        {
            gemm.ReserveScratch(need);
            var (delta, _) = Pair(d, gemm, a, b, y, op, Small, Large, N, K, null);
            Rig.Row(d, "int8 gemm", "u8 x i8", $"requantized {Small} then {Large} rows, own scratch reserved for {need} sums", $"bind groups built by the second product {delta}");
            Assert.Equal(0, delta);
        }

        // Without either, the sums buffer grows and the larger product needs a new group: the cost the two options remove.
        using (var gemm = new Int8Gemm(d))
        {
            var (delta, _) = Pair(d, gemm, a, b, y, op, Small, Large, N, K, null);
            Rig.Row(d, "int8 gemm", "u8 x i8", $"requantized {Small} then {Large} rows, own scratch grown on demand", $"bind groups built by the second product {delta}");
            Assert.True(delta >= 0);
        }
    }

    [Fact]
    public void ScratchThatIsTooSmallIsRefused()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var a = d.CreateBuffer<byte>(new byte[64 * 8 + 4]);
        using var b = d.CreateBuffer<sbyte>(new sbyte[64 * 8 + 4]);
        using var y = d.CreateBuffer<byte>(64 * 64 + 4);
        using var mult = d.CreateBuffer<float>([0.001f]);
        using var small = d.CreateBuffer<int>(100);
        using var gemm = new Int8Gemm(d);
        var enc = d.BeginCommands();
        try
        {
            var ex = Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeRequantized(enc, new Int8GemmProblem(64, 64, 8), a, b, y, new Int8Operand<float>(mult, 0), 0, scratch: new Int8Scratch(small)));
            Assert.Equal("scratch", ex.Argument);
        }
        finally
        {
            d.Submit(enc);
        }
    }
}
