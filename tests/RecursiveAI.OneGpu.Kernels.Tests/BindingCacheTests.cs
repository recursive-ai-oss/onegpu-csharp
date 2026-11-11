namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>The bind-group cache bound of every family, and what ReleaseBindings drops.</summary>
public sealed class BindingCacheTests
{
    // A transformer's weights: 72 distinct weight matrices, each multiplied with an activation into its own output.
    const int Weights = 72;

    static (int First, int Second, int AfterRelease) RunWeights(GpuDevice d, int bound, bool print)
    {
        using var gemm = new Gemm(d, GemmElementType.F32, maxCachedBindGroups: bound);
        var problem = new GemmProblem(16, 16, 16);
        using var x0 = d.CreateBuffer<float>(256);
        using var x1 = d.CreateBuffer<float>(256);
        var w = new GpuBuffer<float>[Weights];
        var o = new GpuBuffer<float>[Weights];
        for (var i = 0; i < Weights; i++)
        {
            w[i] = d.CreateBuffer<float>(256);
            o[i] = d.CreateBuffer<float>(256);
        }

        try
        {
            void Pass()
            {
                var enc = d.BeginCommands();
                for (var i = 0; i < Weights; i++)
                {
                    gemm.Encode(enc, problem, i % 2 == 0 ? x0 : x1, w[i], o[i]);
                }

                d.Submit(enc);
                d.WaitIdle();
            }

            Pass();
            var first = gemm.BindGroupsCreated;
            Pass();
            var second = gemm.BindGroupsCreated;
            gemm.ReleaseBindings();
            Pass();
            var after = gemm.BindGroupsCreated;
            if (print)
            {
                Console.WriteLine($"gemm cache bound {bound}: bind groups built after pass 1 = {first}, pass 2 = {second}, pass 3 after ReleaseBindings = {after}");
            }

            return (first, second, after);
        }
        finally
        {
            d.WaitIdle();
            for (var i = 0; i < Weights; i++)
            {
                w[i].Dispose();
                o[i].Dispose();
            }
        }
    }

    [Fact]
    public void SeventyTwoWeightTriplesAreBuiltOnceWhenTheBoundIsLargeEnough()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var (first, second, after) = RunWeights(d, 72, true);
        Assert.Equal(Weights, first);
        Assert.Equal(Weights, second);
        Assert.Equal(2 * Weights, after);
        var (f2, s2, a2) = RunWeights(d, GpuKernelBindings.DefaultMaxCachedBindGroups, true);
        Assert.Equal((Weights, Weights, 2 * Weights), (f2, s2, a2));

        // With the old bound of 64 the cache is flushed on the 65th set and every pass builds groups again.
        var (f3, s3, _) = RunWeights(d, 64, true);
        Assert.True(s3 > f3, $"a bound below the working set should rebuild groups on the second pass (built {f3} then {s3})");
        Rig.Row(d, "binding cache", "gemm", $"{Weights} weight triples recorded twice: bound 72 builds {first} then {second}; default ({GpuKernelBindings.DefaultMaxCachedBindGroups}) {f2} then {s2}; bound 64 builds {f3} then {s3}; after ReleaseBindings {after - second} more", "ok");
        Assert.Throws<ArgumentOutOfRangeException>(() => new Gemm(d, GemmElementType.F32, maxCachedBindGroups: 0));
    }

    [Fact]
    public async Task EveryFamilyDropsItsBindGroupsOnRelease()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var ew = new ElementwiseKernels(d, 8);
        using var red = new ReduceKernels(d, maxCachedBindGroups: 8);
        using var sm = new SoftmaxKernels(d, maxCachedBindGroups: 8);
        using var norm = new NormKernels(d, maxCachedBindGroups: 8);
        using var tr = new TransposeKernels(d, 8);
        using var ik = new IntegerKernels(d, 8);
        using var a = d.CreateBuffer<float>(1024);
        using var b = d.CreateBuffer<float>(1024);
        using var ia = d.CreateBuffer<int>(1024);
        using var ib = d.CreateBuffer<int>(1024);
        using var flags = d.CreateBuffer<byte>(1024);
        using var longs = d.CreateBuffer<long>(1024);

        var families = new (string Name, Action<GpuCommandEncoder> Record, Func<int> Built, Action Release)[]
        {
            ("elementwise", e => { ew.EncodeBinary(e, GpuBinaryOp.Add, a, 0, [64, 16], a, 0, [16], b, 0); ew.EncodeCast(e, ia, 0, longs, 0, 512); ew.EncodeWhere(e, flags, 0, [1024], a, 0, [1024], b, 0, [1024], a, 0); }, () => ew.BindGroupsCreated, ew.ReleaseBindings),
            ("reduce", e => red.EncodeReduce(e, GpuReduceOp.Sum, a, 0, [64, 16], [1], b, 0), () => red.BindGroupsCreated, red.ReleaseBindings),
            ("softmax", e => sm.Encode(e, a, 0, [64, 16], 1, b, 0), () => sm.BindGroupsCreated, sm.ReleaseBindings),
            ("norm", e => norm.Encode(e, false, a, 0, 64, 16, null, 0, null, 0, b, 0), () => norm.BindGroupsCreated, norm.ReleaseBindings),
            ("transpose", e => tr.Encode(e, a, 0, [64, 16], [1, 0], b, 0), () => tr.BindGroupsCreated, tr.ReleaseBindings),
            ("integer", e => { ik.EncodeBinary(e, GpuIntegerOp.Add, ia, 0, [1024], ib, 0, [1024], ia, 0); ik.EncodeRange(e, longs, 0, 100, 0L, 1L); }, () => ik.BindGroupsCreated, ik.ReleaseBindings),
        };

        var report = new List<string>();
        foreach (var (name, record, built, release) in families)
        {
            Rig.Run(d, record);
            var first = built();
            Rig.Run(d, record);
            var second = built();
            d.WaitIdle();
            release();
            Rig.Run(d, record);
            var third = built();
            await Task.Yield();
            Assert.True(first > 0, name);
            Assert.Equal(first, second);
            Assert.Equal(2 * first, third);
            report.Add($"{name} {first}/{second}/{third}");
        }

        Rig.Row(d, "binding cache", "all families", "bind groups built after pass 1 / pass 2 / pass 3 following ReleaseBindings: " + string.Join(", ", report), "ok");
        Assert.Throws<ArgumentOutOfRangeException>(() => new ElementwiseKernels(d, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TransposeKernels(d, -1));
    }
}
