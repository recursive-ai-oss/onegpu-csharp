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

        // With a bound of 64 the cache is flushed on the 65th set and every pass builds groups again.
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
        using var c = d.CreateBuffer<float>(1024);
        using var ic = d.CreateBuffer<int>(1024);
        using var longs = d.CreateBuffer<long>(1024);

        var families = new (string Name, Action<GpuCommandEncoder> Record, Func<int> Built, Action Release)[]
        {
            ("elementwise", e => { ew.EncodeBinary(e, GpuBinaryOp.Add, a, 0, [64, 16], a, 0, [16], b, 0); ew.EncodeCast(e, ia, 0, longs, 0, 512); ew.EncodeWhere(e, flags, 0, [1024], a, 0, [1024], b, 0, [1024], c, 0); }, () => ew.BindGroupsCreated, ew.ReleaseBindings),
            ("reduce", e => red.EncodeReduce(e, GpuReduceOp.Sum, a, 0, [64, 16], [1], b, 0), () => red.BindGroupsCreated, red.ReleaseBindings),
            ("softmax", e => sm.Encode(e, a, 0, [64, 16], 1, b, 0), () => sm.BindGroupsCreated, sm.ReleaseBindings),
            ("norm", e => norm.Encode(e, false, a, 0, 64, 16, null, 0, null, 0, b, 0), () => norm.BindGroupsCreated, norm.ReleaseBindings),
            ("transpose", e => tr.Encode(e, a, 0, [64, 16], [1, 0], b, 0), () => tr.BindGroupsCreated, tr.ReleaseBindings),
            ("integer", e => { ik.EncodeBinary(e, GpuIntegerOp.Add, ia, 0, [1024], ib, 0, [1024], ic, 0); ik.EncodeRange(e, longs, 0, 100, 0L, 1L); }, () => ik.BindGroupsCreated, ik.ReleaseBindings),
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

    [Fact]
    public async Task AttentionInt8QuantizeGatherScatterLayoutAndDequantizeDropTheirBindGroupsOnRelease()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var attention = new AttentionKernels(d, maxCachedBindGroups: 8);
        using var int8 = new Int8Gemm(d, maxCachedBindGroups: 8);
        using var dq = new DynamicQuantizeKernels(d, 8);
        using var gather = new GatherKernels(d, 8);
        using var scatter = new ScatterKernels(d, 8);
        using var layout = new LayoutKernels(d, 8);
        using var dequantize = new DequantizeKernels(d, 8);
        using var gemm = new Gemm(d, GemmElementType.F32, maxCachedBindGroups: 8);
        using var f = d.CreateBuffer<float>(4096);
        using var f2 = d.CreateBuffer<float>(4096);
        using var f3 = d.CreateBuffer<float>(4096);
        using var f4 = d.CreateBuffer<float>(4096);
        using var bytes = d.CreateBuffer<byte>(4096);
        using var sbytes = d.CreateBuffer<sbyte>(4096);
        using var ints = d.CreateBuffer<int>(4096);
        using var ints2 = d.CreateBuffer<int>(4096);
        using var zero = d.CreateBuffer<byte>(8);

        var families = new (string Name, Action<GpuCommandEncoder> Record, Func<int> Built, Action Release)[]
        {
            ("attention", e => attention.Encode(e, new AttentionProblem(1, 2, 16, 16, 32), f, f2, f3, f4), () => attention.BindGroupsCreated, attention.ReleaseBindings),
            ("int8 gemm", e => int8.EncodeInt32(e, new Int8GemmProblem(16, 16, 16), bytes, sbytes, ints), () => int8.BindGroupsCreated, int8.ReleaseBindings),
            ("dynamic quantize", e => dq.Encode(e, f, 0, 1000, bytes, 0, f2, 0, zero, 0), () => dq.BindGroupsCreated, dq.ReleaseBindings),
            ("gather", e => gather.EncodeGather(e, f, 0, [64, 8], 0, ints, 0, 20, f2, 0), () => gather.BindGroupsCreated, gather.ReleaseBindings),
            ("scatter", e => scatter.EncodeScatterNd(e, f, 0, [64, 8], ints, 0, [4, 1], f2, 0, f3, 0), () => scatter.BindGroupsCreated, scatter.ReleaseBindings),
            ("layout", e => { layout.EncodeExpand(e, f, 0, [1, 16], [8, 16], f2, 0); layout.EncodeConcat<float>(e, [new(f, 0, 2), new(f2, 0, 2)], [4, 16], 0, f3, 0); }, () => layout.BindGroupsCreated, layout.ReleaseBindings),
            ("dequantize", e => dequantize.Encode<byte, float>(e, bytes, 0, [16, 16], 0, false, f, 0, null, 0, f2, 0), () => dequantize.BindGroupsCreated, dequantize.ReleaseBindings),
            ("gemm with epilogue", e => gemm.Encode(e, new GemmProblem(16, 16, 16), f, f2, f3, new GemmEpilogue(f4, 0, GemmActivation.GeluErf)), () => gemm.BindGroupsCreated, gemm.ReleaseBindings),
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

        Rig.Row(d, "binding cache", "newer families", "bind groups built after pass 1 / pass 2 / pass 3 following ReleaseBindings: " + string.Join(", ", report), "ok");
        Assert.Throws<ArgumentOutOfRangeException>(() => new AttentionKernels(d, maxCachedBindGroups: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Int8Gemm(d, null, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DynamicQuantizeKernels(d, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GatherKernels(d, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ScatterKernels(d, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayoutKernels(d, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DequantizeKernels(d, 0));
    }

    // One command buffer records more distinct buffer sets than the cache bound, so the cache evicts (and disposes) bind groups that commands
    // still being recorded refer to. A browser must keep each alive until its command buffer is submitted; on the desktop backends the
    // driver does. Every result is checked, so an evicted group that was destroyed early shows as a wrong value or a validation message.
    [Fact]
    public async Task EvictingBindGroupsInsideOneCommandBufferIsSafe()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        const int Sets = 24;
        using var ew = new ElementwiseKernels(d, 2);
        using var gemm = new Gemm(d, GemmElementType.F32, maxCachedBindGroups: 2);
        using var gather = new GatherKernels(d, 2);
        var a = new GpuBuffer<float>[Sets];
        var o = new GpuBuffer<float>[Sets];
        var g = new GpuBuffer<float>[Sets];
        var m = new GpuBuffer<float>[Sets];
        using var idx = d.CreateBuffer<int>([2, 0, 1]);
        using var b = d.CreateBuffer<float>(Enumerable.Repeat(1f, 4).ToArray());
        using var w = d.CreateBuffer<float>([1f, 2f, 3f, 4f]);
        try
        {
            for (var i = 0; i < Sets; i++)
            {
                a[i] = d.CreateBuffer<float>(Enumerable.Repeat((float)i, 4).ToArray());
                o[i] = d.CreateBuffer<float>(4);
                g[i] = d.CreateBuffer<float>(3);
                m[i] = d.CreateBuffer<float>(4);
            }

            for (var round = 0; round < 2; round++)
            {
                var enc = d.BeginCommands();
                for (var i = 0; i < Sets; i++)
                {
                    ew.EncodeBinary(enc, GpuBinaryOp.Add, a[i], 0, [4], b, 0, [4], o[i], 0);
                    gemm.Encode(enc, new GemmProblem(2, 2, 2), a[i], w, m[i]);
                    gather.EncodeGather(enc, w, 0, [4], 0, idx, 0, 3, g[i], 0);
                }

                if (round == 1)
                {
                    // Dropping every group while the encoder still refers to them.
                    ew.ReleaseBindings();
                    gemm.ReleaseBindings();
                    gather.ReleaseBindings();
                }

                d.Submit(enc);
                var bad = 0;
                var buf = new float[4];
                var gat = new float[3];
                for (var i = 0; i < Sets; i++)
                {
                    await o[i].ReadAsync(buf);
                    bad += buf.All(v => v == i + 1f) ? 0 : 1;
                    await g[i].ReadAsync(gat);
                    bad += gat.SequenceEqual([3f, 1f, 2f]) ? 0 : 1;
                    await m[i].ReadAsync(buf);
                    bad += buf.SequenceEqual([i * 4f, i * 6f, i * 4f, i * 6f]) ? 0 : 1;
                }

                Assert.Equal(0, bad);
            }

            Rig.Row(d, "binding cache", "bound 2", $"{Sets} distinct sets x 3 families in one command buffer, with and without ReleaseBindings before submit", "all results correct");
        }
        finally
        {
            d.WaitIdle();
            for (var i = 0; i < Sets; i++)
            {
                a[i]?.Dispose();
                o[i]?.Dispose();
                g[i]?.Dispose();
                m[i]?.Dispose();
            }
        }
    }
}
