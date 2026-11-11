namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// The bias and activation a Gemm applies in the same dispatch, against the same product followed by the element-wise kernels' add and
/// activation. Both go through the same device, so the arithmetic is the same operation sequence and the bits are compared.
/// </summary>
public sealed class GemmEpilogueTests
{
    static readonly GemmActivation[] Activations = [GemmActivation.None, GemmActivation.Relu, GemmActivation.GeluErf, GemmActivation.GeluTanh];

    static GpuUnaryOp Op(GemmActivation a) => a switch
    {
        GemmActivation.Relu => GpuUnaryOp.Relu,
        GemmActivation.GeluErf => GpuUnaryOp.GeluErf,
        _ => GpuUnaryOp.GeluTanh,
    };

    // Returns (mismatching bits, max |fused - unfused|) for one problem.
    static async Task<(int Bad, double MaxDiff, string Variant)> Case(GpuDevice d, Gemm gemm, ElementwiseKernels ew, GemmProblem p, GemmActivation act, bool withBias, GemmBiasAxis axis, int biasOffset, int seed)
    {
        var lenA = Math.Max(p.OffsetA + ((p.Batch - 1) * p.StrideA) + (p.M * p.K), 1);
        var lenB = Math.Max(p.OffsetB + ((p.Batch - 1) * p.StrideB) + (p.K * p.N), 1);
        var lenC = p.OffsetC + ((p.Batch - 1) * p.StrideC) + (p.M * p.N);
        var a = Rig.Random(lenA, seed, -1f, 1f);
        var b = Rig.Random(lenB, seed + 1, -1f, 1f);
        var biasLength = axis == GemmBiasAxis.Row ? p.M : p.N;
        var bias = Rig.Random(biasOffset + biasLength, seed + 2, -3f, 3f);
        var c0 = Rig.Random(lenC, seed + 3, -2f, 2f);
        var fused = new float[lenC];
        var plain = new float[lenC];
        using var bb = d.CreateBuffer<float>(bias);
        using var cf = d.CreateBuffer<float>(c0);
        using var cp = d.CreateBuffer<float>(c0);
        var epilogue = new GemmEpilogue(withBias ? bb : null, biasOffset, act, axis);
        switch (gemm.ElementType)
        {
            case GemmElementType.F32:
                {
                    using var ab = d.CreateBuffer<float>(a);
                    using var wb = d.CreateBuffer<float>(b);
                    Rig.Run(d, e => { gemm.Encode(e, p, ab, wb, cf, epilogue); gemm.Encode(e, p, ab, wb, cp); });
                    break;
                }

            case GemmElementType.F16:
                {
                    using var ab = d.CreateBuffer<Half>(Rig.ToHalf(a));
                    using var wb = d.CreateBuffer<Half>(Rig.ToHalf(b));
                    Rig.Run(d, e => { gemm.Encode(e, p, ab, wb, cf, epilogue); gemm.Encode(e, p, ab, wb, cp); });
                    break;
                }

            default:
                {
                    using var ab = d.CreateBuffer<ushort>(a.Select(GemmReference.ToBFloat16).ToArray());
                    using var wb = d.CreateBuffer<ushort>(b.Select(GemmReference.ToBFloat16).ToArray());
                    Rig.Run(d, e => { gemm.EncodeBFloat16(e, p, ab, wb, cf, epilogue); gemm.EncodeBFloat16(e, p, ab, wb, cp); });
                    break;
                }
        }

        // The separate passes, product by product, over the rows each product wrote. Each pass writes its own buffer: WebGPU refuses one
        // buffer bound for reading and writing in the same dispatch.
        using var added = d.CreateBuffer<float>(c0);
        using var activated = d.CreateBuffer<float>(c0);
        for (var batch = 0; batch < p.Batch; batch++)
        {
            var at = p.OffsetC + (batch * p.StrideC);
            var count = p.M * p.N;
            var source = cp;
            if (withBias)
            {
                Rig.Run(d, e => ew.EncodeBinary(e, GpuBinaryOp.Add, cp, at, [p.M, p.N], bb, biasOffset, axis == GemmBiasAxis.Row ? [p.M, 1] : [p.N], added, at));
                source = added;
            }

            if (act != GemmActivation.None)
            {
                Rig.Run(d, e => ew.EncodeUnary(e, Op(act), source, at, activated, at, count));
                source = activated;
            }

            if (!ReferenceEquals(source, cp))
            {
                Rig.Run(d, e => e.CopyBuffer(source.Buffer, (ulong)at * 4, cp.Buffer, (ulong)at * 4, (ulong)count * 4));
            }
        }

        await cf.ReadAsync(fused);
        await cp.ReadAsync(plain);
        var bad = Rig.BitMismatches(fused, plain);
        double maxDiff = 0;
        for (var i = 0; i < lenC; i++)
        {
            if (float.IsFinite(fused[i]) && float.IsFinite(plain[i]))
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(fused[i] - plain[i]));
            }
        }

        gemm.ReleaseBindings();
        ew.ReleaseBindings();
        return (bad, maxDiff, gemm.Variant);
    }

    static IEnumerable<GemmProblem> Problems()
    {
        yield return new GemmProblem(1, 1, 1);
        yield return new GemmProblem(5, 7, 3);
        yield return new GemmProblem(64, 64, 16);
        yield return new GemmProblem(65, 129, 17);
        yield return new GemmProblem(130, 200, 70) { Alpha = 0.5f, Beta = 0.5f };
        yield return new GemmProblem(33, 40, 0) { Beta = 1f };
        yield return new GemmProblem(48, 96, 40) { Batch = 3, StrideA = 48 * 40, StrideB = 0, StrideC = 48 * 96, OffsetC = 5, TransposeB = true };
        // BERT feed-forward up-projection on a batch of 8 sequences of 128 tokens, and the same at batch 1.
        yield return new GemmProblem(128, 3072, 768);
        yield return new GemmProblem(1024, 3072, 768);
    }

    static async Task Run(GemmElementType type, GpuKernelMode mode, bool hardware)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var gemm = new Gemm(d, type, mode, new GemmOptions { AllowMatrixHardware = hardware });
        using var ew = new ElementwiseKernels(d);
        long bad = 0, cases = 0;
        double worst = 0;
        var seed = 11;
        foreach (var p in Problems())
        {
            if (Rig.Browser && p.Flops > 4e9)
            {
                continue;
            }

            foreach (var act in Activations)
            {
                foreach (var (withBias, axis) in new[] { (true, GemmBiasAxis.Column), (true, GemmBiasAxis.Row), (false, GemmBiasAxis.Column) })
                {
                    if (!withBias && act == GemmActivation.None)
                    {
                        continue;
                    }

                    var (b, diff, _) = await Case(d, gemm, ew, p, act, withBias, axis, withBias ? 3 : 0, seed++);
                    bad += b;
                    worst = Math.Max(worst, diff);
                    cases++;
                }
            }
        }

        Rig.Row(d, "gemm epilogue", type.ToString(), $"{mode} variant {gemm.Variant}: {cases} cases (column and row bias, activations) over {Problems().Count()} problems vs product then elementwise add and activation", $"mismatching elements {bad}, max abs difference {Rig.Sci(worst)}");
        Assert.Equal(0, bad);
    }

    [Fact] public Task F32Fast() => Run(GemmElementType.F32, GpuKernelMode.Fast, true);

    [Fact] public Task F32Portable() => Run(GemmElementType.F32, GpuKernelMode.Fast, false);

    [Fact] public Task F32Deterministic() => Run(GemmElementType.F32, GpuKernelMode.Deterministic, true);

    [Fact] public Task F16Fast() => Run(GemmElementType.F16, GpuKernelMode.Fast, true);

    [Fact] public Task F16Portable() => Run(GemmElementType.F16, GpuKernelMode.Fast, false);

    [Fact] public Task F16Deterministic() => Run(GemmElementType.F16, GpuKernelMode.Deterministic, true);

    [Fact] public Task BF16Fast() => Run(GemmElementType.BF16, GpuKernelMode.Fast, true);

    [Fact] public Task BF16Portable() => Run(GemmElementType.BF16, GpuKernelMode.Fast, false);

    [Fact] public Task BF16Deterministic() => Run(GemmElementType.BF16, GpuKernelMode.Deterministic, true);

    // The deterministic variant with each epilogue, twice, hashes compared.
    [Fact]
    public async Task DeterministicEpilogueRepeatsBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var report = new List<string>();
        foreach (var type in new[] { GemmElementType.F32, GemmElementType.F16, GemmElementType.BF16 })
        {
            using var gemm = new Gemm(d, type, GpuKernelMode.Deterministic);
            foreach (var act in Activations)
            {
                foreach (var axisUnderTest in new[] { GemmBiasAxis.Column, GemmBiasAxis.Row })
                {
                    var p = new GemmProblem(100, 192, 300);
                    var hashes = new string[2];
                    for (var run = 0; run < 2; run++)
                    {
                        var a = Rig.Random(p.M * p.K, 5, -1f, 1f);
                        var b = Rig.Random(p.K * p.N, 6, -1f, 1f);
                        var bias = Rig.Random(p.N, 7, -2f, 2f);
                        using var bb = d.CreateBuffer<float>(bias);
                        using var c = d.CreateBuffer<float>(p.M * p.N);
                        var epilogue = new GemmEpilogue(bb, 0, act, axisUnderTest);
                        if (type == GemmElementType.F32)
                        {
                            using var ab = d.CreateBuffer<float>(a);
                            using var wb = d.CreateBuffer<float>(b);
                            Rig.Run(d, e => gemm.Encode(e, p, ab, wb, c, epilogue));
                        }
                        else if (type == GemmElementType.F16)
                        {
                            using var ab = d.CreateBuffer<Half>(Rig.ToHalf(a));
                            using var wb = d.CreateBuffer<Half>(Rig.ToHalf(b));
                            Rig.Run(d, e => gemm.Encode(e, p, ab, wb, c, epilogue));
                        }
                        else
                        {
                            using var ab = d.CreateBuffer<ushort>(a.Select(GemmReference.ToBFloat16).ToArray());
                            using var wb = d.CreateBuffer<ushort>(b.Select(GemmReference.ToBFloat16).ToArray());
                            Rig.Run(d, e => gemm.EncodeBFloat16(e, p, ab, wb, c, epilogue));
                        }

                        var got = new float[p.M * p.N];
                        await c.ReadAsync(got);
                        hashes[run] = Rig.Hash<float>(got);
                        gemm.ReleaseBindings();
                    }

                    Assert.Equal(hashes[0], hashes[1]);
                    report.Add($"{type}/{act}/{axisUnderTest} {hashes[0]}");
                }
            }
        }

        Rig.Row(d, "gemm epilogue", "f32/f16/bf16", "deterministic variant, column and row bias + each activation, two runs", "hashes equal: " + string.Join(", ", report));
    }

    [Fact]
    public async Task BadEpilogueArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var gemm = new Gemm(d, GemmElementType.F32);
        using var a = d.CreateBuffer<float>(16);
        using var b = d.CreateBuffer<float>(16);
        using var c = d.CreateBuffer<float>(16);
        using var shortBias = d.CreateBuffer<float>(3);
        var enc = d.BeginCommands();
        var p = new GemmProblem(4, 4, 4);
        var e1 = Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, p, a, b, c, new GemmEpilogue(shortBias)));
        Assert.Contains("bias", e1.Argument, StringComparison.Ordinal);
        Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, p, a, b, c, new GemmEpilogue(shortBias, 0, (GemmActivation)9)));
        Assert.Throws<GpuKernelShapeException>(() => gemm.Encode(enc, p, a, b, c, new GemmEpilogue(c, 14)));
        d.Submit(enc);
        await Task.Yield();
    }

    // Indicative timing of a feed-forward up-projection with bias and GELU: one fused dispatch against the product followed by the element-wise add and GELU.
    // GPU time is the sum of the passes' timestamp durations per iteration (median); wall time is submit to idle per iteration. Runs only with ONEGPU_GEMM_BENCH=1.
    [Fact]
    public void FusedAgainstSeparatePassesTiming()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_GEMM_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_GEMM_BENCH=1 to time the fused epilogue");
        }

        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var ew = new ElementwiseKernels(d);
        foreach (var type in new[] { GemmElementType.F32, GemmElementType.F16, GemmElementType.BF16 })
        {
            using var gemm = new Gemm(d, type);
            foreach (var m in new[] { 1024, 128 })
            {
                const int N = 3072, K = 768;
                var p = new GemmProblem(m, N, K);
                using var c = d.CreateBuffer<float>(m * N);
                using var bias = d.CreateBuffer<float>(N);
                using var a32 = type == GemmElementType.F32 ? d.CreateBuffer<float>(m * K) : null;
                using var b32 = type == GemmElementType.F32 ? d.CreateBuffer<float>(K * N) : null;
                using var a16 = type == GemmElementType.F16 ? d.CreateBuffer<Half>(m * K) : null;
                using var b16 = type == GemmElementType.F16 ? d.CreateBuffer<Half>(K * N) : null;
                using var aB = type == GemmElementType.BF16 ? d.CreateBuffer<ushort>(m * K) : null;
                using var bB = type == GemmElementType.BF16 ? d.CreateBuffer<ushort>(K * N) : null;
                var fusedEpilogue = new GemmEpilogue(bias, 0, GemmActivation.GeluErf);

                void Product(GpuCommandEncoder e, GemmEpilogue epilogue)
                {
                    if (type == GemmElementType.F32)
                    {
                        gemm.Encode(e, p, a32!, b32!, c, epilogue);
                    }
                    else if (type == GemmElementType.F16)
                    {
                        gemm.Encode(e, p, a16!, b16!, c, epilogue);
                    }
                    else
                    {
                        gemm.EncodeBFloat16(e, p, aB!, bB!, c, epilogue);
                    }
                }

                var (fusedGpu, fusedWall) = Measure(d, e => Product(e, fusedEpilogue), 1);
                var (plainGpu, plainWall) = Measure(d, e => Product(e, default), 1);
                using var added = d.CreateBuffer<float>(m * N);
                using var activated = d.CreateBuffer<float>(m * N);
                var (sepGpu, sepWall) = Measure(d, e =>
                {
                    Product(e, default);
                    ew.EncodeBinary(e, GpuBinaryOp.Add, c, 0, [m, N], bias, 0, [N], added, 0);
                    ew.EncodeUnary(e, GpuUnaryOp.GeluErf, added, 0, activated, 0, m * N);
                }, 3);
                Rig.Row(d, "gemm epilogue timing (indicative)", type.ToString(), $"{gemm.Variant}, M={m} N={N} K={K}, bias + GeLU", $"fused {fusedGpu / 1000:F1} us gpu {fusedWall / 1000:F1} us wall; separate passes {sepGpu / 1000:F1} us gpu {sepWall / 1000:F1} us wall; product only {plainGpu / 1000:F1} us gpu {plainWall / 1000:F1} us wall; gpu ratio separate/fused {sepGpu / fusedGpu:F2}");
            }
        }
    }

    static (double Gpu, double Wall) Measure(GpuDevice d, Action<GpuCommandEncoder> record, int passes)
    {
        const int Warm = 3, Iterations = 30;
        var timestamps = d.Capabilities.Timestamps;
        using var set = timestamps ? d.CreateTimestampSet((uint)(2 * passes * Iterations)) : null;
        for (var i = 0; i < Warm; i++)
        {
            Rig.Run(d, record);
        }

        d.WaitIdle();
        var enc = d.BeginCommands();
        if (set is not null)
        {
            enc.BeginPassTiming(new PassTiming(set, 0, (uint)(passes * Iterations)));
        }

        for (var i = 0; i < Iterations; i++)
        {
            record(enc);
        }

        if (set is not null)
        {
            Assert.Equal((uint)(passes * Iterations), enc.EndPassTiming());
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        d.Submit(enc);
        d.WaitIdle();
        var wall = sw.Elapsed.TotalMilliseconds * 1e6 / Iterations;
        if (set is null)
        {
            return (wall, wall);
        }

        var t = new ulong[2 * passes * Iterations];
        Assert.True(d.TryReadTimestamps(set, 0, t));
        var sums = new double[Iterations];
        for (var i = 0; i < Iterations; i++)
        {
            for (var q = 0; q < passes; q++)
            {
                var at = 2 * (i * passes + q);
                sums[i] += t[at + 1] - t[at];
            }
        }

        Array.Sort(sums);
        return (sums[Iterations / 2], wall);
    }

    // A product cut into row pieces (a small submission budget) must give the same row bias per row as the single dispatch.
    [Fact]
    public async Task RowBiasFollowsRowsWhenAProductIsSplit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var whole = new Gemm(d, GemmElementType.F32);
        using var split = new Gemm(d, GemmElementType.F32, GpuKernelMode.Fast, new GemmOptions { SubmissionWorkBudget = 3_000_000 });
        var p = new GemmProblem(300, 70, 100);
        var a = Rig.Random(p.M * p.K, 21, -1f, 1f);
        var b = Rig.Random(p.K * p.N, 22, -1f, 1f);
        var bias = Rig.Random(p.M + 2, 23, -2f, 2f);
        using var ab = d.CreateBuffer<float>(a);
        using var wb = d.CreateBuffer<float>(b);
        using var bb = d.CreateBuffer<float>(bias);
        using var c1 = d.CreateBuffer<float>(p.M * p.N);
        using var c2 = d.CreateBuffer<float>(p.M * p.N);
        var epilogue = new GemmEpilogue(bb, 2, GemmActivation.Relu, GemmBiasAxis.Row);
        Rig.Run(d, e => whole.Encode(e, p, ab, wb, c1, epilogue));
        Rig.Run(d, e => split.Encode(e, p, ab, wb, c2, epilogue));
        var g1 = new float[p.M * p.N];
        var g2 = new float[p.M * p.N];
        await c1.ReadAsync(g1);
        await c2.ReadAsync(g2);
        Assert.Equal(0, Rig.BitMismatches(g1, g2));
        Rig.Row(d, "gemm epilogue", "F32", "row bias with a product split into row pieces (budget 3e6 operations) against one dispatch", "mismatching elements 0");
    }
}
