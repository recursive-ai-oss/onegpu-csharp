using System.Diagnostics;
using static RecursiveAI.OneGpu.Kernels.Tests.AttentionReference;

namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class AttentionTests
{
    static bool Large => Environment.GetEnvironmentVariable("ONEGPU_ATTN_LARGE") == "1";

    // The reference set's shapes (MiniLM-L6: 12 heads of 32; BERT-base and Whisper-small: 12 heads of 64), then the edges.
    static IEnumerable<Case> ModelCases()
    {
        foreach (var b in new[] { 1, 8 })
        {
            foreach (var s in new[] { 1, 7, 64, 128, 384, 512 })
            {
                yield return new Case($"minilm B{b} S{s}", b, 12, 12, s, s, 32, Mask: 3);
            }

            foreach (var s in new[] { 128, 512 })
            {
                yield return new Case($"bert B{b} S{s}", b, 12, 12, s, s, 64, Mask: 3);
            }

            yield return new Case($"whisper encoder B{b} S1500", b, 12, 12, 1500, 1500, 64);
            foreach (var cache in new[] { 1, 2, 33, 100, 447, 448 })
            {
                yield return new Case($"whisper decoder self B{b} cache{cache}", b, 12, 12, 1, cache, 64, Causal: true, Lens: true, Layouts: 2);
            }

            yield return new Case($"whisper decoder prefill B{b}", b, 12, 12, 448, 448, 64, Causal: true, Layouts: 1);
            yield return new Case($"whisper cross 1 query B{b}", b, 12, 12, 1, 1500, 64);
            yield return new Case($"whisper cross 448 queries B{b}", b, 12, 12, 448, 1500, 64, Layouts: 1);
        }
    }

    // The software WebGPU adapter times out on the largest problems, so it gets the problems up to 12 heads of 256 by 256.
    static bool Software { get; set; }

    static IEnumerable<Case> SelectedModelCases() =>
        ModelCases().Where(c => Large || (long)c.B * c.H * c.Sq * c.Sk <= (Software ? 12L * 256 * 256 : 12L * 1500 * 1500));

    static IEnumerable<Case> EdgeCases()
    {
        yield return new Case("S=1", 1, 1, 1, 1, 1, 64);
        yield return new Case("S=1 d=32", 2, 3, 3, 1, 1, 32, Mask: 1);
        foreach (var (sq, sk) in new[] { (63, 31), (64, 32), (65, 33), (129, 127), (130, 257), (200, 300), (1, 255), (1, 256), (2, 257), (3, 1000), (1, 4100), (5, 4100) })
        {
            yield return new Case($"tile edges {sq}x{sk} d64", 2, 4, 4, sq, sk, 64, Mask: 1);
            yield return new Case($"tile edges {sq}x{sk} d32 causal", 1, 5, 5, sq, sk, 32, Causal: true, Lens: true, Layouts: 2);
        }

        yield return new Case("-inf bias", 2, 4, 4, 70, 90, 64, Mask: 1);
        yield return new Case("bias per batch, heads and rows", 2, 3, 3, 40, 50, 32, Mask: 2);
        yield return new Case("fully masked rows -> NaN", 2, 3, 3, 40, 50, 64, Mask: 4);
        yield return new Case("fully masked rows -> zero", 2, 3, 3, 40, 50, 64, Mask: 4, Policy: GpuAttentionMaskedRows.Zero);
        yield return new Case("fully masked rows split -> NaN", 1, 2, 2, 3, 1000, 64, Mask: 4);
        yield return new Case("fully masked rows split -> zero", 1, 2, 2, 3, 1000, 64, Mask: 4, Policy: GpuAttentionMaskedRows.Zero);
        yield return new Case("key lengths with zero", 4, 2, 2, 9, 60, 64, Lens: true, Layouts: 1);
        yield return new Case("key lengths with zero, zero policy", 4, 2, 2, 9, 60, 64, Lens: true, Policy: GpuAttentionMaskedRows.Zero);
        yield return new Case("no keys", 2, 2, 2, 5, 0, 64);
        yield return new Case("no keys, zero policy", 2, 2, 2, 5, 0, 64, Policy: GpuAttentionMaskedRows.Zero);
        yield return new Case("causal offset 3", 1, 2, 2, 20, 60, 64, Causal: true, CausalOff: 3);
        yield return new Case("causal with cache (default offset)", 2, 2, 2, 17, 90, 32, Causal: true);
        yield return new Case("grouped query 8:2", 2, 8, 2, 33, 70, 64, Mask: 3, Layouts: 2);
        yield return new Case("multi query 12:1", 1, 12, 1, 1, 500, 64, Causal: true);
        foreach (var d in new[] { 2, 4, 20, 40, 48, 96, 160, 256 })
        {
            yield return new Case($"head dim {d}", 2, 2, 2, 37, 45, d, Mask: 1, Layouts: d % 8 == 0 ? 1 : 0);
        }

        yield return new Case("head dim 128 split", 1, 2, 2, 2, 600, 128, Causal: true);
        yield return new Case("head dim 256 split", 1, 2, 2, 2, 600, 256);
    }

    sealed class Prepared
    {
        public float[] Q = [], K = [], V = [], Mask = [];
        public int[]? Lens;
        public float Scale;
        public int[] Rows = [];
        public double[] Want = [];
    }

    static Prepared Prepare(Case c, bool half)
    {
        float[] R(float[] x) => half ? x.Select(RoundHalf).ToArray() : x;
        var p = new Prepared
        {
            Q = R(Rig.Random(c.B * c.H * c.Sq * c.D, c.Sq + 11, -2f, 2f)),
            K = R(Rig.Random(c.B * c.Hkv * c.Sk * c.D, c.Sk + 23, -2f, 2f)),
            V = R(Rig.Random(c.B * c.Hkv * c.Sk * c.D, c.Sk + 37, -1f, 1f)),
            Mask = MakeMask(c),
            Lens = c.Lens ? MakeLens(c) : null,
            Scale = 1f / MathF.Sqrt(c.D),
        };
        (p.Rows, p.Want) = Compute(c, p.Scale, p.Q, p.K, p.V, p.Mask, p.Lens);
        return p;
    }

    sealed record Outcome(double Scaled, double Rel, string Hash, int Stray);

    static Layout LayoutOf(Case c, char tensor) => c.Layouts switch
    {
        1 => Layout.PositionsFirst,
        2 => tensor is 'q' or 'v' ? Layout.PositionsFirst : Layout.HeadsFirst,
        _ => Layout.HeadsFirst,
    };

    static async Task<Outcome> Run(GpuDevice d, AttentionKernels k, Case c, Prepared p, bool half, GpuKernelMode mode) =>
        half ? await Run<Half>(d, k, c, p, mode, x => (Half)x, h => (float)h) : await Run<float>(d, k, c, p, mode, x => x, x => x);

    static async Task<Outcome> Run<T>(GpuDevice d, AttentionKernels k, Case c, Prepared p, GpuKernelMode mode, Func<float, T> to, Func<T, float> from)
        where T : unmanaged
    {
        const int Off = 2;
        var (qd, ql, _) = Pack(p.Q, c.B, c.H, c.Sq, c.D, LayoutOf(c, 'q'), Off, to, 0);
        var (kd, kl, _) = Pack(p.K, c.B, c.Hkv, c.Sk, c.D, LayoutOf(c, 'k'), Off, to, 0);
        var (vd, vl, _) = Pack(p.V, c.B, c.Hkv, c.Sk, c.D, LayoutOf(c, 'v'), Off, to, 0);
        // Packing a tensor of ones gives the output buffer its size and the map of elements the kernel may write.
        var (od, ol, used) = Pack(Enumerable.Repeat(1f, c.B * c.H * c.Sq * c.D).ToArray(), c.B, c.H, c.Sq, c.D, LayoutOf(c, 'o'), Off, to, 0);
        Array.Clear(od);

        using var bq = d.CreateBuffer<T>(qd);
        using var bk = d.CreateBuffer<T>(kd);
        using var bv = d.CreateBuffer<T>(vd);
        using var bo = d.CreateBuffer<T>(od);
        using var bm = p.Mask.Length > 0 ? d.CreateBuffer<float>(p.Mask) : null;
        using var bl = p.Lens is not null ? d.CreateBuffer<int>(p.Lens) : null;
        var prob = new AttentionProblem(c.B, c.H, c.Sq, c.Sk, c.D)
        {
            KvHeads = c.Hkv,
            Causal = c.Causal,
            CausalOffset = c.CausalOff,
            MaskedRows = c.Policy,
            Q = ql,
            K = kl,
            V = vl,
            O = ol,
            Scale = p.Scale,
        };
        if (bm is not null)
        {
            prob.Mask = GpuAttentionLayout.ForMask(MaskShape(c), c.B, c.H, c.Sq, c.Sk, 0);
        }

        Rig.Run(d, e => k.Encode(e, prob, bq, bk, bv, bo, bm, bl, mode));
        var got = await Rig.DownloadAsync(bo);
        var stray = 0;
        var zero = default(T);
        for (var i = 0; i < got.Length; i++)
        {
            if (!used[i] && !EqualityComparer<T>.Default.Equals(got[i], zero))
            {
                stray++;
            }
        }

        var gotRows = new double[p.Rows.Length * c.D];
        for (var ri = 0; ri < p.Rows.Length; ri++)
        {
            var r = p.Rows[ri];
            var i = r % c.Sq;
            var h = r / c.Sq % c.H;
            var b = r / (c.Sq * c.H);
            for (var x = 0; x < c.D; x++)
            {
                gotRows[(ri * c.D) + x] = from(got[ol.Offset + (b * ol.BatchStride) + (h * ol.HeadStride) + (i * ol.RowStride) + x]);
            }
        }

        var (scaled, rel) = Rig.Errors(gotRows, p.Want);
        return new Outcome(scaled, rel, Rig.Hash<T>(got), stray);
    }

    static bool Applies(Case c, bool half) => !half || (c.D % 2 == 0);

    [Fact]
    public async Task ModelShapesMatchTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        Software = d.Adapter.Name.Contains("swiftshader", StringComparison.OrdinalIgnoreCase);
        using var kernels = new Kernels(d);
        Rig.Row(d, "attention", "-", "variant chosen for Fast and Deterministic", $"{kernels.Auto.Variant}; forced rows: {kernels.Rows.Variant}; forced pairs: {kernels.Pairs?.Variant ?? "not supported on this device"}");
        await Sweep(d, kernels, SelectedModelCases().ToList(), "model shapes");
    }

    [Fact]
    public async Task EdgeShapesMatchTheCpuReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var kernels = new Kernels(d);
        await Sweep(d, kernels, EdgeCases().ToList(), "edge shapes");
    }

    /// <summary>The instances every sweep runs: the device's choice, and each variant forced (pairs only where the device has shuffles).</summary>
    sealed class Kernels : IDisposable
    {
        public Kernels(GpuDevice d)
        {
            Auto = new AttentionKernels(d);
            Rows = new AttentionKernels(d, variant: GpuAttentionVariant.Rows);
            try
            {
                Pairs = new AttentionKernels(d, variant: GpuAttentionVariant.Pairs);
            }
            catch (GpuUnsupportedException)
            {
            }
        }

        public AttentionKernels Auto { get; }

        public AttentionKernels Rows { get; }

        public AttentionKernels? Pairs { get; }

        public IEnumerable<(string Name, AttentionKernels K, GpuKernelMode Mode)> All()
        {
            yield return ("auto fast", Auto, GpuKernelMode.Fast);
            yield return ("auto deterministic", Auto, GpuKernelMode.Deterministic);
            yield return ("rows deterministic", Rows, GpuKernelMode.Deterministic);
            if (Pairs is not null)
            {
                yield return ("pairs deterministic", Pairs, GpuKernelMode.Deterministic);
            }
        }

        public void Dispose()
        {
            Auto.Dispose();
            Rows.Dispose();
            Pairs?.Dispose();
        }
    }

    static async Task Sweep(GpuDevice d, Kernels kernels, List<Case> cases, string what)
    {
        foreach (var half in new[] { false, true })
        {
            var worst = new Dictionary<string, (double Scaled, double Rel, string At)>();
            var runs = 0;
            var same = 0;
            var used = cases.Where(c => Applies(c, half)).ToList();
            foreach (var c in used)
            {
                var prep = Prepare(c, half);
                var limit = half ? 2e-3 : 1e-5;
                string? fastHash = null;
                foreach (var (name, k, mode) in kernels.All())
                {
                    var o = await Run(d, k, c, prep, half, mode);
                    Assert.True(o.Stray == 0, $"{c.Name} {(half ? "f16" : "f32")} {name}: {o.Stray} elements outside the output tensor were written");
                    Assert.True(o.Scaled <= limit, $"{c.Name} {(half ? "f16" : "f32")} {name}: scaled error {o.Scaled}");
                    if (!worst.TryGetValue(name, out var w))
                    {
                        w = (0, 0, "-");
                    }

                    worst[name] = (Math.Max(w.Scaled, o.Scaled), Math.Max(w.Rel, o.Rel), o.Scaled > w.Scaled ? c.Name : w.At);
                    // Fast and deterministic run the same kernel, so their bits are equal.
                    if (name == "auto fast")
                    {
                        fastHash = o.Hash;
                    }
                    else if (name == "auto deterministic")
                    {
                        same += o.Hash == fastHash ? 1 : 0;
                        runs++;
                    }
                }
            }

            foreach (var (name, (scaled, rel, at)) in worst)
            {
                Rig.Row(d, "attention", half ? "f16" : "f32", $"{what}, {name}, {used.Count} cases", $"max scaled error {Rig.Sci(scaled)} (at {at}), max relative {Rig.Sci(rel)}");
            }

            Rig.Row(d, "attention", half ? "f16" : "f32", $"{what}: fast and deterministic outputs equal bit for bit", $"{same}/{runs}");
            Assert.Equal(runs, same);
        }
    }

    [Fact]
    public async Task DeterministicAttentionIsBitIdenticalAcrossRuns()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        Software = d.Adapter.Name.Contains("swiftshader", StringComparison.OrdinalIgnoreCase);
        using var kernels = new Kernels(d);
        foreach (var half in new[] { false, true })
        {
            foreach (var (name, k, _) in kernels.All().Where(x => x.Name.EndsWith("deterministic", StringComparison.Ordinal)))
            {
                var runs = 0;
                var same = 0;
                var samples = new List<string>();
                foreach (var c in SelectedModelCases().Concat(EdgeCases()).Where(c => Applies(c, half)))
                {
                    var prep = Prepare(c, half);
                    var o1 = await Run(d, k, c, prep, half, GpuKernelMode.Deterministic);
                    var o2 = await Run(d, k, c, prep, half, GpuKernelMode.Deterministic);
                    runs++;
                    same += o1.Hash == o2.Hash ? 1 : 0;
                    if (c.Name is "bert B1 S512" or "whisper decoder self B1 cache448" or "whisper encoder B1 S1500")
                    {
                        samples.Add($"{c.Name}: {o1.Hash} / {o2.Hash}");
                    }
                }

                Rig.Row(d, "attention", half ? "f16" : "f32", $"{name} variant run twice per shape (output hashes, first / second run: " + string.Join("; ", samples) + ")", $"{same}/{runs} identical");
                Assert.Equal(runs, same);
            }
        }
    }

    [Fact]
    public async Task NanScoreMakesTheRowNan()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        if (Rig.SkipInBrowser(d, "attention", "NaN score"))
        {
            return;
        }

        using var k = new AttentionKernels(d);
        foreach (var sk in new[] { 40, 1000 })
        {
            var c = new Case("nan", 1, 2, 2, 8, sk, 32, Mask: 5);
            var prep = Prepare(c, false);
            var o = await Run(d, k, c, prep, false, GpuKernelMode.Fast);
            Assert.True(o.Scaled <= 1e-5, $"a NaN mask entry: {o.Scaled}");
            Assert.True(prep.Want.Any(double.IsNaN) && prep.Want.Any(double.IsFinite), "the reference should have NaN and finite rows");
        }
    }

    [Fact]
    public async Task FusedMatchesTheUnfusedGemmSoftmaxGemmPath()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var fused = new AttentionKernels(d);
        using var gemm = new Gemm(d, GemmElementType.F32);
        using var soft = new SoftmaxKernels(d);
        using var ew = new ElementwiseKernels(d);
        foreach (var (name, b, h, sq, sk, dd, maskKind) in new[] { ("minilm", 2, 12, 100, 100, 32, 3), ("bert", 1, 12, 128, 128, 64, 3), ("cross", 1, 12, 24, 300, 64, 0), ("odd", 3, 2, 37, 51, 48, 1) })
        {
            var c = new Case(name, b, h, h, sq, sk, dd, Mask: maskKind);
            var prep = Prepare(c, false);
            var bh = b * h;
            using var q = d.CreateBuffer<float>(prep.Q);
            using var k = d.CreateBuffer<float>(prep.K);
            using var v = d.CreateBuffer<float>(prep.V);
            using var scores = d.CreateBuffer<float>(bh * sq * sk);
            using var masked = d.CreateBuffer<float>(bh * sq * sk);
            using var probs = d.CreateBuffer<float>(bh * sq * sk);
            using var oRef = d.CreateBuffer<float>(bh * sq * dd);
            using var oFused = d.CreateBuffer<float>(bh * sq * dd);
            using var mask = prep.Mask.Length > 0 ? d.CreateBuffer<float>(prep.Mask) : null;
            var s = new GemmProblem(sq, sk, dd) { Batch = bh, TransposeB = true, Alpha = prep.Scale, Lda = dd, Ldb = dd, Ldc = sk, StrideA = sq * dd, StrideB = sk * dd, StrideC = sq * sk };
            var pv = new GemmProblem(sq, dd, sk) { Batch = bh, Lda = sk, Ldb = dd, Ldc = dd, StrideA = sq * sk, StrideB = sk * dd, StrideC = sq * dd };
            var prob = new AttentionProblem(b, h, sq, sk, dd) { Scale = prep.Scale };
            if (mask is not null)
            {
                prob.Mask = GpuAttentionLayout.ForMask(MaskShape(c), b, h, sq, sk);
            }

            Rig.Run(d, e =>
            {
                gemm.Encode(e, s, q, k, scores);
                var src = scores;
                if (mask is not null)
                {
                    ew.EncodeBinary(e, GpuBinaryOp.Add, scores, 0, [b, h, sq, sk], mask, 0, MaskShape(c), masked, 0);
                    src = masked;
                }

                soft.Encode(e, src, 0, [b, h, sq, sk], 3, probs, 0);
                gemm.Encode(e, pv, probs, v, oRef);
                fused.Encode(e, prob, q, k, v, oFused, mask);
            });
            var a = await Rig.DownloadAsync(oRef);
            var f = await Rig.DownloadAsync(oFused);
            var (scaled, rel) = Rig.Errors(Rig.D(f, f.Length), Rig.D(a, a.Length));
            Rig.Row(d, "attention", "f32", $"fused vs GEMM+softmax+GEMM, {name} B{b} H{h} {sq}x{sk} d{dd}", $"max scaled difference {Rig.Sci(scaled)}, max relative {Rig.Sci(rel)}");
            Assert.True(scaled <= 2e-5, $"{name}: {scaled}");
        }
    }

    [Fact]
    public void BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new AttentionKernels(d);
        using var x = d.CreateBuffer<float>(2 * 3 * 4 * 8);
        using var o = d.CreateBuffer<float>(2 * 3 * 4 * 8);
        using var h = d.CreateBuffer<Half>(2 * 3 * 4 * 8);
        using var ho = d.CreateBuffer<Half>(2 * 3 * 4 * 8);
        using var small = d.CreateBuffer<float>(10);
        using var lens = d.CreateBuffer<int>(1);
        var enc = d.BeginCommands();
        var good = new AttentionProblem(2, 3, 4, 4, 8);
        k.Encode(enc, good, x, x, x, o);
        var e1 = Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, new AttentionProblem(2, 3, 4, 4, 300), x, x, x, o));
        Assert.Equal("HeadDim", e1.Argument);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, good, small, x, x, o));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, good, x, x, x, o, keyLengths: lens));
        var gq = good;
        gq.KvHeads = 2;
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, gq, x, x, x, o));
        var bad = good;
        bad.Q = new GpuAttentionLayout(0, -1, 0, 0);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, bad, x, x, x, o));
        var odd = new AttentionProblem(2, 3, 4, 4, 7);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, odd, h, h, h, ho));
        var nan = good;
        nan.Scale = float.NaN;
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, nan, x, x, x, o));
        Assert.Throws<GpuKernelShapeException>(() => GpuAttentionLayout.ForMask([2, 1, 3, 5], 2, 3, 4, 4));
        Assert.Throws<GpuKernelShapeException>(() => GpuAttentionLayout.ForMask([5, 1, 1, 4], 2, 3, 4, 4));
        d.Submit(enc);
    }

    // Opt-in timing of the fused kernel against the unfused path, from GPU timestamps; the numbers are indicative.
    [Fact]
    public void Throughput()
    {
        if (Environment.GetEnvironmentVariable("ONEGPU_ATTN_BENCH") != "1")
        {
            Assert.Skip("set ONEGPU_ATTN_BENCH=1 to time the large shapes");
        }

        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var timestamps = d.Capabilities.Timestamps;
        using var kernels = new Kernels(d);
        using var gemm = new Gemm(d, GemmElementType.F32);
        using var soft = new SoftmaxKernels(d);
        foreach (var (name, b, h, sq, sk, dd, causal) in new[]
        {
            ("bert B1 S512", 1, 12, 512, 512, 64, false), ("bert B8 S512", 8, 12, 512, 512, 64, false), ("minilm B8 S128", 8, 12, 128, 128, 32, false),
            ("whisper encoder S1500", 1, 12, 1500, 1500, 64, false), ("whisper decoder cache448", 1, 12, 1, 448, 64, true), ("whisper cross 1 query", 1, 12, 1, 1500, 64, false),
        })
        {
            var bh = b * h;
            using var q = d.CreateBuffer<float>(Rig.Random(bh * sq * dd, 1, -2f, 2f));
            using var k = d.CreateBuffer<float>(Rig.Random(bh * sk * dd, 2, -2f, 2f));
            using var v = d.CreateBuffer<float>(Rig.Random(bh * sk * dd, 3, -1f, 1f));
            using var scores = d.CreateBuffer<float>(bh * sq * sk);
            using var probs = d.CreateBuffer<float>(bh * sq * sk);
            using var o1 = d.CreateBuffer<float>(bh * sq * dd);
            using var o2 = d.CreateBuffer<float>(bh * sq * dd);
            var s = new GemmProblem(sq, sk, dd) { Batch = bh, TransposeB = true, Alpha = 1f / MathF.Sqrt(dd), Lda = dd, Ldb = dd, Ldc = sk, StrideA = sq * dd, StrideB = sk * dd, StrideC = sq * sk };
            var pv = new GemmProblem(sq, dd, sk) { Batch = bh, Lda = sk, Ldb = dd, Ldc = dd, StrideA = sq * sk, StrideB = sk * dd, StrideC = sq * dd };
            var prob = new AttentionProblem(b, h, sq, sk, dd) { Causal = causal };
            var unfusedGpu = Time(d, timestamps, e =>
            {
                gemm.Encode(e, s, q, k, scores);
                soft.Encode(e, scores, 0, [b, h, sq, sk], 3, probs, 0);
                gemm.Encode(e, pv, probs, v, o2);
            }, out var unfusedWall);
            var flops = 4.0 * bh * sq * sk * dd;
            foreach (var (vname, fk) in new[] { ("rows", kernels.Rows), ("pairs", kernels.Pairs) }.Where(x => x.Item2 is not null))
            {
                var fusedGpu = Time(d, timestamps, e => fk!.Encode(e, prob, q, k, v, o1), out var fusedWall);
                var auto = fk!.Variant == kernels.Auto.Variant ? " (auto)" : string.Empty;
                var line = $"THROUGHPUT\t{d.Adapter.Name}\t{name}\t{vname}{auto}: fused {fusedGpu / 1000:F1} us gpu {fusedWall / 1000:F1} us wall {flops / fusedGpu:F0} GFLOP/s\tunfused {unfusedGpu / 1000:F1} us gpu {unfusedWall / 1000:F1} us wall\tspeed-up {unfusedGpu / fusedGpu:F2}x gpu";
                Console.WriteLine(line);
#if !WEB_BACKEND
                if (Environment.GetEnvironmentVariable("ONEGPU_REPORT") is { Length: > 0 } path)
                {
                    File.AppendAllText(path, line + "\n");
                }
#endif
            }
        }
    }

    // Median GPU nanoseconds of one call (the sum of its passes), and host nanoseconds per call over a whole batch of calls.
    static double Time(GpuDevice d, bool timestamps, Action<GpuCommandEncoder> record, out double wall)
    {
        const int Warm = 3, Iterations = 20, MaxPassesPerCall = 4;
        using var set = timestamps ? d.CreateTimestampSet(2 * MaxPassesPerCall * Iterations) : null;
        for (var i = 0; i < Warm; i++)
        {
            var w = d.BeginCommands();
            record(w);
            d.Submit(w);
        }

        d.WaitIdle();
        var enc = d.BeginCommands();
        if (set is not null)
        {
            enc.BeginPassTiming(new PassTiming(set, 0, MaxPassesPerCall * Iterations));
        }

        for (var i = 0; i < Iterations; i++)
        {
            record(enc);
        }

        uint passes = 0;
        if (set is not null)
        {
            passes = enc.EndPassTiming();
        }

        var sw = Stopwatch.StartNew();
        d.Submit(enc);
        d.WaitIdle();
        wall = sw.Elapsed.TotalMilliseconds * 1e6 / Iterations;
        if (set is null)
        {
            return wall;
        }

        var t = new ulong[2 * (int)passes];
        Assert.True(d.TryReadTimestamps(set, 0, t));
        var per = (int)passes / Iterations;
        var durations = new double[Iterations];
        for (var i = 0; i < Iterations; i++)
        {
            for (var j = 0; j < per; j++)
            {
                var at = (i * per) + j;
                durations[i] += t[(2 * at) + 1] - t[2 * at];
            }
        }

        Array.Sort(durations);
        return durations[Iterations / 2];
    }
}
