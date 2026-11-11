namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Convolution and transposed convolution against the CPU reference, on edge-case shapes and on the layer shapes of the vision models.</summary>
public sealed class ConvTests
{
    const float Sentinel = -777f;

    /// <summary>Kernel configurations every parity test runs: the default, matrix hardware off, the deterministic mode, and the column-matrix path in place of the implicit kernel.</summary>
    internal static readonly (string Name, GpuKernelMode Mode, bool Hardware, bool Implicit)[] Configurations =
    [
        ("fast", GpuKernelMode.Fast, true, true),
        ("fast-portable", GpuKernelMode.Fast, false, true),
        ("deterministic", GpuKernelMode.Deterministic, true, true),
        ("fast-im2col", GpuKernelMode.Fast, true, false),
    ];

    static bool Full => Environment.GetEnvironmentVariable("ONEGPU_CONV_FULL") == "1";

    internal static ConvKernels Make(GpuDevice d, GpuKernelMode mode, bool hardware, bool useImplicit = true) =>
        new(d, mode, new GemmOptions { AllowMatrixHardware = hardware }) { AllowImplicit = useImplicit };

    internal static ConvProblem P(int batch, int cin, int h, int w, int cout, int kh, int kw, int stride = 1, int dilation = 1, int pad = 0, int groups = 1) =>
        new(batch, cin, h, w, cout, kh, kw)
        {
            StrideHeight = stride,
            StrideWidth = stride,
            DilationHeight = dilation,
            DilationWidth = dilation,
            PadTop = pad,
            PadLeft = pad,
            PadBottom = pad,
            PadRight = pad,
            Groups = groups,
        };

    internal static GpuBuffer<T> Upload<T>(GpuDevice d, float[] v, int offset, float fill)
        where T : unmanaged
    {
        if (typeof(T) == typeof(Half))
        {
            var h = new Half[GpuKernelBuffers.PaddedHalfLength(offset + v.Length + 1)];
            Array.Fill(h, (Half)fill);
            for (var i = 0; i < v.Length; i++)
            {
                h[offset + i] = (Half)v[i];
            }

            return (GpuBuffer<T>)(object)d.CreateBuffer<Half>(h);
        }

        var f = new float[offset + v.Length + 1];
        Array.Fill(f, fill);
        v.CopyTo(f, offset);
        return (GpuBuffer<T>)(object)d.CreateBuffer<float>(f);
    }

    /// <summary>Reads [offset, offset + count) of a buffer; outside[0] receives the largest deviation from the sentinel everywhere else.</summary>
    internal static async Task<double[]> Read<T>(GpuBuffer<T> b, int offset, int count, double[] outside)
        where T : unmanaged
    {
        var r = new double[count];
        double bad = 0;
        if (typeof(T) == typeof(Half))
        {
            var h = await Rig.DownloadAsync((GpuBuffer<Half>)(object)b);
            for (var i = 0; i < h.Length; i++)
            {
                if (i >= offset && i < offset + count)
                {
                    r[i - offset] = (double)h[i];
                }
                else if (i < offset || i >= offset + count + 1)
                {
                    bad = Math.Max(bad, Math.Abs((double)h[i] - Sentinel));
                }
            }
        }
        else
        {
            var f = await Rig.DownloadAsync((GpuBuffer<float>)(object)b);
            for (var i = 0; i < f.Length; i++)
            {
                if (i >= offset && i < offset + count)
                {
                    r[i - offset] = f[i];
                }
                else if (i < offset || i >= offset + count + 1)
                {
                    bad = Math.Max(bad, Math.Abs(f[i] - Sentinel));
                }
            }
        }

        outside[0] = bad;
        return r;
    }

    internal static float[] Rounded(float[] v, bool half)
    {
        if (!half)
        {
            return v;
        }

        var r = new float[v.Length];
        for (var i = 0; i < v.Length; i++)
        {
            r[i] = (float)(Half)v[i];
        }

        return r;
    }

    internal sealed record Outcome(double Scaled, double Rel, double Outside, string Hash, ConvVariant Variant);

    /// <summary>Runs one convolution on the GPU and against the reference.</summary>
    internal static async Task<Outcome> RunConv(GpuDevice d, ConvKernels k, ConvProblem problem, bool half, bool hasBias, bool hasResidual, int seed, bool offsets, int? chunkImages = null)
    {
        var p = problem;
        if (offsets)
        {
            p.OffsetX = 3;
            p.OffsetW = 2;
            p.OffsetBias = 1;
            p.OffsetResidual = 5;
            p.OffsetY = 7;
        }

        var cinG = p.InChannels / p.Groups;
        var nx = p.Batch * p.InChannels * p.Height * p.Width;
        var nw = p.OutChannels * cinG * p.KernelHeight * p.KernelWidth;
        var ny = p.Batch * p.OutChannels * p.OutHeight * p.OutWidth;
        var x = Rounded(Rig.Random(nx, seed, -1f, 1f), half);
        var scale = 1f / MathF.Sqrt(Math.Max(1, cinG * p.KernelHeight * p.KernelWidth));
        var w = Rounded(Rig.Random(nw, seed + 1, -scale, scale), half);
        var bias = hasBias ? Rounded(Rig.Random(p.OutChannels, seed + 2, -0.5f, 0.5f), half) : null;
        var res = hasResidual ? Rounded(Rig.Random(ny, seed + 3, -1f, 1f), half) : null;
        var want = ConvReference.Conv(problem, x, w, bias, res);
        var pad = new double[1];
        var perImageColumns = Math.Max(ConvKernels.ColumnElements(problem) / p.Batch, k.Scratch(problem, half).ColumnElementsPerImage);
        if (half)
        {
            using var bx = Upload<Half>(d, x, p.OffsetX, 0.25f);
            using var bw = Upload<Half>(d, w, p.OffsetW, 0.25f);
            using var bb = bias is null ? null : Upload<Half>(d, bias, p.OffsetBias, 0.25f);
            using var br = res is null ? null : Upload<Half>(d, res, p.OffsetResidual, 0.25f);
            using var by = Upload<Half>(d, new float[ny], p.OffsetY, Sentinel);
            using var cols = chunkImages is not null && perImageColumns > 0 ? d.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength((int)(perImageColumns * chunkImages.Value) + 2)) : null;
            using var acc = chunkImages is not null ? d.CreateBuffer<float>(Math.Max(1, p.OutChannels * p.OutHeight * p.OutWidth * chunkImages.Value)) : null;
            Rig.Run(d, e => k.Encode(e, p, bx, bw, bb, br, by, cols, acc));
            var got = await Read(by, p.OffsetY, ny, pad);
            var (s, r) = Rig.Errors(got, want);
            k.ReleaseBindings();
            return new Outcome(s, r, pad[0], Rig.Hash<double>(got), ConvKernels.ChooseVariant(problem));
        }
        else
        {
            using var bx = Upload<float>(d, x, p.OffsetX, 0.25f);
            using var bw = Upload<float>(d, w, p.OffsetW, 0.25f);
            using var bb = bias is null ? null : Upload<float>(d, bias, p.OffsetBias, 0.25f);
            using var br = res is null ? null : Upload<float>(d, res, p.OffsetResidual, 0.25f);
            using var by = Upload<float>(d, new float[ny], p.OffsetY, Sentinel);
            using var cols = chunkImages is not null && perImageColumns > 0 ? d.CreateBuffer<float>((int)(perImageColumns * chunkImages.Value)) : null;
            Rig.Run(d, e => k.Encode(e, p, bx, bw, bb, br, by, cols));
            var got = await Read(by, p.OffsetY, ny, pad);
            var (s, r) = Rig.Errors(got, want);
            k.ReleaseBindings();
            return new Outcome(s, r, pad[0], Rig.Hash<double>(got), ConvKernels.ChooseVariant(problem));
        }
    }

    static readonly GpuActivation[] Activations =
    [
        default,
        new(GpuUnaryOp.Relu),
        new(GpuUnaryOp.Clip, 0f, 6f),
        new(GpuUnaryOp.HardSwish),
        new(GpuUnaryOp.HardSigmoid, 1f / 6f, 0.5f),
        new(GpuUnaryOp.Silu),
        new(GpuUnaryOp.Sigmoid),
        new(GpuUnaryOp.LeakyRelu, 0.1f),
    ];

    /// <summary>Shapes that hit tile edges, remainders, size 1, odd sizes and every window feature.</summary>
    static IEnumerable<(string Name, ConvProblem P)> EdgeShapes()
    {
        yield return ("1x1", P(2, 16, 9, 7, 24, 1, 1));
        yield return ("1x1 odd channels", P(1, 13, 5, 5, 67, 1, 1));
        yield return ("1x1 on 1x1 input", P(3, 70, 1, 1, 33, 1, 1));
        yield return ("1x1 stride 2", P(2, 20, 11, 10, 18, 1, 1, stride: 2));
        yield return ("3x3 pad 1", P(2, 8, 10, 13, 20, 3, 3, pad: 1));
        yield return ("3x3 stride 2 pad 1", P(1, 16, 15, 15, 32, 3, 3, stride: 2, pad: 1));
        yield return ("3x3 no pad", P(2, 5, 8, 8, 7, 3, 3));
        yield return ("3x3 dilation 2", P(1, 6, 12, 11, 9, 3, 3, dilation: 2, pad: 2));
        yield return ("5x5 pad 2", P(1, 3, 14, 14, 10, 5, 5, pad: 2));
        yield return ("7x7 stride 2 pad 3 stem", P(1, 3, 32, 32, 16, 7, 7, stride: 2, pad: 3));
        yield return ("1x7", P(2, 8, 9, 20, 12, 1, 7) with { PadLeft = 3, PadRight = 3 });
        yield return ("7x1", P(2, 8, 20, 9, 12, 7, 1) with { PadTop = 3, PadBottom = 3 });
        yield return ("asymmetric pad (same upper)", P(1, 4, 10, 10, 6, 4, 4, stride: 2).WithAutoPad(GpuAutoPad.SameUpper));
        yield return ("asymmetric pad (same lower)", P(1, 4, 9, 9, 6, 4, 4, stride: 3).WithAutoPad(GpuAutoPad.SameLower));
        yield return ("pad larger than kernel reach", P(1, 3, 6, 6, 4, 3, 3, pad: 3));
        yield return ("kernel as big as input", P(2, 6, 5, 5, 8, 5, 5));
        yield return ("one input channel", P(1, 1, 17, 17, 16, 3, 3, pad: 1));
        yield return ("one output channel", P(1, 32, 9, 9, 1, 3, 3, pad: 1));
        yield return ("groups 2", P(2, 8, 9, 9, 12, 3, 3, pad: 1, groups: 2));
        yield return ("groups 4 1x1", P(1, 16, 6, 6, 8, 1, 1, groups: 4));
        yield return ("depthwise 3x3", P(2, 24, 15, 15, 24, 3, 3, pad: 1, groups: 24));
        yield return ("depthwise 3x3 stride 2", P(1, 33, 14, 14, 33, 3, 3, stride: 2, pad: 1, groups: 33));
        yield return ("depthwise 5x5", P(2, 16, 12, 12, 16, 5, 5, pad: 2, groups: 16));
        yield return ("depthwise multiplier 2", P(1, 6, 9, 9, 12, 3, 3, pad: 1, groups: 6));
        yield return ("depthwise dilation 2", P(1, 8, 11, 11, 8, 3, 3, dilation: 2, pad: 2, groups: 8));
        yield return ("1d", ConvProblem.Conv1D(2, 8, 50, 16, 5, padBegin: 2, padEnd: 2));
        yield return ("1d stride 2", ConvProblem.Conv1D(1, 4, 33, 6, 3, stride: 2, padBegin: 1, padEnd: 1));
        yield return ("1d depthwise", ConvProblem.Conv1D(2, 12, 40, 12, 7, padBegin: 3, padEnd: 3, groups: 12));
        yield return ("larger", P(2, 64, 28, 28, 96, 3, 3, pad: 1));

        // Small grids and deep k: the matrix-hardware kernel cuts k into ranges (whole 16-wide steps, the last one ragged) and a second pass adds them.
        yield return ("split k 3x3", P(1, 128, 7, 7, 64, 3, 3, pad: 1));
        yield return ("split k 3x3 ragged k", P(2, 100, 7, 7, 96, 3, 3, pad: 1));
        yield return ("split k 1x1", P(1, 1024, 7, 7, 64, 1, 1));
        yield return ("split k 1x1 odd depth", P(1, 997, 5, 5, 33, 1, 1));
        yield return ("split k groups 2", P(1, 256, 7, 7, 128, 3, 3, pad: 1, groups: 2));
        yield return ("split k stride 2", P(1, 256, 14, 14, 256, 3, 3, stride: 2, pad: 1));
    }

    [Fact]
    public async Task ConvMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var shapes = EdgeShapes().ToArray();
        foreach (var half in new[] { false, true })
        {
            var tol = half ? 2e-3 : 1e-5;
            foreach (var (cfg, mode, hw, imp) in Configurations)
            {
                if (!hw && !half)
                {
                    continue; // the float32 matrix product has one portable kernel on Vulkan; forcing hardware off changes nothing
                }

                using var k = Make(d, mode, hw, imp);
                double worst = 0, worstRel = 0, outside = 0;
                var cases = 0;
                var variants = new Dictionary<ConvVariant, int>();
                for (var i = 0; i < shapes.Length; i++)
                {
                    var (name, p) = shapes[i];
                    p.Activation = Activations[i % Activations.Length];
                    var o = await RunConv(d, k, p, half, hasBias: i % 3 != 1, hasResidual: i % 4 == 2, seed: 100 + i, offsets: i % 2 == 0);
                    worst = Math.Max(worst, o.Scaled);
                    worstRel = Math.Max(worstRel, o.Rel);
                    outside = Math.Max(outside, o.Outside);
                    cases++;
                    variants[o.Variant] = variants.GetValueOrDefault(o.Variant) + 1;
                    Assert.True(o.Scaled <= tol, $"{name} ({(half ? "f16" : "f32")}, {cfg}): scaled error {o.Scaled:e2} above {tol:e0}");
                    Assert.Equal(0.0, o.Outside);
                }

                Rig.Row(d, "conv", half ? "f16" : "f32", $"{cfg}: {cases} edge shapes ({string.Join(", ", variants.Select(v => $"{v.Key} {v.Value}"))}); gemm {(half ? k.GemmVariantFloat16 : k.GemmVariantFloat32)}",
                    $"max scaled {Rig.Sci(worst)}, max rel {Rig.Sci(worstRel)}, untouched outside {outside == 0}");
            }
        }
    }

    [Fact]
    public async Task ConvIsBitIdenticalRunToRun()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = Make(d, GpuKernelMode.Deterministic, true);
        var layers = new[] { P(2, 16, 14, 14, 32, 3, 3, pad: 1), P(2, 32, 14, 14, 64, 1, 1), P(2, 24, 15, 15, 24, 3, 3, pad: 1, groups: 24), P(1, 3, 40, 40, 16, 7, 7, stride: 2, pad: 3), P(1, 8, 9, 9, 12, 3, 3, pad: 1, groups: 2), P(1, 512, 7, 7, 512, 3, 3, pad: 1), P(1, 1024, 7, 7, 128, 1, 1) };
        foreach (var half in new[] { false, true })
        {
            var differ = 0;
            var hashes = new List<string>();
            foreach (var p in layers)
            {
                var q = p with { Activation = new GpuActivation(GpuUnaryOp.Relu) };
                var a = await RunConv(d, k, q, half, true, true, 7, true);
                var b = await RunConv(d, k, q, half, true, true, 7, true);
                if (a.Hash != b.Hash)
                {
                    differ++;
                }

                hashes.Add(a.Hash[..6]);
            }

            Rig.Row(d, "conv", half ? "f16" : "f32", "deterministic: two runs of 7 layer shapes (the last two with split k)", $"runs with different hashes {differ}; hashes {string.Join(" ", hashes)}");
            Assert.Equal(0, differ);
        }
    }

    [Fact]
    public async Task ChunkingTheBatchChangesNothing()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = Make(d, GpuKernelMode.Deterministic, true);
        // The second shape has a grid small enough for k to be split: the ranges, and so the order of the sums, must not depend on how the batch is chunked.
        foreach (var p in new[] { P(5, 8, 12, 12, 16, 3, 3, pad: 1) with { Activation = new GpuActivation(GpuUnaryOp.Relu) }, P(3, 192, 7, 7, 96, 3, 3, pad: 1) with { Activation = new GpuActivation(GpuUnaryOp.Relu) } })
        {
            foreach (var half in new[] { false, true })
            {
                var whole = await RunConv(d, k, p, half, true, false, 5, false);
                foreach (var images in new[] { 1, 2 })
                {
                    var chunked = await RunConv(d, k, p, half, true, false, 5, false, chunkImages: images);
                    Assert.Equal(whole.Hash, chunked.Hash);
                }

                Rig.Row(d, "conv", half ? "f16" : "f32", $"batch of {p.Batch} ({p.InChannels}->{p.OutChannels} {p.Height}x{p.Width}) in chunks of {p.Batch}, 2 and 1 images with caller-supplied scratch", "hashes equal");
            }
        }
    }

    [Fact]
    public void BindGroupsAreReusedOnceWarm()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ConvKernels(d);
        var p = P(1, 8, 10, 10, 8, 3, 3, pad: 1) with { Activation = new GpuActivation(GpuUnaryOp.Relu) };
        using var x = d.CreateBuffer<float>(8 * 100);
        using var w = d.CreateBuffer<float>(8 * 8 * 9);
        using var b = d.CreateBuffer<float>(8);
        using var y = d.CreateBuffer<float>(8 * 100);
        using var cols = d.CreateBuffer<float>((int)ConvKernels.ColumnElements(p));
        Rig.Run(d, e => k.Encode(e, p, x, w, b, null, y, cols));
        var created = k.BindGroupsCreated;
        for (var i = 0; i < 5; i++)
        {
            Rig.Run(d, e => k.Encode(e, p, x, w, b, null, y, cols));
        }

        Assert.Equal(created, k.BindGroupsCreated);
        Rig.Row(d, "conv", "f32", "warm calls", $"bind groups after the first call {created}, after five more {k.BindGroupsCreated}");
    }

    [Fact]
    public void BadShapesAreRejectedByName()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ConvKernels(d);
        using var x = d.CreateBuffer<float>(4 * 8 * 8);
        using var w = d.CreateBuffer<float>(4 * 4 * 9);
        using var y = d.CreateBuffer<float>(4 * 8 * 8);
        var enc = d.BeginCommands();
        var ok = P(1, 4, 8, 8, 4, 3, 3, pad: 1);
        Assert.Contains("Groups", Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { Groups = 3 }, x, w, null, null, y)).Argument);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { StrideWidth = 0 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { DilationHeight = -1 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { PadTop = -1 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { KernelHeight = 11, PadTop = 0, PadBottom = 0 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { Batch = 0 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { OffsetX = 1 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { OutChannels = 8 }, x, w, null, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { Height = int.MaxValue / 2, Width = 4 }, x, w, null, null, y));
        using var tiny = d.CreateBuffer<float>(16);
        using var column = new ConvKernels(d) { AllowImplicit = false };
        Assert.Throws<GpuKernelShapeException>(() => column.Encode(enc, ok, x, w, null, null, y, tiny));
        d.Submit(enc);
    }

    // ---- the layer shapes of the vision models -------------------------------------------------------------------------------

    internal sealed record Layer(string Name, ConvProblem P, GpuActivation Act, bool Bias)
    {
        /// <summary>How many times the model runs this shape.</summary>
        public int Count { get; set; } = 1;
    }

    /// <summary>The distinct convolutions of ResNet-50 v1.5 at <paramref name="size"/> x <paramref name="size"/>.</summary>
    internal static List<Layer> ResNet50(int batch, int size)
    {
        var layers = new Dictionary<string, Layer>();
        void Add(string name, ConvProblem p, GpuActivation act, bool bias = true)
        {
            var key = $"{p.InChannels}x{p.Height}x{p.Width}->{p.OutChannels} k{p.KernelHeight} s{p.StrideHeight} p{p.PadTop}";
            if (layers.TryGetValue(key, out var seen)) { seen.Count++; } else { layers.Add(key, new Layer(name, p, act, bias)); }
        }

        var relu = new GpuActivation(GpuUnaryOp.Relu);
        Add("stem 7x7/2", P(batch, 3, size, size, 64, 7, 7, stride: 2, pad: 3), relu);
        var h = ((size + 6 - 7) / 2) + 1;
        h = ((h + 2 - 3) / 2) + 1; // after the 3x3/2 max pool
        var cin = 64;
        foreach (var (mid, cout, blocks, stride) in new[] { (64, 256, 3, 1), (128, 512, 4, 2), (256, 1024, 6, 2), (512, 2048, 3, 2) })
        {
            for (var b = 0; b < blocks; b++)
            {
                var s = b == 0 ? stride : 1;
                var ho = ((h + 2 - 3) / s) + 1;
                Add("1x1 reduce", P(batch, cin, h, h, mid, 1, 1), relu);
                Add("3x3", P(batch, mid, h, h, mid, 3, 3, stride: s, pad: 1), relu);
                Add("1x1 expand", P(batch, mid, ho, ho, cout, 1, 1), relu);
                if (b == 0)
                {
                    Add("1x1 downsample", P(batch, cin, h, h, cout, 1, 1, stride: s), default);
                }

                cin = cout;
                h = ho;
            }
        }

        return [.. layers.Values];
    }

    /// <summary>The distinct convolutions of MobileNetV3-large (Howard et al., 2019, table 1) at <paramref name="size"/> x <paramref name="size"/>.</summary>
    internal static List<Layer> MobileNetV3(int batch, int size)
    {
        var layers = new Dictionary<string, Layer>();
        void Add(string name, ConvProblem p, GpuActivation act)
        {
            var key = $"{p.InChannels}x{p.Height}x{p.Width}->{p.OutChannels} k{p.KernelHeight} s{p.StrideHeight} g{p.Groups}";
            if (layers.TryGetValue(key, out var seen)) { seen.Count++; } else { layers.Add(key, new Layer(name, p, act, true)); }
        }

        var relu = new GpuActivation(GpuUnaryOp.Relu);
        var hs = new GpuActivation(GpuUnaryOp.HardSwish);
        var hsig = new GpuActivation(GpuUnaryOp.HardSigmoid, 1f / 6f, 0.5f);
        var h = ((size + 2 - 3) / 2) + 1;
        Add("stem 3x3/2", P(batch, 3, size, size, 16, 3, 3, stride: 2, pad: 1), hs);
        var cin = 16;
        foreach (var (k, exp, cout, se, act, s) in new (int, int, int, bool, bool, int)[]
        {
            (3, 16, 16, false, false, 1), (3, 64, 24, false, false, 2), (3, 72, 24, false, false, 1), (5, 72, 40, true, false, 2), (5, 120, 40, true, false, 1),
            (5, 120, 40, true, false, 1), (3, 240, 80, false, true, 2), (3, 200, 80, false, true, 1), (3, 184, 80, false, true, 1), (3, 184, 80, false, true, 1),
            (3, 480, 112, true, true, 1), (3, 672, 112, true, true, 1), (5, 672, 160, true, true, 2), (5, 960, 160, true, true, 1), (5, 960, 160, true, true, 1),
        })
        {
            var a = act ? hs : relu;
            if (exp != cin)
            {
                Add("1x1 expand", P(batch, cin, h, h, exp, 1, 1), a);
            }

            var ho = ((h + (2 * (k / 2)) - k) / s) + 1;
            Add($"depthwise {k}x{k}/{s}", P(batch, exp, h, h, exp, k, k, stride: s, pad: k / 2, groups: exp), a);
            if (se)
            {
                var sq = ((exp / 4) + 7) / 8 * 8;
                Add("SE reduce", P(batch, exp, 1, 1, sq, 1, 1), relu);
                Add("SE expand", P(batch, sq, 1, 1, exp, 1, 1), hsig);
            }

            Add("1x1 project", P(batch, exp, ho, ho, cout, 1, 1), default);
            cin = cout;
            h = ho;
        }

        Add("1x1 last", P(batch, 160, h, h, 960, 1, 1), hs);
        Add("1x1 head", P(batch, 960, 1, 1, 1280, 1, 1), hs);
        return [.. layers.Values];
    }

    /// <summary>The convolutions of the document detector (depthwise-separable backbone, 1x1 laterals, heads) on a [batch, 1, size, size] page.</summary>
    internal static List<Layer> DbNet(int batch, int size)
    {
        var layers = new Dictionary<string, Layer>();
        void Add(string name, ConvProblem p, GpuActivation act = default)
        {
            var key = $"{p.InChannels}x{p.Height}x{p.Width}->{p.OutChannels} k{p.KernelHeight} s{p.StrideHeight} g{p.Groups}";
            if (layers.TryGetValue(key, out var seen)) { seen.Count++; } else { layers.Add(key, new Layer(name, p, act, true)); }
        }

        var relu = new GpuActivation(GpuUnaryOp.Relu);
        void Dw(int c, int o, int hh, int stride)
        {
            Add("depthwise 3x3", P(batch, c, hh, hh, c, 3, 3, stride: stride, pad: 1, groups: c), relu);
            var ho = ((hh + 2 - 3) / stride) + 1;
            Add("pointwise 1x1", P(batch, c, ho, ho, o, 1, 1), relu);
        }

        var h = ((size + 2 - 3) / 2) + 1;
        Add("stem 3x3/2", P(batch, 1, size, size, 16, 3, 3, stride: 2, pad: 1), relu);
        var s2 = h;
        var cin = 16;
        var sizes = new List<int>();
        foreach (var (w, depth) in new[] { (32, 2), (72, 3), (144, 4), (224, 3) })
        {
            Dw(cin, w, h, 2);
            h = ((h + 2 - 3) / 2) + 1;
            for (var i = 1; i < depth; i++)
            {
                Dw(w, w, h, 1);
            }

            sizes.Add(h);
            Add("lateral 1x1", P(batch, w, h, h, 56, 1, 1));
            cin = w;
        }

        Dw(56, 56, sizes[1], 1);
        Dw(56, 56, sizes[0], 1);
        Dw(56, 112, sizes[1], 1);
        Dw(112, 112, sizes[1], 1);
        Add("layout out 1x1", P(batch, 112, sizes[1], sizes[1], 8, 1, 1));
        Dw(56, 32, sizes[0], 1);
        Add("stem projection 1x1", P(batch, 16, s2, s2, 32, 1, 1));
        Dw(32, 32, s2, 1);
        Add("prob out 1x1", P(batch, 32, s2, s2, 1, 1, 1));
        return [.. layers.Values];
    }

    static async Task RunLayers(string family, List<Layer> layers)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        foreach (var half in new[] { false, true })
        {
            var tol = half ? 2e-3 : 1e-5;
            foreach (var (cfg, mode, hw, imp) in Configurations)
            {
                if (!hw && !half)
                {
                    continue;
                }

                using var k = Make(d, mode, hw, imp);
                double worst = 0, worstRel = 0;
                var count = 0;
                var perVariant = new Dictionary<ConvVariant, double>();
                var worstName = "";
                foreach (var l in layers)
                {
                    var p = l.P;
                    p.Activation = l.Act;
                    var o = await RunConv(d, k, p, half, l.Bias, hasResidual: false, seed: 900 + count, offsets: false);
                    count++;
                    if (o.Scaled > worst)
                    {
                        worst = o.Scaled;
                        worstName = $"{l.Name} {p.InChannels}->{p.OutChannels} {p.Height}x{p.Width}";
                    }

                    worstRel = Math.Max(worstRel, o.Rel);
                    perVariant[o.Variant] = Math.Max(perVariant.GetValueOrDefault(o.Variant), o.Scaled);
                    Assert.True(o.Scaled <= tol, $"{family} {l.Name} {p.InChannels}->{p.OutChannels} {p.Height}x{p.Width} b{p.Batch} ({(half ? "f16" : "f32")}, {cfg}): scaled error {o.Scaled:e2}");
                    Assert.Equal(0.0, o.Outside);
                }

                Rig.Row(d, "conv", half ? "f16" : "f32", $"{cfg}: {family} {count} distinct layers, batch {layers[0].P.Batch}, {string.Join(", ", perVariant.Select(v => $"{v.Key} max {Rig.Sci(v.Value)}"))}",
                    $"max scaled {Rig.Sci(worst)} ({worstName}), max rel {Rig.Sci(worstRel)}");
            }
        }
    }

    static int Size => Rig.Browser ? 64 : 224;

    [Fact]
    public Task ResNet50LayersBatch1() => RunLayers("ResNet-50", ResNet50(1, Size));

    [Fact]
    public Task MobileNetV3LayersBatch1() => RunLayers("MobileNetV3-large", MobileNetV3(1, Size));

    [Fact]
    public Task DbNetLayers() => RunLayers("DBNet", DbNet(1, Rig.Browser ? 128 : 512));

    [Fact]
    public async Task VisionLayersBatch8()
    {
        if (!Full || Rig.Browser)
        {
            Assert.Skip("set ONEGPU_CONV_FULL=1 to run the batch-8 layer sets (heavy on the CPU reference)");
        }

        await RunLayers("ResNet-50", ResNet50(8, 224));
        await RunLayers("MobileNetV3-large", MobileNetV3(8, 224));
    }

    // ---- transposed convolution ----------------------------------------------------------------------------------------------

    static IEnumerable<(string Name, ConvTransposeProblem P)> TransposeShapes()
    {
        yield return ("2x2 stride 2 (upsample)", new ConvTransposeProblem(2, 16, 9, 7, 8, 2, 2) { StrideHeight = 2, StrideWidth = 2 });
        yield return ("2x2 stride 2 to one channel", new ConvTransposeProblem(1, 32, 16, 16, 1, 2, 2) { StrideHeight = 2, StrideWidth = 2 });
        yield return ("3x3 stride 2 pad 1 output padding 1", new ConvTransposeProblem(1, 8, 10, 10, 6, 3, 3) { StrideHeight = 2, StrideWidth = 2, PadTop = 1, PadLeft = 1, PadBottom = 1, PadRight = 1, OutputPaddingHeight = 1, OutputPaddingWidth = 1 });
        yield return ("3x3 stride 1", new ConvTransposeProblem(2, 5, 8, 9, 7, 3, 3));
        yield return ("4x4 stride 2 pad 1", new ConvTransposeProblem(1, 12, 7, 7, 12, 4, 4) { StrideHeight = 2, StrideWidth = 2, PadTop = 1, PadLeft = 1, PadBottom = 1, PadRight = 1 });
        yield return ("3x3 dilation 2", new ConvTransposeProblem(1, 4, 6, 6, 4, 3, 3) { DilationHeight = 2, DilationWidth = 2 });
        yield return ("groups 2", new ConvTransposeProblem(2, 8, 6, 5, 6, 3, 3) { Groups = 2, StrideHeight = 2, StrideWidth = 2 });
        yield return ("depthwise stride 2", new ConvTransposeProblem(1, 8, 6, 6, 8, 2, 2) { Groups = 8, StrideHeight = 2, StrideWidth = 2 });
        yield return ("stride 3, 1x1 input", new ConvTransposeProblem(2, 9, 1, 1, 5, 3, 3) { StrideHeight = 3, StrideWidth = 3 });
        yield return ("padding beyond the output edge", new ConvTransposeProblem(1, 3, 5, 5, 4, 3, 3) { PadTop = 2, PadLeft = 2, PadBottom = 2, PadRight = 2 });
        yield return ("1d", new ConvTransposeProblem(2, 6, 1, 30, 4, 1, 4) { StrideWidth = 2, PadLeft = 1, PadRight = 1 });
        yield return ("DBNet head 64->64 2x2/2 at 64x64", new ConvTransposeProblem(1, 64, 64, 64, 64, 2, 2) { StrideHeight = 2, StrideWidth = 2 });
        yield return ("DBNet head 64->1 2x2/2 at 128x128", new ConvTransposeProblem(1, 64, 128, 128, 1, 2, 2) { StrideHeight = 2, StrideWidth = 2 });
    }

    [Fact]
    public async Task ConvTransposeMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        foreach (var half in new[] { false, true })
        {
            var tol = half ? 2e-3 : 1e-5;
            foreach (var (cfg, mode, hw, imp) in Configurations)
            {
                if (!hw && !half)
                {
                    continue;
                }

                using var k = Make(d, mode, hw, imp);
                double worst = 0, worstRel = 0;
                var cases = 0;
                foreach (var (name, p0) in TransposeShapes())
                {
                    var p = p0 with { Activation = Activations[cases % Activations.Length] };
                    if (cases % 2 == 0)
                    {
                        p.OffsetX = 3;
                        p.OffsetW = 2;
                        p.OffsetBias = 1;
                        p.OffsetY = 7;
                    }

                    var hasBias = cases % 3 != 1;
                    var coutG = p.OutChannels / p.Groups;
                    var nx = p.Batch * p.InChannels * p.Height * p.Width;
                    var nw = p.InChannels * coutG * p.KernelHeight * p.KernelWidth;
                    var ny = p.Batch * p.OutChannels * p.OutHeight * p.OutWidth;
                    var x = Rounded(Rig.Random(nx, 50 + cases, -1f, 1f), half);
                    var w = Rounded(Rig.Random(nw, 60 + cases, -0.3f, 0.3f), half);
                    var bias = hasBias ? Rounded(Rig.Random(p.OutChannels, 70 + cases, -0.5f, 0.5f), half) : null;
                    var want = ConvReference.ConvTranspose(p0 with { Activation = p.Activation }, x, w, bias);
                    var pad = new double[1];
                    double[] got;
                    if (half)
                    {
                        using var bx = Upload<Half>(d, x, p.OffsetX, 0.25f);
                        using var bw = Upload<Half>(d, w, p.OffsetW, 0.25f);
                        using var bb = bias is null ? null : Upload<Half>(d, bias, p.OffsetBias, 0.25f);
                        using var by = Upload<Half>(d, new float[ny], p.OffsetY, Sentinel);
                        Rig.Run(d, e => k.EncodeTranspose(e, p, bx, bw, bb, by));
                        got = await Read(by, p.OffsetY, ny, pad);
                    }
                    else
                    {
                        using var bx = Upload<float>(d, x, p.OffsetX, 0.25f);
                        using var bw = Upload<float>(d, w, p.OffsetW, 0.25f);
                        using var bb = bias is null ? null : Upload<float>(d, bias, p.OffsetBias, 0.25f);
                        using var by = Upload<float>(d, new float[ny], p.OffsetY, Sentinel);
                        Rig.Run(d, e => k.EncodeTranspose(e, p, bx, bw, bb, by));
                        got = await Read(by, p.OffsetY, ny, pad);
                    }

                    k.ReleaseBindings();
                    var (s, r) = Rig.Errors(got, want);
                    worst = Math.Max(worst, s);
                    worstRel = Math.Max(worstRel, r);
                    cases++;
                    Assert.True(s <= tol, $"{name} ({(half ? "f16" : "f32")}, {cfg}): scaled error {s:e2}");
                    Assert.Equal(0.0, pad[0]);
                }

                Rig.Row(d, "conv-transpose", half ? "f16" : "f32", $"{cfg}: {cases} shapes (upsampling heads, output padding, groups, dilation, 1-D)", $"max scaled {Rig.Sci(worst)}, max rel {Rig.Sci(worstRel)}");
            }
        }
    }

    [Fact]
    public void TransposeBadShapesAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ConvKernels(d);
        using var x = d.CreateBuffer<float>(1000);
        using var w = d.CreateBuffer<float>(1000);
        using var y = d.CreateBuffer<float>(1000);
        var enc = d.BeginCommands();
        var ok = new ConvTransposeProblem(1, 4, 4, 4, 4, 2, 2) { StrideHeight = 2, StrideWidth = 2 };
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, ok with { Groups = 3 }, x, w, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, ok with { StrideHeight = 0 }, x, w, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, ok with { PadTop = 20 }, x, w, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, ok with { OutputPaddingHeight = -1 }, x, w, null, y));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, ok with { Height = 1000 }, x, w, null, y));
        d.Submit(enc);
    }

    [Fact]
    public void VariantChoiceReport()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var fast = Make(d, GpuKernelMode.Fast, true);
        using var portable = Make(d, GpuKernelMode.Fast, false);
        var classes = new (string Name, ConvProblem P)[]
        {
            ("1x1", P(1, 256, 56, 56, 64, 1, 1)),
            ("1x1 stride 2", P(1, 256, 56, 56, 512, 1, 1, stride: 2)),
            ("3x3", P(1, 64, 56, 56, 64, 3, 3, pad: 1)),
            ("3x3 stride 2", P(1, 128, 56, 56, 128, 3, 3, stride: 2, pad: 1)),
            ("7x7 stem", P(1, 3, 224, 224, 64, 7, 7, stride: 2, pad: 3)),
            ("depthwise 3x3", P(1, 96, 112, 112, 96, 3, 3, pad: 1, groups: 96)),
            ("grouped 3x3", P(1, 64, 56, 56, 64, 3, 3, pad: 1, groups: 4)),
        };
        foreach (var (name, p) in classes)
        {
            Rig.Row(d, "conv", "variant", name, $"f32: {fast.DescribeVariant(p, false)}; f16: {fast.DescribeVariant(p, true)}; f16 with matrix hardware off: {portable.DescribeVariant(p, true)}");
        }

        Assert.Equal(ConvVariant.MatrixProduct, ConvKernels.ChooseVariant(classes[0].P));
        Assert.Equal(ConvVariant.Im2ColMatrixProduct, ConvKernels.ChooseVariant(classes[1].P));
        Assert.Equal(ConvVariant.Depthwise, ConvKernels.ChooseVariant(classes[5].P));
        Assert.Equal(ConvVariant.Im2ColMatrixProduct, ConvKernels.ChooseVariant(classes[6].P));
    }
}
