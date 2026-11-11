namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Pooling and resizing against the CPU reference.</summary>
public sealed class PoolResizeTests
{
    const float Sentinel = -777f;

    static PoolProblem Pp(int batch, int c, int h, int w, int kh, int kw, int stride = 1, int dilation = 1, int pad = 0, bool ceil = false) =>
        new(batch, c, h, w, kh, kw)
        {
            StrideHeight = stride,
            StrideWidth = stride,
            DilationHeight = dilation,
            DilationWidth = dilation,
            PadTop = pad,
            PadLeft = pad,
            PadBottom = pad,
            PadRight = pad,
            CeilMode = ceil,
        };

    static IEnumerable<(string Name, PoolProblem P)> PoolShapes()
    {
        yield return ("3x3 stride 2 pad 1 (ResNet stem pool)", Pp(2, 8, 28, 28, 3, 3, stride: 2, pad: 1));
        yield return ("2x2 stride 2", Pp(1, 5, 9, 9, 2, 2, stride: 2));
        yield return ("2x2 stride 2 ceil", Pp(1, 5, 9, 9, 2, 2, stride: 2, ceil: true));
        yield return ("3x3 stride 2 ceil pad 1", Pp(2, 3, 10, 12, 3, 3, stride: 2, pad: 1, ceil: true));
        yield return ("3x3 dilation 2 pad 2", Pp(1, 4, 11, 11, 3, 3, dilation: 2, pad: 2));
        yield return ("window as big as the input", Pp(2, 6, 7, 7, 7, 7));
        yield return ("1x1 window stride 2", Pp(1, 3, 8, 8, 1, 1, stride: 2));
        yield return ("pad larger than half the window", Pp(1, 2, 6, 6, 2, 2, pad: 1) with { PadBottom = 1, PadRight = 3 });
        yield return ("asymmetric window", Pp(1, 4, 9, 14, 3, 5, pad: 0) with { StrideWidth = 2, PadLeft = 2, PadRight = 2 });
        yield return ("same upper", Pp(1, 3, 10, 10, 3, 3, stride: 2).WithAutoPad(GpuAutoPad.SameUpper));
        yield return ("same lower", Pp(1, 3, 9, 9, 4, 4, stride: 3).WithAutoPad(GpuAutoPad.SameLower));
        yield return ("1d", new PoolProblem(2, 5, 1, 30, 1, 3) { StrideWidth = 2, PadLeft = 1, PadRight = 1 });
        yield return ("odd plane count", Pp(1, 7, 5, 5, 3, 3, pad: 1));
        yield return ("ResNet stem pool 64 x 112 x 112", Pp(1, 64, 112, 112, 3, 3, stride: 2, pad: 1));
    }

    static float[] Data(int n, int seed, bool nans)
    {
        var v = Rig.Random(n, seed, -3f, 3f);
        if (nans)
        {
            for (var i = 7; i < n; i += 13)
            {
                v[i] = float.NaN;
            }
        }

        return v;
    }

    [Fact]
    public async Task PoolingMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new PoolKernels(d);
        foreach (var half in new[] { false, true })
        {
            double worstAvg = 0;
            long maxMismatch = 0;
            var cases = 0;
            foreach (var (name, p0) in PoolShapes())
            {
                foreach (var kind in new[] { GpuPoolKind.Max, GpuPoolKind.Average, GpuPoolKind.AverageIncludePad })
                {
                    foreach (var nans in Rig.Browser ? new[] { false } : new[] { false, true })
                    {
                        var p = p0;
                        if (cases % 2 == 0)
                        {
                            p.OffsetX = 3;
                            p.OffsetY = 5;
                        }

                        var nx = p.Batch * p.Channels * p.Height * p.Width;
                        var ny = p.Batch * p.Channels * p.OutHeight * p.OutWidth;
                        var x = ConvTests.Rounded(Data(nx, 31 + cases, nans && kind == GpuPoolKind.Max), half);
                        var want = ConvReference.Pool(p0, kind, x, half ? -65504.0 : -3.4028234663852886e38);
                        var pad = new double[1];
                        double[] got;
                        if (half)
                        {
                            using var bx = ConvTests.Upload<Half>(d, x, p.OffsetX, 0.25f);
                            using var by = ConvTests.Upload<Half>(d, new float[ny], p.OffsetY, Sentinel);
                            Rig.Run(d, e => k.Encode(e, p, kind, bx, by));
                            got = await ConvTests.Read(by, p.OffsetY, ny, pad);
                        }
                        else
                        {
                            using var bx = ConvTests.Upload<float>(d, x, p.OffsetX, 0.25f);
                            using var by = ConvTests.Upload<float>(d, new float[ny], p.OffsetY, Sentinel);
                            Rig.Run(d, e => k.Encode(e, p, kind, bx, by));
                            got = await ConvTests.Read(by, p.OffsetY, ny, pad);
                        }

                        k.ReleaseBindings();
                        Assert.Equal(0.0, pad[0]);
                        cases++;
                        if (kind == GpuPoolKind.Max)
                        {
                            // Max picks an element: bit-exact (NaN with NaN).
                            long bad = 0;
                            for (var i = 0; i < want.Length; i++)
                            {
                                if (!(double.IsNaN(want[i]) ? double.IsNaN(got[i]) : got[i] == want[i]))
                                {
                                    bad++;
                                }
                            }

                            maxMismatch += bad;
                            Assert.True(bad == 0, $"{name} max ({(half ? "f16" : "f32")}, nans {nans}): {bad} mismatching elements");
                        }
                        else
                        {
                            var (s, _) = Rig.Errors(got, want);
                            worstAvg = Math.Max(worstAvg, s);
                            Assert.True(s <= (half ? 2e-3 : 1e-5), $"{name} {kind} ({(half ? "f16" : "f32")}): scaled error {s:e2}");
                        }
                    }
                }
            }

            Rig.Row(d, "pool", half ? "f16" : "f32", $"{cases} cases ({PoolShapes().Count()} shapes: ceil mode, dilation, auto pad, include-pad, 1-D{(Rig.Browser ? "" : ", NaN windows")})",
                $"max pooling mismatching elements {maxMismatch}, average max scaled {Rig.Sci(worstAvg)}");
        }
    }

    [Fact]
    public async Task GlobalPoolingMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new PoolKernels(d);
        var shapes = new[] { (2, 8, 49), (1, 960, 49), (1, 3, 1), (4, 5, 255), (1, 7, 257), (3, 11, 1000), (1, 2, 12544), (5, 1, 3) };
        foreach (var half in new[] { false, true })
        {
            double worst = 0;
            long maxBad = 0;
            var cases = 0;
            foreach (var (n, c, size) in shapes)
            {
                foreach (var kind in new[] { GpuPoolKind.Max, GpuPoolKind.Average })
                {
                    foreach (var off in new[] { 0, 3 })
                    {
                        var planes = n * c;
                        var x = ConvTests.Rounded(Data(planes * size, 5 + cases, !Rig.Browser && kind == GpuPoolKind.Max), half);
                        var want = ConvReference.GlobalPool(kind, planes, size, x, half ? -65504.0 : -3.4028234663852886e38);
                        var pad = new double[1];
                        double[] got;
                        if (half)
                        {
                            using var bx = ConvTests.Upload<Half>(d, x, off, 0.25f);
                            using var by = ConvTests.Upload<Half>(d, new float[planes], off, Sentinel);
                            Rig.Run(d, e => k.EncodeGlobal(e, kind, planes, size, bx, off, by, off));
                            got = await ConvTests.Read(by, off, planes, pad);
                        }
                        else
                        {
                            using var bx = ConvTests.Upload<float>(d, x, off, 0.25f);
                            using var by = ConvTests.Upload<float>(d, new float[planes], off, Sentinel);
                            Rig.Run(d, e => k.EncodeGlobal(e, kind, planes, size, bx, off, by, off));
                            got = await ConvTests.Read(by, off, planes, pad);
                        }

                        k.ReleaseBindings();
                        Assert.Equal(0.0, pad[0]);
                        cases++;
                        if (kind == GpuPoolKind.Max)
                        {
                            long bad = 0;
                            for (var i = 0; i < want.Length; i++)
                            {
                                if (got[i] != want[i])
                                {
                                    bad++;
                                }
                            }

                            maxBad += bad;
                            Assert.Equal(0, bad);
                        }
                        else
                        {
                            var (s, _) = Rig.Errors(got, want);
                            worst = Math.Max(worst, s);
                            Assert.True(s <= (half ? 2e-3 : 1e-5), $"global average ({n}x{c}x{size}, {(half ? "f16" : "f32")}): scaled error {s:e2}");
                        }
                    }
                }
            }

            Rig.Row(d, "global-pool", half ? "f16" : "f32", $"{cases} cases ({shapes.Length} shapes, odd output offsets)", $"max mismatching elements {maxBad}, average max scaled {Rig.Sci(worst)}");
        }
    }

    [Fact]
    public async Task GlobalPoolingIsBitIdenticalRunToRun()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new PoolKernels(d);
        var x = Rig.Random(4 * 64 * 3137, 9, -1f, 1f);
        using var bx = d.CreateBuffer<float>(x);
        using var by = d.CreateBuffer<float>(4 * 64);
        string? first = null;
        for (var run = 0; run < 3; run++)
        {
            Rig.Run(d, e => k.EncodeGlobal(e, GpuPoolKind.Average, 4 * 64, 3137, bx, 0, by, 0));
            var h = Rig.Hash<float>(await Rig.DownloadAsync(by));
            first ??= h;
            Assert.Equal(first, h);
        }

        Rig.Row(d, "global-pool", "f32", "three runs of 256 planes of 3137", $"hash {first} each time");
    }

    [Fact]
    public void PoolingBadShapesAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new PoolKernels(d);
        using var x = d.CreateBuffer<float>(1000);
        using var y = d.CreateBuffer<float>(1000);
        var enc = d.BeginCommands();
        var ok = Pp(1, 4, 8, 8, 2, 2, stride: 2);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { StrideHeight = 0 }, GpuPoolKind.Max, x, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { KernelWidth = 20 }, GpuPoolKind.Max, x, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { PadLeft = -1 }, GpuPoolKind.Max, x, y));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, ok with { Channels = 100 }, GpuPoolKind.Max, x, y));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGlobal(enc, GpuPoolKind.AverageIncludePad, 4, 4, x, 0, y, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGlobal(enc, GpuPoolKind.Max, 4, 4000, x, 0, y, 0));
        d.Submit(enc);
    }

    // ---- resize ------------------------------------------------------------------------------------------------------------------

    static ResizeProblem R(int planes, int h, int w, int oh, int ow, GpuResizeMode mode, GpuResizeCoordinate coord, GpuResizeNearest nearest = GpuResizeNearest.RoundPreferFloor, float sy = 0f, float sx = 0f) =>
        new(planes, h, w, oh, ow, mode) { Coordinate = coord, Nearest = nearest, ScaleHeight = sy, ScaleWidth = sx };

    static IEnumerable<(string Name, ResizeProblem P)> ResizeShapes()
    {
        yield return ("nearest x2 asymmetric floor (torch interpolate)", R(6, 7, 9, 14, 18, GpuResizeMode.Nearest, GpuResizeCoordinate.Asymmetric, GpuResizeNearest.Floor, 2f, 2f));
        yield return ("nearest x2 half_pixel round_prefer_floor", R(4, 5, 5, 10, 10, GpuResizeMode.Nearest, GpuResizeCoordinate.HalfPixel, GpuResizeNearest.RoundPreferFloor, 2f, 2f));
        yield return ("nearest x3 half_pixel round_prefer_ceil", R(2, 4, 6, 12, 18, GpuResizeMode.Nearest, GpuResizeCoordinate.HalfPixel, GpuResizeNearest.RoundPreferCeil));
        yield return ("nearest odd sizes ceil", R(3, 7, 11, 10, 17, GpuResizeMode.Nearest, GpuResizeCoordinate.Asymmetric, GpuResizeNearest.Ceil));
        yield return ("nearest shrink", R(2, 16, 16, 5, 7, GpuResizeMode.Nearest, GpuResizeCoordinate.HalfPixel));
        yield return ("nearest align_corners", R(1, 6, 6, 11, 11, GpuResizeMode.Nearest, GpuResizeCoordinate.AlignCorners, GpuResizeNearest.RoundPreferCeil));
        yield return ("nearest to 1x1", R(5, 8, 8, 1, 1, GpuResizeMode.Nearest, GpuResizeCoordinate.PytorchHalfPixel));
        yield return ("linear x2 half_pixel", R(3, 6, 8, 12, 16, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel, sy: 2f, sx: 2f));
        yield return ("linear x2 align_corners", R(2, 5, 7, 10, 14, GpuResizeMode.Linear, GpuResizeCoordinate.AlignCorners));
        yield return ("linear pytorch_half_pixel odd", R(2, 9, 13, 17, 25, GpuResizeMode.Linear, GpuResizeCoordinate.PytorchHalfPixel));
        yield return ("linear asymmetric", R(2, 5, 5, 8, 9, GpuResizeMode.Linear, GpuResizeCoordinate.Asymmetric));
        yield return ("linear half_pixel_symmetric", R(1, 6, 6, 9, 9, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixelSymmetric));
        yield return ("linear shrink", R(2, 17, 15, 6, 5, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel));
        yield return ("linear one axis only", R(3, 8, 8, 8, 20, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel));
        yield return ("linear from 1x1", R(4, 1, 1, 5, 5, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel));
        yield return ("1d linear", R(6, 1, 20, 1, 37, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel));
        yield return ("identity", R(2, 6, 6, 6, 6, GpuResizeMode.Linear, GpuResizeCoordinate.HalfPixel));
        yield return ("DBNet-sized nearest x2 (56 x 64 x 64)", R(56, 64, 64, 128, 128, GpuResizeMode.Nearest, GpuResizeCoordinate.Asymmetric, GpuResizeNearest.Floor, 2f, 2f));
    }

    [Fact]
    public async Task ResizeMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ResizeKernels(d);
        foreach (var half in new[] { false, true })
        {
            long nearestBad = 0;
            double worstLinear = 0;
            var cases = 0;
            foreach (var (name, p) in ResizeShapes())
            {
                var offsetX = cases % 2 == 0 ? 3 : 0;
                var offsetY = cases % 2 == 0 ? 5 : 0;
                var nx = p.Planes * p.Height * p.Width;
                var ny = p.Planes * p.OutHeight * p.OutWidth;
                var x = ConvTests.Rounded(Rig.Random(nx, 77 + cases, -2f, 2f), half);
                var want = ConvReference.Resize(p, x);
                using var plan = k.CreatePlan(p);
                var pad = new double[1];
                double[] got;
                if (half)
                {
                    using var bx = ConvTests.Upload<Half>(d, x, offsetX, 0.25f);
                    using var by = ConvTests.Upload<Half>(d, new float[ny], offsetY, Sentinel);
                    Rig.Run(d, e => k.Encode(e, plan, bx, offsetX, by, offsetY));
                    got = await ConvTests.Read(by, offsetY, ny, pad);
                }
                else
                {
                    using var bx = ConvTests.Upload<float>(d, x, offsetX, 0.25f);
                    using var by = ConvTests.Upload<float>(d, new float[ny], offsetY, Sentinel);
                    Rig.Run(d, e => k.Encode(e, plan, bx, offsetX, by, offsetY));
                    got = await ConvTests.Read(by, offsetY, ny, pad);
                }

                k.ReleaseBindings();
                Assert.Equal(0.0, pad[0]);
                cases++;
                if (p.Mode == GpuResizeMode.Nearest)
                {
                    long bad = 0;
                    for (var i = 0; i < want.Length; i++)
                    {
                        // A nearest resize outputs the element itself, so it equals the reference exactly.
                        if (got[i] != want[i])
                        {
                            bad++;
                        }
                    }

                    nearestBad += bad;
                    Assert.True(bad == 0, $"{name} ({(half ? "f16" : "f32")}): {bad} mismatching elements");
                }
                else
                {
                    var (s, _) = Rig.Errors(got, want);
                    worstLinear = Math.Max(worstLinear, s);
                    Assert.True(s <= (half ? 2e-3 : 1e-5), $"{name} ({(half ? "f16" : "f32")}): scaled error {s:e2}");
                }
            }

            Rig.Row(d, "resize", half ? "f16" : "f32", $"{cases} shapes (nearest rounding modes, linear, every coordinate mode, 1-D, identity)", $"nearest mismatching elements {nearestBad}, linear max scaled {Rig.Sci(worstLinear)}");
        }
    }

    [Fact]
    public void ResizeBadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ResizeKernels(d);
        Assert.Throws<GpuKernelShapeException>(() => k.CreatePlan(new ResizeProblem(1, 0, 4, 4, 4, GpuResizeMode.Nearest)));
        Assert.Throws<GpuKernelShapeException>(() => k.CreatePlan(new ResizeProblem(1, 4, 4, 4, 4, GpuResizeMode.Nearest) { ScaleHeight = -1f }));
        Assert.Throws<GpuKernelShapeException>(() => k.CreatePlan(new ResizeProblem(1, 4, 4, 4, 4, GpuResizeMode.Nearest) { ScaleWidth = float.NaN }));
        Assert.Throws<GpuKernelShapeException>(() => k.CreatePlan(new ResizeProblem(1, 4, 4, 4, 4, (GpuResizeMode)5)));
        using var plan = k.CreatePlan(new ResizeProblem(1, 4, 4, 8, 8, GpuResizeMode.Nearest));
        using var x = d.CreateBuffer<float>(16);
        using var y = d.CreateBuffer<float>(60);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, plan, x, 0, y, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, plan, x, 1, y, 0));
        d.Submit(enc);
    }
}
