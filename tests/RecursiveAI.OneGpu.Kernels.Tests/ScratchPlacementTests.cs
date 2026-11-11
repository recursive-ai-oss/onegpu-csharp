namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// Scratch the caller places (a buffer and an element offset, possibly inside the buffer that also holds the operands) and Concat inputs that live in the
/// output's buffer: the results must equal the ones from separate buffers bit for bit, and overlapping ranges must be refused by name.
/// </summary>
public sealed class ScratchPlacementTests
{
    static T Cast<T>(float v)
        where T : unmanaged => typeof(T) == typeof(Half) ? (T)(object)(Half)v : (T)(object)v;

    static double ToD<T>(T v)
        where T : unmanaged => v is Half h ? (double)h : Convert.ToDouble(v);

    static bool SameBits<T>(T[] a, T[] b)
        where T : unmanaged
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (var i = 0; i < a.Length; i++)
        {
            if (!IndexRig.Same(a[i], b[i]))
            {
                return false;
            }
        }

        return true;
    }

    static void Encode<T>(GpuCommandEncoder e, ConvKernels k, ConvProblem p, GpuBuffer<T> x, GpuBuffer<T> w, GpuBuffer<T>? bias, GpuBuffer<T> y, GpuBuffer<T>? cols, int colsOffset, GpuBuffer<float>? acc, int accOffset)
        where T : unmanaged
    {
        if (typeof(T) == typeof(Half))
        {
            k.Encode(e, p, (GpuBuffer<Half>)(object)x, (GpuBuffer<Half>)(object)w, bias is null ? null : (GpuBuffer<Half>)(object)bias, null, (GpuBuffer<Half>)(object)y, cols is null ? null : (GpuBuffer<Half>)(object)cols, acc, colsOffset, accOffset);
        }
        else
        {
            k.Encode(e, p, (GpuBuffer<float>)(object)x, (GpuBuffer<float>)(object)w, bias is null ? null : (GpuBuffer<float>)(object)bias, null, (GpuBuffer<float>)(object)y, cols is null ? null : (GpuBuffer<float>)(object)cols, colsOffset);
        }
    }

    static async Task<(int Cases, int Bad)> ConvCases<T>(GpuDevice d)
        where T : unmanaged
    {
        var half = typeof(T) == typeof(Half);
        var problems = new[]
        {
            ConvTests.P(3, 8, 12, 12, 16, 3, 3, pad: 1),
            ConvTests.P(2, 16, 9, 9, 32, 1, 1),
            ConvTests.P(2, 16, 11, 11, 32, 3, 3, stride: 2, pad: 1),
            ConvTests.P(2, 8, 10, 10, 8, 3, 3, pad: 1, groups: 2),
        };
        var cases = 0;
        var bad = 0;
        foreach (var (implicitOn, coopOn) in new[] { (true, true), (true, false), (false, true) })
        {
            using var k = new ConvKernels(d) { AllowImplicit = implicitOn, AllowCoopImplicit = coopOn };
            foreach (var p0 in problems)
            {
                var p = p0 with { Activation = new GpuActivation(GpuUnaryOp.Relu) };
                var cin = p.InChannels / p.Groups;
                var nx = p.Batch * p.InChannels * p.Height * p.Width;
                var nw = p.OutChannels * cin * p.KernelHeight * p.KernelWidth;
                var ny = p.Batch * p.OutChannels * p.OutHeight * p.OutWidth;
                var xs = Rig.Random(nx, 11, -1f, 1f);
                var ws = Rig.Random(nw, 12, -0.2f, 0.2f);
                var bs = Rig.Random(p.OutChannels, 13, -0.5f, 0.5f);
                var need = k.Scratch(p, half);

                // Reference: separate buffers, the instance's own scratch.
                using var x0 = ConvTests.Upload<T>(d, xs, 0, 0f);
                using var w0 = ConvTests.Upload<T>(d, ws, 0, 0f);
                using var b0 = ConvTests.Upload<T>(d, bs, 0, 0f);
                using var y0 = ConvTests.Upload<T>(d, new float[ny], 0, 0f);
                Rig.Run(d, e => Encode(e, k, p, x0, w0, b0, y0, null, 0, null, 0));
                var wantFull = await Rig.DownloadAsync(y0);
                var want = wantFull[..ny];

                // Arena: operands and the column scratch in one buffer, the scratch at an odd element offset; accumulators in their own buffer at an offset.
                int ox = 1, ow = ox + nx + 3, ob = ow + nw + 5, oy = ob + p.OutChannels + 3, oc = oy + ny + 7;
                // A browser does not let one buffer be read and written in a dispatch, so there the operands keep their own buffers and only the scratch sits at an offset in a buffer of its own.
                var browser = d.Backend == GpuBackend.WebGpu;
                var arena = new T[GpuKernelBuffers.PaddedHalfLength(oc + (int)need.ColumnElements + 9)];
                Array.Fill(arena, Cast<T>(-7f));
                for (var i = 0; i < nx; i++) { arena[ox + i] = Cast<T>(xs[i]); }
                for (var i = 0; i < nw; i++) { arena[ow + i] = Cast<T>(ws[i]); }
                for (var i = 0; i < bs.Length; i++) { arena[ob + i] = Cast<T>(bs[i]); }
                using var a = d.CreateBuffer<T>(arena);
                using var bx = browser ? ConvTests.Upload<T>(d, xs, ox, -7f) : null;
                using var bw = browser ? ConvTests.Upload<T>(d, ws, ow, -7f) : null;
                using var bb = browser ? ConvTests.Upload<T>(d, bs, ob, -7f) : null;
                using var by = browser ? ConvTests.Upload<T>(d, new float[ny], oy, -7f) : null;
                var placed = p with { OffsetX = ox, OffsetW = ow, OffsetBias = ob, OffsetY = oy };
                using var acc = half ? d.CreateBuffer<float>((int)need.AccumulatorElements + 11) : null;
                var outBuffer = browser ? by! : a;
                Rig.Run(d, e => Encode(e, k, placed, browser ? bx! : a, browser ? bw! : a, browser ? bb! : a, outBuffer, need.ColumnElements > 0 ? a : null, oc, acc, 5));
                var got = await Rig.DownloadAsync(outBuffer);
                var gotY = new T[ny];
                Array.Copy(got, oy, gotY, 0, ny);
                cases++;
                bad += SameBits(want, gotY) ? 0 : 1;
                var diffs = 0;
                double worstDiff = 0;
                var firstBad = -1;
                for (var i = 0; i < ny; i++)
                {
                    if (!IndexRig.Same(want[i], gotY[i]))
                    {
                        diffs++;
                        firstBad = firstBad < 0 ? i : firstBad;
                        worstDiff = Math.Max(worstDiff, Math.Abs(ToD(want[i]) - ToD(gotY[i])));
                    }
                }

                Assert.True(diffs == 0, $"{diffs} of {ny} elements differ (first {firstBad}, worst {worstDiff}); output differs: {p.InChannels}->{p.OutChannels} k{p.KernelHeight} g{p.Groups} implicit {implicitOn} coop {coopOn} half {half}");

                // The operands and the memory around the output must be untouched (in a browser only the output's buffer is checked).
                var untouched = true;
                for (var i = 0; i < got.Length; i++)
                {
                    var inScratch = !browser && need.ColumnElements > 0 && i >= oc && i < oc + (int)need.ColumnElements;
                    var inY = i >= oy && i < oy + ny;
                    var original = browser ? Cast<T>(-7f) : arena[i];
                    if (!inScratch && !inY && !IndexRig.Same(got[i], original))
                    {
                        untouched = false;
                    }
                }

                bad += untouched ? 0 : 1;
                Assert.True(untouched, $"memory outside the output and the scratch changed: {p.InChannels}->{p.OutChannels} k{p.KernelHeight} g{p.Groups} implicit {implicitOn} coop {coopOn} half {half}");

                // A scratch range inside the output's range is refused.
                if (need.ColumnElements > 0)
                {
                    var enc = d.BeginCommands();
                    var ex = Assert.Throws<GpuKernelShapeException>(() => Encode(enc, k, placed, browser ? bx! : a, browser ? bw! : a, browser ? bb! : a, outBuffer, outBuffer, oy + 1, acc, 5));
                    if (!browser)
                    {
                        Assert.Contains("overlaps", ex.Message, StringComparison.Ordinal);
                    }

                    d.Submit(enc);
                }
            }
        }

        return (cases, bad);
    }

    [Fact]
    public async Task ConvScratchCanLiveInTheCallersArena()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var (c32, b32) = await ConvCases<float>(d);
        var (c16, b16) = await ConvCases<Half>(d);
        Rig.Row(d, "conv", "f32/f16", $"{c32 + c16} cases with the column scratch at an odd offset inside the operands' buffer and accumulators at an offset, portable implicit, matrix-hardware implicit and column-matrix variants", $"mismatching cases {b32 + b16}");
        Assert.Equal(0, b32 + b16);
    }

    [Fact]
    public async Task ConvTransposeScratchOffset()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ConvKernels(d);
        var p = new ConvTransposeProblem(2, 8, 6, 6, 4, 3, 3) { StrideHeight = 2, StrideWidth = 2, PadTop = 1, PadLeft = 1, PadBottom = 1, PadRight = 1 };
        var nx = p.Batch * p.InChannels * p.Height * p.Width;
        var nw = p.InChannels * p.OutChannels * 9;
        var ny = p.Batch * p.OutChannels * p.OutHeight * p.OutWidth;
        var xs = Rig.Random(nx, 21, -1f, 1f);
        var ws = Rig.Random(nw, 22, -0.2f, 0.2f);
        using var x0 = ConvTests.Upload<float>(d, xs, 0, 0f);
        using var w0 = ConvTests.Upload<float>(d, ws, 0, 0f);
        using var y0 = ConvTests.Upload<float>(d, new float[ny], 0, 0f);
        Rig.Run(d, e => k.EncodeTranspose(e, p, x0, w0, null, y0));
        var want = (await Rig.DownloadAsync(y0))[..ny];
        var need = (int)ConvKernels.Scratch(p).ColumnElements;
        using var cols = d.CreateBuffer<float>(need + 13);
        using var y1 = ConvTests.Upload<float>(d, new float[ny], 0, 0f);
        Rig.Run(d, e => k.EncodeTranspose(e, p, x0, w0, null, y1, cols, 13));
        Assert.True(SameBits(want, (await Rig.DownloadAsync(y1))[..ny]));
        var enc = d.BeginCommands();
        var perImage = (int)ConvKernels.Scratch(p).ColumnElementsPerImage;
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeTranspose(enc, p, x0, w0, null, y1, cols, 13 + need - perImage + 1));
        d.Submit(enc);
        Rig.Row(d, "conv-transpose", "f32", "column scratch at offset 13 of a larger buffer", "bit-identical to the instance's own scratch; an offset leaving too little room is refused");
    }

    static async Task<int> ConcatInArena<T>(GpuDevice d, int[] lens, int[] rest, int axis, int seed)
        where T : unmanaged
    {
        using var k = new LayoutKernels(d);
        var rank = rest.Length + 1;
        var shape = new int[rank];
        for (int i = 0, s = 0; i < rank; i++)
        {
            shape[i] = i == axis ? lens.Sum() : rest[s++];
        }

        long outer = 1, inner = 1;
        for (var i = 0; i < axis; i++) { outer *= shape[i]; }
        for (var i = axis + 1; i < rank; i++) { inner *= shape[i]; }
        var total = (int)(outer * shape[axis] * inner);

        // Reference with separate buffers.
        var datas = new T[lens.Length][];
        var separate = new List<GpuConcatInput<T>>();
        var buffers = new List<GpuBuffer<T>>();
        for (var i = 0; i < lens.Length; i++)
        {
            datas[i] = IndexRig.Fill<T>((int)(outer * lens[i] * inner), seed + i);
            buffers.Add(d.CreateBuffer<T>(datas[i]));
            separate.Add(new GpuConcatInput<T>(buffers[i], 0, lens[i]));
        }

        using var want = d.CreateBuffer<T>(new T[IndexRig.Padded<T>(total)]);
        Rig.Run(d, e => k.EncodeConcat<T>(e, separate.ToArray(), shape, axis, want, 0));
        var expected = await Rig.DownloadAsync(want);

        // One arena: the inputs at odd offsets, then the output after them.
        var offsets = new int[lens.Length];
        var at = 1;
        for (var i = 0; i < lens.Length; i++)
        {
            offsets[i] = at;
            at += datas[i].Length + 3;
        }

        var outOffset = at + 1;
        var arena = new T[IndexRig.Padded<T>(outOffset + total + 3)];
        Array.Fill(arena, IndexRig.Make<T>(-999));
        for (var i = 0; i < lens.Length; i++) { datas[i].CopyTo(arena, offsets[i]); }
        using var a = d.CreateBuffer<T>(arena);
        var inArena = new GpuConcatInput<T>[lens.Length];
        for (var i = 0; i < lens.Length; i++) { inArena[i] = new GpuConcatInput<T>(a, offsets[i], lens[i]); }
        Rig.Run(d, e => k.EncodeConcat<T>(e, inArena, shape, axis, a, outOffset));
        var got = await Rig.DownloadAsync(a);
        var bad = 0;
        for (var i = 0; i < total; i++) { bad += IndexRig.Same(got[outOffset + i], expected[i]) ? 0 : 1; }
        for (var i = 0; i < arena.Length; i++)
        {
            if ((i < outOffset || i >= outOffset + total) && !IndexRig.Same(got[i], arena[i])) { bad++; }
        }

        // An input overlapping the output range is refused.
        var enc = d.BeginCommands();
        var overlapping = new GpuConcatInput<T>[lens.Length];
        Array.Copy(inArena, overlapping, lens.Length);
        overlapping[0] = new GpuConcatInput<T>(a, outOffset + 1, lens[0]);
        var ex = Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<T>(enc, overlapping, shape, axis, a, outOffset));
        Assert.Contains("overlaps", ex.Message, StringComparison.Ordinal);
        d.Submit(enc);
        foreach (var b in buffers) { b.Dispose(); }
        return bad;
    }

    [Fact]
    public async Task ConcatInputsMayLiveInTheOutputsBuffer()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        long bad = 0, cases = 0;
        var perType = new long[3];
        (int[] Lens, int[] Others, int Axis)[] shapes =
        [
            ([2, 3], [4], 0), ([2, 3, 1], [5], 1), ([3, 3, 3, 3, 2], [2, 3], 1), ([1, 4], [3, 2], 2), ([5, 1, 2], [3], 0),
        ];
        var seed = 5000;
        foreach (var (lens, others, axis) in shapes)
        {
            // Rest dimensions are those other than the axis, in order.
            var ax = Math.Min(axis, others.Length);
            var f = await ConcatInArena<float>(d, lens, others, ax, seed++);
            var h = await ConcatInArena<Half>(d, lens, others, ax, seed++);
            var l = await ConcatInArena<long>(d, lens, others, ax, seed++);
            perType[0] += f;
            perType[1] += h;
            perType[2] += l;
            bad += f + h + l;
            cases += 3;
        }

        Rig.Row(d, "concat", "f32/f16/i64", $"{cases} cases with the inputs at odd offsets in the output's buffer ({(d.Backend == GpuBackend.WebGpu ? "copied through a scratch buffer, as a browser cannot read and write one buffer in a dispatch" : "read in place")})", $"mismatching elements {bad} (f32 {perType[0]}, f16 {perType[1]}, i64 {perType[2]})");
        Assert.Equal(0, bad);
    }
}
