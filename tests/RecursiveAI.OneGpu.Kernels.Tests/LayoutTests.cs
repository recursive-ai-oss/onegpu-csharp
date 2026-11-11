namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Slice, Expand and Concat against index-map references, bit for bit.</summary>
public sealed class LayoutTests
{
    const int OffIn = 1, OffOut = 3;

    static async Task<int> SliceCase<T>(GpuDevice d, LayoutKernels k, int[] shape, long[] starts, long[] ends, int[] axes, long[] steps, int seed)
        where T : unmanaged
    {
        var rank = shape.Length;
        var n = IndexRig.Count(shape);
        var data = IndexRig.Fill<T>(n + OffIn, seed);
        var outShape = new int[rank];
        var first = new int[rank];
        var step = new int[rank];
        LayoutKernels.SliceGeometry(shape, starts, ends, axes, steps, outShape, first, step);
        var total = IndexRig.Count(outShape);
        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(total + OffOut + 2)];
        Array.Fill(output, sentinel);
        using var bi = d.CreateBuffer<T>(data);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeSlice(e, bi, OffIn, shape, starts, ends, axes, steps, bo, OffOut));
        var got = await Rig.DownloadAsync(bo);
        var strides = IndexRig.Strides(shape);
        var bad = 0;
        for (var j = 0; j < total; j++)
        {
            var c = IndexRig.Unravel(outShape, j);
            long off = 0;
            for (var i = 0; i < rank; i++)
            {
                off += (first[i] + (long)c[i] * step[i]) * strides[i];
            }

            bad += IndexRig.Same(got[OffOut + j], data[OffIn + (int)off]) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + total; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public async Task SliceGeometryFollowsTheOperatorRules()
    {
        await Task.CompletedTask;
        var o = new int[2];
        var f = new int[2];
        var s = new int[2];
        void Geo(int[] shape, long[] st, long[] en, int[] ax, long[] sp, int[] wantShape, int[] wantFirst)
        {
            var os = new int[shape.Length];
            var fi = new int[shape.Length];
            var ss = new int[shape.Length];
            LayoutKernels.SliceGeometry(shape, st, en, ax, sp, os, fi, ss);
            Assert.Equal(wantShape, os);
            Assert.Equal(wantFirst, fi);
        }

        // Examples from the operator's documentation and its clamping rules.
        Geo([20, 10, 5], [0, 0], [3, 10], [0, 1], [], [3, 10, 5], [0, 0, 0]);
        Geo([20, 10, 5], [1], [1000], [1], [], [20, 9, 5], [0, 1, 0]);
        Geo([20, 10, 5], [-1], [-1000], [0], [-1], [20, 10, 5], [19, 0, 0]);
        Geo([20, 10, 5], [0, 1], [3, 1000], [0, 1], [1, 2], [3, 5, 5], [0, 1, 0]);
        Geo([20, 10, 5], [1000], [-1000], [0], [-3], [7, 10, 5], [19, 0, 0]);
        Geo([20, 10, 5], [5], [5], [0], [], [0, 10, 5], [0, 0, 0]);
        Geo([20, 10, 5], [-3], [long.MaxValue], [-1], [], [20, 10, 3], [0, 0, 2]);
        Geo([20, 10, 5], [long.MaxValue], [long.MinValue], [1], [-1], [20, 10, 5], [0, 9, 0]);
        Assert.Throws<GpuKernelShapeException>(() => LayoutKernels.SliceGeometry([4, 4], [0], [2], [0], [0], o, f, s));
        Assert.Throws<GpuKernelShapeException>(() => LayoutKernels.SliceGeometry([4, 4], [0, 0], [2, 2], [1, 1], [], o, f, s));
        Assert.Throws<GpuKernelShapeException>(() => LayoutKernels.SliceGeometry([4, 4], [0], [2], [2], [], o, f, s));
        Assert.Throws<GpuKernelShapeException>(() => LayoutKernels.SliceGeometry([4, 4], [0, 0, 0], [2, 2, 2], [], [], o, f, s));
    }

    [Fact]
    public async Task SliceIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new LayoutKernels(d);
        (int[] Shape, long[] Starts, long[] Ends, int[] Axes, long[] Steps)[] cases =
        [
            ([10], [2], [9], [0], [1]), ([10], [-3], [long.MaxValue], [], []), ([10], [9], [-100], [0], [-1]), ([10], [8], [1], [0], [-3]), ([11], [1], [10], [0], [4]),
            ([6, 7], [1, 2], [5, 6], [0, 1], [1, 1]), ([6, 7], [0], [4], [1], [2]), ([6, 7], [5, 6], [-100, -100], [0, 1], [-1, -2]), ([4, 6, 8], [1, 1, 1], [3, 5, 7], [], []),
            ([1, 128, 768], [0], [64], [1], []), ([8, 512, 384], [0], [1], [1], []), ([2, 3, 4, 5], [1, 0], [2, 4], [0, 3], [1, 3]), ([5, 5, 5], [4], [0], [2], [-2]), ([3, 4], [0], [0], [0], []),
            ([33, 129], [3, 5], [30, 120], [0, 1], [2, 5]),
        ];
        long bad = 0, rows = 0;
        var seed = 700;
        foreach (var (shape, starts, ends, axes, steps) in cases)
        {
            bad += await SliceCase<float>(d, k, shape, starts, ends, axes, steps, seed++);
            bad += await SliceCase<Half>(d, k, shape, starts, ends, axes, steps, seed++);
            bad += await SliceCase<int>(d, k, shape, starts, ends, axes, steps, seed++);
            bad += await SliceCase<long>(d, k, shape, starts, ends, axes, steps, seed++);
            rows += 4;
        }

        Rig.Row(d, "slice", "f32/f16/i32/i64", $"{rows} cases (positive and negative steps, clamped bounds, empty result, rank 1 to 4)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }

    static async Task<int> ExpandCase<T>(GpuDevice d, LayoutKernels k, int[] shape, int[] outShape, int seed)
        where T : unmanaged
    {
        var n = IndexRig.Count(shape);
        var total = IndexRig.Count(outShape);
        var data = IndexRig.Fill<T>(n + OffIn, seed);
        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(total + OffOut + 2)];
        Array.Fill(output, sentinel);
        using var bi = d.CreateBuffer<T>(data);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeExpand(e, bi, OffIn, shape, outShape, bo, OffOut));
        var got = await Rig.DownloadAsync(bo);
        var strides = IndexRig.Strides(shape);
        var lead = outShape.Length - shape.Length;
        var bad = 0;
        for (var j = 0; j < total; j++)
        {
            var c = IndexRig.Unravel(outShape, j);
            long off = 0;
            for (var i = 0; i < shape.Length; i++)
            {
                off += (shape[i] == 1 ? 0 : c[i + lead]) * strides[i];
            }

            bad += IndexRig.Same(got[OffOut + j], data[OffIn + (int)off]) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + total; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public async Task ExpandIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new LayoutKernels(d);
        (int[] Shape, int[] Out)[] cases =
        [
            ([1], [7]), ([3], [4, 3]), ([3, 1], [3, 5]), ([1, 1], [6, 9]), ([2, 1, 4], [2, 3, 4]), ([5], [2, 3, 5]), ([1, 7], [1, 7]), ([4, 1, 1], [2, 4, 3, 5]), ([1, 1, 1, 1], [3, 3, 3, 3]),
            ([8, 1, 1, 128], [8, 12, 128, 128]), ([1, 1, 128, 128], [4, 12, 128, 128]), ([3, 3], [0, 3, 3]),
        ];
        long bad = 0, rows = 0;
        var seed = 900;
        foreach (var (shape, outShape) in cases)
        {
            bad += await ExpandCase<float>(d, k, shape, outShape, seed++);
            bad += await ExpandCase<Half>(d, k, shape, outShape, seed++);
            bad += await ExpandCase<long>(d, k, shape, outShape, seed++);
            rows += 3;
        }

        using var a = d.CreateBuffer<float>(8);
        using var o = d.CreateBuffer<float>(64);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeExpand(enc, a, 0, [2, 4], [3, 4], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeExpand(enc, a, 0, [2, 4], [4], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeExpand(enc, a, 0, [3, 4], [3, 4], o, 0));
        d.Submit(enc);
        Rig.Row(d, "expand", "f32/f16/i64", $"{rows} cases (leading dimensions, repeated size-1 dimensions, up to 4 x 12 x 128 x 128, empty)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }

    static async Task<int> ConcatCase<T>(GpuDevice d, LayoutKernels k, int[] lens, int[] shapeWithoutAxis, int axis, int seed)
        where T : unmanaged
    {
        var rank = shapeWithoutAxis.Length + 1;
        var ax = axis < 0 ? axis + rank : axis;
        var outShape = new int[rank];
        for (int i = 0, s = 0; i < rank; i++)
        {
            outShape[i] = i == ax ? lens.Sum() : shapeWithoutAxis[s++];
        }

        long outer = 1, inner = 1;
        for (var i = 0; i < ax; i++)
        {
            outer *= outShape[i];
        }

        for (var i = ax + 1; i < rank; i++)
        {
            inner *= outShape[i];
        }

        var total = IndexRig.Count(outShape);
        var buffers = new List<GpuBuffer<T>>();
        var datas = new List<T[]>();
        var inputs = new List<GpuConcatInput<T>>();
        try
        {
            for (var i = 0; i < lens.Length; i++)
            {
                var count = (int)(outer * lens[i] * inner);
                var data = IndexRig.Fill<T>(count + 1 + (i % 3), seed + i);
                var off = 1 + (i % 3);
                var buf = d.CreateBuffer<T>(data);
                buffers.Add(buf);
                datas.Add(data);
                inputs.Add(new GpuConcatInput<T>(buf, off, lens[i]));
            }

            var sentinel = IndexRig.Make<T>(-999);
            var output = new T[IndexRig.Padded<T>(total + OffOut + 2)];
            Array.Fill(output, sentinel);
            using var bo = d.CreateBuffer<T>(output);
            Rig.Run(d, e => k.EncodeConcat<T>(e, inputs.ToArray(), outShape, axis, bo, OffOut));
            var got = await Rig.DownloadAsync(bo);
            var bad = 0;
            long start = 0;
            for (var i = 0; i < lens.Length; i++)
            {
                for (var o = 0; o < outer; o++)
                {
                    for (var a = 0; a < lens[i]; a++)
                    {
                        for (var m = 0; m < inner; m++)
                        {
                            var want = datas[i][inputs[i].Offset + (int)(((o * lens[i]) + a) * inner + m)];
                            bad += IndexRig.Same(got[OffOut + (int)(((o * outShape[ax]) + start + a) * inner + m)], want) ? 0 : 1;
                        }
                    }
                }

                start += lens[i];
            }

            for (var j = 0; j < OffOut; j++)
            {
                bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
            }

            for (var j = OffOut + total; j < output.Length; j++)
            {
                bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
            }

            return bad;
        }
        finally
        {
            foreach (var b in buffers)
            {
                b.Dispose();
            }
        }
    }

    [Fact]
    public async Task ConcatIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new LayoutKernels(d);
        (int[] Lens, int[] Other, int Axis)[] cases =
        [
            ([3], [], 0), ([3, 4], [], 0), ([1, 1, 1], [], -1), ([3, 5, 2, 7, 1], [], 0), ([2, 3], [4], 0), ([2, 3], [4], 1), ([1, 2, 3, 4, 5, 6, 7], [3, 5], 1), ([1, 2, 3, 4, 5, 6, 7], [3, 5], 0),
            ([3, 5], [2, 4], 1), ([1, 3, 5, 2, 1], [4, 3, 3], 2), ([7, 1, 1, 9, 3, 2, 2, 2, 5], [2, 5], -1), ([128, 1], [8, 12], 1), ([64, 64], [8, 12, 64], 2), ([0, 5, 0], [3], 0),
        ];
        long bad = 0, rows = 0;
        var seed = 1100;
        foreach (var (lens, other, axis) in cases)
        {
            bad += await ConcatCase<float>(d, k, lens, other, axis, seed++);
            bad += await ConcatCase<Half>(d, k, lens, other, axis, seed++);
            bad += await ConcatCase<int>(d, k, lens, other, axis, seed++);
            bad += await ConcatCase<long>(d, k, lens, other, axis, seed++);
            rows += 4;
        }

        using var a = d.CreateBuffer<float>(8);
        using var o = d.CreateBuffer<float>(16);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<float>(enc, [new GpuConcatInput<float>(a, 0, 2)], [2, 3], 1, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<float>(enc, [new GpuConcatInput<float>(a, 0, 5)], [2, 5], 1, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<float>(enc, [new GpuConcatInput<float>(a, 0, 2)], [2, 2], 3, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<float>(enc, [new GpuConcatInput<float>(o, 0, 2)], [2, 2], 1, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeConcat<float>(enc, [], [2, 2], 1, o, 0));
        d.Submit(enc);
        Rig.Row(d, "concat", "f32/f16/i32/i64", $"{rows} cases (1 to 9 inputs, so up to three passes, odd offsets, empty inputs, rank 1 to 5)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }
}
