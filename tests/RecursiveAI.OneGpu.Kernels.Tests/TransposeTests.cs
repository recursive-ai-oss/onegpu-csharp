namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class TransposeTests
{
    static IEnumerable<int[]> Permutations(int n)
    {
        var p = Enumerable.Range(0, n).ToArray();
        return Heap(p, n);

        static IEnumerable<int[]> Heap(int[] a, int k)
        {
            if (k <= 1)
            {
                yield return (int[])a.Clone();
                yield break;
            }

            for (var i = 0; i < k; i++)
            {
                foreach (var r in Heap(a, k - 1))
                {
                    yield return r;
                }

                var j = k % 2 == 0 ? i : 0;
                (a[j], a[k - 1]) = (a[k - 1], a[j]);
            }
        }
    }

    static int[] ExpectedIndex(int[] shape, int[] perm)
    {
        var rank = shape.Length;
        var outShape = perm.Select(p => shape[p]).ToArray();
        var n = shape.Aggregate(1, (a, b) => a * b);
        var inStride = new int[rank];
        var s = 1;
        for (var i = rank - 1; i >= 0; i--)
        {
            inStride[i] = s;
            s *= shape[i];
        }

        var src = new int[n];
        var idx = new int[rank];
        for (var j = 0; j < n; j++)
        {
            var rem = j;
            for (var i = rank - 1; i >= 0; i--)
            {
                idx[i] = rem % outShape[i];
                rem /= outShape[i];
            }

            var off = 0;
            for (var i = 0; i < rank; i++)
            {
                off += idx[i] * inStride[perm[i]];
            }

            src[j] = off;
        }

        return src;
    }

    static int Check<T>(GpuDevice d, TransposeKernels k, int[] shape, int[] perm, Func<int, T> make, Func<T, T, bool> same, Action<GpuCommandEncoder, GpuBuffer<T>, GpuBuffer<T>>? custom = null)
        where T : unmanaged
    {
        const int OffIn = 1, OffOut = 3;
        var n = shape.Aggregate(1, (a, b) => a * b);
        var src = ExpectedIndex(shape, perm);
        var input = new T[typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n + OffIn) : n + OffIn];
        for (var i = 0; i < n; i++)
        {
            input[OffIn + i] = make(i);
        }

        var sentinel = make(-12345);
        var output = new T[typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n + OffOut + 2) : n + OffOut + 2];
        Array.Fill(output, sentinel);
        using var bi = d.CreateBuffer<T>(input);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e =>
        {
            if (custom is null)
            {
                k.Encode(e, bi, OffIn, shape, perm, bo, OffOut);
            }
            else
            {
                custom(e, bi, bo);
            }
        });
        var got = Rig.Download(bo);
        var bad = 0;
        for (var j = 0; j < n; j++)
        {
            bad += same(got[OffOut + j], make(src[j])) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + n; j < output.Length; j++)
        {
            bad += same(got[j], sentinel) ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public void PermutationsAreExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new TransposeKernels(d);
        List<int[]> shapes = [[5], [1], [3, 7], [1, 7], [2, 3, 4, 5], [1, 4, 1, 5], [2, 1, 3, 1, 2], [3, 2, 2, 3, 2, 2], [4, 512, 768 / 4], [1, 128, 12, 64], [2, 3, 17, 19], [33, 65]];
        var cases = 0;
        long bad32 = 0, bad16 = 0, bad64 = 0;
        foreach (var shape in shapes)
        {
            IEnumerable<int[]> perms = shape.Length <= 4 ? Permutations(shape.Length) : Permutations(shape.Length).Where((_, i) => i % 37 == 0);
            foreach (var perm in perms)
            {
                bad32 += Check<float>(d, k, shape, perm, i => i * 0.5f + 1f, (a, b) => BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(b));
                bad16 += Check<Half>(d, k, shape, perm, i => BitConverter.UInt16BitsToHalf((ushort)(i * 7 + 1)), (a, b) => BitConverter.HalfToUInt16Bits(a) == BitConverter.HalfToUInt16Bits(b));
                bad64 += Check<long>(d, k, shape, perm, i => (long)i * 4294967311L + 5, (a, b) => a == b);
                cases++;
            }
        }

        Rig.Row(d, "transpose", "f32/f16/i64", $"{cases} permutations of {shapes.Count} shapes (rank 1 to 6, ones, odd offsets), checked against the index map", $"mismatching elements: f32 {bad32}, f16 {bad16}, i64 {bad64}");
        Assert.Equal(0, bad32 + bad16 + bad64);
    }

    [Fact]
    public void LayoutHelpersMatchTheirPermutations()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new TransposeKernels(d);
        long bad = 0;
        bool Same(float a, float b) => BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(b);
        bad += Check<float>(d, k, [2, 3, 17, 19], [0, 2, 3, 1], i => i, Same, (e, i, o) => k.EncodeNchwToNhwc(e, i, 1, 2, 3, 17, 19, o, 3));
        bad += Check<float>(d, k, [2, 17, 19, 3], [0, 3, 1, 2], i => i, Same, (e, i, o) => k.EncodeNhwcToNchw(e, i, 1, 2, 3, 17, 19, o, 3));
        foreach (var (b, s, h, dim) in new[] { (1, 128, 12, 64), (2, 7, 12, 32), (1, 1, 6, 64), (3, 512, 12, 64) })
        {
            bad += Check<float>(d, k, [b, s, h, dim], [0, 2, 1, 3], i => i, Same, (e, i, o) => k.EncodeSplitHeads(e, i, 1, b, s, h, dim, o, 3));
            bad += Check<float>(d, k, [b, h, s, dim], [0, 2, 1, 3], i => i, Same, (e, i, o) => k.EncodeMergeHeads(e, i, 1, b, s, h, dim, o, 3));
            bad += Check<Half>(d, k, [b, s, h, dim], [0, 2, 1, 3], i => BitConverter.UInt16BitsToHalf((ushort)(i + 1)), (x, y) => BitConverter.HalfToUInt16Bits(x) == BitConverter.HalfToUInt16Bits(y), (e, i, o) => k.EncodeSplitHeads(e, i, 1, b, s, h, dim, o, 3));
        }

        Rig.Row(d, "transpose", "f32/f16", "NCHW<->NHWC and head split/merge for BERT/MiniLM/Whisper-sized shapes", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public void BadPermutationsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new TransposeKernels(d);
        using var a = d.CreateBuffer<float>(24);
        using var o = d.CreateBuffer<float>(24);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, a, 0, [2, 3, 4], [0, 1], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, a, 0, [2, 3, 4], [0, 1, 1], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, a, 0, [2, 3, 4], [0, 1, 3], o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, a, 1, [2, 3, 4], [2, 1, 0], o, 0));
        // Seven dimensions that stay separate after merging are more than the kernel addresses.
        using var big = d.CreateBuffer<float>(256);
        Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, big, 0, [2, 2, 2, 2, 2, 2, 2], [6, 5, 4, 3, 2, 1, 0], big, 0));
        d.Submit(enc);
    }
}
