namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>ScatterND and ScatterElements against a sequential CPU reference (updates applied in index order), bit for bit.</summary>
public sealed class ScatterTests
{
    const int OffData = 1, OffIdx = 2, OffUpd = 3, OffOut = 5;

    static T Combine<T>(GpuScatterReduction op, T cur, T upd)
        where T : unmanaged
    {
        if (op == GpuScatterReduction.None)
        {
            return upd;
        }

        if (typeof(T) == typeof(float))
        {
            float a = (float)(object)cur, b = (float)(object)upd;
            return (T)(object)F(op, a, b);
        }

        if (typeof(T) == typeof(Half))
        {
            float a = (float)(Half)(object)cur, b = (float)(Half)(object)upd;
            return (T)(object)(Half)F(op, a, b);
        }

        if (typeof(T) == typeof(int))
        {
            int a = (int)(object)cur, b = (int)(object)upd;
            return (T)(object)unchecked(op switch { GpuScatterReduction.Add => a + b, GpuScatterReduction.Mul => a * b, GpuScatterReduction.Min => b < a ? b : a, _ => b > a ? b : a });
        }

        {
            long a = (long)(object)cur, b = (long)(object)upd;
            return (T)(object)unchecked(op switch { GpuScatterReduction.Add => a + b, GpuScatterReduction.Mul => a * b, GpuScatterReduction.Min => b < a ? b : a, _ => b > a ? b : a });
        }

        static float F(GpuScatterReduction op, float a, float b) => op switch
        {
            GpuScatterReduction.Add => a + b,
            GpuScatterReduction.Mul => a * b,
            GpuScatterReduction.Min => (b < a || float.IsNaN(b)) ? b : a,
            _ => (b > a || float.IsNaN(b)) ? b : a,
        };
    }

    // Payload values that keep float sums and products finite and make rounding matter: small multiples of 2^-6 plus an odd tail.
    static T[] Updates<T>(int length, int seed, GpuScatterReduction op)
        where T : unmanaged
    {
        var a = IndexRig.Fill<T>(length, seed);
        if (op == GpuScatterReduction.Mul && (typeof(T) == typeof(float) || typeof(T) == typeof(Half)))
        {
            for (var i = 0; i < a.Length; i++)
            {
                a[i] = typeof(T) == typeof(float) ? (T)(object)(1f + (((i * 37 + seed) % 17) - 8) / 64f) : (T)(object)(Half)(1f + (((i * 37 + seed) % 17) - 8) / 64f);
            }
        }

        return a;
    }

    static string Name(GpuScatterReduction r) => r.ToString().ToLowerInvariant();

    static async Task<(int Bad, string Hash)> NdCase<T, TIdx>(GpuDevice d, ScatterKernels k, int[] shape, int[] ishape, GpuScatterReduction op, bool distinct, int seed, int range)
        where T : unmanaged
        where TIdx : unmanaged
    {
        var rank = shape.Length;
        var last = ishape[^1];
        var tuples = IndexRig.Count(ishape[..^1]);
        var n = IndexRig.Count(shape);
        var strides = IndexRig.Strides(shape);
        var slice = 1;
        for (var i = last; i < rank; i++)
        {
            slice *= shape[i];
        }

        var data = IndexRig.Fill<T>(n + OffData, seed);
        var updates = Updates<T>(tuples * slice + OffUpd, seed + 1, op);
        var idxValues = new long[tuples * last];
        var idx = new TIdx[tuples * last + OffIdx];
        // Distinct tuples come from a stride walk through the slice grid; otherwise the first components are drawn from a narrow range to force duplicates.
        var slots = n / slice;
        for (var t = 0; t < tuples; t++)
        {
            long[] comps;
            if (distinct)
            {
                var slot = (int)(((long)t * 7919 + 13) % slots);
                if (tuples <= slots && slots % 7919 != 0)
                {
                    comps = new long[last];
                    var rem = slot;
                    for (var i = last - 1; i >= 0; i--)
                    {
                        comps[i] = rem % shape[i];
                        rem /= shape[i];
                    }
                }
                else
                {
                    throw new InvalidOperationException("distinct case needs tuples <= slots");
                }
            }
            else
            {
                comps = new long[last];
                for (var i = 0; i < last; i++)
                {
                    comps[i] = IndexRig.RandomIndex(t * last + i, Math.Min(shape[i], range), seed);
                }
            }

            for (var i = 0; i < last; i++)
            {
                idxValues[(t * last) + i] = comps[i];
                idx[OffIdx + (t * last) + i] = IndexRig.FromLong<TIdx>(comps[i]);
            }
        }

        // Reference.
        var want = (T[])data.Clone();
        for (var t = 0; t < tuples; t++)
        {
            long off = 0;
            for (var i = 0; i < last; i++)
            {
                off += IndexRig.Norm(idxValues[(t * last) + i], shape[i]) * strides[i];
            }

            for (var r = 0; r < slice; r++)
            {
                want[OffData + (int)off + r] = Combine(op, want[OffData + (int)off + r], updates[OffUpd + (t * slice) + r]);
            }
        }

        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(n + OffOut + 3)];
        Array.Fill(output, sentinel);
        using var status = new GpuIndexStatus(d);
        using var bd = d.CreateBuffer<T>(data);
        using var bi = d.CreateBuffer<TIdx>(idx);
        using var bu = d.CreateBuffer<T>(updates);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeScatterNd(e, bd, OffData, shape, bi, OffIdx, ishape, bu, OffUpd, bo, OffOut, op, distinct, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var j = 0; j < n; j++)
        {
            bad += IndexRig.Same(got[OffOut + j], want[OffData + j]) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + n; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        return (bad, IndexRig.Digest(got));
    }

    static readonly GpuScatterReduction[] Reductions = [GpuScatterReduction.None, GpuScatterReduction.Add, GpuScatterReduction.Mul, GpuScatterReduction.Min, GpuScatterReduction.Max];

    [Fact]
    public async Task ScatterNdMatchesSequentialReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ScatterKernels(d);
        // Duplicate-heavy cases: many tuples into few slices.
        (int[] Shape, int[] Indices, int Range)[] dup =
        [
            ([40], [300, 1], 6), ([8, 9], [200, 2], 3), ([8, 9], [50, 1], 4), ([6, 5, 4], [90, 2], 3), ([6, 5, 4], [40, 3], 2), ([30, 257], [20, 1], 5), ([2, 3, 4, 5, 6, 7], [70, 6], 2), ([100], [5000, 1], 100),
        ];
        long bad = 0, cases = 0;
        var seed = 300;
        var hashes = new List<string>();
        foreach (var (shape, ishape, range) in dup)
        {
            foreach (var op in Reductions)
            {
                var r1 = await NdCase<float, int>(d, k, shape, ishape, op, false, seed, range);
                var r2 = await NdCase<float, int>(d, k, shape, ishape, op, false, seed, range);
                Assert.Equal(r1.Hash, r2.Hash);
                bad += r1.Bad;
                bad += (await NdCase<Half, long>(d, k, shape, ishape, op, false, seed + 1, range)).Bad;
                bad += (await NdCase<int, int>(d, k, shape, ishape, op, false, seed + 2, range)).Bad;
                bad += (await NdCase<long, long>(d, k, shape, ishape, op, false, seed + 3, range)).Bad;
                cases += 5;
                if (shape.Length == 1 && ishape[0] == 5000)
                {
                    hashes.Add($"{Name(op)}={r1.Hash}");
                }
            }

            seed += 10;
        }

        // Distinct targets, both flavours of the kernel.
        (int[] Shape, int[] Indices)[] distinct = [([64], [40, 1]), ([8, 9], [30, 2]), ([8, 9], [8, 1]), ([6, 5, 4], [25, 2]), ([30, 257], [20, 1]), ([2, 3, 4, 5, 6, 7], [20, 3])];
        foreach (var (shape, ishape) in distinct)
        {
            foreach (var op in Reductions)
            {
                bad += (await NdCase<float, int>(d, k, shape, ishape, op, true, seed, 0)).Bad;
                bad += (await NdCase<Half, long>(d, k, shape, ishape, op, true, seed + 1, 0)).Bad;
                bad += (await NdCase<long, int>(d, k, shape, ishape, op, true, seed + 2, 0)).Bad;
                bad += (await NdCase<float, int>(d, k, shape, ishape, op, false, seed + 3, 3)).Bad;
                cases += 4;
            }

            seed += 10;
        }

        Rig.Row(d, "scatter nd", "f32/f16/i32/i64", $"{cases} cases ({Reductions.Length} reductions, heavy duplicates, distinct targets, negative indices); two runs of every f32 duplicate case hashed equal", $"mismatching elements {bad}");
        Rig.Row(d, "scatter nd", "f32", "determinism: 5000 updates into 100 slots, run twice", string.Join(" ", hashes) + " (each run equal)");
        Assert.Equal(0, bad);
    }

    static async Task<(int Bad, string Hash)> ElementsCase<T, TIdx>(GpuDevice d, ScatterKernels k, int[] shape, int[] ishape, int axis, GpuScatterReduction op, bool distinct, int seed)
        where T : unmanaged
        where TIdx : unmanaged
    {
        var rank = shape.Length;
        var ax = axis < 0 ? axis + rank : axis;
        var n = IndexRig.Count(shape);
        var m = IndexRig.Count(ishape);
        var strides = IndexRig.Strides(shape);
        var data = IndexRig.Fill<T>(n + OffData, seed);
        var updates = Updates<T>(m + OffUpd, seed + 1, op);
        var idxValues = new long[m];
        var idx = new TIdx[m + OffIdx];
        for (var j = 0; j < m; j++)
        {
            long v;
            if (distinct)
            {
                // Along the axis, position c maps to 3c mod len: injective when len is not a multiple of 3 and the indices cover at most len positions.
                var c = IndexRig.Unravel(ishape, j);
                v = ((long)c[ax] * 3) % shape[ax];
            }
            else
            {
                v = IndexRig.RandomIndex(j, Math.Min(shape[ax], 4), seed);
            }

            idxValues[j] = v;
            idx[OffIdx + j] = IndexRig.FromLong<TIdx>(v);
        }

        var want = (T[])data.Clone();
        for (var j = 0; j < m; j++)
        {
            var c = IndexRig.Unravel(ishape, j);
            long off = 0;
            for (var i = 0; i < rank; i++)
            {
                off += (i == ax ? IndexRig.Norm(idxValues[j], shape[ax]) : c[i]) * strides[i];
            }

            want[OffData + (int)off] = Combine(op, want[OffData + (int)off], updates[OffUpd + j]);
        }

        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(n + OffOut + 3)];
        Array.Fill(output, sentinel);
        using var status = new GpuIndexStatus(d);
        using var bd = d.CreateBuffer<T>(data);
        using var bi = d.CreateBuffer<TIdx>(idx);
        using var bu = d.CreateBuffer<T>(updates);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeScatterElements(e, bd, OffData, shape, axis, bi, OffIdx, ishape, bu, OffUpd, bo, OffOut, op, distinct, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var j = 0; j < n; j++)
        {
            bad += IndexRig.Same(got[OffOut + j], want[OffData + j]) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + n; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        return (bad, IndexRig.Digest(got));
    }

    [Fact]
    public async Task ScatterElementsMatchesSequentialReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ScatterKernels(d);
        // Duplicates: indices come from at most four positions along the axis.
        (int[] Shape, int[] Indices, int Axis)[] dup =
        [
            ([9], [60], 0), ([5, 6], [5, 6], 0), ([5, 6], [20, 3], 0), ([5, 6], [4, 30], 1), ([3, 4, 5], [3, 40, 2], 1), ([3, 4, 5], [2, 4, 5], -1), ([2, 3, 4, 5, 6], [2, 3, 6, 5, 6], 2),
            ([50], [3000], 0), ([4, 70], [4, 2000], -1),
        ];
        long bad = 0, cases = 0;
        var seed = 500;
        var hashes = new List<string>();
        foreach (var (shape, ishape, axis) in dup)
        {
            foreach (var op in Reductions)
            {
                var r1 = await ElementsCase<float, int>(d, k, shape, ishape, axis, op, false, seed);
                var r2 = await ElementsCase<float, int>(d, k, shape, ishape, axis, op, false, seed);
                Assert.Equal(r1.Hash, r2.Hash);
                bad += r1.Bad;
                bad += (await ElementsCase<Half, long>(d, k, shape, ishape, axis, op, false, seed + 1)).Bad;
                bad += (await ElementsCase<int, int>(d, k, shape, ishape, axis, op, false, seed + 2)).Bad;
                bad += (await ElementsCase<long, long>(d, k, shape, ishape, axis, op, false, seed + 3)).Bad;
                cases += 5;
                if (ishape.Length == 1 && ishape[0] == 3000)
                {
                    hashes.Add($"{Name(op)}={r1.Hash}");
                }
            }

            seed += 10;
        }

        (int[] Shape, int[] Indices, int Axis)[] distinct = [([7], [7], 0), ([5, 4], [5, 4], 1), ([5, 7], [3, 7], 0), ([3, 4, 5], [3, 2, 5], 1), ([2, 3, 4, 5, 6], [2, 3, 2, 5, 6], 3)];
        foreach (var (shape, ishape, axis) in distinct)
        {
            foreach (var op in Reductions)
            {
                bad += (await ElementsCase<float, int>(d, k, shape, ishape, axis, op, true, seed)).Bad;
                bad += (await ElementsCase<Half, long>(d, k, shape, ishape, axis, op, true, seed + 1)).Bad;
                bad += (await ElementsCase<long, int>(d, k, shape, ishape, axis, op, true, seed + 2)).Bad;
                cases += 3;
            }

            seed += 10;
        }

        Rig.Row(d, "scatter elements", "f32/f16/i32/i64", $"{cases} cases ({Reductions.Length} reductions, rank 1 to 5, heavy duplicates, distinct targets, negative indices); two runs of every f32 duplicate case hashed equal", $"mismatching elements {bad}");
        Rig.Row(d, "scatter elements", "f32", "determinism: 3000 updates into 50 elements, run twice", string.Join(" ", hashes) + " (each run equal)");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task OutOfRangeIndicesAreSkippedAndReported()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ScatterKernels(d);
        var data = IndexRig.Fill<float>(6 * 3, 9);
        var upd = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        using var bd = d.CreateBuffer<float>(data);
        using var bu = d.CreateBuffer<float>(upd);
        using var bi = d.CreateBuffer<long>([0, 7, 5]);
        using var bo = d.CreateBuffer<float>(data.Length);
        using var status = new GpuIndexStatus(d);
        Rig.Run(d, e => k.EncodeScatterNd(e, bd, 0, [6, 3], bi, 0, [3, 1], bu, 0, bo, 0, GpuScatterReduction.Add, false, status));
        var ex = await Assert.ThrowsAsync<GpuKernelIndexException>(async () => await status.ThrowIfFailedAsync());
        Assert.Equal(1, ex.Position);
        Assert.Equal("ScatterND", ex.Operation);
        var got = await Rig.DownloadAsync(bo);
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(data[i] + upd[i], got[i]);
            Assert.Equal(data[15 + i] + upd[6 + i], got[15 + i]);
            Assert.Equal(data[3 + i], got[3 + i]);
        }

        using var bie = d.CreateBuffer<int>([0, 1, 9]);
        using var bue = d.CreateBuffer<float>([1, 2, 3]);
        using var bd1 = d.CreateBuffer<float>([10, 20, 30, 40]);
        using var bo1 = d.CreateBuffer<float>(4);
        status.Reset();
        Rig.Run(d, e => k.EncodeScatterElements(e, bd1, 0, [4], 0, bie, 0, [3], bue, 0, bo1, 0, GpuScatterReduction.None, false, status));
        var ex2 = await Assert.ThrowsAsync<GpuKernelIndexException>(async () => await status.ThrowIfFailedAsync());
        Assert.Equal(2, ex2.Position);
        var after1 = await Rig.DownloadAsync(bo1);
        Assert.Equal(1f, after1[0]);
        Assert.Equal(2f, after1[1]);
        Assert.Equal(30f, after1[2]);
        Assert.Equal(40f, after1[3]);
        // In place: the output may be the data buffer.
        status.Reset();
        using var bie2 = d.CreateBuffer<int>([3, 0, 3]);
        Rig.Run(d, e => k.EncodeScatterElements(e, bo1, 0, [4], 0, bie2, 0, [3], bue, 0, bo1, 0, GpuScatterReduction.Add, false, status));
        await status.ThrowIfFailedAsync();
        var after2 = await Rig.DownloadAsync(bo1);
        Assert.Equal(3f, after2[0]);
        Assert.Equal(2f, after2[1]);
        Assert.Equal(30f, after2[2]);
        Assert.Equal(44f, after2[3]);
        Rig.Row(d, "scatter", "f32", "out-of-range indices and in-place update", "invalid updates skipped, smallest position reported");
    }

    [Fact]
    public void BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ScatterKernels(d);
        using var data = d.CreateBuffer<float>(12);
        using var idx = d.CreateBuffer<int>(8);
        using var upd = d.CreateBuffer<float>(12);
        using var outp = d.CreateBuffer<float>(12);
        using var uints = d.CreateBuffer<uint>(12);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterNd(enc, data, 0, [3, 4], idx, 0, [2, 3], upd, 0, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterNd(enc, data, 0, [3, 4], idx, 0, [4, 1], upd, 0, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterNd(enc, uints, 0, [3, 4], idx, 0, [2, 1], uints, 0, uints, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterNd(enc, data, 0, [3, 4], idx, 0, [2, 1], upd, 0, outp, 0, (GpuScatterReduction)9));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterNd(enc, data, 0, [3, 4], idx, 0, [2, 1], upd, 0, data, 1));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterElements(enc, data, 0, [3, 4], 2, idx, 0, [2, 2], upd, 0, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterElements(enc, data, 0, [3, 4], 0, idx, 0, [2, 5], upd, 0, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeScatterElements(enc, data, 0, [3, 4], 0, idx, 0, [2], upd, 0, outp, 0));
        d.Submit(enc);
        d.WaitIdle();
    }
}
