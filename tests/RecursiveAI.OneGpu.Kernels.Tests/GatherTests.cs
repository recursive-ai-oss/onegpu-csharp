namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Gather, GatherElements and GatherND against index-map references, bit for bit.</summary>
public sealed class GatherTests
{
    const int OffData = 1, OffIdx = 2, OffOut = 3;

    static async Task<int> GatherCase<T, TIdx>(GpuDevice d, GatherKernels k, int[] shape, int axis, int indexCount, int seed)
        where T : unmanaged
        where TIdx : unmanaged
    {
        var rank = shape.Length;
        var ax = axis < 0 ? axis + rank : axis;
        var n = IndexRig.Count(shape);
        var data = IndexRig.Fill<T>(n + OffData, seed);
        var idxValues = new long[indexCount];
        var idx = new TIdx[indexCount + OffIdx];
        for (var i = 0; i < indexCount; i++)
        {
            idxValues[i] = IndexRig.RandomIndex(i, shape[ax], seed);
            idx[OffIdx + i] = IndexRig.FromLong<TIdx>(idxValues[i]);
        }

        long outer = 1, inner = 1;
        for (var i = 0; i < ax; i++)
        {
            outer *= shape[i];
        }

        for (var i = ax + 1; i < rank; i++)
        {
            inner *= shape[i];
        }

        var total = (int)(outer * indexCount * inner);
        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(total + OffOut + 2)];
        Array.Fill(output, sentinel);
        using var status = new GpuIndexStatus(d);
        using var bd = d.CreateBuffer<T>(data);
        using var bi = d.CreateBuffer<TIdx>(idx.Length == 0 ? new TIdx[1] : idx);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeGather(e, bd, OffData, shape, axis, bi, OffIdx, indexCount, bo, OffOut, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var o = 0; o < outer; o++)
        {
            for (var i = 0; i < indexCount; i++)
            {
                for (var m = 0; m < inner; m++)
                {
                    var want = data[OffData + (int)((o * shape[ax] + IndexRig.Norm(idxValues[i], shape[ax])) * inner + m)];
                    bad += IndexRig.Same(got[OffOut + (int)((o * indexCount + i) * inner + m)], want) ? 0 : 1;
                }
            }
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
    public async Task GatherIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        (int[] Shape, int Axis, int Count)[] cases =
        [
            ([10], 0, 7), ([10], -1, 1), ([5, 6], 0, 9), ([5, 6], 1, 4), ([5, 6], -1, 11), ([2, 3, 4, 5], 2, 6), ([2, 3, 4, 5], -4, 3),
            ([1, 7], 1, 5), ([4, 1, 9], 0, 8), ([3, 257], 1, 300), ([1031, 3], 0, 1500), ([2, 3, 4, 5, 6], 3, 2),
        ];
        long bad = 0, rows = 0;
        var seed = 1;
        foreach (var (shape, axis, count) in cases)
        {
            bad += await GatherCase<float, int>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<float, long>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<Half, int>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<Half, long>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<int, int>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<uint, long>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<long, int>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<long, long>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<byte, int>(d, k, shape, axis, count, seed++);
            bad += await GatherCase<sbyte, long>(d, k, shape, axis, count, seed++);
            rows += 10;
        }

        Rig.Row(d, "gather", "f32/f16/i32/u32/i64/u8/i8", $"{rows} cases ({cases.Length} shapes x payload x int32/int64 indices, negative indices, odd offsets)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }

    // The embedding lookups of the reference models: a table of vocab x hidden values and batch x sequence token ids.
    [Fact]
    public async Task EmbeddingLookupsAreExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        var maxBinding = d.Capabilities.Compute.MaxStorageBufferBindingBytes;
        long bad = 0, checkedElements = 0;
        foreach (var (vocab, hidden, half) in new[] { (30522, 384, false), (30522, 768, false), (250002, 384, false), (250002, 768, true) })
        {
            var bytes = (long)vocab * hidden * (half ? 2 : 4);
            if ((maxBinding != 0 && (ulong)bytes > maxBinding) || OperatingSystem.IsBrowser() && bytes > (96L << 20))
            {
                Rig.Row(d, "gather", half ? "f16" : "f32", $"embedding {vocab} x {hidden}", $"skipped: {bytes >> 20} MB is more than one storage binding covers here");
                continue;
            }

            foreach (var batch in new[] { 1, 8 })
            {
                var (b, c) = half ? await Embedding<Half>(d, k, vocab, hidden, batch, 512) : await Embedding<float>(d, k, vocab, hidden, batch, 512);
                bad += b;
                checkedElements += c;
                Rig.Row(d, "gather", half ? "f16" : "f32", $"embedding {vocab} x {hidden}, {batch} x 512 tokens", $"mismatching elements {b} of {c}");
            }
        }

        Assert.Equal(0, bad);
    }

    static async Task<(long Bad, long Count)> Embedding<T>(GpuDevice d, GatherKernels k, int vocab, int hidden, int batch, int seq)
        where T : unmanaged
    {
        // The table is generated chunk by chunk and never held on the host: element i holds Make(i).
        long elements = (long)vocab * hidden;
        using var table = d.CreateBuffer<T>(checked((int)IndexRig.Padded<T>((int)elements)));
        const int Chunk = 1 << 22;
        var buf = new T[Chunk];
        for (long at = 0; at < elements; at += Chunk)
        {
            var len = (int)Math.Min(Chunk, elements - at);
            for (var i = 0; i < len; i++)
            {
                buf[i] = IndexRig.Make<T>((int)(at + i));
            }

            table.Write(buf.AsSpan(0, len), (int)at);
        }

        var tokens = batch * seq;
        var ids = new int[tokens];
        for (var i = 0; i < tokens; i++)
        {
            ids[i] = (int)IndexRig.Norm(IndexRig.RandomIndex(i, vocab, batch), vocab);
        }

        ids[0] = 0;
        ids[^1] = vocab - 1;
        using var bi = d.CreateBuffer<int>(ids);
        var count = tokens * hidden;
        using var bo = d.CreateBuffer<T>(IndexRig.Padded<T>(count));
        using var status = new GpuIndexStatus(d);
        Rig.Run(d, e => k.EncodeGather(e, table, 0, [vocab, hidden], 0, bi, 0, tokens, bo, 0, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo, count);
        long bad = 0;
        for (var t = 0; t < tokens; t++)
        {
            for (var h = 0; h < hidden; h++)
            {
                bad += IndexRig.Same(got[(t * hidden) + h], IndexRig.Make<T>((int)((long)ids[t] * hidden + h))) ? 0 : 1;
            }
        }

        return (bad, count);
    }

    [Fact]
    public async Task OutOfRangeIndicesAreReportedAndNeverReadOutside()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        var data = IndexRig.Fill<float>(5 * 4, 3);
        using var bd = d.CreateBuffer<float>(data);
        foreach (var bad in new long[] { 5, -6, 1000, int.MaxValue, 1L << 40, -(1L << 40) })
        {
            // Position 3 is the first bad index (and 6 is a second one).
            var idx = new long[] { 0, 4, -1, bad, 2, 3, 7 };
            using var bi = d.CreateBuffer<long>(idx);
            using var bo = d.CreateBuffer<float>(idx.Length * 4);
            using var status = new GpuIndexStatus(d);
            Rig.Run(d, e => k.EncodeGather(e, bd, 0, [5, 4], 0, bi, 0, idx.Length, bo, 0, status));
            var ex = await Assert.ThrowsAsync<GpuKernelIndexException>(async () => await status.ThrowIfFailedAsync());
            Assert.Equal(3, ex.Position);
            Assert.Equal("Gather", ex.Operation);
            var got = await Rig.DownloadAsync(bo);
            for (var m = 0; m < 4; m++)
            {
                Assert.Equal(0f, got[(3 * 4) + m]);
                Assert.Equal(0f, got[(6 * 4) + m]);
                Assert.Equal(data[(4 * 4) + m], got[(1 * 4) + m]);
                Assert.Equal(data[(4 * 4) + m], got[(2 * 4) + m]);
            }
        }

        // The int32 form, and the host-side check.
        using var bi32 = d.CreateBuffer<int>([0, 5]);
        using var bo32 = d.CreateBuffer<float>(8);
        using var st32 = new GpuIndexStatus(d);
        Rig.Run(d, e => k.EncodeGather(e, bd, 0, [5, 4], 0, bi32, 0, 2, bo32, 0, st32));
        var e32 = await Assert.ThrowsAsync<GpuKernelIndexException>(async () => await st32.ThrowIfFailedAsync());
        Assert.Equal(1, e32.Position);
        st32.Reset();
        Rig.Run(d, e => k.EncodeGather(e, bd, 0, [5, 4], 0, bi32, 0, 1, bo32, 0, st32));
        await st32.ThrowIfFailedAsync();
        var host = Assert.Throws<GpuKernelIndexException>(() => GpuKernelIndexException.Validate("Gather", new[] { 0, 1, -5, 5 }, 5));
        Assert.Equal(3, host.Position);
        Assert.Equal(5L, host.Index);
        Rig.Row(d, "gather", "f32", "out-of-range indices (int32 and int64, up to 2^40, negative below -len)", "reported with the smallest position, outputs zero, nothing read outside the data");
    }

    [Fact]
    public async Task EmptyAndBadArguments()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        using var data = d.CreateBuffer<float>(12);
        using var idx = d.CreateBuffer<int>(4);
        using var outp = d.CreateBuffer<float>(12);
        var enc = d.BeginCommands();
        k.EncodeGather(enc, data, 0, [3, 4], 0, idx, 0, 0, outp, 0);
        k.EncodeGather(enc, data, 0, [0, 4], 0, idx, 0, 2, outp, 0);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 4], 2, idx, 0, 2, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 4], 0, idx, 3, 2, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 4], 0, idx, 0, 4, outp, 1));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 4], 0, idx, 0, -1, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [], 0, idx, 0, 1, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 5], 0, idx, 0, 1, outp, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [int.MaxValue, 2], 0, idx, 0, 1, outp, 0));
        using var shorts = d.CreateBuffer<ushort>(16);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, shorts, 0, [16], 0, idx, 0, 1, shorts, 0));
        // Byte payloads are packed four to a word: a buffer that ends inside a word is refused.
        using var bytes = d.CreateBuffer<byte>(18);
        using var padded = d.CreateBuffer<byte>(20);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, bytes, 0, [1, 18], 0, idx, 0, 1, padded, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, padded, 0, [1, 18], 0, idx, 0, 1, bytes, 0));
        using var floats = d.CreateBuffer<float>(4);
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeGather(enc, data, 0, [3, 4], 0, floats, 0, 1, outp, 0));
        d.Submit(enc);
        d.WaitIdle();
        await Task.CompletedTask;
    }

    static async Task<int> ElementsCase<T, TIdx>(GpuDevice d, GatherKernels k, int[] shape, int[] ishape, int axis, int seed)
        where T : unmanaged
        where TIdx : unmanaged
    {
        var rank = shape.Length;
        var ax = axis < 0 ? axis + rank : axis;
        var n = IndexRig.Count(shape);
        var m = IndexRig.Count(ishape);
        var data = IndexRig.Fill<T>(n + OffData, seed);
        var idxValues = new long[m];
        var idx = new TIdx[m + OffIdx];
        for (var i = 0; i < m; i++)
        {
            idxValues[i] = IndexRig.RandomIndex(i, shape[ax], seed);
            idx[OffIdx + i] = IndexRig.FromLong<TIdx>(idxValues[i]);
        }

        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(m + OffOut + 2)];
        Array.Fill(output, sentinel);
        using var status = new GpuIndexStatus(d);
        using var bd = d.CreateBuffer<T>(data);
        using var bi = d.CreateBuffer<TIdx>(idx);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeGatherElements(e, bd, OffData, shape, axis, bi, OffIdx, ishape, bo, OffOut, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo);
        var strides = IndexRig.Strides(shape);
        var bad = 0;
        for (var j = 0; j < m; j++)
        {
            var c = IndexRig.Unravel(ishape, j);
            long off = 0;
            for (var i = 0; i < rank; i++)
            {
                off += (i == ax ? IndexRig.Norm(idxValues[j], shape[ax]) : c[i]) * strides[i];
            }

            bad += IndexRig.Same(got[OffOut + j], data[OffData + (int)off]) ? 0 : 1;
        }

        for (var j = 0; j < OffOut; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffOut + m; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public async Task GatherElementsIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        (int[] Shape, int[] Indices, int Axis)[] cases =
        [
            ([8], [5], 0), ([4, 5], [4, 5], 0), ([4, 5], [3, 5], 0), ([4, 5], [4, 3], 1), ([4, 5], [2, 9], -1), ([3, 4, 5], [3, 4, 5], 1),
            ([3, 4, 5], [2, 7, 5], 1), ([3, 4, 5], [3, 2, 1], 2), ([1, 6], [1, 3], 1), ([2, 3, 4, 5, 6], [2, 2, 4, 3, 6], 3), ([2, 3, 2, 3, 2, 3, 2, 3], [2, 3, 2, 3, 2, 3, 2, 3], 4),
            ([70, 129], [70, 129], 1),
        ];
        long bad = 0, rows = 0;
        var seed = 100;
        foreach (var (shape, ishape, axis) in cases)
        {
            bad += await ElementsCase<float, int>(d, k, shape, ishape, axis, seed++);
            bad += await ElementsCase<Half, long>(d, k, shape, ishape, axis, seed++);
            bad += await ElementsCase<int, long>(d, k, shape, ishape, axis, seed++);
            bad += await ElementsCase<long, int>(d, k, shape, ishape, axis, seed++);
            bad += await ElementsCase<byte, long>(d, k, shape, ishape, axis, seed++);
            bad += await ElementsCase<sbyte, int>(d, k, shape, ishape, axis, seed++);
            rows += 6;
        }

        Rig.Row(d, "gather elements", "f32/f16/i32/i64/u8/i8", $"{rows} cases (rank 1 to 8, index shapes smaller than the data off the axis, negative indices)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }

    static async Task<int> NdCase<T, TIdx>(GpuDevice d, GatherKernels k, int[] shape, int[] ishape, int batchDims, int seed)
        where T : unmanaged
        where TIdx : unmanaged
    {
        var rank = shape.Length;
        var last = ishape[^1];
        var tuples = IndexRig.Count(ishape[..^1]);
        var n = IndexRig.Count(shape);
        var data = IndexRig.Fill<T>(n + OffData, seed);
        var idxValues = new long[tuples * last];
        var idx = new TIdx[tuples * last + OffIdx];
        for (var t = 0; t < tuples; t++)
        {
            for (var i = 0; i < last; i++)
            {
                idxValues[(t * last) + i] = IndexRig.RandomIndex(t * last + i, shape[batchDims + i], seed);
                idx[OffIdx + (t * last) + i] = IndexRig.FromLong<TIdx>(idxValues[(t * last) + i]);
            }
        }

        var slice = 1;
        for (var i = batchDims + last; i < rank; i++)
        {
            slice *= shape[i];
        }

        var total = tuples * slice;
        var sentinel = IndexRig.Make<T>(-999);
        var output = new T[IndexRig.Padded<T>(total + OffOut + 2)];
        Array.Fill(output, sentinel);
        using var status = new GpuIndexStatus(d);
        using var bd = d.CreateBuffer<T>(data);
        using var bi = d.CreateBuffer<TIdx>(idx);
        using var bo = d.CreateBuffer<T>(output);
        Rig.Run(d, e => k.EncodeGatherNd(e, bd, OffData, shape, bi, OffIdx, ishape, batchDims, bo, OffOut, status));
        await status.ThrowIfFailedAsync();
        var got = await Rig.DownloadAsync(bo);
        var strides = IndexRig.Strides(shape);
        var batches = 1;
        for (var i = 0; i < batchDims; i++)
        {
            batches *= shape[i];
        }

        var perBatch = tuples / batches;
        var bad = 0;
        for (var t = 0; t < tuples; t++)
        {
            long off = (long)(t / perBatch) * (n / batches);
            for (var i = 0; i < last; i++)
            {
                off += IndexRig.Norm(idxValues[(t * last) + i], shape[batchDims + i]) * strides[batchDims + i];
            }

            for (var r = 0; r < slice; r++)
            {
                bad += IndexRig.Same(got[OffOut + (t * slice) + r], data[OffData + (int)off + r]) ? 0 : 1;
            }
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
    public async Task GatherNdIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new GatherKernels(d);
        (int[] Shape, int[] Indices, int Batch)[] cases =
        [
            ([6], [4, 1], 0), ([6], [1], 0), ([3, 4], [5, 2], 0), ([3, 4], [5, 1], 0), ([3, 4, 5], [2, 2], 0), ([3, 4, 5], [2, 3, 3], 0), ([3, 4, 5], [6, 1], 0),
            ([2, 3, 4], [2, 3, 1], 1), ([2, 3, 4, 5], [2, 4, 2], 1), ([2, 3, 4, 5], [2, 3, 2], 2), ([4, 5, 6, 7, 2, 3], [9, 6], 0), ([100, 33], [257, 1], 0),
        ];
        long bad = 0, rows = 0;
        var seed = 200;
        foreach (var (shape, ishape, batch) in cases)
        {
            bad += await NdCase<float, int>(d, k, shape, ishape, batch, seed++);
            bad += await NdCase<Half, long>(d, k, shape, ishape, batch, seed++);
            bad += await NdCase<int, long>(d, k, shape, ishape, batch, seed++);
            bad += await NdCase<long, int>(d, k, shape, ishape, batch, seed++);
            bad += await NdCase<byte, int>(d, k, shape, ishape, batch, seed++);
            bad += await NdCase<sbyte, long>(d, k, shape, ishape, batch, seed++);
            rows += 6;
        }

        Rig.Row(d, "gather nd", "f32/f16/i32/i64/u8/i8", $"{rows} cases (tuple length 1 to 6, batch_dims 0 to 2, scalar and slice results, negative indices)", $"mismatching elements {bad}");
        Assert.Equal(0, bad);
    }
}
