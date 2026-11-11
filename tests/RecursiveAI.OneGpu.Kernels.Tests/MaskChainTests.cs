namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// The attention-mask chains that exporters emit for BERT-style models, recorded into one command buffer the way a session would, against the
/// same arithmetic on the CPU. Batches of 1 and 8, sequences of 128 and 512.
/// </summary>
public sealed class MaskChainTests
{
    public static IEnumerable<object[]> Sizes() => [[1, 128], [1, 512], [8, 128], [8, 512]];

    // attention_mask: ones, then padding; every batch row has its own length (and one row is all padding when b is large enough).
    static long[] Mask(int b, int s)
    {
        var m = new long[b * s];
        for (var r = 0; r < b; r++)
        {
            var len = r == 0 ? s : Math.Max(0, s - (r * 61 % (s / 2 + 1)) - (r == 7 ? s : 0));
            for (var i = 0; i < len; i++)
            {
                m[r * s + i] = 1;
            }
        }

        return m;
    }

    static int Mismatches(ReadOnlySpan<float> got, ReadOnlySpan<float> want) => Rig.BitMismatches(got, want);

    // attention_mask int64 [b,s] -> Unsqueeze x2 -> Cast to float32 -> Sub from 1 -> Mul by a large negative: the additive mask [b,1,1,s].
    [Theory]
    [MemberData(nameof(Sizes))]
    public async Task CastSubMulChainGivesTheAdditiveMask(int b, int s)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var ew = new ElementwiseKernels(d);
        var mask = Mask(b, s);
        var n = b * s;
        using var bm = Rig.Upload(d, mask);
        using var asFloat = d.CreateBuffer<float>(n);
        using var one = Rig.Scalar(d, 1f);
        using var scale = Rig.Scalar(d, -10000f);
        using var inverted = d.CreateBuffer<float>(n);
        using var additive = d.CreateBuffer<float>(n);
        int[] shape = [b, 1, 1, s];
        Rig.Run(d, e =>
        {
            ew.EncodeCast(e, bm, 0, asFloat, 0, n);
            ew.EncodeBinary(e, GpuBinaryOp.Sub, one, 0, [], asFloat, 0, shape, inverted, 0);
            ew.EncodeBinary(e, GpuBinaryOp.Mul, inverted, 0, shape, scale, 0, [], additive, 0);
        });
        var got = await Rig.DownloadAsync(additive);
        var want = mask.Select(m => (1f - (float)m) * -10000f).ToArray();
        var bad = Mismatches(got, want);
        Rig.Row(d, "mask chain", "i64->f32", $"Cast, Sub, Mul to additive mask b={b} s={s}", $"mismatches {bad} of {n}");
        Assert.Equal(0, bad);
    }

    // Equal(mask, 0) -> Where(isPad, lowest, 0): the variant exporters emit with a comparison.
    [Theory]
    [MemberData(nameof(Sizes))]
    public async Task EqualWhereChainGivesTheAdditiveMask(int b, int s)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var ew = new ElementwiseKernels(d);
        using var ik = new IntegerKernels(d);
        var mask = Mask(b, s);
        var n = b * s;
        using var bm = Rig.Upload(d, mask);
        using var zero64 = Rig.Scalar(d, 0L);
        using var isPad = d.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(n));
        using var low = Rig.Scalar(d, float.MinValue);
        using var zero = Rig.Scalar(d, 0f);
        using var additive = d.CreateBuffer<float>(n);
        int[] shape = [b, 1, 1, s];
        Rig.Run(d, e =>
        {
            ik.EncodeCompare(e, GpuCompareOp.Equal, bm, 0, shape, zero64, 0, [], isPad, 0);
            ew.EncodeWhere(e, isPad, 0, shape, low, 0, [], zero, 0, [], additive, 0);
        });
        var got = await Rig.DownloadAsync(additive);
        var want = mask.Select(m => m == 0 ? float.MinValue : 0f).ToArray();
        var bad = Mismatches(got, want);
        Rig.Row(d, "mask chain", "i64->bool->f32", $"Equal, Where to additive mask b={b} s={s}", $"mismatches {bad} of {n}");
        Assert.Equal(0, bad);
    }

    // The causal chain: Range(0, s) -> GreaterOrEqual(row, column) -> And with the padding mask -> Where, and the bool cast to float.
    [Theory]
    [MemberData(nameof(Sizes))]
    public async Task RangeGreaterOrEqualAndWhereChainGivesTheCausalMask(int b, int s)
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var ew = new ElementwiseKernels(d);
        using var ik = new IntegerKernels(d);
        var mask = Mask(b, s);
        var count = IntegerKernels.RangeLength(0L, s, 1L);
        Assert.Equal(s, count);
        using var bm = Rig.Upload(d, mask);
        using var positions = d.CreateBuffer<long>(s);
        using var zero64 = Rig.Scalar(d, 0L);
        using var padding = d.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(b * s));
        using var notPad = d.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(b * s));
        using var lower = d.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(s * s));
        using var allowed = d.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(b * s * s));
        using var low = Rig.Scalar(d, float.MinValue);
        using var zero = Rig.Scalar(d, 0f);
        using var additive = d.CreateBuffer<float>(b * s * s);
        using var allowedFloat = d.CreateBuffer<float>(b * s * s);
        Rig.Run(d, e =>
        {
            ik.EncodeRange(e, positions, 0, count, 0L, 1L);
            ik.EncodeCompare(e, GpuCompareOp.GreaterOrEqual, positions, 0, [1, 1, s, 1], positions, 0, [1, 1, 1, s], lower, 0);
            ik.EncodeCompare(e, GpuCompareOp.Equal, bm, 0, [b, 1, 1, s], zero64, 0, [], padding, 0);
            ik.EncodeNot(e, padding, 0, notPad, 0, b * s);
            ik.EncodeLogic(e, GpuLogicOp.And, lower, 0, [1, 1, s, s], notPad, 0, [b, 1, 1, s], allowed, 0);
            ew.EncodeWhere(e, allowed, 0, [b, 1, s, s], zero, 0, [], low, 0, [], additive, 0);
            ew.EncodeCast(e, allowed, 0, allowedFloat, 0, b * s * s);
        });
        var gotAdd = await Rig.DownloadAsync(additive);
        var gotAllowed = await Rig.DownloadAsync(allowedFloat);
        var wantAdd = new float[b * s * s];
        var wantAllowed = new float[b * s * s];
        for (var r = 0; r < b; r++)
        {
            for (var i = 0; i < s; i++)
            {
                for (var j = 0; j < s; j++)
                {
                    var ok = i >= j && mask[r * s + j] != 0;
                    wantAdd[(r * s + i) * s + j] = ok ? 0f : float.MinValue;
                    wantAllowed[(r * s + i) * s + j] = ok ? 1f : 0f;
                }
            }
        }

        var bad = Mismatches(gotAdd, wantAdd) + Mismatches(gotAllowed, wantAllowed);
        Rig.Row(d, "mask chain", "i64->bool->f32", $"Range, GreaterOrEqual, Not, And, Where, Cast b={b} s={s}", $"mismatches {bad} of {2 * b * s * s}");
        Assert.Equal(0, bad);
    }
}
