namespace RecursiveAI.OneGpu.Kernels.Tests;

public sealed class CastTests
{
    static readonly long[] Longs =
    [
        0, 1, -1, 2, 255, 256, 16777215, 16777216, 16777217, 16777219, -16777217, 33554431, 33554433, 1L << 31, -(1L << 31), (1L << 31) - 1, (1L << 32) + 1, 4294967295L,
        (1L << 40) + 12345, -((1L << 40) + 12345), (1L << 53) + 1, (1L << 62) + (1L << 38) + 1, (1L << 62) + (1L << 38), long.MaxValue, long.MinValue, long.MinValue + 1,
        123456789012345678, -123456789012345678, 0x7FFFFF7FFFFFFFFF, 0x0000008000000001, 0x00FFFFFF80000000,
    ];

    static readonly float[] Floats =
    [
        0f, -0f, 0.5f, -0.5f, 0.99999994f, 1f, -1f, 1.5f, -1.5f, 2.9999998f, 255.5f, 16777216f, -16777216f, 2147483520f, 2147483648f, -2147483648f, 3e9f, -3e9f,
        4294967296f, 1099511627776f, 9.2233715e18f, -9.2233715e18f, 9.223372e18f, -9.223372e18f, 1e30f, -1e30f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 123456.789f,
    ];

    [Fact]
    public async Task CastsMatchTheCpuBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        var bad = 0;
        var checks = new int[1];

        var big = new List<long>(Longs);
        var rnd = Rig.Random(4000, 9, -1f, 1f);
        for (var i = 0; i < rnd.Length; i += 2)
        {
            big.Add((long)(((ulong)BitConverter.SingleToUInt32Bits(rnd[i]) << 32) | BitConverter.SingleToUInt32Bits(rnd[i + 1])) >> (i % 60));
        }

        var longs = big.ToArray();
        var ints = longs.Select(v => unchecked((int)v)).Concat([int.MinValue, int.MaxValue, 0, 1, -1, 16777217, -16777217, 2147483583]).ToArray();
        var floats = Floats.Concat(Rig.Random(3000, 3, -3e9f, 3e9f)).Concat(Rig.Random(2000, 4, -70000f, 70000f)).ToArray();

        // Offsets inside Run are odd on purpose so that half-precision words are shared with untouched neighbours.
        // long -> float, half, int; int -> float, half, long; float -> int, long, half; half -> float, int, long; same-type copies.
        bad += await Run<long, float>(d, k, longs, v => (float)v, (a, b) => Same(a, b), checks);
        bad += await Run<long, int>(d, k, longs, v => unchecked((int)v), (a, b) => a == b, checks);
        bad += await Run<long, Half>(d, k, longs, v => (Half)(float)v, (a, b) => a == b, checks);
        bad += await Run<int, float>(d, k, ints, v => (float)v, (a, b) => Same(a, b), checks);
        bad += await Run<int, long>(d, k, ints, v => v, (a, b) => a == b, checks);
        bad += await Run<int, Half>(d, k, ints, v => (Half)(float)v, (a, b) => a == b, checks);
        bad += await Run<float, int>(d, k, floats, SatInt, (a, b) => a == b, checks);
        bad += await Run<float, long>(d, k, floats, SatLong, (a, b) => a == b, checks);
        bad += await Run<float, Half>(d, k, floats, v => (Half)v, (a, b) => Half.IsNaN(a) ? Half.IsNaN(b) : a == b && BitConverter.HalfToUInt16Bits(a) == BitConverter.HalfToUInt16Bits(b), checks);
        var halves = floats.Select(v => (Half)v).Concat(Enumerable.Range(0, 65536).Select(i => BitConverter.UInt16BitsToHalf((ushort)i))).ToArray();
        bad += await Run<Half, float>(d, k, halves, v => (float)v, (a, b) => Same(a, b), checks);
        bad += await Run<Half, int>(d, k, halves, v => SatInt((float)v), (a, b) => a == b, checks);
        bad += await Run<Half, long>(d, k, halves, v => SatLong((float)v), (a, b) => a == b, checks);
        bad += await Run<int, int>(d, k, ints, v => v, (a, b) => a == b, checks);
        bad += await Run<Half, Half>(d, k, halves, v => v, (a, b) => Half.IsNaN(a) ? Half.IsNaN(b) : BitConverter.HalfToUInt16Bits(a) == BitConverter.HalfToUInt16Bits(b), checks);

        Rig.Row(d, "cast", "f32/f16/i32/i64", $"all 12 pairs plus copies, {checks[0]} values incl. every one of the 65536 half bit patterns", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    // The saturating conversions the kernel promises, written out: the platform's own float-to-integer cast differs between runtimes.
    static int SatInt(float v) => float.IsNaN(v) ? 0 : v >= 2147483648f ? int.MaxValue : v <= -2147483648f ? int.MinValue : (int)v;

    static long SatLong(float v) => float.IsNaN(v) ? 0 : v >= 9223372036854775808f ? long.MaxValue : v <= -9223372036854775808f ? long.MinValue : (long)v;

    static bool Same(float a, float b) => float.IsNaN(a) ? float.IsNaN(b) : BitConverter.SingleToUInt32Bits(a) == BitConverter.SingleToUInt32Bits(b);

    static int Padded<T>(int n) => typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n) : (typeof(T) == typeof(byte) ? GpuKernelBuffers.PaddedBoolLength(n) : n);

    [Fact]
    public async Task CastsToAndFromBoolMatchTheCpu()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new ElementwiseKernels(d);
        var longs = Longs.Concat([0, 0, 1, 0, 2]).ToArray();
        var ints = longs.Select(v => unchecked((int)v)).Concat([int.MinValue, 0, 0, 1]).ToArray();
        var floats = Floats.Concat(Rig.Random(500, 5, -2f, 2f).Select(v => MathF.Round(v))).Concat([float.NaN, -0f, 0f, float.Epsilon, -float.Epsilon]).ToArray();
        var halves = Enumerable.Range(0, 65536).Select(i => BitConverter.UInt16BitsToHalf((ushort)i)).ToArray();
        var bools = Enumerable.Range(0, 1001).Select(i => (byte)((i * 7 / 3) & 1)).ToArray();
        var checks = new int[1];
        var bad = 0;
        bad += await Run<long, byte>(d, k, longs, v => (byte)(v != 0 ? 1 : 0), (a, b) => a == b, checks);
        bad += await Run<int, byte>(d, k, ints, v => (byte)(v != 0 ? 1 : 0), (a, b) => a == b, checks);
        bad += await Run<float, byte>(d, k, floats, v => (byte)(v != 0 || float.IsNaN(v) ? 1 : 0), (a, b) => a == b, checks);
        bad += await Run<Half, byte>(d, k, halves, v => (byte)(v != Half.Zero || Half.IsNaN(v) ? 1 : 0), (a, b) => a == b, checks);
        bad += await Run<byte, long>(d, k, bools, v => v, (a, b) => a == b, checks);
        bad += await Run<byte, int>(d, k, bools, v => v, (a, b) => a == b, checks);
        bad += await Run<byte, float>(d, k, bools, v => v, (a, b) => Same(a, b), checks);
        bad += await Run<byte, Half>(d, k, bools, v => (Half)(float)v, (a, b) => a == b, checks);
        bad += await Run<byte, byte>(d, k, bools, v => v, (a, b) => a == b, checks);
        Rig.Row(d, "cast", "bool", $"bool to and from i32, i64, f32, f16 (all 65536 half patterns), {checks[0]} values, odd offsets", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    static async Task<int> Run<TFrom, TTo>(GpuDevice d, ElementwiseKernels k, TFrom[] input, Func<TFrom, TTo> convert, Func<TTo, TTo, bool> equal, int[] checks)
        where TFrom : unmanaged
        where TTo : unmanaged
    {
        const int OffS = 1, OffD = 3;
        var n = input.Length;
        var src = new TFrom[Padded<TFrom>(n + OffS)];
        input.CopyTo(src, OffS);
        var dstLen = Padded<TTo>(n + OffD + 1);
        using var bs = Rig.Upload(d, src);
        using var bd = d.CreateBuffer<TTo>(dstLen);
        Rig.Run(d, e => k.EncodeCast(e, bs, OffS, bd, OffD, n));
        var got = await Rig.DownloadAsync(bd);
        var bad = 0;
        for (var i = 0; i < n; i++)
        {
            // The CPU reference of an out-of-range float -> long or int conversion is the saturating one, as in the kernel.
            if (!equal(got[OffD + i], convert(input[i])))
            {
                if (bad < 5)
                {
                    Console.WriteLine($"cast {typeof(TFrom).Name}->{typeof(TTo).Name}: input {input[i]} gpu {got[OffD + i]} cpu {convert(input[i])}");
                }

                bad++;
            }
        }

        checks[0] += n;
        return bad;
    }
}
