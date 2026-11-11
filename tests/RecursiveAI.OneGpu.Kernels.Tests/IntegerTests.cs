using System.Numerics;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Integer and bool kernels against plain C# on the CPU, bit for bit.</summary>
public sealed class IntegerTests
{
    // ---- reference arithmetic (the semantics onemodel's CPU kernels have) ----

    static T Arith<T>(GpuIntegerOp op, T a, T b)
        where T : IBinaryInteger<T>, IMinMaxValue<T>
    {
        var minusOne = T.Zero - T.One;
        switch (op)
        {
            case GpuIntegerOp.Add: return unchecked(a + b);
            case GpuIntegerOp.Sub: return unchecked(a - b);
            case GpuIntegerOp.Mul: return unchecked(a * b);
            case GpuIntegerOp.Div: return b == T.Zero ? T.Zero : b == minusOne ? unchecked(T.Zero - a) : a / b;
            case GpuIntegerOp.Fmod: return b == T.Zero || b == minusOne ? T.Zero : a % b;
            case GpuIntegerOp.Mod:
            {
                var r = Arith(GpuIntegerOp.Fmod, a, b);
                return r != T.Zero && T.IsNegative(r) != T.IsNegative(b) ? unchecked(r + b) : r;
            }

            case GpuIntegerOp.Min: return a < b ? a : b;
            default: return a > b ? a : b;
        }
    }

    static bool Cmp<T>(GpuCompareOp op, T a, T b)
        where T : IComparable<T> =>
        op switch
        {
            GpuCompareOp.Equal => a.CompareTo(b) == 0,
            GpuCompareOp.Less => a.CompareTo(b) < 0,
            GpuCompareOp.Greater => a.CompareTo(b) > 0,
            GpuCompareOp.LessOrEqual => a.CompareTo(b) <= 0,
            _ => a.CompareTo(b) >= 0,
        };

    // ---- data ----

    static long[] Special64 =>
    [
        0, 1, -1, 2, -2, 3, 7, -7, 10, 255, 256, 65535, 65536, int.MaxValue, int.MinValue, (long)int.MaxValue + 1, (long)int.MinValue - 1, uint.MaxValue, 1L << 32, -(1L << 32), (1L << 32) + 5,
        3000000000L, -3000000000L, 1L << 40, (1L << 40) + 12345, -(1L << 40), 1L << 62, long.MaxValue, long.MinValue, long.MinValue + 1, long.MaxValue - 1, 0x7FFFFFFF00000000, -0x1234567890L,
    ];

    // Values of every magnitude: the shift spreads results between 0 and the full 64 bits, so quotients and remainders are interesting.
    static long[] Values64(int n, int seed)
    {
        var r = Rig.Random(2 * n, seed, -1f, 1f);
        var v = new long[n];
        for (var i = 0; i < n; i++)
        {
            var bits = ((ulong)BitConverter.SingleToUInt32Bits(r[2 * i]) << 32) | BitConverter.SingleToUInt32Bits(r[2 * i + 1]);
            v[i] = (long)bits >> ((i * 7 + seed) % 63);
            if (i % 11 == 3)
            {
                v[i] = Special64[(i + seed) % Special64.Length];
            }
        }

        return v;
    }

    static int[] Values32(int n, int seed) => Values64(n, seed + 100).Select((x, i) => i % 5 == 2 ? new[] { 0, 1, -1, int.MinValue, int.MaxValue, 7, -7 }[(i + seed) % 7] : unchecked((int)x)).ToArray();

    static int Count(int[] s) => s.Aggregate(1, (x, y) => x * y);

    // Index into a broadcast operand of shape s for output element i of shape rs.
    static int Idx(int[] s, int[] rs, int i)
    {
        var idx = 0;
        var stride = 1;
        var rem = i;
        for (var d = rs.Length - 1; d >= 0; d--)
        {
            var c = rem % rs[d];
            rem /= rs[d];
            var sd = d - (rs.Length - s.Length);
            if (sd >= 0)
            {
                if (s[sd] != 1)
                {
                    idx += c * stride;
                }

                stride *= s[sd];
            }
        }

        return idx;
    }

    static T[] Pad<T>(T[] v, int lead)
        where T : unmanaged
    {
        var len = v.Length + lead;
        len = typeof(T) == typeof(byte) ? GpuKernelBuffers.PaddedBoolLength(Math.Max(1, len)) : Math.Max(1, len);
        var r = new T[len];
        v.CopyTo(r, lead);
        return r;
    }

    static IEnumerable<(int[] A, int[] B)> Cases() => ElementwiseTests.BroadcastCases().Where((_, i) => i % 4 == 0)
        .Concat([([0], [1]), ([2, 0, 3], [3]), ([], [5]), ([7], []), ([1], [1])]);

    // ---- binary arithmetic ----

    static async Task<int> RunBinary<T>(GpuDevice d, IntegerKernels k, GpuIntegerOp op, T[] a, int[] sa, T[] b, int[] sb)
        where T : unmanaged, IBinaryInteger<T>, IMinMaxValue<T>
    {
        var rs = ElementwiseKernels.BroadcastShape(sa, sb);
        var n = Count(rs);
        const int OffA = 1, OffB = 2, OffO = 3;
        using var ba = Rig.Upload(d, Pad(a, OffA));
        using var bb = Rig.Upload(d, Pad(b, OffB));
        using var bo = Rig.Upload(d, new T[n + OffO + 1]);
        Rig.Run(d, e => k.EncodeBinary(e, op, ba, OffA, sa, bb, OffB, sb, bo, OffO));
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var i = 0; i < n; i++)
        {
            var want = Arith(op, a[Idx(sa, rs, i)], b[Idx(sb, rs, i)]);
            if (got[OffO + i] != want)
            {
                if (bad < 3)
                {
                    Console.WriteLine($"{op} {typeof(T).Name}: {a[Idx(sa, rs, i)]} , {b[Idx(sb, rs, i)]} gpu {got[OffO + i]} cpu {want}");
                }

                bad++;
            }
        }

        // The element after the result must not be touched.
        bad += got[OffO + n] == default ? 0 : 1;
        return bad;
    }

    [Fact]
    public async Task ArithmeticMatchesTheCpuBitForBit()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        foreach (var is64 in new[] { false, true })
        {
            foreach (var op in Enum.GetValues<GpuIntegerOp>())
            {
                var bad = 0;
                var checks = 0;
                var seed = 1;
                foreach (var (sa, sb) in Cases())
                {
                    var na = Count(sa);
                    var nb = Count(sb);
                    var rs = ElementwiseKernels.BroadcastShape(sa, sb);
                    if (is64)
                    {
                        bad += await RunBinary(d, k, op, Values64(na, seed), sa, Values64(nb, seed + 1), sb);
                    }
                    else
                    {
                        bad += await RunBinary(d, k, op, Values32(na, seed), sa, Values32(nb, seed + 1), sb);
                    }

                    checks += Count(rs);
                    seed++;
                }

                Rig.Row(d, "integer", is64 ? "i64" : "i32", $"{op}, {checks} elements over broadcast shapes incl. empty", $"mismatches {bad}");
                Assert.Equal(0, bad);
            }
        }
    }

    [Fact]
    public async Task DivisionAndRemainderHandleTheAwkwardPairs()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        // Every pair of the special values, so zero, -1, the minimum and the carry cases of the shift-and-subtract division are all hit.
        var all = Special64;
        var a = all.SelectMany(x => all.Select(_ => x)).ToArray();
        var b = all.SelectMany(_ => all).ToArray();
        var a32 = a.Select(x => unchecked((int)x)).ToArray();
        var b32 = b.Select(x => unchecked((int)x)).ToArray();
        var bad = 0;
        foreach (var op in new[] { GpuIntegerOp.Div, GpuIntegerOp.Mod, GpuIntegerOp.Fmod, GpuIntegerOp.Mul, GpuIntegerOp.Sub })
        {
            bad += await RunBinary(d, k, op, a, [a.Length], b, [b.Length]);
            bad += await RunBinary(d, k, op, a32, [a.Length], b32, [b.Length]);
        }

        Rig.Row(d, "integer", "i32/i64", $"Div, Mod, Fmod, Mul, Sub over all {a.Length} pairs of special values", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    // ---- comparison, logic, unary ----

    static async Task<int> RunCompare<T>(GpuDevice d, IntegerKernels k, GpuCompareOp op, T[] a, int[] sa, T[] b, int[] sb, Func<T, T, bool> want)
        where T : unmanaged
    {
        var rs = ElementwiseKernels.BroadcastShape(sa, sb);
        var n = Count(rs);
        const int OffA = 1, OffB = 2, OffO = 3;
        using var ba = Rig.Upload(d, Pad(a, OffA));
        using var bb = Rig.Upload(d, Pad(b, OffB));
        var init = Enumerable.Repeat((byte)0x55, GpuKernelBuffers.PaddedBoolLength(n + OffO + 4)).ToArray();
        using var bo = Rig.Upload(d, init);
        Rig.Run(d, e => k.EncodeCompare(e, op, ba, OffA, sa, bb, OffB, sb, bo, OffO));
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var i = 0; i < got.Length; i++)
        {
            var expected = i >= OffO && i < OffO + n ? (want(a[Idx(sa, rs, i - OffO)], b[Idx(sb, rs, i - OffO)]) ? (byte)1 : (byte)0) : (byte)0x55;
            bad += got[i] == expected ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public async Task ComparisonsMatchTheCpuAndLeaveNeighbouringBytesAlone()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        foreach (var kind in new[] { "i32", "i64", "bool" })
        {
            foreach (var op in Enum.GetValues<GpuCompareOp>())
            {
                var bad = 0;
                var seed = 5;
                foreach (var (sa, sb) in Cases())
                {
                    var na = Count(sa);
                    var nb = Count(sb);
                    if (kind == "i64")
                    {
                        // Equal values are common so that Equal and the non-strict orders see both outcomes.
                        var a = Values64(na, seed).Select(x => x % 4 == 0 ? x : x >> 33).ToArray();
                        var b = Values64(nb, seed + 1).Select(x => x % 3 == 0 ? x >> 33 : x).ToArray();
                        if (na > 0 && nb > 0)
                        {
                            b[0] = a[0];
                        }

                        bad += await RunCompare(d, k, op, a, sa, b, sb, (x, y) => Cmp(op, x, y));
                    }
                    else if (kind == "i32")
                    {
                        var a = Values32(na, seed);
                        var b = Values32(nb, seed + 1);
                        if (na > 0 && nb > 0)
                        {
                            b[0] = a[0];
                        }

                        bad += await RunCompare(d, k, op, a, sa, b, sb, (x, y) => Cmp(op, x, y));
                    }
                    else
                    {
                        var a = Values64(na, seed).Select(x => (byte)(x & 1)).ToArray();
                        var b = Values64(nb, seed + 1).Select(x => (byte)((x >> 3) & 1)).ToArray();
                        bad += await RunCompare(d, k, op, a, sa, b, sb, (x, y) => Cmp(op, x, y));
                    }

                    seed++;
                }

                Rig.Row(d, "integer", kind, $"compare {op} to bool, odd offsets, neighbours checked", $"mismatches {bad}");
                Assert.Equal(0, bad);
            }
        }
    }

    [Fact]
    public async Task LogicAndNotMatchTheCpu()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        var bad = 0;
        foreach (var op in Enum.GetValues<GpuLogicOp>())
        {
            var seed = 9;
            foreach (var (sa, sb) in Cases())
            {
                var a = Values64(Count(sa), seed).Select(x => (byte)(x & 1)).ToArray();
                var b = Values64(Count(sb), seed + 3).Select(x => (byte)((x >> 5) & 1)).ToArray();
                var rs = ElementwiseKernels.BroadcastShape(sa, sb);
                var n = Count(rs);
                using var ba = Rig.Upload(d, Pad(a, 1));
                using var bb = Rig.Upload(d, Pad(b, 2));
                using var bo = Rig.Upload(d, Enumerable.Repeat((byte)0x55, GpuKernelBuffers.PaddedBoolLength(n + 7)).ToArray());
                Rig.Run(d, e => k.EncodeLogic(e, op, ba, 1, sa, bb, 2, sb, bo, 3));
                var got = await Rig.DownloadAsync(bo);
                for (var i = 0; i < got.Length; i++)
                {
                    byte expected = 0x55;
                    if (i >= 3 && i < 3 + n)
                    {
                        var (x, y) = (a[Idx(sa, rs, i - 3)] != 0, b[Idx(sb, rs, i - 3)] != 0);
                        expected = (byte)((op switch { GpuLogicOp.And => x && y, GpuLogicOp.Or => x || y, _ => x ^ y }) ? 1 : 0);
                    }

                    bad += got[i] == expected ? 0 : 1;
                }

                seed++;
            }
        }

        // Not, and the integer unary functions.
        foreach (var n in new[] { 0, 1, 3, 4, 5, 17, 1000 })
        {
            var x = Values64(n, 4).Select(v => (byte)(v & 1)).ToArray();
            using var bx = Rig.Upload(d, Pad(x, 1));
            using var by = Rig.Upload(d, Enumerable.Repeat((byte)0x55, GpuKernelBuffers.PaddedBoolLength(n + 6)).ToArray());
            Rig.Run(d, e => k.EncodeNot(e, bx, 1, by, 2, n));
            var got = await Rig.DownloadAsync(by);
            for (var i = 0; i < got.Length; i++)
            {
                var expected = i >= 2 && i < 2 + n ? (byte)(x[i - 2] == 0 ? 1 : 0) : (byte)0x55;
                bad += got[i] == expected ? 0 : 1;
            }

            var l = Values64(n, 6).Concat([long.MinValue]).ToArray();
            var i32 = l.Select(v => unchecked((int)v)).Concat([int.MinValue]).ToArray();
            foreach (var op in Enum.GetValues<GpuIntegerUnaryOp>())
            {
                using var bl = Rig.Upload(d, Pad(l, 1));
                using var bol = d.CreateBuffer<long>(l.Length + 3);
                Rig.Run(d, e => k.EncodeUnary(e, op, bl, 1, bol, 2, l.Length));
                var gl = await Rig.DownloadAsync(bol);
                using var bi = Rig.Upload(d, Pad(i32, 1));
                using var boi = d.CreateBuffer<int>(i32.Length + 3);
                Rig.Run(d, e => k.EncodeUnary(e, op, bi, 1, boi, 2, i32.Length));
                var gi = await Rig.DownloadAsync(boi);
                for (var j = 0; j < l.Length; j++)
                {
                    var wl = op == GpuIntegerUnaryOp.Neg ? unchecked(-l[j]) : (l[j] < 0 ? unchecked(-l[j]) : l[j]);
                    var wi = op == GpuIntegerUnaryOp.Neg ? unchecked(-i32[j]) : (i32[j] < 0 ? unchecked(-i32[j]) : i32[j]);
                    bad += (gl[2 + j] == wl ? 0 : 1) + (gi[2 + j] == wi ? 0 : 1);
                }
            }
        }

        Rig.Row(d, "integer", "bool/i32/i64", "And, Or, Xor over broadcast shapes; Not; Neg and Abs incl. the minimum values", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    // ---- generators ----

    [Fact]
    public async Task FillAndRangeMatchTheCpu()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        var bad = 0;
        var checks = 0;
        foreach (var n in new[] { 0, 1, 2, 3, 4, 5, 7, 255, 256, 257, 1000, 70000 })
        {
            const int Off = 3;

            async Task<int> Fill<T>(T value, T sentinel)
                where T : unmanaged
            {
                var len = typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n + Off + 5) : n + Off + 5;
                var init = new T[len];
                Array.Fill(init, sentinel);
                using var b = Rig.Upload(d, init);
                Rig.Run(d, e => k.EncodeFill(e, b, Off, n, value));
                var got = await Rig.DownloadAsync(b);
                var wrong = 0;
                for (var i = 0; i < got.Length; i++)
                {
                    var want = i >= Off && i < Off + n ? value : sentinel;
                    wrong += System.Runtime.InteropServices.MemoryMarshal.AsBytes<T>([got[i]]).SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes<T>([want])) ? 0 : 1;
                }

                checks += len;
                return wrong;
            }

            bad += await Fill(1.5f, -3f);
            bad += await Fill(float.NaN, 2f);
            bad += await Fill((Half)(-2.5f), (Half)7f);
            bad += await Fill(-123456, 99);
            bad += await Fill(4000000000u, 5u);
            bad += await Fill(-(1L << 40) - 7, 1L << 50);

            {
                var init = Enumerable.Repeat((byte)0x55, GpuKernelBuffers.PaddedBoolLength(n + Off + 5)).ToArray();
                foreach (var v in new[] { true, false })
                {
                    using var b = Rig.Upload(d, init);
                    Rig.Run(d, e => k.EncodeFillBool(e, b, Off, n, v));
                    var got = await Rig.DownloadAsync(b);
                    for (var i = 0; i < got.Length; i++)
                    {
                        bad += got[i] == (i >= Off && i < Off + n ? (v ? 1 : 0) : 0x55) ? 0 : 1;
                    }

                    checks += got.Length;
                }
            }

            // Range: int32, int64 (starts and steps past 2^32, negative steps), float32 with fractional steps.
            foreach (var (s, delta) in new (long, long)[] { (0, 1), (5, -1), (-100, 7), (1L << 33, 3), ((1L << 40) + 5, -(1L << 35) - 11), (long.MaxValue - 5, 1), (long.MinValue + 9, -1), (0, 0) })
            {
                {
                    var (s32, d32) = (unchecked((int)s), unchecked((int)delta));
                    using var b = Rig.Upload(d, new int[n + Off + 2]);
                    Rig.Run(d, e => k.EncodeRange(e, b, Off, n, s32, d32));
                    var got = await Rig.DownloadAsync(b);
                    for (var i = 0; i < n; i++)
                    {
                        bad += got[Off + i] == unchecked(s32 + i * d32) ? 0 : 1;
                    }

                    bad += got[Off + n] == 0 ? 0 : 1;
                    checks += n;
                }

                {
                    using var b = Rig.Upload(d, new long[n + Off + 2]);
                    Rig.Run(d, e => k.EncodeRange(e, b, Off, n, s, delta));
                    var got = await Rig.DownloadAsync(b);
                    for (var i = 0; i < n; i++)
                    {
                        bad += got[Off + i] == unchecked(s + i * delta) ? 0 : 1;
                    }

                    bad += got[Off + n] == 0 ? 0 : 1;
                    checks += n;
                }
            }

            foreach (var (s, delta) in new (float, float)[] { (0f, 1f), (10f, -0.1f), (-3.5f, 0.3f), (1e7f, 1.5f), (0.1f, 0.7f), (-1f, 0f), (123456.789f, -0.001f) })
            {
                using var b = Rig.Upload(d, new float[n + Off + 2]);
                Rig.Run(d, e => k.EncodeRange(e, b, Off, n, s, delta));
                var got = await Rig.DownloadAsync(b);
                for (var i = 0; i < n; i++)
                {
                    var p = (float)i * delta;
                    var want = s + p;
                    bad += BitConverter.SingleToUInt32Bits(got[Off + i]) == BitConverter.SingleToUInt32Bits(want) ? 0 : 1;
                }

                checks += n;
            }
        }

        Rig.Row(d, "integer", "fill/range", $"Fill f32 f16 i32 u32 i64 bool and Range i32 i64 f32, {checks} elements, odd offsets, sentinels checked", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public void RangeLengthFollowsOnnx()
    {
        Assert.Equal(0, IntegerKernels.RangeLength(5, 5, 1));
        Assert.Equal(0, IntegerKernels.RangeLength(5, 0, 1));
        Assert.Equal(5, IntegerKernels.RangeLength(0, 5, 1));
        Assert.Equal(3, IntegerKernels.RangeLength(0, 5, 2));
        Assert.Equal(5, IntegerKernels.RangeLength(5, 0, -1));
        Assert.Equal(2, IntegerKernels.RangeLength(5L, 0L, -3L));
        Assert.Equal(10, IntegerKernels.RangeLength(0f, 1f, 0.1f));
        Assert.Equal(0, IntegerKernels.RangeLength(3f, 1f, 1f));
        Assert.Equal(1, IntegerKernels.RangeLength(long.MinValue, long.MinValue + 1, long.MaxValue));
        Assert.Throws<GpuKernelShapeException>(() => IntegerKernels.RangeLength(0, 5, 0));
        Assert.Throws<GpuKernelShapeException>(() => IntegerKernels.RangeLength(0f, 5f, 0f));
        Assert.Throws<GpuKernelShapeException>(() => IntegerKernels.RangeLength(long.MinValue, long.MaxValue, 1L));
        Assert.Throws<GpuKernelShapeException>(() => IntegerKernels.RangeLength((byte)0, (byte)5, (byte)1));
    }

    // ---- Where over integer words, and the typed errors ----

    [Fact]
    public async Task WhereCopiesIntegerWordsAndPairs()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var e1 = new ElementwiseKernels(d);
        var bad = 0;
        var cases = 0;
        foreach (var (sa, sb) in Cases().Where((_, i) => i % 2 == 0))
        {
            var rs = ElementwiseKernels.BroadcastShape(sa, sb);
            var n = Count(rs);
            var cond = Values64(n, cases + 2).Select(v => (byte)(v & 1)).ToArray();
            using var bc = Rig.Upload(d, Rig.PadBool(cond));
            var a32 = Values32(Count(sa), cases);
            var b32 = Values32(Count(sb), cases + 1);
            using (var ba = Rig.Upload(d, a32.Length == 0 ? new int[1] : a32))
            using (var bb = Rig.Upload(d, b32.Length == 0 ? new int[1] : b32))
            using (var bo = d.CreateBuffer<int>(Math.Max(1, n)))
            {
                Rig.Run(d, e => e1.EncodeWhere(e, bc, 0, rs, ba, 0, sa, bb, 0, sb, bo, 0));
                var got = await Rig.DownloadAsync(bo, n);
                for (var i = 0; i < n; i++)
                {
                    bad += got[i] == (cond[i] != 0 ? a32[Idx(sa, rs, i)] : b32[Idx(sb, rs, i)]) ? 0 : 1;
                }
            }

            var a64 = Values64(Count(sa), cases);
            var b64 = Values64(Count(sb), cases + 1);
            if (rs.Length <= 5 && sa.Length <= 5 && sb.Length <= 5)
            {
                using var ba = Rig.Upload(d, a64.Length == 0 ? new long[1] : a64);
                using var bb = Rig.Upload(d, b64.Length == 0 ? new long[1] : b64);
                using var bo = d.CreateBuffer<long>(Math.Max(1, n));
                try
                {
                    Rig.Run(d, e => e1.EncodeWhere(e, bc, 0, rs, ba, 0, sa, bb, 0, sb, bo, 0));
                }
                catch (GpuKernelShapeException ex) when (ex.Argument == "shape")
                {
                    // Eight dimensions after merging the word pair do not fit; a typed error, not a bad dispatch.
                    cases++;
                    continue;
                }

                var got = await Rig.DownloadAsync(bo, n);
                for (var i = 0; i < n; i++)
                {
                    bad += got[i] == (cond[i] != 0 ? a64[Idx(sa, rs, i)] : b64[Idx(sb, rs, i)]) ? 0 : 1;
                }
            }

            cases++;
        }

        Rig.Row(d, "integer", "i32/i64", $"where with bool condition, {cases} broadcast cases", $"mismatches {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public void BadArgumentsThrowTypedErrors()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new IntegerKernels(d);
        using var a = d.CreateBuffer<int>(8);
        using var f = d.CreateBuffer<float>(8);
        using var bools = d.CreateBuffer<byte>(8);
        using var odd = d.CreateBuffer<byte>(6);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuIntegerOp.Add, a, 0, [3], a, 0, [4], a, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuIntegerOp.Add, a, 0, [9], a, 0, [9], a, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, (GpuIntegerOp)99, a, 0, [8], a, 0, [8], a, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuIntegerOp.Add, f, 0, [8], f, 0, [8], f, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeBinary(enc, GpuIntegerOp.Add, bools, 0, [8], bools, 0, [8], bools, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeCompare(enc, GpuCompareOp.Equal, a, 0, [8], a, 0, [8], odd, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeCompare(enc, GpuCompareOp.Equal, a, 0, [8], a, -1, [8], bools, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeFill(enc, a, 4, 5, 1));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeFillBool(enc, bools, 0, -1, true));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeLogic(enc, (GpuLogicOp)3, bools, 0, [8], bools, 0, [8], bools, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IntegerKernels(d, 0));
        d.Submit(enc);
        d.WaitIdle();
    }
}
