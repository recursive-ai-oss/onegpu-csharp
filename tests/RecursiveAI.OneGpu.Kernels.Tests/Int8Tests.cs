using System.Text;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>8-bit products and dynamic quantization against CPU references that reproduce onemodel's integer semantics, bit for bit.</summary>
public sealed class Int8Tests
{
    // The model shapes: projection and feed-forward products of BERT and MiniLM at 128 tokens, batch 1 and batch 8 (rows folded together).
    static readonly (string Name, int M, int N, int K)[] ModelShapes =
    [
        ("minilm qkv/proj b1x128", 128, 384, 384),
        ("minilm ffn-up b1x128", 128, 1536, 384),
        ("minilm ffn-down b1x128", 128, 384, 1536),
        ("bert proj b1x128", 128, 768, 768),
        ("bert ffn-up b1x128", 128, 3072, 768),
        ("bert ffn-down b1x128", 128, 768, 3072),
        ("minilm ffn-up b8x128", 1024, 1536, 384),
        ("bert proj b8x128", 1024, 768, 768),
        ("bert ffn-up b8x128", 1024, 3072, 768),
        ("bert ffn-down b8x128", 1024, 768, 3072),
    ];

    // Every way the kernel can run on the device: as chosen, the portable variant, and each tile of the packed-dot variant forced.
    static List<(string Name, Int8Gemm Gemm)> Configs(GpuDevice d, bool all)
    {
        var list = new List<(string, Int8Gemm)> { ("default", new Int8Gemm(d)), ("portable", new Int8Gemm(d, new Int8GemmOptions { AllowPackedDot = false })) };
        if (all)
        {
            list.Add(("default tile 32", new Int8Gemm(d, new Int8GemmOptions { Tile = 32 })));
            list.Add(("default tile 64", new Int8Gemm(d, new Int8GemmOptions { Tile = 64 })));
        }

        return list;
    }

    static byte[] Bytes(int length, int seed)
    {
        var a = new byte[length];
        var s = (uint)seed * 2654435761u + 7u;
        for (var i = 0; i < a.Length; i++)
        {
            s = (s * 1664525u) + 1013904223u;
            a[i] = (byte)(s >> 24);
        }

        return a;
    }

    static int Val(byte b, bool isSigned) => isSigned ? (sbyte)b : b;

    // C[batch, m, n] = sum_k (A - za) * (B - zb) with 32-bit wrap-around, rows in parallel.
    static int[] Reference(Int8GemmProblem p, byte[] a, bool aSigned, byte[] b, bool bSigned, byte[]? za, bool zaVec, byte[]? zb, bool zbVec)
    {
        var c = new int[p.Batch * p.M * p.N];
        var ldb = p.Ldb == 0 ? (p.TransposeB ? p.K : p.N) : p.Ldb;
        var lda = p.Lda == 0 ? p.K : p.Lda;
        Parallel.For(0, p.Batch * p.M, row =>
        {
            var bt = row / p.M;
            var m = row % p.M;
            var zA = za is null ? 0 : Val(za[zaVec ? m : 0], aSigned);
            var arow = new int[p.K];
            for (var k = 0; k < p.K; k++)
            {
                arow[k] = Val(a[p.OffsetA + (bt * p.StrideA) + (m * lda) + k], aSigned) - zA;
            }

            for (var n = 0; n < p.N; n++)
            {
                var zB = zb is null ? 0 : Val(zb[zbVec ? n : 0], bSigned);
                var acc = 0;
                for (var k = 0; k < p.K; k++)
                {
                    var bv = Val(b[p.OffsetB + (bt * p.StrideB) + (p.TransposeB ? (n * ldb) + k : (k * ldb) + n)], bSigned) - zB;
                    acc = unchecked(acc + (arow[k] * bv));
                }

                c[(row * p.N) + n] = acc;
            }
        });
        return c;
    }

    static int Requant(int acc, float mult, int zero, bool unsigned)
    {
        var lo = unsigned ? 0 : -128;
        var hi = unsigned ? 255 : 127;
        var v = (float)acc * mult;
        if (float.IsNaN(v))
        {
            return lo;
        }

        var r = MathF.Round(v);
        r = Math.Clamp(r, lo - zero, hi - zero);
        return (int)r + zero;
    }

    sealed class Case
    {
        public string Name = "";
        public Int8GemmProblem P;
        public bool ASigned, BSigned;
        public int ZaMode, ZbMode;     // 0 none, 1 scalar, 2 vector
        public int Fill = -1;          // 0..255: every byte of A, B and the zero points holds this value (extremes), -1 random
    }

    static T[] Cast<T>(byte[] data)
        where T : unmanaged
    {
        var padded = new byte[(data.Length + 3) & ~3];
        data.CopyTo(padded, 0);
        return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(padded).ToArray();
    }

    static async Task<(int Bad, string Variant)> RunInt32(GpuDevice d, Int8Gemm gemm, Case c, int seed)
    {
        var p = c.P;
        var lda = p.Lda == 0 ? p.K : p.Lda;
        var ldb = p.Ldb == 0 ? (p.TransposeB ? p.K : p.N) : p.Ldb;
        var lenA = p.OffsetA + ((p.Batch - 1) * p.StrideA) + (p.M * lda) + 4;
        var lenB = p.OffsetB + ((p.Batch - 1) * p.StrideB) + ((p.TransposeB ? p.N : p.K) * ldb) + 4;
        var a = Bytes(lenA, seed);
        var b = Bytes(lenB, seed + 1);
        var za = c.ZaMode == 0 ? null : Bytes(1 + (c.ZaMode == 2 ? p.M : 0), seed + 2);
        var zb = c.ZbMode == 0 ? null : Bytes(1 + (c.ZbMode == 2 ? p.N : 0), seed + 3);
        if (c.Fill >= 0)
        {
            Array.Fill(a, (byte)c.Fill);
            Array.Fill(b, (byte)(c.Fill ^ 0xFF));
            za?.AsSpan().Fill((byte)c.Fill);
            zb?.AsSpan().Fill((byte)(c.Fill ^ 0x80));
        }
        var want = Reference(p, a, c.ASigned, b, c.BSigned, za, c.ZaMode == 2, zb, c.ZbMode == 2);
        var count = want.Length;
        var sentinel = 0x5A5A5A5A;
        var outArr = new int[p.OffsetC + count + 2];
        Array.Fill(outArr, sentinel);
        using var bc = d.CreateBuffer<int>(outArr);
        using var ba8 = d.CreateBuffer<byte>(a);
        using var bb8 = d.CreateBuffer<byte>(b);
        using var bza = za is null ? null : d.CreateBuffer<byte>(za);
        using var bzb = zb is null ? null : d.CreateBuffer<byte>(zb);
        using var ba = d.CreateBuffer<sbyte>(Cast<sbyte>(a));
        using var bb = d.CreateBuffer<sbyte>(Cast<sbyte>(b));
        using var bzas = za is null ? null : d.CreateBuffer<sbyte>(Cast<sbyte>(za));
        using var bzbs = zb is null ? null : d.CreateBuffer<sbyte>(Cast<sbyte>(zb));
        Rig.Run(d, e => Dispatch(gemm, e, p, c, ba8, ba, bb8, bb, bza, bzas, bzb, bzbs, bc));
        var got = await Rig.DownloadAsync(bc);
        var bad = 0;
        var shown = 0;
        for (var i = 0; i < count; i++)
        {
            if (got[p.OffsetC + i] != want[i])
            {
                bad++;
                if (shown++ < 6)
                {
                    Console.WriteLine($"  differs at row {i / p.N} col {i % p.N} (flat {i}): got {got[p.OffsetC + i]} want {want[i]} [{gemm.Variant}]");
                }
            }
        }

        for (var i = 0; i < p.OffsetC; i++)
        {
            bad += got[i] == sentinel ? 0 : 1;
        }

        for (var i = p.OffsetC + count; i < got.Length; i++)
        {
            bad += got[i] == sentinel ? 0 : 1;
        }

        return (bad, gemm.Variant);
    }

    static void Dispatch(Int8Gemm gemm, GpuCommandEncoder e, Int8GemmProblem p, Case c, GpuBuffer<byte> au, GpuBuffer<sbyte> ai, GpuBuffer<byte> bu, GpuBuffer<sbyte> bi, GpuBuffer<byte>? zau, GpuBuffer<sbyte>? zai, GpuBuffer<byte>? zbu, GpuBuffer<sbyte>? zbi, GpuBuffer<int> cOut)
    {
        var zaV = c.ZaMode == 2;
        var zbV = c.ZbMode == 2;
        switch (c.ASigned, c.BSigned)
        {
            case (false, false):
                gemm.EncodeInt32(e, p, au, bu, cOut, zau is null ? null : new Int8Operand<byte>(zau, 0, zaV), zbu is null ? null : new Int8Operand<byte>(zbu, 0, zbV));
                break;
            case (false, true):
                gemm.EncodeInt32(e, p, au, bi, cOut, zau is null ? null : new Int8Operand<byte>(zau, 0, zaV), zbi is null ? null : new Int8Operand<sbyte>(zbi, 0, zbV));
                break;
            case (true, false):
                gemm.EncodeInt32(e, p, ai, bu, cOut, zai is null ? null : new Int8Operand<sbyte>(zai, 0, zaV), zbu is null ? null : new Int8Operand<byte>(zbu, 0, zbV));
                break;
            default:
                gemm.EncodeInt32(e, p, ai, bi, cOut, zai is null ? null : new Int8Operand<sbyte>(zai, 0, zaV), zbi is null ? null : new Int8Operand<sbyte>(zbi, 0, zbV));
                break;
        }
    }

    static Int8GemmProblem Problem(int m, int n, int k, int batch = 1, bool transB = false, int offA = 0, int offB = 0, int offC = 0, int ldaExtra = 0, int ldbExtra = 0, bool shareB = false)
    {
        var p = new Int8GemmProblem(m, n, k) { Batch = batch, TransposeB = transB, OffsetA = offA, OffsetB = offB, OffsetC = offC };
        p.Lda = k + ldaExtra;
        p.Ldb = (transB ? k : n) + ldbExtra;
        p.StrideA = m * p.Lda;
        p.StrideB = shareB ? 0 : (transB ? n : k) * p.Ldb;
        if (batch == 1)
        {
            p.StrideA = 0;
            p.StrideB = 0;
        }

        return p;
    }

    [Fact]
    public async Task EdgeShapesAreBitExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var cases = new List<Case>();
        (int M, int N, int K)[] shapes = [(1, 1, 1), (1, 1, 0), (3, 5, 7), (64, 64, 16), (65, 63, 17), (1, 257, 33), (130, 1, 31), (17, 129, 1), (2, 2, 4096), (33, 31, 300), (4, 4, 32), (7, 9, 31), (40, 40, 65), (5, 3, 63), (33, 40, 96)];
        var variants = 0;
        foreach (var (m, n, k) in shapes)
        {
            for (var s = 0; s < 4; s++)
            {
                for (var za = 0; za < 3; za++)
                {
                    var zb = (za + s) % 3;
                    cases.Add(new Case { Name = $"{m}x{n}x{k}", P = Problem(m, n, k, offA: za, offB: 1 + zb, offC: s), ASigned = (s & 1) != 0, BSigned = (s & 2) != 0, ZaMode = za, ZbMode = zb });
                }
            }
        }

        // Transposed B, row padding, batches (shared and separate B), offsets that are not word aligned.
        cases.Add(new Case { Name = "transposed B", P = Problem(40, 70, 50, transB: true, offA: 3, offB: 1), ASigned = false, BSigned = true, ZaMode = 1, ZbMode = 2 });
        cases.Add(new Case { Name = "padded rows", P = Problem(21, 22, 23, ldaExtra: 5, ldbExtra: 3, offA: 2, offB: 3), ASigned = true, BSigned = true, ZaMode = 2, ZbMode = 1 });
        cases.Add(new Case { Name = "batch 5 shared B", P = Problem(9, 70, 33, batch: 5, shareB: true, offA: 1, offB: 2), ASigned = false, BSigned = true, ZaMode = 1, ZbMode = 2 });
        cases.Add(new Case { Name = "batch 3", P = Problem(70, 9, 33, batch: 3, offA: 3, offB: 1, offC: 3), ASigned = false, BSigned = false, ZaMode = 2, ZbMode = 1 });
        // Extreme bytes (-128, 127, 0, 255) with extreme zero points, in every signedness mix, to exercise the signed-domain rewrite.
        foreach (var fill in new[] { 0x00, 0x7F, 0x80, 0xFF })
        {
            for (var s = 0; s < 4; s++)
            {
                cases.Add(new Case { Name = $"extreme {fill:X2}", P = Problem(37, 41, 77, transB: (s & 1) != 0), ASigned = (s & 1) != 0, BSigned = (s & 2) != 0, ZaMode = 1 + (s & 1), ZbMode = 2 - (s & 1), Fill = fill });
            }
        }

        // The longest sums: every byte at an extreme, K up to 4096 (the matrix variant adds 256 to 1024 terms in float32 before it folds into int32; each of
        // these sums reaches a quarter of the int32 range or more, so a lost bit anywhere shows), with and without zero points, and with few rows for the row kernel.
        foreach (var fill in new[] { 0x00, 0x7F, 0x80, 0xFF })
        {
            for (var s = 0; s < 4; s++)
            {
                foreach (var (m, n, k) in new[] { (33, 35, 4096), (3, 70, 4096), (65, 64, 1024) })
                {
                    cases.Add(new Case { Name = $"long extreme {fill:X2} {m}x{n}x{k}", P = Problem(m, n, k, transB: (s & 1) != 0), ASigned = (s & 1) != 0, BSigned = (s & 2) != 0, ZaMode = (s + fill) % 3, ZbMode = (s + 1) % 3, Fill = fill });
                }
            }
        }

        foreach (var (name, gemm) in Configs(d, all: true))
        {
            using (gemm)
            {
                long bad = 0;
                var seed = 1;
                foreach (var c in cases)
                {
                    var (b, v) = await RunInt32(d, gemm, c, seed++);
                    bad += b;
                    if (b != 0)
                    {
                        Console.WriteLine($"MISMATCH [{name}] {c.Name} signed {c.ASigned}/{c.BSigned} zp {c.ZaMode}/{c.ZbMode}: {b} elements");
                    }

                    variants++;
                }

                Rig.Row(d, "int8 gemm", "u8/i8 mixes", $"{cases.Count} cases (ragged tiles, K 0 to 4096, scalar and vector zero points, transposed B, padded rows, batches, odd offsets, extreme bytes), {name}, variant {gemm.Variant}", $"mismatching int32 sums {bad}");
                Assert.Equal(0, bad);
            }
        }
    }

    [Fact]
    public async Task ModelShapesAreBitExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        long bad = 0, sums = 0;
        foreach (var (name, m, n, k) in ModelShapes)
        {
#if WEB_BACKEND
            if ((long)m * n * k > 400_000_000L)
            {
                Rig.Row(d, "int8 gemm", "u8 x i8", name, "skipped in the browser: the CPU reference would take too long in WebAssembly");
                continue;
            }
#endif
            // Activations are uint8 with a zero point (dynamic quantization), weights int8 per tensor or per output channel with zero point 0.
            var c = new Case { Name = name, P = Problem(m, n, k, transB: true), ASigned = false, BSigned = true, ZaMode = 1, ZbMode = 0 };
            // Small products run every tile and both variants; the large ones the default and the portable variant.
            foreach (var (config, gemm) in Configs(d, all: m <= 128))
            {
                using (gemm)
                {
                    var (b, _) = await RunInt32(d, gemm, c, m + n + k);
                    bad += b;
                    sums += (long)m * n;
                    Rig.Row(d, "int8 gemm", "u8 x i8", $"{name} (M {m}, N {n}, K {k}, B stored N x K), {config}", $"mismatching int32 sums {b} of {(long)m * n}, variant {gemm.Variant}");
                }
            }
        }

        Assert.Equal(0, bad);
        _ = sums;
    }

    [Fact]
    public async Task ForcingTheDotVariantOffChangesNothing()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var chosen = new Int8Gemm(d);
        using var forced = new Int8Gemm(d, new Int8GemmOptions { AllowPackedDot = false });
        var c = new Case { Name = "forced", P = Problem(70, 90, 100), ASigned = false, BSigned = true, ZaMode = 1, ZbMode = 2 };
        var (b1, v1) = await RunInt32(d, chosen, c, 5);
        var (b2, v2) = await RunInt32(d, forced, c, 5);
        Rig.Row(d, "int8 gemm", "u8 x i8", $"variant per device: device int8-dot capability {d.Capabilities.Compute.Int8DotProduct}; default chose {v1}; dot forced off chose {v2}", $"mismatching sums {b1 + b2}");
        Assert.Equal(0, b1 + b2);
        Assert.Equal("portable-tile64x64x16", v2);
    }

    static async Task<int> FloatCase(GpuDevice d, Int8Gemm gemm, int m, int n, int k, bool scaleVec, bool withIntBias, bool withFloatBias, int seed)
    {
        var p = Problem(m, n, k);
        var a = Bytes((m * k) + 4, seed);
        var b = Bytes((k * n) + 4, seed + 1);
        var za = Bytes(1, seed + 2);
        var zb = Bytes(n, seed + 3);
        var want = Reference(p, a, false, b, true, za, false, zb, true);
        var scale = new float[scaleVec ? n : 1];
        for (var i = 0; i < scale.Length; i++)
        {
            scale[i] = 0.0007f + (((i * 37) + seed) % 101) * 1.3e-5f;
        }

        var ib = new int[n];
        var fb = new float[n];
        for (var i = 0; i < n; i++)
        {
            ib[i] = ((i * 7919) % 20001) - 10000;
            fb[i] = (((i * 31) % 201) - 100) * 0.0137f;
        }

        var expected = new float[m * n];
        for (var i = 0; i < expected.Length; i++)
        {
            var col = i % n;
            var acc = want[i];
            if (withIntBias)
            {
                acc = unchecked(acc + ib[col]);
            }

            var f = (float)acc * scale[scaleVec ? col : 0];
            if (withFloatBias)
            {
                f += fb[col];
            }

            expected[i] = f;
        }

        using var ba = d.CreateBuffer<byte>(a);
        using var bb = d.CreateBuffer<sbyte>(Cast<sbyte>(b));
        using var bza = d.CreateBuffer<byte>(za);
        using var bzb = d.CreateBuffer<sbyte>(Cast<sbyte>(zb));
        using var bs = d.CreateBuffer<float>(scale);
        using var bib = d.CreateBuffer<int>(ib);
        using var bfb = d.CreateBuffer<float>(fb);
        using var bc = d.CreateBuffer<float>(m * n);
        Rig.Run(d, e => gemm.EncodeFloat(e, p, ba, bb, bc, new Int8Operand<float>(bs, 0, scaleVec), new Int8Operand<byte>(bza, 0), new Int8Operand<sbyte>(bzb, 0, true), withFloatBias ? new Int8Operand<float>(bfb, 0, true) : null, withIntBias ? new Int8Operand<int>(bib, 0, true) : null));
        var got = await Rig.DownloadAsync(bc);
        return Rig.BitMismatches(got, expected);
    }

    static async Task<int> RequantCase(GpuDevice d, Int8Gemm gemm, int m, int n, int k, bool multVec, bool unsigned, int zeroY, bool withBias, int seed, bool aSigned, bool bSigned)
    {
        var p = Problem(m, n, k, offC: 3);
        var a = Bytes((m * k) + 4, seed);
        var b = Bytes((k * n) + 4, seed + 1);
        var za = Bytes(1, seed + 2);
        var want = Reference(p, a, aSigned, b, bSigned, za, false, null, false);
        var mult = new float[multVec ? n : 1];
        for (var i = 0; i < mult.Length; i++)
        {
            mult[i] = 0.00011f + (((i * 13) + seed) % 59) * 2.7e-6f;
        }

        var ib = new int[n];
        for (var i = 0; i < n; i++)
        {
            ib[i] = ((i * 6007) % 4001) - 2000;
        }

        var expected = new byte[m * n];
        for (var i = 0; i < expected.Length; i++)
        {
            var col = i % n;
            var acc = want[i];
            if (withBias)
            {
                acc = unchecked(acc + ib[col]);
            }

            expected[i] = (byte)Requant(acc, mult[multVec ? col : 0], zeroY, unsigned);
        }

        var outBytes = new byte[IndexRig.Padded<byte>(3 + (m * n) + 3) + 4];
        Array.Fill(outBytes, (byte)0xA7);
        using var bm = d.CreateBuffer<float>(mult);
        using var bib = d.CreateBuffer<int>(ib);
        using var ba8 = d.CreateBuffer<byte>(a);
        using var bb8 = d.CreateBuffer<byte>(b);
        using var bz8 = d.CreateBuffer<byte>(za);
        using var bas = d.CreateBuffer<sbyte>(Cast<sbyte>(a));
        using var bbs = d.CreateBuffer<sbyte>(Cast<sbyte>(b));
        using var bzs = d.CreateBuffer<sbyte>(Cast<sbyte>(za));
        using var yu = d.CreateBuffer<byte>(outBytes);
        using var ys = d.CreateBuffer<sbyte>(Cast<sbyte>(outBytes));
        var mo = new Int8Operand<float>(bm, 0, multVec);
        var bias = withBias ? new Int8Operand<int>(bib, 0, true) : (Int8Operand<int>?)null;
        Rig.Run(d, e =>
        {
            switch (aSigned, bSigned, unsigned)
            {
                case (false, false, true): gemm.EncodeRequantized(e, p, ba8, bb8, yu, mo, zeroY, new Int8Operand<byte>(bz8, 0), null, bias); break;
                case (false, true, true): gemm.EncodeRequantized(e, p, ba8, bbs, yu, mo, zeroY, new Int8Operand<byte>(bz8, 0), null, bias); break;
                case (true, false, true): gemm.EncodeRequantized(e, p, bas, bb8, yu, mo, zeroY, new Int8Operand<sbyte>(bzs, 0), null, bias); break;
                case (true, true, true): gemm.EncodeRequantized(e, p, bas, bbs, yu, mo, zeroY, new Int8Operand<sbyte>(bzs, 0), null, bias); break;
                case (false, false, false): gemm.EncodeRequantized(e, p, ba8, bb8, ys, mo, zeroY, new Int8Operand<byte>(bz8, 0), null, bias); break;
                case (false, true, false): gemm.EncodeRequantized(e, p, ba8, bbs, ys, mo, zeroY, new Int8Operand<byte>(bz8, 0), null, bias); break;
                case (true, false, false): gemm.EncodeRequantized(e, p, bas, bb8, ys, mo, zeroY, new Int8Operand<sbyte>(bzs, 0), null, bias); break;
                default: gemm.EncodeRequantized(e, p, bas, bbs, ys, mo, zeroY, new Int8Operand<sbyte>(bzs, 0), null, bias); break;
            }
        });
        var got = unsigned ? await Rig.DownloadAsync(yu) : System.Runtime.InteropServices.MemoryMarshal.AsBytes<sbyte>(await Rig.DownloadAsync(ys)).ToArray();
        var bad = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            bad += got[3 + i] == expected[i] ? 0 : 1;
        }

        for (var i = 0; i < 3; i++)
        {
            bad += got[i] == 0xA7 ? 0 : 1;
        }

        for (var i = 3 + expected.Length; i < outBytes.Length; i++)
        {
            bad += got[i] == 0xA7 ? 0 : 1;
        }

        return bad;
    }

    [Fact]
    public async Task FloatAndRequantizedFormsAreBitExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var gemm = new Int8Gemm(d);
        long badF = 0, badQ = 0, nf = 0, nq = 0;
        var seed = 10;
        foreach (var (m, n, k) in new[] { (1, 1, 1), (5, 7, 9), (64, 65, 100), (130, 70, 384), (128, 384, 384), (33, 257, 77) })
        {
            foreach (var vec in new[] { false, true })
            {
                foreach (var (ibias, fbias) in new[] { (false, false), (true, false), (false, true), (true, true) })
                {
                    badF += await FloatCase(d, gemm, m, n, k, vec, ibias, fbias, seed++);
                    nf++;
                }
            }

            foreach (var vec in new[] { false, true })
            {
                foreach (var (unsigned, zero) in new[] { (true, 128), (false, 0), (true, 0), (false, -5), (true, 255), (false, 127) })
                {
                    badQ += await RequantCase(d, gemm, m, n, k, vec, unsigned, zero, (seed & 1) == 0, seed, (seed & 2) != 0, (seed & 4) != 0);
                    seed++;
                    nq++;
                }
            }
        }

        Rig.Row(d, "int8 gemm", "float32 out", $"{nf} cases (scalar and per-column scale, int and float bias, ragged shapes)", $"mismatching bits {badF}");
        Rig.Row(d, "int8 gemm", "u8/i8 out", $"{nq} cases (QLinearMatMul requantization: scalar and per-column multiplier, zero points 0/128/255/-5/127, bias, odd output offset)", $"mismatching bytes {badQ}");
        Assert.Equal(0, badF + badQ);
    }

    [Fact]
    public async Task RequantizationRoundsHalfToEvenAndSaturates()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var gemm = new Int8Gemm(d);
        // A 1x1 product with K = 1 and a multiplier of 0.5 or 2.5 puts exact ties and out-of-range values through the epilogue.
        var bad = 0;
        var cases = 0;
        foreach (var (mult, zeroY, unsigned) in new[] { (0.5f, 10, true), (0.5f, -10, false), (2.5f, 0, true), (1e9f, 3, true), (-1e9f, 3, false), (float.NaN, 5, true), (float.NaN, 5, false), (float.PositiveInfinity, 1, false), (0f, 7, true) })
        {
            for (var av = -9; av <= 9; av++)
            {
                var a = new sbyte[4];
                a[0] = (sbyte)av;
                var b = new sbyte[4];
                b[0] = 1;
                var want = (byte)Requant(av, mult, zeroY, unsigned);
                using var ba = d.CreateBuffer<sbyte>(a);
                using var bb = d.CreateBuffer<sbyte>(b);
                using var bm = d.CreateBuffer<float>([mult]);
                using var y = d.CreateBuffer<byte>(4);
                if (unsigned)
                {
                    Rig.Run(d, e => gemm.EncodeRequantized(e, new Int8GemmProblem(1, 1, 1), ba, bb, y, new Int8Operand<float>(bm, 0), zeroY));
                }
                else
                {
                    using var ys = d.CreateBuffer<sbyte>(4);
                    Rig.Run(d, e => gemm.EncodeRequantized(e, new Int8GemmProblem(1, 1, 1), ba, bb, ys, new Int8Operand<float>(bm, 0), zeroY));
                    var gs = await Rig.DownloadAsync(ys);
                    bad += (byte)gs[0] == want ? 0 : 1;
                    cases++;
                    continue;
                }

                var g = await Rig.DownloadAsync(y);
                bad += g[0] == want ? 0 : 1;
                cases++;
            }
        }

        Rig.Row(d, "int8 gemm", "u8/i8 out", $"{cases} tie, saturation, NaN and infinity cases of the requantization", $"mismatching bytes {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public void BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var gemm = new Int8Gemm(d);
        using var a = d.CreateBuffer<byte>(64);
        using var b = d.CreateBuffer<sbyte>(64);
        using var c = d.CreateBuffer<int>(64);
        using var f = d.CreateBuffer<float>(64);
        using var fl = d.CreateBuffer<float>(1);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(9, 9, 9), a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(-1, 2, 2), a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(2, 2, 2) { Batch = 0 }, a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(2, 2, 2) { Lda = 1 }, a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(2, 2, 2) { OffsetC = 62 }, a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(8, 8, 8) { OffsetA = 1 }, a, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(2, 2, 2), a, b, c, new Int8Operand<byte>(a, 63, true)));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeInt32(enc, new Int8GemmProblem(2, 2, 2), f, b, c));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeFloat(enc, new Int8GemmProblem(2, 8, 2), a, b, f, new Int8Operand<float>(fl, 0, true)));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeRequantized(enc, new Int8GemmProblem(2, 2, 2), a, b, a, new Int8Operand<float>(fl, 0), 300));
        Assert.Throws<GpuKernelShapeException>(() => gemm.EncodeRequantized(enc, new Int8GemmProblem(2, 2, 2), a, b, b, new Int8Operand<float>(fl, 0), 128));
        d.Submit(enc);
        d.WaitIdle();
    }

    // onemodel's DynamicQuantizeLinear for one tensor.
    static (byte[] Y, float Scale, byte Zero) DynamicReference(float[] x)
    {
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in x)
        {
            if (v < min)
            {
                min = v;
            }

            if (v > max)
            {
                max = v;
            }
        }

        float lo = Math.Min(0f, min), hi = Math.Max(0f, max);
        var scale = (hi - lo) / 255f;
        var zp = 0;
        if (scale != 0f && float.IsFinite(scale))
        {
            var z = MathF.Round(0f - (lo / scale));
            zp = z < 0f ? 0 : z > 255f ? 255 : (int)z;
        }

        var y = new byte[x.Length];
        if (scale == 0f)
        {
            return (y, 1f, 0);
        }

        for (var i = 0; i < x.Length; i++)
        {
            var v = MathF.Round(x[i] / scale) + zp;
            y[i] = float.IsNaN(v) ? (byte)0 : v < 0 ? (byte)0 : v > 255 ? (byte)255 : (byte)(int)v;
        }

        return (y, scale, (byte)zp);
    }

    [Fact]
    public async Task DynamicQuantizeIsBitExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new DynamicQuantizeKernels(d);
        long bad = 0, checkedValues = 0;
        var cases = 0;
        // Shapes of the activations that reach the first int8 product of the encoders, plus tiny and ragged ones.
        var counts = new[] { 1, 2, 3, 255, 256, 257, 1023, 1024, 1025, 4097, 128 * 384, 128 * 768, 1024 * 768, 1024 * 384 + 17, 262144 + 3 };
        var seed = 1;
        foreach (var count in counts)
        {
            foreach (var kind in new[] { 0, 1, 2, 3, 4, 5, 6 })
            {
                var x = new float[count];
                var s = (uint)(seed * 2654435761u) + 99u;
                for (var i = 0; i < count; i++)
                {
                    s = (s * 1664525u) + 1013904223u;
                    var u = ((s >> 8) / 16777216f * 2f) - 1f;
                    x[i] = kind switch
                    {
                        0 => u * 4.7f,                          // symmetric
                        1 => (u + 1f) * 3.3f,                   // all positive
                        2 => -((u + 1f) * 0.0123f),             // all negative, small
                        3 => (u * u * u) * 100f + 0.001f,       // heavy tails
                        4 => (MathF.Round(u * 127f) * 0.125f),  // values on a grid: many ties after division
                        5 => 0f,                                // all zero
                        _ => u * 1e-3f,
                    };
                }

                var (wy, wscale, wzero) = DynamicReference(x);
                const int OffX = 1, OffY = 3, OffS = 2, OffZ = 5;
                var xs = new float[count + OffX];
                x.CopyTo(xs, OffX);
                using var bx = d.CreateBuffer<float>(xs);
                var yInit = new byte[((OffY + count + 3) & ~3) + 4];
                Array.Fill(yInit, (byte)0x3C);
                using var by = d.CreateBuffer<byte>(yInit);
                using var bs = d.CreateBuffer<float>(4);
                var zInit = new byte[8];
                Array.Fill(zInit, (byte)0xEE);
                using var bz = d.CreateBuffer<byte>(zInit);
                Rig.Run(d, e => k.Encode(e, bx, OffX, count, by, OffY, bs, OffS, bz, OffZ));
                var y = await Rig.DownloadAsync(by);
                var sc = await Rig.DownloadAsync(bs);
                var zp = await Rig.DownloadAsync(bz);
                var badHere = 0;
                for (var i = 0; i < count; i++)
                {
                    badHere += y[OffY + i] == wy[i] ? 0 : 1;
                }

                badHere += BitConverter.SingleToUInt32Bits(sc[OffS]) == BitConverter.SingleToUInt32Bits(wscale) ? 0 : 1;
                badHere += zp[OffZ] == wzero ? 0 : 1;
                for (var i = 0; i < OffY; i++)
                {
                    badHere += y[i] == 0x3C ? 0 : 1;
                }

                for (var i = OffY + count; i < y.Length; i++)
                {
                    badHere += y[i] == 0x3C ? 0 : 1;
                }

                for (var i = 0; i < zp.Length; i++)
                {
                    badHere += i == OffZ || zp[i] == 0xEE ? 0 : 1;
                }

                if (badHere != 0)
                {
                    Console.WriteLine($"MISMATCH dynamic quantize count {count} kind {kind}: {badHere} (scale {sc[OffS]:R} want {wscale:R}, zero {zp[OffZ]} want {wzero})");
                }

                bad += badHere;
                checkedValues += count;
                cases++;
                seed++;
            }
        }

        Rig.Row(d, "dynamic quantize", "f32 to u8", $"{cases} tensors, {checkedValues} values (symmetric, positive, negative, heavy-tailed, grid values with ties, all zero; 1 to 786432 elements)", $"mismatching values, scales and zero points {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task DynamicQuantizeFeedsTheProductWithoutLeavingTheDevice()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var dq = new DynamicQuantizeKernels(d);
        using var gemm = new Int8Gemm(d);
        // A dynamically quantized activation, int8 weights with a per-channel scale, the float output of the fused chain, all in one submission.
        const int M = 128, N = 384, K = 384;
        var x = Rig.Random(M * K, 3, -3f, 5f);
        var (wy, wscale, wzero) = DynamicReference(x);
        var w = Bytes(N * K, 4);
        var wscales = new float[N];
        for (var i = 0; i < N; i++)
        {
            wscales[i] = 0.002f + (i % 17) * 1e-4f;
        }

        var p = Problem(M, N, K, transB: true);
        var sums = Reference(p, wy, false, w, true, [wzero], false, null, false);
        using var bx = d.CreateBuffer<float>(x);
        using var by = d.CreateBuffer<byte>(M * K);
        using var bs = d.CreateBuffer<float>(1);
        using var bz = d.CreateBuffer<byte>(1);
        using var bw = d.CreateBuffer<sbyte>(Cast<sbyte>(w));
        using var bws = d.CreateBuffer<float>(wscales);
        using var bscale = d.CreateBuffer<float>(N);
        using var bc = d.CreateBuffer<float>(M * N);
        using var elementwise = new ElementwiseKernels(d);
        Rig.Run(d, e =>
        {
            dq.Encode(e, bx, 0, M * K, by, 0, bs, 0, bz, 0);
            // scale = activation scale * weight scale, computed on the device, then used by the product.
            elementwise.EncodeBinary(e, GpuBinaryOp.Mul, bs, 0, [1], bws, 0, [N], bscale, 0);
            gemm.EncodeFloat(e, p, by, bw, bc, new Int8Operand<float>(bscale, 0, true), new Int8Operand<byte>(bz, 0));
        });
        var got = await Rig.DownloadAsync(bc);
        var bad = 0;
        for (var i = 0; i < got.Length; i++)
        {
            var want = (float)sums[i] * (wscale * wscales[i % N]);
            bad += BitConverter.SingleToUInt32Bits(got[i]) == BitConverter.SingleToUInt32Bits(want) ? 0 : 1;
        }

        Rig.Row(d, "dynamic int8 chain", "f32", "DynamicQuantizeLinear, scale product and MatMulIntegerToFloat in one submission (128 x 384 x 384)", $"mismatching bits {bad}");
        Assert.Equal(0, bad);
    }
}
