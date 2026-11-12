using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>ConvInteger against a plain CPU definition of the operator (ONNX operator documentation), bit for bit, plus the wide-grid case of the 8-bit product.</summary>
public sealed class ConvIntegerTests
{
    sealed class Case
    {
        public string Name = "";
        public ConvProblem P;
        public bool XSigned, WSigned;
        public int XZeroMode;      // 0 none, 1 one value
        public int WZeroMode;      // 0 none, 1 one value, 2 one per output channel
        public int Fill = -1;      // 0..255: every byte holds this value (extremes), -1 random
        public int ScratchImages;  // 0: the instance's own scratch; n: caller scratch for n images
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

    static T[] Cast<T>(byte[] data)
        where T : unmanaged
    {
        var padded = new byte[(data.Length + 3) & ~3];
        data.CopyTo(padded, 0);
        return MemoryMarshal.Cast<byte, T>(padded).ToArray();
    }

    // y[n, m, oh, ow] = sum over c, kh, kw of (x - xz) * (w - wz[m]); padded taps are the real value zero and add nothing. 32-bit wrap-around.
    static int[] Reference(ConvProblem p, byte[] x, bool xs, byte[] w, bool ws, byte[]? xz, byte[]? wz, bool wzVec)
    {
        int oh = p.OutHeight, ow = p.OutWidth;
        int cinG = p.InChannels / p.Groups, coutG = p.OutChannels / p.Groups;
        var y = new int[p.Batch * p.OutChannels * oh * ow];
        var xzv = xz is null ? 0 : Val(xz[0], xs);
        Parallel.For(0, p.Batch * p.OutChannels, row =>
        {
            var n = row / p.OutChannels;
            var m = row % p.OutChannels;
            var g = m / coutG;
            var wzv = wz is null ? 0 : Val(wz[wzVec ? m : 0], ws);
            for (var i = 0; i < oh; i++)
            {
                for (var j = 0; j < ow; j++)
                {
                    var acc = 0;
                    for (var c = 0; c < cinG; c++)
                    {
                        for (var kh = 0; kh < p.KernelHeight; kh++)
                        {
                            var ih = (i * p.StrideHeight) + (kh * p.DilationHeight) - p.PadTop;
                            if ((uint)ih >= (uint)p.Height)
                            {
                                continue;
                            }

                            for (var kw = 0; kw < p.KernelWidth; kw++)
                            {
                                var iw = (j * p.StrideWidth) + (kw * p.DilationWidth) - p.PadLeft;
                                if ((uint)iw >= (uint)p.Width)
                                {
                                    continue;
                                }

                                var xv = Val(x[p.OffsetX + ((((n * p.InChannels) + (g * cinG) + c) * p.Height) + ih) * p.Width + iw], xs) - xzv;
                                var wv = Val(w[p.OffsetW + ((((m * cinG) + c) * p.KernelHeight) + kh) * p.KernelWidth + kw], ws) - wzv;
                                acc = unchecked(acc + (xv * wv));
                            }
                        }
                    }

                    y[((((n * p.OutChannels) + m) * oh) + i) * ow + j] = acc;
                }
            }
        });
        return y;
    }

    static async Task<int> Run<TX, TW>(GpuDevice d, ConvIntegerKernels conv, Case c, int seed)
        where TX : unmanaged
        where TW : unmanaged
    {
        var p = c.P;
        int oh = p.OutHeight, ow = p.OutWidth;
        var cinG = p.InChannels / p.Groups;
        var xLen = p.OffsetX + (p.Batch * p.InChannels * p.Height * p.Width) + 4;
        var wLen = p.OffsetW + (p.OutChannels * cinG * p.KernelHeight * p.KernelWidth) + 4;
        var x = Bytes(xLen, seed);
        var w = Bytes(wLen, seed + 1);
        var xz = c.XZeroMode == 0 ? null : Bytes(1, seed + 2);
        var wz = c.WZeroMode == 0 ? null : Bytes(c.WZeroMode == 2 ? p.OutChannels : 1, seed + 3);
        if (c.Fill >= 0)
        {
            Array.Fill(x, (byte)c.Fill);
            Array.Fill(w, (byte)(c.Fill ^ 0xFF));
            xz?.AsSpan().Fill((byte)c.Fill);
            wz?.AsSpan().Fill((byte)(c.Fill ^ 0x80));
        }

        var want = Reference(p, x, c.XSigned, w, c.WSigned, xz, wz, c.WZeroMode == 2);
        var sentinel = 0x5A5A5A5A;
        var got = new int[p.OffsetY + want.Length + 2];
        Array.Fill(got, sentinel);
        using var bx = d.CreateBuffer<TX>(Cast<TX>(x));
        using var bw = d.CreateBuffer<TW>(Cast<TW>(w));
        using var bxz = xz is null ? null : d.CreateBuffer<TX>(Cast<TX>(xz));
        using var bwz = wz is null ? null : d.CreateBuffer<TW>(Cast<TW>(wz));
        using var by = d.CreateBuffer<int>(got);
        GpuBuffer<TX>? scratch = null;
        var scratchOffset = 0;
        if (c.ScratchImages > 0)
        {
            scratchOffset = 8;
            scratch = d.CreateBuffer<TX>((int)(scratchOffset + (c.ScratchImages * ConvIntegerKernels.ScratchBytesPerImage(p))));
        }

        try
        {
            Rig.Run(d, e => conv.Encode(e, p, bx, bw, by, bxz is null ? null : new Int8Operand<TX>(bxz, 0), bwz is null ? null : new Int8Operand<TW>(bwz, 0, c.WZeroMode == 2), scratch, scratchOffset));
            var result = await Rig.DownloadAsync(by);
            var bad = 0;
            var shown = 0;
            for (var i = 0; i < want.Length; i++)
            {
                if (result[p.OffsetY + i] != want[i])
                {
                    bad++;
                    if (shown++ < 4)
                    {
                        Console.WriteLine($"  {c.Name}: differs at flat {i}: got {result[p.OffsetY + i]} want {want[i]} [{conv.Variant}]");
                    }
                }
            }

            for (var i = 0; i < p.OffsetY; i++)
            {
                bad += result[i] == sentinel ? 0 : 1;
            }

            for (var i = p.OffsetY + want.Length; i < result.Length; i++)
            {
                bad += result[i] == sentinel ? 0 : 1;
            }

            return bad;
        }
        finally
        {
            scratch?.Dispose();
        }
    }

    static Task<int> Dispatch(GpuDevice d, ConvIntegerKernels conv, Case c, int seed) => (c.XSigned, c.WSigned) switch
    {
        (false, false) => Run<byte, byte>(d, conv, c, seed),
        (false, true) => Run<byte, sbyte>(d, conv, c, seed),
        (true, false) => Run<sbyte, byte>(d, conv, c, seed),
        _ => Run<sbyte, sbyte>(d, conv, c, seed),
    };

    static ConvProblem P(int batch, int cin, int h, int w, int cout, int kh, int kw, int stride = 1, int dilation = 1, int pad = 0, int groups = 1) =>
        new(batch, cin, h, w, cout, kh, kw)
        {
            StrideHeight = stride,
            StrideWidth = stride,
            DilationHeight = dilation,
            DilationWidth = dilation,
            PadTop = pad,
            PadBottom = pad,
            PadLeft = pad,
            PadRight = pad,
            Groups = groups,
        };

    static async Task Check(GpuDevice d, List<Case> cases)
    {
        var convs = new List<ConvIntegerKernels> { new(d), new(d, new Int8GemmOptions { AllowPackedDot = false }) };
        try
        {
            var total = 0;
            var seed = 100;
            foreach (var conv in convs)
            {
                foreach (var c in cases)
                {
                    var bad = await Dispatch(d, conv, c, seed++);
                    Assert.True(bad == 0, $"{c.Name} [{conv.Variant}]: {bad} elements differ");
                    total++;
                }
            }

            TestContext.Current.SendDiagnosticMessage($"conv integer: {total} runs, 0 mismatches, variants {string.Join(" | ", convs.ConvertAll(v => v.Variant))}");
        }
        finally
        {
            foreach (var conv in convs)
            {
                conv.Dispose();
            }
        }
    }

    static List<Case> Spread(string name, ConvProblem p)
    {
        // All four type combinations; zero points none/one/per channel in turn.
        var list = new List<Case>();
        for (var s = 0; s < 4; s++)
        {
            list.Add(new Case { Name = $"{name} s{s}", P = p, XSigned = (s & 1) != 0, WSigned = (s & 2) != 0, XZeroMode = 1 - (s & 1), WZeroMode = (s + 1) % 3 });
        }

        return list;
    }

    [Fact]
    public async Task EdgeGeometriesAreBitExact()
    {
        using var gpu = new GpuFixture();
        var cases = new List<Case>();
        cases.AddRange(Spread("1x1 pointwise", P(2, 5, 7, 9, 6, 1, 1)));
        cases.AddRange(Spread("3x3 pad1", P(1, 3, 10, 13, 4, 3, 3, pad: 1)));
        cases.AddRange(Spread("3x3 stride2 pad1 odd", P(2, 4, 9, 11, 5, 3, 3, stride: 2, pad: 1)));
        cases.AddRange(Spread("5x5 dilation2", P(1, 2, 14, 15, 3, 5, 5, dilation: 2, pad: 4)));
        cases.AddRange(Spread("groups 2", P(2, 4, 8, 8, 6, 3, 3, pad: 1, groups: 2)));
        cases.AddRange(Spread("depthwise", P(2, 6, 9, 10, 6, 3, 3, pad: 1, groups: 6)));
        cases.AddRange(Spread("1d k3", ConvProblem.Conv1D(2, 5, 31, 7, 3, padBegin: 1, padEnd: 1)));
        cases.AddRange(Spread("1d stride2", ConvProblem.Conv1D(1, 4, 40, 6, 3, stride: 2, padBegin: 1, padEnd: 1)));
        cases.AddRange(Spread("1x1 image", P(3, 8, 1, 1, 4, 1, 1)));
        cases.AddRange(Spread("kernel = image", P(1, 3, 5, 5, 2, 5, 5)));
        var asym = P(1, 3, 12, 9, 4, 3, 2);
        asym.PadTop = 2;
        asym.PadBottom = 0;
        asym.PadLeft = 0;
        asym.PadRight = 3;
        asym.StrideHeight = 2;
        cases.AddRange(Spread("asymmetric padding", asym));
        var off = P(2, 3, 6, 7, 5, 3, 3, pad: 1);
        off.OffsetX = 3;
        off.OffsetW = 5;
        off.OffsetY = 7;
        cases.AddRange(Spread("offsets", off));
        for (var s = 0; s < 4; s++)
        {
            foreach (var fill in new[] { 0, 127, 128, 255 })
            {
                cases.Add(new Case { Name = $"extremes {fill} s{s}", P = P(1, 17, 6, 6, 3, 3, 3, pad: 1), XSigned = (s & 1) != 0, WSigned = (s & 2) != 0, XZeroMode = 1, WZeroMode = 2, Fill = fill });
            }
        }

        // Caller-placed scratch of exactly one image makes a batch run in chunks; the result is the same.
        cases.Add(new Case { Name = "chunked batch, scratch for 1 image", P = P(3, 3, 8, 8, 4, 3, 3, pad: 1), XZeroMode = 1, WZeroMode = 1, ScratchImages = 1 });
        cases.Add(new Case { Name = "grouped, scratch for 2 images", P = P(3, 4, 8, 8, 6, 3, 3, pad: 1, groups: 2), XSigned = true, WSigned = true, XZeroMode = 1, WZeroMode = 2, ScratchImages = 2 });
        await Check(gpu.Device, cases);
    }

    [Fact]
    public async Task WhisperStemIsBitExact()
    {
        using var gpu = new GpuFixture();
        var cases = new List<Case>
        {
            // The two convolutions in front of the encoder (Whisper-small): 80 mel bins to 768 channels over 3000 frames, then stride 2 to 1500. The second runs at 1500
            // input frames here, which is the same layer at half the length (the CPU reference is the slow part).
            new() { Name = "whisper conv1 u8 x i8 w", P = ConvProblem.Conv1D(1, 80, 3000, 768, 3, padBegin: 1, padEnd: 1), XSigned = false, WSigned = true, XZeroMode = 1, WZeroMode = 1 },
            new() { Name = "whisper conv1 u8 x u8 w, per channel", P = ConvProblem.Conv1D(1, 80, 3000, 768, 3, padBegin: 1, padEnd: 1), XSigned = false, WSigned = false, XZeroMode = 1, WZeroMode = 2 },
            new() { Name = "whisper conv2 u8 x i8 w", P = ConvProblem.Conv1D(1, 768, 1500, 768, 3, stride: 2, padBegin: 1, padEnd: 1), XSigned = false, WSigned = true, XZeroMode = 1, WZeroMode = 1 },
        };
        await Check(gpu.Device, cases);
    }

    [Fact]
    public async Task ResNetShapesAreBitExact()
    {
        using var gpu = new GpuFixture();
        var cases = new List<Case>
        {
            new() { Name = "resnet stem 7x7/2", P = P(1, 3, 224, 224, 64, 7, 7, stride: 2, pad: 3), XSigned = false, WSigned = true, XZeroMode = 1, WZeroMode = 1 },
            new() { Name = "resnet 3x3 56x56", P = P(1, 64, 56, 56, 64, 3, 3, pad: 1), XSigned = false, WSigned = true, XZeroMode = 1, WZeroMode = 2 },
            new() { Name = "resnet 1x1 56x56 b2", P = P(2, 64, 56, 56, 256, 1, 1), XSigned = true, WSigned = true, XZeroMode = 1, WZeroMode = 1 },
            new() { Name = "mobilenet depthwise 3x3", P = P(1, 96, 28, 28, 96, 3, 3, pad: 1, groups: 96), XSigned = false, WSigned = true, XZeroMode = 1, WZeroMode = 2 },
        };
        await Check(gpu.Device, cases);
    }

    [Fact]
    public void BadShapesThrowTypedErrors()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var conv = new ConvIntegerKernels(d);
        using var x = d.CreateBuffer<byte>(64);
        using var w = d.CreateBuffer<sbyte>(64);
        using var y = d.CreateBuffer<int>(64);
        using var f = d.CreateBuffer<float>(64);
        var enc = d.BeginCommands();
        try
        {
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode(enc, P(1, 3, 4, 4, 4, 3, 3, groups: 2), x, w, y));
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode(enc, P(1, 3, 2, 2, 4, 3, 3), x, w, y));
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode(enc, P(1, 3, 8, 8, 4, 3, 3), x, w, y));
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode(enc, P(1, 1, 4, 4, 1, 1, 1), x, w, y, null, new Int8Operand<sbyte>(w, 64, true)));
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode<float, sbyte>(enc, P(1, 1, 4, 4, 1, 1, 1), f, w, y));
            Assert.Throws<GpuKernelShapeException>(() => conv.Encode(enc, P(1, 1, 4, 4, 1, 1, 1), x, w, y, null, null, x, 2));
        }
        finally
        {
            d.Submit(enc);
            d.WaitIdle();
        }
    }

    [Fact]
    public async Task WarmCallsBuildNoBindGroups()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var conv = new ConvIntegerKernels(d);
        var p = P(1, 3, 10, 10, 4, 3, 3, pad: 1);
        using var x = d.CreateBuffer<byte>(new byte[304]);
        using var w = d.CreateBuffer<sbyte>(new sbyte[112]);
        using var y = d.CreateBuffer<int>(400);
        Rig.Run(d, e => conv.Encode(e, p, x, w, y));
        d.WaitIdle();
        var created = conv.BindGroupsCreated;
        Rig.Run(d, e => conv.Encode(e, p, x, w, y));
        await Rig.DownloadAsync(y, 4);
        Assert.Equal(created, conv.BindGroupsCreated);
    }

    [Fact]
    public async Task ProductsWiderThanOneGridAxisAreBitExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        // 65535 workgroups per axis at 64-wide tiles is 4194240 columns (2097120 at 32): these products need more than one grid axis holds, wide and tall.
        foreach (var (m, n) in new[] { (3, 4_300_001), (4_300_001, 3), (2, 4_194_241) })
        {
            const int k = 5;
            var a = Bytes(m * k, 11);
            var b = Bytes(k * n, 12);
            var za = Bytes(1, 13);
            var zb = Bytes(n, 14);
            var want = new int[m * n];
            Parallel.For(0, m, i =>
            {
                for (var j = 0; j < n; j++)
                {
                    var acc = 0;
                    for (var t = 0; t < k; t++)
                    {
                        acc = unchecked(acc + ((a[(i * k) + t] - za[0]) * ((sbyte)b[(t * n) + j] - (sbyte)zb[j])));
                    }

                    want[(i * n) + j] = acc;
                }
            });
            foreach (var options in new[] { new Int8GemmOptions(), new Int8GemmOptions { AllowPackedDot = false }, new Int8GemmOptions { Tile = 32 } })
            {
                using var gemm = new Int8Gemm(d, options);
                using var ba = d.CreateBuffer<byte>(Cast<byte>(a));
                using var bb = d.CreateBuffer<sbyte>(Cast<sbyte>(b));
                using var bza = d.CreateBuffer<byte>(Cast<byte>(za));
                using var bzb = d.CreateBuffer<sbyte>(Cast<sbyte>(zb));
                using var bc = d.CreateBuffer<int>(m * n);
                var problem = new Int8GemmProblem(m, n, k);
                Rig.Run(d, e => gemm.EncodeInt32(e, problem, ba, bb, bc, new Int8Operand<byte>(bza, 0), new Int8Operand<sbyte>(bzb, 0, true)));
                var got = await Rig.DownloadAsync(bc);
                var bad = 0;
                for (var i = 0; i < want.Length; i++)
                {
                    bad += got[i] == want[i] ? 0 : 1;
                }

                Assert.True(bad == 0, $"{m} x {n} [{gemm.Variant}, {options.Tile}]: {bad} elements differ");
            }
        }
    }
}
