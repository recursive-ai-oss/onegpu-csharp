namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>DequantizeLinear against the definition, bit for bit: (float)(x - zeroPoint) * scale, as onemodel's CPU kernel computes it.</summary>
public sealed class DequantizeTests
{
    const int OffX = 3, OffS = 1, OffZ = 2, OffO = 5;

    static T[] Quantized<T>(int n, int seed)
        where T : unmanaged
    {
        var a = new T[typeof(T) == typeof(int) ? n : GpuKernelBuffers.PaddedBoolLength(Math.Max(n, 1))];
        var h = (uint)seed * 2654435761u + 1u;
        for (var i = 0; i < a.Length; i++)
        {
            h = h * 1664525u + 1013904223u;
            if (typeof(T) == typeof(int))
            {
                // Mostly large values, whose conversion to float has to round, plus the extremes, whose subtraction has to wrap.
                a[i] = (T)(object)(i % 17 == 0 ? int.MinValue : i % 19 == 0 ? int.MaxValue : (int)h >> (i % 7));
            }
            else if (typeof(T) == typeof(byte))
            {
                a[i] = (T)(object)(byte)(h >> 16);
            }
            else
            {
                a[i] = (T)(object)(sbyte)(h >> 16);
            }
        }

        return a;
    }

    static int Value<T>(T v) => v switch { byte b => b, sbyte s => s, int i => i, _ => throw new NotSupportedException() };

    static async Task<int> Case<TIn, TOut>(GpuDevice d, DequantizeKernels k, int[] shape, int axis, bool perAxis, bool withZero, int seed)
        where TIn : unmanaged
        where TOut : unmanaged
    {
        var n = IndexRig.Count(shape);
        var rank = shape.Length;
        var ax = axis < 0 ? axis + rank : axis;
        var axisLen = perAxis ? shape[ax] : 1;
        long inner = 1;
        if (perAxis)
        {
            for (var i = ax + 1; i < rank; i++)
            {
                inner *= shape[i];
            }
        }

        var x = Quantized<TIn>(n + OffX, seed);
        var scale = Rig.Random(axisLen + OffS, seed + 1, 0.001f, 2f);
        var zeros = Quantized<TIn>(axisLen + OffZ, seed + 2);
        var half = typeof(TOut) == typeof(Half);
        var output = new TOut[half ? GpuKernelBuffers.PaddedHalfLength(n + OffO + 1) : n + OffO + 1];
        var sentinel = half ? (TOut)(object)(Half)(-77f) : (TOut)(object)(-77f);
        Array.Fill(output, sentinel);
        using var bx = d.CreateBuffer<TIn>(x);
        using var bs = d.CreateBuffer<float>(scale);
        using var bz = d.CreateBuffer<TIn>(zeros);
        using var bo = d.CreateBuffer<TOut>(output);
        Rig.Run(d, e => k.Encode(e, bx, OffX, shape, axis, perAxis, bs, OffS, withZero ? bz : null, OffZ, bo, OffO));
        var got = await Rig.DownloadAsync(bo);
        var bad = 0;
        for (var j = 0; j < n; j++)
        {
            var a = perAxis ? (int)((j / inner) % axisLen) : 0;
            var zp = withZero ? Value(zeros[OffZ + a]) : 0;
            var want = (float)unchecked(Value(x[OffX + j]) - zp) * scale[OffS + a];
            var expected = half ? (TOut)(object)(Half)want : (TOut)(object)want;
            bad += IndexRig.Same(got[OffO + j], expected) ? 0 : 1;
        }

        for (var j = 0; j < OffO; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        for (var j = OffO + n; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], sentinel) ? 0 : 1;
        }

        k.ReleaseBindings();
        return bad;
    }

    static async Task<(int Bad, int Cases)> Sweep<TIn>(GpuDevice d, DequantizeKernels k, int seed)
        where TIn : unmanaged
    {
        (int[] Shape, int Axis, bool PerAxis)[] cases =
        [
            ([1], 0, false), ([7], 0, true), ([5, 6], 0, true), ([5, 6], 1, true), ([5, 6], -1, false), ([2, 3, 4, 5], 1, true),
            ([2, 3, 4, 5], -1, true), ([1031, 3], 0, true), ([3, 257], 1, true), ([4, 1, 9], 2, true), ([100003], 0, false), ([384, 768], 1, true),
        ];
        var bad = 0;
        var count = 0;
        foreach (var (shape, axis, perAxis) in cases)
        {
            foreach (var withZero in new[] { true, false })
            {
                bad += await Case<TIn, float>(d, k, shape, axis, perAxis, withZero, seed++);
                bad += await Case<TIn, Half>(d, k, shape, axis, perAxis, withZero, seed++);
                count += 2;
            }
        }

        return (bad, count);
    }

    [Fact]
    public async Task DequantizeIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new DequantizeKernels(d);
        var (b1, c1) = await Sweep<byte>(d, k, 1);
        var (b2, c2) = await Sweep<sbyte>(d, k, 100);
        var (b3, c3) = await Sweep<int>(d, k, 200);
        Rig.Row(d, "dequantize", "u8/i8/i32 to f32/f16", $"{c1 + c2 + c3} cases (per tensor and per axis, with and without zero point, odd offsets, int32 extremes)", $"mismatching elements {b1 + b2 + b3} (u8 {b1}, i8 {b2}, i32 {b3})");
        Assert.Equal(0, b1 + b2 + b3);
    }

    [Fact]
    public async Task BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new DequantizeKernels(d);
        using var x = d.CreateBuffer<byte>(16);
        using var s = d.CreateBuffer<float>(4);
        using var o = d.CreateBuffer<float>(16);
        using var big = d.CreateBuffer<uint>(16);
        using var ox = d.CreateBuffer<ushort>(16);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.Encode<byte, float>(enc, x, 0, [4, 4], 1, true, s, 1, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode<byte, float>(enc, x, 0, [4, 4], 5, true, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode<byte, float>(enc, x, 8, [4, 4], 0, false, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode<uint, float>(enc, big, 0, [4], 0, false, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.Encode<byte, ushort>(enc, x, 0, [4], 0, false, s, 0, null, 0, ox, 0));
        d.Submit(enc);
        await Task.Yield();
    }

    // onemodel's CPU kernel: round_half_even(x / scale) + zeroPoint in float32, NaN to the lowest value, then clamp.
    static int QuantizeOne(float x, float scale, int zeroPoint, int lo, int hi)
    {
        var v = MathF.Round(x / scale) + zeroPoint;
        if (float.IsNaN(v))
        {
            return lo;
        }

        return v < lo ? lo : v > hi ? hi : (int)v;
    }

    // Mostly random values around the scale's unit; some exact ties (x / scale = k + 0.5 when the scale is a power of two), saturating values, and with
    // special = true NaN, infinities, subnormals and zeros (a browser's shader language gives no guarantees for those).
    static float[] QuantizeInput(int n, int seed, bool special)
    {
        var x = Rig.Random(n, seed, -40f, 40f);
        for (var i = 0; i < n; i++)
        {
            switch (i % 13)
            {
                case 3: x[i] = (i % 5 - 2) * 0.25f; break;            // ties for scale 0.5 and 0.25-multiples
                case 5: x[i] = (i % 2 == 0 ? 1 : -1) * (100f + i); break; // saturates
                case 7: x[i] = (i % 9 - 4) + 0.5f; break;             // ties for scale 1
                case 9 when special: x[i] = i % 4 == 0 ? float.NaN : (i % 4 == 1 ? float.PositiveInfinity : (i % 4 == 2 ? float.NegativeInfinity : 1e-42f)); break;
                case 11 when special: x[i] = i % 2 == 0 ? 0f : -0f; break;
            }
        }

        return x;
    }

    static async Task<int> QuantizeCase<TOut>(GpuDevice d, DequantizeKernels k, int[] shape, int axis, bool perAxis, bool withZero, bool halfInput, float[] scales, int seed)
        where TOut : unmanaged
    {
        var signed = typeof(TOut) == typeof(sbyte);
        var n = IndexRig.Count(shape);
        var rank = shape.Length;
        var ax = axis < 0 ? axis + rank : axis;
        var axisLen = perAxis ? shape[ax] : 1;
        long inner = 1;
        if (perAxis)
        {
            for (var i = ax + 1; i < rank; i++)
            {
                inner *= shape[i];
            }
        }

        var x = QuantizeInput(n + OffX, seed, !Rig.Browser);
        var xh = Rig.ToHalf(x);
        if (halfInput)
        {
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (float)xh[i];
            }
        }

        var scale = new float[axisLen + OffS];
        for (var i = 0; i < scale.Length; i++)
        {
            scale[i] = scales[i % scales.Length];
        }

        var zeros = Quantized<TOut>(axisLen + OffZ, seed + 2);
        var output = new TOut[GpuKernelBuffers.PaddedBoolLength(n + OffO + 3)];
        Array.Fill(output, (TOut)(object)(signed ? (sbyte)-77 : (byte)177));
        using var bx = halfInput ? null : d.CreateBuffer<float>(x);
        using var bh = halfInput ? d.CreateBuffer<Half>(xh) : null;
        using var bs = d.CreateBuffer<float>(scale);
        using var bz = d.CreateBuffer<TOut>(zeros);
        using var bo = d.CreateBuffer<TOut>(output);
        Rig.Run(d, e =>
        {
            if (halfInput)
            {
                k.EncodeQuantize(e, bh!, OffX, shape, axis, perAxis, bs, OffS, withZero ? bz : null, OffZ, bo, OffO);
            }
            else
            {
                k.EncodeQuantize(e, bx!, OffX, shape, axis, perAxis, bs, OffS, withZero ? bz : null, OffZ, bo, OffO);
            }
        });
        var got = await Rig.DownloadAsync(bo);
        int lo = signed ? -128 : 0, hi = signed ? 127 : 255;
        var bad = 0;
        for (var j = 0; j < n; j++)
        {
            var a = perAxis ? (int)((j / inner) % axisLen) : 0;
            var zp = withZero ? Value(zeros[OffZ + a]) : 0;
            var want = QuantizeOne(x[OffX + j], scale[OffS + a], zp, lo, hi);
            var g = Value(got[OffO + j]);
            bad += g == want ? 0 : 1;
        }

        for (var j = 0; j < OffO; j++)
        {
            bad += IndexRig.Same(got[j], output[j]) ? 0 : 1;
        }

        for (var j = OffO + n; j < output.Length; j++)
        {
            bad += IndexRig.Same(got[j], output[j]) ? 0 : 1;
        }

        k.ReleaseBindings();
        return bad;
    }

    [Fact]
    public async Task QuantizeIsExact()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new DequantizeKernels(d);
        (int[] Shape, int Axis, bool PerAxis)[] cases =
        [
            ([1], 0, false), ([7], 0, true), ([5, 6], 0, true), ([5, 6], 1, true), ([5, 6], -1, false), ([2, 3, 4, 5], 1, true),
            ([2, 3, 4, 5], -1, true), ([1031, 3], 0, true), ([3, 257], 1, true), ([4, 1, 9], 2, true), ([100003], 0, false), ([384, 768], 1, true),
        ];

        // Powers of two (exact ties), values whose division rounds, a scale near 1/255 as DynamicQuantizeLinear makes, and a very small and a large one.
        float[][] scaleSets = [[0.5f], [1f], [0.0137f, 0.25f, 3.3f], [1f / 255f, 0.09999f], [1e-6f, 1e3f, 0.7f]];
        var bad = 0;
        var count = 0;
        var seed = 1;
        foreach (var (shape, axis, perAxis) in cases)
        {
            foreach (var withZero in new[] { true, false })
            {
                var scales = scaleSets[count % scaleSets.Length];
                bad += await QuantizeCase<byte>(d, k, shape, axis, perAxis, withZero, false, scales, seed++);
                bad += await QuantizeCase<sbyte>(d, k, shape, axis, perAxis, withZero, false, scales, seed++);
                bad += await QuantizeCase<byte>(d, k, shape, axis, perAxis, withZero, true, scales, seed++);
                bad += await QuantizeCase<sbyte>(d, k, shape, axis, perAxis, withZero, true, scales, seed++);
                count += 4;
            }
        }

        Rig.Row(d, "quantize", "f32/f16 to u8/i8", $"{count} cases (per tensor and per axis, with and without zero point, odd offsets, ties, saturation{(Rig.Browser ? "" : ", NaN, infinities, subnormals")}, five scale sets)", $"mismatching bytes {bad}");
        Assert.Equal(0, bad);
    }

    [Fact]
    public async Task QuantizeRejectsBadArguments()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new DequantizeKernels(d);
        using var x = d.CreateBuffer<float>(16);
        using var s = d.CreateBuffer<float>(4);
        using var o = d.CreateBuffer<byte>(16);
        using var o16 = d.CreateBuffer<short>(16);
        using var xi = d.CreateBuffer<int>(16);
        var enc = d.BeginCommands();
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<float, byte>(enc, x, 0, [4, 4], 1, true, s, 1, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<float, byte>(enc, x, 0, [4, 4], 5, true, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<float, byte>(enc, x, 8, [4, 4], 0, false, s, 0, null, 0, o, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<float, byte>(enc, x, 0, [4, 4], 0, false, s, 0, null, 0, o, 3));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<float, short>(enc, x, 0, [4], 0, false, s, 0, null, 0, o16, 0));
        Assert.Throws<GpuKernelShapeException>(() => k.EncodeQuantize<int, byte>(enc, xi, 0, [4], 0, false, s, 0, null, 0, o, 0));
        d.Submit(enc);
        await Task.Yield();
    }
}
