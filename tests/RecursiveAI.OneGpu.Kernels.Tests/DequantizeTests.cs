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
}
