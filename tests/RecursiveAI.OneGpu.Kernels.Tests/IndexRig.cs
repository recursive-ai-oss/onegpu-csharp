using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Payload generation, bit-exact comparison and index maps shared by the gather, scatter and layout tests.</summary>
internal static class IndexRig
{
    public static bool Same<T>(T a, T b)
        where T : unmanaged =>
        MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in a)).SequenceEqual(MemoryMarshal.AsBytes(new ReadOnlySpan<T>(in b)));

    /// <summary>A buffer length that keeps half-precision buffers on a 32-bit word.</summary>
    public static int Padded<T>(int n)
        where T : unmanaged => typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n) : (typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte)) ? GpuKernelBuffers.PaddedBoolLength(Math.Max(n, 1)) : n;

    /// <summary>A pseudo-random payload element: all bit patterns of the type's width are possible except NaN halves and floats, which are made finite.</summary>
    public static T Make<T>(int i)
        where T : unmanaged
    {
        var h = (uint)i * 2654435761u + 0x9E3779B9u;
        h ^= h >> 15;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        if (typeof(T) == typeof(float))
        {
            return (T)(object)(((int)(h & 0xFFFFF) - 0x80000) / 1024f);
        }

        if (typeof(T) == typeof(Half))
        {
            return (T)(object)(Half)(((int)(h & 0xFFF) - 0x800) / 16f);
        }

        if (typeof(T) == typeof(int))
        {
            return (T)(object)(int)h;
        }

        if (typeof(T) == typeof(uint))
        {
            return (T)(object)h;
        }

        if (typeof(T) == typeof(byte))
        {
            return (T)(object)(byte)(h >> 8);
        }

        if (typeof(T) == typeof(sbyte))
        {
            return (T)(object)(sbyte)(h >> 8);
        }

        if (typeof(T) == typeof(long))
        {
            return (T)(object)(((long)h << 32) ^ (uint)(i * 40503 + 17));
        }

        throw new NotSupportedException(typeof(T).Name);
    }

    public static T[] Fill<T>(int length, int seed)
        where T : unmanaged
    {
        var a = new T[Padded<T>(length)];
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = Make<T>(i + seed * 7919);
        }

        return a;
    }

    public static TIdx FromLong<TIdx>(long v)
        where TIdx : unmanaged => typeof(TIdx) == typeof(int) ? (TIdx)(object)(int)v : (TIdx)(object)v;

    public static string TypeName<T>() => typeof(T) == typeof(float) ? "f32" : typeof(T) == typeof(Half) ? "f16" : typeof(T) == typeof(int) ? "i32" : typeof(T) == typeof(uint) ? "u32" : typeof(T) == typeof(long) ? "i64" : typeof(T).Name;

    public static long[] Strides(int[] shape)
    {
        var s = new long[shape.Length];
        long run = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            s[i] = run;
            run *= shape[i];
        }

        return s;
    }

    public static int Count(int[] shape) => shape.Aggregate(1, (a, b) => a * b);

    /// <summary>Row-major coordinates of flat position <paramref name="j"/>.</summary>
    public static int[] Unravel(int[] shape, int j)
    {
        var c = new int[shape.Length];
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            c[i] = shape[i] == 0 ? 0 : j % shape[i];
            j = shape[i] == 0 ? 0 : j / shape[i];
        }

        return c;
    }

    public static long Norm(long v, long len) => v < 0 ? v + len : v;

    /// <summary>A deterministic index in [-len, len), with some negative values.</summary>
    public static long RandomIndex(int i, long len, int seed)
    {
        var h = (uint)(i + seed * 104729) * 2246822519u;
        h ^= h >> 13;
        h *= 3266489917u;
        var v = (long)(h % (uint)(2 * len)) - len;
        return v;
    }

    public static string Digest<T>(T[] data)
        where T : unmanaged => Rig.Hash<T>(data);
}
