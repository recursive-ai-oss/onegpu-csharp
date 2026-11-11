using System.Globalization;
using System.Security.Cryptography;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>Deterministic data, buffer round trips and error measures shared by the kernel tests.</summary>
internal static class Rig
{
    public static float[] Random(int n, int seed, float lo, float hi)
    {
        var s = (uint)seed * 2654435761u + 12345u;
        var r = new float[n];
        for (var i = 0; i < n; i++)
        {
            s = s * 1664525u + 1013904223u;
            r[i] = lo + (hi - lo) * ((s >> 8) / 16777216f);
        }

        return r;
    }

    public static Half[] ToHalf(float[] v)
    {
        var h = new Half[GpuKernelBuffers.PaddedHalfLength(v.Length)];
        for (var i = 0; i < v.Length; i++)
        {
            h[i] = (Half)v[i];
        }

        return h;
    }

    public static GpuBuffer<T> Upload<T>(GpuDevice d, T[] data)
        where T : unmanaged => d.CreateBuffer<T>(data);

    public static T[] Download<T>(GpuBuffer<T> b, int count = -1)
        where T : unmanaged
    {
        var r = new T[count < 0 ? b.Length : count];
        b.ReadAsync(r).AsTask().GetAwaiter().GetResult();
        return r;
    }

    public static void Run(GpuDevice d, Action<GpuCommandEncoder> record)
    {
        var enc = d.BeginCommands();
        record(enc);
        d.Submit(enc);
    }

    /// <summary>max |g - r| / max |r| over the finite reference values, and the largest element-wise relative error.</summary>
    public static (double Scaled, double Rel) Errors(ReadOnlySpan<double> got, ReadOnlySpan<double> want)
    {
        double maxRef = 0, maxDiff = 0, maxRel = 0;
        for (var i = 0; i < want.Length; i++)
        {
            if (double.IsFinite(want[i]))
            {
                maxRef = Math.Max(maxRef, Math.Abs(want[i]));
            }
        }

        for (var i = 0; i < want.Length; i++)
        {
            if (!double.IsFinite(want[i]) || !double.IsFinite(got[i]))
            {
                // Non-finite values must agree exactly (NaN with NaN, signed infinities with themselves).
                var same = double.IsNaN(want[i]) ? double.IsNaN(got[i]) : got[i] == want[i];
                if (!same)
                {
                    return (double.PositiveInfinity, double.PositiveInfinity);
                }

                continue;
            }

            var diff = Math.Abs(got[i] - want[i]);
            maxDiff = Math.Max(maxDiff, diff);
            if (want[i] != 0)
            {
                maxRel = Math.Max(maxRel, diff / Math.Abs(want[i]));
            }
        }

        return (maxRef == 0 ? maxDiff : maxDiff / maxRef, maxRel);
    }

    public static double[] D(IReadOnlyList<float> v, int n)
    {
        var r = new double[n];
        for (var i = 0; i < n; i++)
        {
            r[i] = v[i];
        }

        return r;
    }

    public static double[] D(IReadOnlyList<Half> v, int n)
    {
        var r = new double[n];
        for (var i = 0; i < n; i++)
        {
            r[i] = (double)v[i];
        }

        return r;
    }

    public static int BitMismatches(ReadOnlySpan<float> got, ReadOnlySpan<float> want)
    {
        var bad = 0;
        for (var i = 0; i < want.Length; i++)
        {
            var same = float.IsNaN(want[i]) ? float.IsNaN(got[i]) : BitConverter.SingleToUInt32Bits(got[i]) == BitConverter.SingleToUInt32Bits(want[i]);
            if (!same)
            {
                bad++;
            }
        }

        return bad;
    }

    public static int BitMismatches(ReadOnlySpan<Half> got, ReadOnlySpan<Half> want)
    {
        var bad = 0;
        for (var i = 0; i < want.Length; i++)
        {
            var same = Half.IsNaN(want[i]) ? Half.IsNaN(got[i]) : BitConverter.HalfToUInt16Bits(got[i]) == BitConverter.HalfToUInt16Bits(want[i]);
            if (!same)
            {
                bad++;
            }
        }

        return bad;
    }

    public static string Hash<T>(ReadOnlySpan<T> data)
        where T : unmanaged =>
        Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data)))[..16];

    static readonly object Gate = new();

    /// <summary>Prints one result row and appends it to the file named by ONEGPU_PARITY_REPORT, so a run on each device leaves a table.</summary>
    public static void Row(GpuDevice d, string family, string dtype, string what, string result)
    {
        var line = string.Create(CultureInfo.InvariantCulture, $"{d.Adapter.Name} | {d.Backend} | {family} | {dtype} | {what} | {result}");
        Console.WriteLine("ROW " + line);
#if !WEB_BACKEND
        if (Environment.GetEnvironmentVariable("ONEGPU_PARITY_REPORT") is { Length: > 0 } path)
        {
            lock (Gate)
            {
                File.AppendAllText(path, line + "\n");
            }
        }
#endif
    }

    public static string Sci(double v) => v.ToString("0.###e+0", CultureInfo.InvariantCulture);
}
