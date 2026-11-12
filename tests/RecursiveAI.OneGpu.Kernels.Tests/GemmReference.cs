using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>One matrix-multiplication case: the problem plus how the operands are laid out in their buffers.</summary>
internal sealed record GemmCase(string Name, GemmProblem Problem, int PadA = 0, int PadB = 0, int PadC = 0)
{
    public override string ToString() => Name;
}

/// <summary>A plain double-precision matrix multiplication and the error measures used to judge the GPU against it.</summary>
internal static class GemmReference
{
    public static float[] RandomFloats(int count, uint seed)
    {
        var r = new float[count];
        var s = seed * 2654435761u + 12345u;
        for (var i = 0; i < count; i++)
        {
            s = (s * 1664525u) + 1013904223u;
            r[i] = ((int)(s >> 8) - (1 << 23)) / (float)(1 << 23);
        }

        return r;
    }

    public static ushort ToBFloat16(float f)
    {
        var bits = BitConverter.SingleToUInt32Bits(f);
        // Round to nearest even on the 16 dropped bits.
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    public static float FromBFloat16(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    /// <summary>Reference result in double. <paramref name="a"/> and <paramref name="b"/> are the float values the GPU sees (already rounded to the input type).</summary>
    public static double[] Multiply(GemmProblem p, float[] a, float[] b, float[] cOld, int lda, int ldb, int ldc)
    {
        var result = new double[cOld.Length];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = cOld[i];
        }

        for (var z = 0; z < p.Batch; z++)
        {
            var baseA = p.OffsetA + (z * p.StrideA);
            var baseB = p.OffsetB + (z * p.StrideB);
            var baseC = p.OffsetC + (z * p.StrideC);
            Parallel.For(0, p.M, m =>
            {
                for (var n = 0; n < p.N; n++)
                {
                    double acc = 0;
                    for (var k = 0; k < p.K; k++)
                    {
                        var av = a[baseA + (p.TransposeA ? (k * lda) + m : (m * lda) + k)];
                        var bv = b[baseB + (p.TransposeB ? (n * ldb) + k : (k * ldb) + n)];
                        acc += (double)av * bv;
                    }

                    var at = baseC + (m * ldc) + n;
                    result[at] = (p.Alpha * acc) + (p.Beta != 0f ? p.Beta * (double)cOld[at] : 0.0);
                }
            });
        }

        return result;
    }

    /// <summary>Scaled error max|g-r|/max|r|, and the largest element-wise relative error over elements with |r| above 1e-3 of the largest (smaller ones are cancellation noise).</summary>
    public static (double Scaled, double Relative) Compare(float[] gpu, double[] reference, ReadOnlySpan<bool> written)
    {
        double maxRef = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            if (written[i])
            {
                maxRef = Math.Max(maxRef, Math.Abs(reference[i]));
            }
        }

        double maxAbs = 0;
        double maxRel = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            if (!written[i])
            {
                // Elements outside the output region must be untouched.
                if (gpu[i] != (float)reference[i])
                {
                    return (double.PositiveInfinity, double.PositiveInfinity);
                }

                continue;
            }

            var err = Math.Abs(gpu[i] - reference[i]);
            if (double.IsNaN(err))
            {
                return (double.NaN, double.NaN);
            }

            maxAbs = Math.Max(maxAbs, err);
            if (Math.Abs(reference[i]) > 1e-3 * maxRef)
            {
                maxRel = Math.Max(maxRel, err / Math.Abs(reference[i]));
            }
        }

        return (maxRef == 0 ? maxAbs : maxAbs / maxRef, maxRel);
    }

    public static string Hash(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes))[..16].ToLowerInvariant();
    }

    public static string Fmt(double v) => v.ToString("0.00e+00", CultureInfo.InvariantCulture);

    /// <summary>Which elements of C the problem writes.</summary>
    public static bool[] WrittenMask(in GemmProblem p, int length, int ldc)
    {
        var mask = new bool[length];
        for (var z = 0; z < p.Batch; z++)
        {
            for (var m = 0; m < p.M; m++)
            {
                for (var n = 0; n < p.N; n++)
                {
                    mask[p.OffsetC + (z * p.StrideC) + (m * ldc) + n] = true;
                }
            }
        }

        return mask;
    }

    public static IEnumerable<GemmCase> EdgeCases()
    {
        static GemmCase C(string name, int m, int n, int k, Action<GemmProblemBuilder>? tweak = null, int padA = 0, int padB = 0, int padC = 0)
        {
            var b = new GemmProblemBuilder(m, n, k);
            tweak?.Invoke(b);
            return new GemmCase(name, b.P, padA, padB, padC);
        }

        yield return C("1x1x1", 1, 1, 1);
        yield return C("k=1", 33, 70, 1);
        yield return C("m=1", 1, 200, 77);
        yield return C("n=1", 150, 1, 77);
        yield return C("tile-1", 63, 63, 15);
        yield return C("tile", 64, 64, 16);
        yield return C("tile+1", 65, 65, 17);
        yield return C("odd", 129, 257, 31);
        yield return C("prime", 97, 101, 103);
        yield return C("k=0 beta", 20, 20, 0, t => t.P.Beta = 0.5f);
        yield return C("transA", 70, 90, 50, t => t.P.TransposeA = true);
        yield return C("transB", 70, 90, 50, t => t.P.TransposeB = true);
        yield return C("transAB", 71, 91, 49, t => { t.P.TransposeA = true; t.P.TransposeB = true; });
        yield return C("alpha beta", 80, 60, 40, t => { t.P.Alpha = -0.75f; t.P.Beta = 0.5f; });
        yield return C("beta=0 nan", 40, 40, 20, t => t.P.Alpha = 2f);
        yield return C("ld padded", 50, 60, 33, null, 7, 5, 9);
        yield return C("ld padded transAB", 50, 60, 33, t => { t.P.TransposeA = true; t.P.TransposeB = true; t.P.Beta = 1f; }, 3, 11, 2);
        yield return C("offsets", 45, 55, 35, t => { t.P.OffsetA = 3; t.P.OffsetB = 5; t.P.OffsetC = 7; });
        yield return C("batch", 33, 47, 19, t => { t.P.Batch = 5; t.P.StrideA = (33 * 19) + 3; t.P.StrideB = (19 * 47) + 1; t.P.StrideC = (33 * 47) + 2; });
        yield return C("batch shared B", 40, 50, 30, t => { t.P.Batch = 3; t.P.StrideA = 40 * 30; t.P.StrideB = 0; t.P.StrideC = 40 * 50; t.P.Beta = 1f; });
        // At most four rows of A: a kernel of its own on Vulkan and WebGPU (float32), in both layouts of B, with remainders in k and in the columns.
        yield return C("skinny m=2", 2, 130, 77);
        yield return C("skinny m=3 transB", 3, 45, 301, t => t.P.TransposeB = true);
        yield return C("skinny m=4 alpha beta", 4, 100, 130, t => { t.P.Alpha = 0.5f; t.P.Beta = 1.5f; });
        yield return C("skinny m=4 transB alpha beta", 4, 19, 200, t => { t.P.Alpha = -1.25f; t.P.Beta = 0.5f; t.P.TransposeB = true; });
        yield return C("skinny padded offsets", 3, 70, 150, t => { t.P.OffsetA = 5; t.P.OffsetB = 9; t.P.OffsetC = 3; }, 7, 5, 4);
        yield return C("skinny batch", 1, 90, 100, t => { t.P.Batch = 6; t.P.StrideA = 100 + 3; t.P.StrideB = 100 * 90; t.P.StrideC = 90 + 1; });
        yield return C("skinny batch transB", 2, 33, 70, t => { t.P.Batch = 3; t.P.TransposeB = true; t.P.StrideA = 2 * 70; t.P.StrideB = 33 * 70; t.P.StrideC = 2 * 33; });
        yield return C("m=5 not skinny", 5, 120, 200);
        // A small grid with a long k: k is cut into ranges whose planes a second pass adds in order (ranges are multiples of 32 terms; k not a multiple of 32 in several).
        yield return C("split k=700", 70, 90, 700);
        yield return C("split k=1000 alpha beta", 65, 65, 1000, t => { t.P.Alpha = 0.75f; t.P.Beta = -0.5f; });
        yield return C("split transA transB", 64, 130, 1500, t => { t.P.TransposeA = true; t.P.TransposeB = true; });
        yield return C("split batch offsets padded", 40, 70, 777, t => { t.P.Batch = 3; t.P.StrideA = (40 * 777) + 2; t.P.StrideB = 0; t.P.StrideC = (40 * 71) + 5; t.P.OffsetA = 4; t.P.OffsetC = 6; t.P.Beta = 1f; }, 3, 2, 1);
        yield return C("split k=129", 20, 20, 129);
        yield return C("batch transB", 64, 64, 64, t => { t.P.Batch = 4; t.P.TransposeB = true; t.P.StrideA = 64 * 64; t.P.StrideB = 64 * 64; t.P.StrideC = 64 * 64; });
    }

    /// <summary>GEMM shapes of BERT-base, MiniLM-L6 and Whisper-small at one sequence and a batch. <paramref name="full"/> adds the big ones.</summary>
    public static IEnumerable<GemmCase> ModelCases(bool full)
    {
        static GemmCase C(string name, int m, int n, int k, bool transB = false, int batch = 1, bool shareB = true)
        {
            var b = new GemmProblemBuilder(m, n, k);
            b.P.TransposeB = transB;
            if (batch > 1)
            {
                b.P.Batch = batch;
                b.P.StrideA = m * k;
                b.P.StrideB = shareB ? 0 : k * n;
                b.P.StrideC = m * n;
            }

            return new GemmCase(name, b.P);
        }

        // MiniLM-L6: hidden 384, feed-forward 1536, 12 heads of 64... (head size 32 for 384/12).
        yield return C("minilm qkv 128", 128, 384, 384);
        yield return C("minilm ffn-up 128", 128, 1536, 384);
        yield return C("minilm ffn-down 128", 128, 384, 1536);
        yield return C("minilm ffn-up 8x128", 1024, 1536, 384);
        yield return C("minilm scores 12x128", 128, 128, 32, transB: true, batch: 12, shareB: false);
        // BERT-base: hidden 768, feed-forward 3072, 12 heads of 64.
        yield return C("bert proj 128", 128, 768, 768);
        yield return C("bert ffn-up 128", 128, 3072, 768);
        yield return C("bert ffn-down 128", 128, 768, 3072);
        yield return C("bert ffn-up 8x128", 1024, 3072, 768);
        yield return C("bert scores 12x128", 128, 128, 64, transB: true, batch: 12, shareB: false);
        yield return C("bert context 12x128", 128, 64, 128, batch: 12, shareB: false);
        // Whisper-small decoder at one token: 768 hidden, 3072 feed-forward, 51865-token vocabulary.
        yield return C("whisper dec proj 1", 1, 768, 768);
        yield return C("whisper dec ffn-up 1", 1, 3072, 768);
        yield return C("whisper dec logits 1", 1, 51865, 768, transB: true);
        if (full)
        {
            yield return C("whisper enc proj 1500", 1500, 768, 768);
            yield return C("whisper enc ffn-up 1500", 1500, 3072, 768);
            yield return C("whisper enc ffn-down 1500", 1500, 768, 3072);
            yield return C("whisper enc scores 12x1500", 1500, 1500, 64, transB: true, batch: 12, shareB: false);
            yield return C("bert ffn-down 8x128", 1024, 768, 3072);
            yield return C("whisper dec logits 448", 448, 51865, 768, transB: true);
        }
    }
}

/// <summary>Mutable holder so case lists can tweak the struct problem in a lambda.</summary>
internal sealed class GemmProblemBuilder(int m, int n, int k)
{
    public GemmProblem P = new(m, n, k);
}
