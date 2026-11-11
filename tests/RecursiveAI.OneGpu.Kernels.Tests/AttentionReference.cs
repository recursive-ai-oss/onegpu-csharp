namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>A CPU model of the attention kernels in double precision, and the helpers that put logical tensors into buffers in any layout.</summary>
internal static class AttentionReference
{
    /// <summary>Which sequence layout a tensor is stored in.</summary>
    internal enum Layout
    {
        HeadsFirst,
        PositionsFirst,
    }

    internal sealed record Case(string Name, int B, int H, int Hkv, int Sq, int Sk, int D, bool Causal = false, int? CausalOff = null, int Mask = 0, bool Lens = false,
        int Layouts = 0, GpuAttentionMaskedRows Policy = GpuAttentionMaskedRows.Nan);

    // Mask kinds: 0 none; 1 additive [1,1,Sq,Sk] random with some -inf; 2 additive [B,H,Sq,Sk]; 3 key padding [B,1,1,Sk] with -inf at the tail;
    // 4 [1,1,Sq,Sk] in which every third row is entirely -inf; 5 one NaN in the [1,1,Sq,Sk] mask.
    internal static int[] MaskShape(Case c) => c.Mask switch
    {
        0 => [],
        1 or 4 or 5 => [1, 1, c.Sq, c.Sk],
        2 => [c.B, c.H, c.Sq, c.Sk],
        _ => [c.B, 1, 1, c.Sk],
    };

    internal static float[] MakeMask(Case c)
    {
        var shape = MaskShape(c);
        if (shape.Length == 0)
        {
            return [];
        }

        var n = 1;
        foreach (var s in shape)
        {
            n *= s;
        }

        // WGSL does not promise infinities behave, so the browser hides keys with a large finite value instead.
        var hide = Rig.Browser ? -1e30f : float.NegativeInfinity;
        var m = Rig.Random(n, c.Sq + 3 * c.Sk, -1f, 1f);
        switch (c.Mask)
        {
            case 1:
                for (var i = 0; i < n; i++)
                {
                    if (i % 7 == 3)
                    {
                        m[i] = hide;
                    }
                }

                break;
            case 3:
                for (var b = 0; b < c.B; b++)
                {
                    var valid = Math.Max(1, c.Sk - (b + 1) * c.Sk / (c.B + 3));
                    for (var j = 0; j < c.Sk; j++)
                    {
                        m[(b * c.Sk) + j] = j < valid ? 0f : hide;
                    }
                }

                break;
            case 4:
                for (var i = 0; i < c.Sq; i += 3)
                {
                    for (var j = 0; j < c.Sk; j++)
                    {
                        m[(i * c.Sk) + j] = hide;
                    }
                }

                break;
            case 5:
                m[(c.Sq / 2 * c.Sk) + (c.Sk / 2)] = float.NaN;
                break;
        }

        return m;
    }

    internal static int[] MakeLens(Case c)
    {
        var l = new int[c.B];
        for (var b = 0; b < c.B; b++)
        {
            l[b] = c.Sk - (b * c.Sk / (c.B + 1)) - (b == 1 ? 1 : 0);
        }

        if (c.B > 2)
        {
            l[2] = 0;
        }

        return l;
    }

    /// <summary>Row-wise reference for the sampled rows (every <paramref name="step"/>th row of the flattened batch, head, query index, and the last one).</summary>
    internal static (int[] Rows, double[] Want) Compute(Case c, float scale, float[] q, float[] k, float[] v, float[] mask, int[]? lens)
    {
        var rowsTotal = c.B * c.H * c.Sq;
        var step = Math.Max(1, rowsTotal / 192);
        var rows = new List<int>();
        for (var r = 0; r < rowsTotal; r += step)
        {
            rows.Add(r);
        }

        if (rows[^1] != rowsTotal - 1)
        {
            rows.Add(rowsTotal - 1);
        }

        var want = new double[rows.Count * c.D];
        var group = c.H / c.Hkv;
        var maskShape = MaskShape(c);
        var off = c.CausalOff ?? (c.Sk - c.Sq);
        var s = new double[c.Sk];
        for (var ri = 0; ri < rows.Count; ri++)
        {
            var r = rows[ri];
            var i = r % c.Sq;
            var h = r / c.Sq % c.H;
            var b = r / (c.Sq * c.H);
            var kvh = h / group;
            var kEnd = lens is null ? c.Sk : Math.Clamp(lens[b], 0, c.Sk);
            double max = double.NegativeInfinity;
            var poisoned = false;
            for (var j = 0; j < c.Sk; j++)
            {
                var visible = j < kEnd && (!c.Causal || j <= i + off);
                if (!visible)
                {
                    s[j] = double.NegativeInfinity;
                    continue;
                }

                double dot = 0;
                var qo = ((((b * c.H) + h) * c.Sq) + i) * c.D;
                var ko = ((((b * c.Hkv) + kvh) * c.Sk) + j) * c.D;
                for (var x = 0; x < c.D; x++)
                {
                    dot += (double)q[qo + x] * k[ko + x];
                }

                var sc = (double)(float)(dot * scale);
                if (maskShape.Length > 0)
                {
                    var mb = maskShape[0] == 1 ? 0 : b;
                    var mh = maskShape[1] == 1 ? 0 : h;
                    var mi = maskShape[2] == 1 ? 0 : i;
                    sc += mask[((((mb * maskShape[1]) + mh) * maskShape[2] + mi) * c.Sk) + j];
                }

                if (double.IsNaN(sc) || double.IsPositiveInfinity(sc))
                {
                    poisoned = true;
                }

                s[j] = sc;
                max = Math.Max(max, sc);
            }

            double sum = 0;
            var acc = new double[c.D];
            if (!double.IsNegativeInfinity(max) && !poisoned)
            {
                for (var j = 0; j < c.Sk; j++)
                {
                    if (double.IsNegativeInfinity(s[j]))
                    {
                        continue;
                    }

                    var w = Math.Exp(s[j] - max);
                    sum += w;
                    var vo = ((((b * c.Hkv) + kvh) * c.Sk) + j) * c.D;
                    for (var x = 0; x < c.D; x++)
                    {
                        acc[x] += w * v[vo + x];
                    }
                }
            }

            for (var x = 0; x < c.D; x++)
            {
                want[(ri * c.D) + x] = poisoned ? double.NaN : (sum == 0 ? (c.Policy == GpuAttentionMaskedRows.Zero ? 0 : double.NaN) : acc[x] / sum);
            }
        }

        return (rows.ToArray(), want);
    }

    /// <summary>Stores a logical [b, heads, rows, d] tensor in a buffer array with the given layout and a start offset; returns the array and its description.</summary>
    internal static (T[] Data, GpuAttentionLayout Layout, bool[] Used) Pack<T>(float[] logical, int batch, int heads, int rows, int d, Layout layout, int offset, Func<float, T> convert, int pad)
    {
        var l = layout == Layout.HeadsFirst ? GpuAttentionLayout.HeadsFirst(heads, rows, d, offset) : GpuAttentionLayout.PositionsFirst(heads, rows, d, offset);
        var length = offset + (batch == 0 ? 0 : ((batch - 1) * l.BatchStride) + ((heads - 1) * l.HeadStride) + ((rows - 1) * l.RowStride) + d) + pad;
        var data = new T[(Math.Max(length, 2) + 1) & ~1];
        var used = new bool[data.Length];
        if (logical.Length > 0)
        {
            for (var b = 0; b < batch; b++)
            {
                for (var h = 0; h < heads; h++)
                {
                    for (var r = 0; r < rows; r++)
                    {
                        for (var x = 0; x < d; x++)
                        {
                            var at = l.Offset + (b * l.BatchStride) + (h * l.HeadStride) + (r * l.RowStride) + x;
                            data[at] = convert(logical[((((b * heads) + h) * rows + r) * d) + x]);
                            used[at] = true;
                        }
                    }
                }
            }
        }

        return (data, l, used);
    }

    internal static float RoundHalf(float x) => (float)(Half)x;
}
