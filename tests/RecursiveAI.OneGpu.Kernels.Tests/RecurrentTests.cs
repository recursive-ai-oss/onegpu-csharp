using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>RNN, GRU and LSTM against a double-precision reference: every cell, direction, layout and option, odd shapes and the shapes of a text-line recognizer.</summary>
public sealed class RecurrentTests
{
    const int Lead = 3;

    sealed record Config(GpuRecurrentCell Cell, GpuRecurrentDirection Direction, bool Layout1, int Seq, int Batch, int Input, int Hidden, bool Bias, bool Lens, bool Init, bool Peep, float Clip, bool InputForget, bool LinearBeforeReset, bool Y, bool Yh, bool Yc, bool MaxLength);

    sealed class Data
    {
        public required RecurrentProblem Problem { get; init; }
        public required RecurrentReference.Input Input { get; init; }
        public required float[] X { get; init; }
        public required float[] W { get; init; }
        public required float[] R { get; init; }
        public float[]? B { get; init; }
        public float[]? P { get; init; }
        public int[]? Lens { get; init; }
        public float[]? H0 { get; init; }
        public float[]? C0 { get; init; }
    }

    static RecurrentProblem ProblemOf(Config c, GpuRecurrentActivations? acts = null, GpuRecurrentActivations? reverse = null) =>
        new(c.Cell, c.Seq, c.Batch, c.Input, c.Hidden)
        {
            Direction = c.Direction,
            Layout1 = c.Layout1,
            Clip = c.Clip,
            InputForget = c.InputForget,
            LinearBeforeReset = c.LinearBeforeReset,
            Activations = acts,
            ReverseActivations = reverse,
        };

    // Values that a Half holds exactly, so the reference sees what the device sees.
    static float[] Make(int n, int seed, float amplitude, bool half)
    {
        var v = Rig.Random(n, seed, -amplitude, amplitude);
        if (half)
        {
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(Half)v[i];
            }
        }

        return v;
    }

    static Data MakeData(Config c, bool half, int seed, GpuRecurrentActivations? acts = null, GpuRecurrentActivations? reverse = null)
    {
        var p = ProblemOf(c, acts, reverse);
        int dirs = p.Directions, gh = p.Gates * p.Hidden;
        var w = 1.2f / MathF.Sqrt(Math.Max(c.Input, c.Hidden));
        var x = Make(c.Seq * c.Batch * c.Input, seed, 1f, half);
        var ww = Make(dirs * gh * c.Input, seed + 1, w, half);
        var rr = Make(dirs * gh * c.Hidden, seed + 2, w, half);
        var b = c.Bias ? Make(dirs * 2 * gh, seed + 3, 0.3f, half) : null;
        var pp = c.Peep ? Make(dirs * 3 * c.Hidden, seed + 4, 0.5f, half) : null;
        var h0 = c.Init ? Make(dirs * c.Batch * c.Hidden, seed + 5, 0.5f, half) : null;
        var c0 = c.Init && c.Cell == GpuRecurrentCell.Lstm ? Make(dirs * c.Batch * c.Hidden, seed + 6, 0.5f, half) : null;
        int[]? lens = null;
        if (c.Lens)
        {
            lens = new int[c.Batch];
            for (var i = 0; i < lens.Length; i++)
            {
                lens[i] = i % 4 == 0 ? c.Seq : (i * 7 + seed) % (c.Seq + 1);
            }

            if (c.Batch > 1)
            {
                lens[1] = 0;
            }
        }

        if (c.MaxLength && lens is not null)
        {
            p.MaxLength = lens.Max();
        }

        var input = new RecurrentReference.Input
        {
            X = Array.ConvertAll(x, f => (double)f),
            W = Array.ConvertAll(ww, f => (double)f),
            R = Array.ConvertAll(rr, f => (double)f),
            B = b is null ? null : Array.ConvertAll(b, f => (double)f),
            P = pp is null ? null : Array.ConvertAll(pp, f => (double)f),
            Lens = lens,
            H0 = h0 is null ? null : Array.ConvertAll(h0, f => (double)f),
            C0 = c0 is null ? null : Array.ConvertAll(c0, f => (double)f),
        };
        return new Data { Problem = p, Input = input, X = x, W = ww, R = rr, B = b, P = pp, Lens = lens, H0 = h0, C0 = c0 };
    }

    sealed class Outputs
    {
        public float[]? Y;
        public float[]? Yh;
        public float[]? Yc;
        public int Violations;
    }

    static T[] Convert<T>(float[] v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return (T[])(object)v;
        }

        var h = new Half[v.Length];
        for (var i = 0; i < v.Length; i++)
        {
            h[i] = (Half)v[i];
        }

        return (T[])(object)h;
    }

    static float[] ToFloats<T>(T[] data, int lead, int n)
        where T : unmanaged
    {
        var r = new float[n];
        for (var i = 0; i < n; i++)
        {
            r[i] = data[lead + i] is float f ? f : (float)(Half)(object)data[lead + i];
        }

        return r;
    }

    // Uploads a tensor with Lead spare elements before it and a few after, all filled with a value the kernel must never read or overwrite.
    static GpuTensorRef<T> Place<T>(GpuDevice d, List<IDisposable> owned, T[] data, T fill, int extra = 3)
        where T : unmanaged
    {
        var padded = new T[IndexIfHalf<T>(Lead + data.Length + extra)];
        Array.Fill(padded, fill);
        Array.Copy(data, 0, padded, Lead, data.Length);
        var buffer = d.CreateBuffer<T>(padded);
        owned.Add(buffer);
        return new GpuTensorRef<T>(buffer, Lead);
    }

    static int IndexIfHalf<T>(int n)
        where T : unmanaged => typeof(T) == typeof(Half) ? GpuKernelBuffers.PaddedHalfLength(n) : n;

    static async Task<Outputs> Execute<T>(GpuDevice d, RecurrentKernels k, Config c, Data data, GpuBuffer<float>? xw = null, GpuBuffer<float>? state = null)
        where T : unmanaged
    {
        var owned = new List<IDisposable>();
        try
        {
            var fill = Convert<T>([777f])[0];
            var o = new RecurrentOperands<T>
            {
                X = Place(d, owned, Convert<T>(data.X), fill),
                W = Place(d, owned, Convert<T>(data.W), fill),
                R = Place(d, owned, Convert<T>(data.R), fill),
                B = data.B is null ? null : Place(d, owned, Convert<T>(data.B), fill),
                P = data.P is null ? null : Place(d, owned, Convert<T>(data.P), fill),
                SequenceLengths = data.Lens is null ? null : Place(d, owned, data.Lens, 99),
                InitialH = data.H0 is null ? null : Place(d, owned, Convert<T>(data.H0), fill),
                InitialC = data.C0 is null ? null : Place(d, owned, Convert<T>(data.C0), fill),
                Y = c.Y ? Place(d, owned, Convert<T>(new float[c.Seq * data.Problem.Directions * c.Batch * c.Hidden]), fill) : null,
                YH = c.Yh ? Place(d, owned, Convert<T>(new float[data.Problem.Directions * c.Batch * c.Hidden]), fill) : null,
                YC = c.Yc && c.Cell == GpuRecurrentCell.Lstm ? Place(d, owned, Convert<T>(new float[data.Problem.Directions * c.Batch * c.Hidden]), fill) : null,
            };

            // The output buffers start as the sentinel everywhere, including inside the tensor, so an element the kernel forgets shows up.
            Rig.Run(d, e => k.Encode(e, data.Problem, o, xw, state));
            var r = new Outputs();
            async Task<float[]?> Read(GpuTensorRef<T>? t, int n)
            {
                if (t is null)
                {
                    return null;
                }

                var all = await Rig.DownloadAsync(t.Value.Buffer);
                for (var i = 0; i < all.Length; i++)
                {
                    if (i >= Lead && i < Lead + n)
                    {
                        continue;
                    }

                    if (!IndexRig.Same(all[i], fill))
                    {
                        r.Violations++;
                    }
                }

                return ToFloats(all, Lead, n);
            }

            r.Y = await Read(o.Y, c.Seq * data.Problem.Directions * c.Batch * c.Hidden);
            r.Yh = await Read(o.YH, data.Problem.Directions * c.Batch * c.Hidden);
            r.Yc = await Read(o.YC, data.Problem.Directions * c.Batch * c.Hidden);
            return r;
        }
        finally
        {
            d.WaitIdle();
            foreach (var b in owned)
            {
                b.Dispose();
            }
        }
    }

    sealed class Worst
    {
        public double Scaled;
        public double MaxAbs;
        public int Cases;
        public int Violations;
        public int NonIdentical;
    }

    static void Measure(float[]? got, double[] want, bool wanted, Worst w)
    {
        if (!wanted || got is null)
        {
            return;
        }

        var g = Array.ConvertAll(got, f => (double)f);
        var (scaled, _) = Rig.Errors(g, want);
        w.Scaled = Math.Max(w.Scaled, scaled);
        for (var i = 0; i < want.Length; i++)
        {
            w.MaxAbs = Math.Max(w.MaxAbs, Math.Abs(g[i] - want[i]));
        }
    }

    static async Task<string> Check<T>(GpuDevice d, RecurrentKernels k, Config c, int seed, Worst w, bool twice, GpuRecurrentActivations? acts = null, GpuRecurrentActivations? reverse = null)
        where T : unmanaged
    {
        var data = MakeData(c, typeof(T) == typeof(Half), seed, acts, reverse);
        var want = RecurrentReference.Run(data.Problem, data.Input);
        var got = await Execute<T>(d, k, c, data);
        w.Cases++;
        w.Violations += got.Violations;
        Measure(got.Y, want.Y, c.Y, w);
        Measure(got.Yh, want.Yh, c.Yh, w);
        Measure(got.Yc, want.Yc, c.Yc && c.Cell == GpuRecurrentCell.Lstm, w);
        if (twice)
        {
            var again = await Execute<T>(d, k, c, data);
            var same = Same(got.Y, again.Y) && Same(got.Yh, again.Yh) && Same(got.Yc, again.Yc);
            if (!same)
            {
                w.NonIdentical++;
            }
        }

        return $"{c.Cell} {c.Direction} layout{(c.Layout1 ? 1 : 0)} S{c.Seq} B{c.Batch} I{c.Input} H{c.Hidden}";
    }

    static bool Same(float[]? a, float[]? b) => a is null || (b is not null && MemoryMarshal.AsBytes<float>(a).SequenceEqual(MemoryMarshal.AsBytes<float>(b)));

    static readonly (int Seq, int Batch, int Input, int Hidden)[] Shapes =
    [
        (1, 1, 1, 1), (5, 3, 7, 9), (6, 2, 33, 31), (9, 1, 40, 64), (4, 10, 17, 35), (3, 17, 5, 2), (7, 4, 20, 128), (2, 5, 65, 33),
    ];

    static IEnumerable<Config> Sweep()
    {
        var n = 0;
        foreach (var cell in new[] { GpuRecurrentCell.Lstm, GpuRecurrentCell.Gru, GpuRecurrentCell.Rnn })
        {
            foreach (var dir in new[] { GpuRecurrentDirection.Forward, GpuRecurrentDirection.Reverse, GpuRecurrentDirection.Bidirectional })
            {
                foreach (var layout1 in new[] { false, true })
                {
                    for (var opt = 0; opt < (Rig.Browser ? 2 : 4); opt++)
                    {
                        var (s, b, i, h) = Shapes[n++ % Shapes.Length];
                        yield return new Config(cell, dir, layout1, s, b, i, h,
                            Bias: opt != 1,
                            Lens: opt >= 2,
                            Init: opt == 1 || opt == 3,
                            Peep: cell == GpuRecurrentCell.Lstm && opt >= 1,
                            Clip: opt == 2 ? 0.8f : 0f,
                            InputForget: cell == GpuRecurrentCell.Lstm && opt == 3,
                            LinearBeforeReset: cell == GpuRecurrentCell.Gru && (opt & 1) == 1,
                            Y: opt != 3, Yh: true, Yc: opt != 2,
                            MaxLength: opt == 3);
                    }
                }
            }
        }
    }

    static async Task SweepDtype<T>(string dtype, double tolerance)
        where T : unmanaged
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d);
        var w = new Worst();
        var seed = 1000;
        string worstCase = "";
        double worst = -1;
        foreach (var c in Sweep())
        {
            var before = w.Scaled;
            var name = await Check<T>(d, k, c, seed++, w, twice: true);
            if (w.Scaled > worst)
            {
                worst = w.Scaled;
                worstCase = name;
            }

            _ = before;
        }

        Rig.Row(d, "recurrent", dtype, $"{w.Cases} cases (RNN, GRU, LSTM x forward, reverse, bidirectional x layouts 0 and 1, bias, sequence lengths incl. 0, initial state, peepholes, clip, input_forget, linear_before_reset, odd offsets and sentinels)",
            $"max scaled error {Rig.Sci(w.Scaled)} (limit {Rig.Sci(tolerance)}, worst {worstCase}), max abs {Rig.Sci(w.MaxAbs)}, bytes outside outputs changed {w.Violations}, reruns not bit-identical {w.NonIdentical}, projection {k.ProjectionVariant}");
        Assert.Equal(0, w.Violations);
        Assert.Equal(0, w.NonIdentical);
        Assert.True(w.Scaled <= tolerance, $"scaled error {w.Scaled} above {tolerance}");
    }

    [Fact]
    public Task EveryCellDirectionLayoutAndOptionMatchesTheReferenceF32() => SweepDtype<float>("f32", 1e-5);

    [Fact]
    public Task EveryCellDirectionLayoutAndOptionMatchesTheReferenceF16() => SweepDtype<Half>("f16", 2e-3);

    [Fact]
    public async Task EveryActivationMatchesTheReference()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d);
        var w = new Worst();
        var kinds = Enum.GetValues<GpuRecurrentActivationKind>();
        var seed = 5000;
        for (var i = 0; i < kinds.Length; i++)
        {
            GpuRecurrentActivation Of(int n)
            {
                var kind = kinds[(i + n) % kinds.Length];
                return new GpuRecurrentActivation(kind, 0.3f + 0.1f * n, 0.4f - 0.05f * n);
            }

            foreach (var cell in new[] { GpuRecurrentCell.Lstm, GpuRecurrentCell.Gru, GpuRecurrentCell.Rnn })
            {
                var c = new Config(cell, GpuRecurrentDirection.Bidirectional, false, 5, 3, 9, 13, true, false, false, cell == GpuRecurrentCell.Lstm, 0f, false, false, true, true, true, false);
                var fwd = new GpuRecurrentActivations(Of(0), Of(1), Of(2));
                var rev = new GpuRecurrentActivations(Of(3), Of(4), Of(5));
                await Check<float>(d, k, c, seed++, w, twice: false, fwd, rev);
            }
        }

        Rig.Row(d, "recurrent", "f32", $"{w.Cases} cases (each of the 11 activations in every slot of both directions, with parameters)", $"max scaled error {Rig.Sci(w.Scaled)} (limit 1e-5), max abs {Rig.Sci(w.MaxAbs)}, bytes outside outputs changed {w.Violations}");
        Assert.Equal(0, w.Violations);
        Assert.True(w.Scaled <= 1e-5, $"scaled error {w.Scaled}");
    }

    // The recognizer's shapes: a bidirectional LSTM over a text line's columns, batch 1 and 8.
    static readonly (int Seq, int Batch, int Input, int Hidden)[] Crnn = [(80, 1, 256, 256), (80, 8, 256, 256), (48, 8, 192, 96)];

    static async Task CrnnCase<T>(string dtype, double tolerance, GpuKernelMode mode)
        where T : unmanaged
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d, mode);
        var w = new Worst();
        foreach (var (seq, batch, input, hidden) in Crnn)
        {
            var s = Rig.Browser ? Math.Min(seq, 24) : seq;
            var c = new Config(GpuRecurrentCell.Lstm, GpuRecurrentDirection.Bidirectional, false, s, batch, input, hidden, true, false, false, false, 0f, false, false, true, true, true, false);
            await Check<T>(d, k, c, 77 + batch, w, twice: true);
        }

        Rig.Row(d, "recurrent", dtype, $"{w.Cases} recognizer shapes (bidirectional LSTM, S 80/48 x B 1/8 x I 256/192 x H 256/96), mode {mode}",
            $"max scaled error {Rig.Sci(w.Scaled)} (limit {Rig.Sci(tolerance)}), max abs {Rig.Sci(w.MaxAbs)}, reruns not bit-identical {w.NonIdentical}, bytes outside outputs changed {w.Violations}");
        Assert.Equal(0, w.NonIdentical);
        Assert.Equal(0, w.Violations);
        Assert.True(w.Scaled <= tolerance, $"scaled error {w.Scaled}");
    }

    [Fact]
    public Task RecognizerShapesF32() => CrnnCase<float>("f32", 1e-5, GpuKernelMode.Fast);

    [Fact]
    public Task RecognizerShapesF16() => CrnnCase<Half>("f16", 2e-3, GpuKernelMode.Fast);

    [Fact]
    public Task RecognizerShapesDeterministicF32() => CrnnCase<float>("f32", 1e-5, GpuKernelMode.Deterministic);

    [Fact]
    public Task RecognizerShapesDeterministicF16() => CrnnCase<Half>("f16", 2e-3, GpuKernelMode.Deterministic);

    [Fact]
    public async Task CallerScratchGivesTheSameBits()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d);
        var c = new Config(GpuRecurrentCell.Gru, GpuRecurrentDirection.Bidirectional, true, 6, 3, 11, 20, true, true, true, false, 0f, false, false, true, true, false, false);
        var data = MakeData(c, false, 31);
        var own = await Execute<float>(d, k, c, data);
        using var xw = d.CreateBuffer<float>((int)RecurrentKernels.XwLength(data.Problem) + 5);
        using var state = d.CreateBuffer<float>((int)RecurrentKernels.StateLength(data.Problem));
        var given = await Execute<float>(d, k, c, data, xw, state);
        Assert.True(Same(own.Y, given.Y) && Same(own.Yh, given.Yh));
        using var tiny = d.CreateBuffer<float>(10);
        await Assert.ThrowsAsync<GpuKernelShapeException>(() => Execute<float>(d, k, c, data, tiny, state));
        await Assert.ThrowsAsync<GpuKernelShapeException>(() => Execute<float>(d, k, c, data, xw, tiny));
        Rig.Row(d, "recurrent", "f32", "scratch buffers supplied by the caller (exact and one-short sizes)", $"identical bits {Same(own.Y, given.Y) && Same(own.Yh, given.Yh)}, short scratch rejected");
    }

    [Fact]
    public async Task EmptyInputsAndOddSizesBehave()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d);
        var w = new Worst();

        // Seq 0: nothing to step through, the final states are the initial ones (or zeros). Batch 0: nothing at all.
        var empty = new Config(GpuRecurrentCell.Lstm, GpuRecurrentDirection.Bidirectional, false, 0, 3, 4, 5, true, false, true, false, 0f, false, false, true, true, true, false);
        await Check<float>(d, k, empty, 1, w, twice: false);
        var noInit = empty with { Init = false, Layout1 = true };
        await Check<float>(d, k, noInit, 2, w, twice: false);
        var data = MakeData(empty with { Seq = 4, Batch = 0 }, false, 3);
        var owned = new List<IDisposable>();
        try
        {
            var o = new RecurrentOperands<float>
            {
                X = Place(d, owned, data.X, 0f),
                W = Place(d, owned, data.W, 0f),
                R = Place(d, owned, data.R, 0f),
                Y = Place(d, owned, new float[4], 0f),
            };
            Rig.Run(d, e => k.Encode(e, data.Problem, o));
        }
        finally
        {
            d.WaitIdle();
            foreach (var b in owned)
            {
                b.Dispose();
            }
        }

        // A hidden size above the units of one workgroup in either direction, and a batch above one chunk of eight rows.
        await Check<float>(d, k, new Config(GpuRecurrentCell.Lstm, GpuRecurrentDirection.Forward, false, 3, 19, 6, 257, true, true, true, true, 0f, false, false, true, true, true, false), 4, w, twice: false);
        await Check<float>(d, k, new Config(GpuRecurrentCell.Gru, GpuRecurrentDirection.Reverse, true, 3, 9, 6, 97, true, true, true, false, 0f, false, true, true, true, false, false), 5, w, twice: false);
        Rig.Row(d, "recurrent", "f32", "empty sequence, empty batch, hidden 257 with batch 19, GRU hidden 97 with batch 9", $"max scaled error {Rig.Sci(w.Scaled)}, bytes outside outputs changed {w.Violations}");
        Assert.Equal(0, w.Violations);
        Assert.True(w.Scaled <= 1e-5, $"scaled error {w.Scaled}");
    }

    [Fact]
    public async Task BadArgumentsAreRejected()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        using var k = new RecurrentKernels(d);
        var c = new Config(GpuRecurrentCell.Gru, GpuRecurrentDirection.Forward, false, 4, 2, 5, 6, true, true, true, false, 0f, false, false, true, true, false, false);
        var data = MakeData(c, false, 9);
        using var lensBuffer = d.CreateBuffer<int>(8);
        using var ints = d.CreateBuffer<int>(4);

        async Task<GpuKernelShapeException> Fails(Config cfg, Func<RecurrentProblem, RecurrentProblem>? change = null, Func<RecurrentOperands<float>, RecurrentOperands<float>>? operands = null)
        {
            var dd = MakeData(cfg, false, 9);
            var owned = new List<IDisposable>();
            try
            {
                var o = new RecurrentOperands<float>
                {
                    X = Place(d, owned, dd.X, 0f),
                    W = Place(d, owned, dd.W, 0f),
                    R = Place(d, owned, dd.R, 0f),
                    Y = Place(d, owned, new float[cfg.Seq * dd.Problem.Directions * cfg.Batch * cfg.Hidden], 0f),
                    YH = Place(d, owned, new float[dd.Problem.Directions * cfg.Batch * cfg.Hidden], 0f),
                };
                if (operands is not null)
                {
                    o = operands(o);
                }

                var p = change is null ? dd.Problem : change(dd.Problem);
                var enc = d.BeginCommands();
                try
                {
                    return Assert.Throws<GpuKernelShapeException>(() => k.Encode(enc, p, o));
                }
                finally
                {
                    d.Submit(enc);
                }
            }
            finally
            {
                d.WaitIdle();
                foreach (var b in owned)
                {
                    b.Dispose();
                }
            }
        }

        var e1 = await Fails(c, p => { p.Hidden = 0; return p; });
        Assert.Equal("GRU", e1.Operation);
        Assert.Equal("Hidden", e1.Argument);
        await Fails(c, p => { p.Clip = -1f; return p; });
        await Fails(c, p => { p.Clip = float.NaN; return p; });
        await Fails(c, p => { p.InputForget = true; return p; });
        await Fails(c with { Cell = GpuRecurrentCell.Lstm }, p => { p.LinearBeforeReset = true; return p; });
        await Fails(c, p => { p.Input = 1000; return p; });
        await Fails(c, p => { p.Seq = -1; return p; });
        await Fails(c, p => { p.Direction = (GpuRecurrentDirection)7; return p; });
        await Fails(c, p => { p.Activations = new GpuRecurrentActivations(new((GpuRecurrentActivationKind)99), GpuRecurrentActivation.Tanh, GpuRecurrentActivation.Tanh); return p; });
        await Fails(c, p => { p.Seq = int.MaxValue / 2; p.Batch = 100000; return p; });
        await Fails(c with { Lens = true }, p => { p.MaxLength = 99; return p; }, o => new RecurrentOperands<float> { X = o.X, W = o.W, R = o.R, Y = o.Y, YH = o.YH, SequenceLengths = new GpuTensorRef<int>(lensBuffer) });
        await Fails(c, null, o => new RecurrentOperands<float> { X = o.X, W = o.W, R = o.R, P = o.X, Y = o.Y });
        await Fails(c, null, o => new RecurrentOperands<float> { X = o.X, W = o.W, R = o.R, YC = o.YH });
        await Fails(c, null, o => new RecurrentOperands<float> { X = new GpuTensorRef<float>(o.X.Buffer, 50), W = o.W, R = o.R, Y = o.Y });
        Assert.Throws<GpuKernelShapeException>(() =>
        {
            var enc = d.BeginCommands();
            try
            {
                k.Encode(enc, data.Problem, new RecurrentOperands<int> { X = new(ints), W = new(ints), R = new(ints) });
            }
            finally
            {
                d.Submit(enc);
            }
        });
        Rig.Row(d, "recurrent", "-", "bad arguments (zero hidden, negative clip, options of the wrong cell, ranges, bad enum values, oversized layers, integer tensors)", "typed GpuKernelShapeException each");
    }
}
