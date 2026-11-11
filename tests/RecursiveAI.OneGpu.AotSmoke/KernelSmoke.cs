using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;

/// <summary>Runs one kernel of each family in the AOT-published app and returns the number of wrong values.</summary>
internal static class KernelSmoke
{
    public static async Task<int> RunAsync(GpuDevice device)
    {
        const int Rows = 8, Len = 300;
        var x = new float[Rows * Len];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = ((i * 37) % 101 - 50) / 7f;
        }

        using var input = device.CreateBuffer<float>(x);
        using var soft = device.CreateBuffer<float>(x.Length);
        using var sums = device.CreateBuffer<float>(Rows);
        using var added = device.CreateBuffer<float>(x.Length);
        using var transposed = device.CreateBuffer<float>(x.Length);
        using var softmax = new SoftmaxKernels(device);
        using var reduce = new ReduceKernels(device);
        using var elementwise = new ElementwiseKernels(device);
        using var transpose = new TransposeKernels(device);
        using var integer = new IntegerKernels(device);
        using var positions = device.CreateBuffer<long>(Rows);
        using var lower = device.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(Rows * Rows));

        // A reduction over one long axis runs in two passes through the family's scratch buffer.
        var big = new float[20000];
        for (var i = 0; i < big.Length; i++)
        {
            big[i] = (i % 13) / 4f;
        }

        using var bigInput = device.CreateBuffer<float>(big);
        using var bigSum = device.CreateBuffer<float>(1);

        // Causal attention over 300 keys is cut into two key ranges, so both passes of the family run.
        const int AH = 2, ASq = 5, ASk = 300, AD = 32;
        var aq = new float[AH * ASq * AD];
        var ak = new float[AH * ASk * AD];
        var av = new float[AH * ASk * AD];
        for (var i = 0; i < aq.Length; i++)
        {
            aq[i] = ((i * 29) % 61 - 30) / 20f;
        }

        for (var i = 0; i < ak.Length; i++)
        {
            ak[i] = ((i * 17) % 53 - 26) / 20f;
            av[i] = ((i * 13) % 47 - 23) / 23f;
        }

        using var attnQ = device.CreateBuffer<float>(aq);
        using var attnK = device.CreateBuffer<float>(ak);
        using var attnV = device.CreateBuffer<float>(av);
        using var attnO = device.CreateBuffer<float>(aq.Length);
        using var attention = new AttentionKernels(device);
        var attnProblem = new AttentionProblem(1, AH, ASq, ASk, AD) { Causal = true };

        // The same attention with half tensors (the matrix-hardware kernel where the device has one), compared with the float32 reference below.
        Half[] ToHalf(float[] v)
        {
            var h = new Half[GpuKernelBuffers.PaddedHalfLength(v.Length)];
            for (var i = 0; i < v.Length; i++)
            {
                h[i] = (Half)v[i];
            }

            return h;
        }

        using var attnQh = device.CreateBuffer<Half>(ToHalf(aq));
        using var attnKh = device.CreateBuffer<Half>(ToHalf(ak));
        using var attnVh = device.CreateBuffer<Half>(ToHalf(av));
        using var attnOh = device.CreateBuffer<Half>(GpuKernelBuffers.PaddedHalfLength(aq.Length));

        var commands = device.BeginCommands();
        attention.Encode(commands, attnProblem, attnQ, attnK, attnV, attnO, mode: GpuKernelMode.Deterministic);
        attention.Encode(commands, attnProblem, attnQh, attnKh, attnVh, attnOh, mode: GpuKernelMode.Deterministic);
        softmax.Encode(commands, input, 0, [Rows, Len], 1, soft, 0, mode: GpuKernelMode.Deterministic);
        reduce.EncodeReduce(commands, GpuReduceOp.Sum, input, 0, [Rows, Len], [1], sums, 0);
        elementwise.EncodeBinary(commands, GpuBinaryOp.Add, input, 0, [Rows, Len], input, 0, [Rows, Len], added, 0, new GpuActivation(GpuUnaryOp.Relu));
        transpose.Encode(commands, input, 0, [Rows, Len], [1, 0], transposed, 0);
        reduce.EncodeReduce(commands, GpuReduceOp.Sum, bigInput, 0, [big.Length], [0], bigSum, 0, GpuKernelMode.Deterministic);
        integer.EncodeRange(commands, positions, 0, Rows, 5L, -2L);
        integer.EncodeCompare(commands, GpuCompareOp.GreaterOrEqual, positions, 0, [Rows, 1], positions, 0, [1, Rows], lower, 0);
        device.Submit(commands);

        var gotSoft = new float[x.Length];
        var gotSums = new float[Rows];
        var gotAdded = new float[x.Length];
        var gotT = new float[x.Length];
        await soft.ReadAsync(gotSoft);
        await sums.ReadAsync(gotSums);
        await added.ReadAsync(gotAdded);
        await transposed.ReadAsync(gotT);
        var gotBig = new float[1];
        await bigSum.ReadAsync(gotBig);
        var gotPositions = new long[Rows];
        var gotLower = new byte[GpuKernelBuffers.PaddedBoolLength(Rows * Rows)];
        await positions.ReadAsync(gotPositions);
        await lower.ReadAsync(gotLower);

        var gotAttn = new float[aq.Length];
        await attnO.ReadAsync(gotAttn);
        var gotAttnHalf = new Half[GpuKernelBuffers.PaddedHalfLength(aq.Length)];
        await attnOh.ReadAsync(gotAttnHalf);

        var wrong = 0;
        for (var h = 0; h < AH; h++)
        {
            for (var i = 0; i < ASq; i++)
            {
                var visible = ASk - ASq + i + 1;
                var s = new double[visible];
                var top = double.NegativeInfinity;
                for (var j = 0; j < visible; j++)
                {
                    double dot = 0;
                    for (var e = 0; e < AD; e++)
                    {
                        dot += (double)aq[(((h * ASq) + i) * AD) + e] * ak[(((h * ASk) + j) * AD) + e];
                    }

                    s[j] = dot / Math.Sqrt(AD);
                    top = Math.Max(top, s[j]);
                }

                var norm = s.Sum(v => Math.Exp(v - top));
                for (var e = 0; e < AD; e++)
                {
                    double acc = 0;
                    for (var j = 0; j < visible; j++)
                    {
                        acc += Math.Exp(s[j] - top) * av[(((h * ASk) + j) * AD) + e];
                    }

                    wrong += Math.Abs(gotAttn[(((h * ASq) + i) * AD) + e] - (acc / norm)) < 1e-5 ? 0 : 1;
                    wrong += Math.Abs((double)gotAttnHalf[(((h * ASq) + i) * AD) + e] - (acc / norm)) < 5e-3 ? 0 : 1;
                }
            }
        }

        for (var r = 0; r < Rows; r++)
        {
            double max = double.NegativeInfinity, total = 0, sum = 0;
            for (var c = 0; c < Len; c++)
            {
                max = Math.Max(max, x[r * Len + c]);
                sum += x[r * Len + c];
            }

            for (var c = 0; c < Len; c++)
            {
                total += Math.Exp(x[r * Len + c] - max);
            }

            for (var c = 0; c < Len; c++)
            {
                wrong += Math.Abs(gotSoft[r * Len + c] - Math.Exp(x[r * Len + c] - max) / total) < 1e-5 ? 0 : 1;
                wrong += gotAdded[r * Len + c] == Math.Max(0f, 2f * x[r * Len + c]) ? 0 : 1;
                wrong += gotT[c * Rows + r] == x[r * Len + c] ? 0 : 1;
            }

            wrong += Math.Abs(gotSums[r] - sum) < 1e-3 ? 0 : 1;
        }

        for (var r = 0; r < Rows; r++)
        {
            wrong += gotPositions[r] == 5 - 2 * r ? 0 : 1;
            for (var c = 0; c < Rows; c++)
            {
                wrong += gotLower[r * Rows + c] == (gotPositions[r] >= gotPositions[c] ? 1 : 0) ? 0 : 1;
            }
        }

        wrong += Math.Abs(gotBig[0] - big.Sum(v => (double)v)) < 1e-2 ? 0 : 1;
        return wrong + await GatherAndInt8Async(device) + await EpilogueAndDequantizeAsync(device) + await VisionAsync(device) + await RecurrentAndQuantizeAsync(device);
    }

    // Embedding-style gather with a negative index, then an int8 GEMM with a zero point and a dynamic quantize, against plain C#.
    static async Task<int> GatherAndInt8Async(GpuDevice device)
    {
        var wrong = 0;
        const int Vocab = 100, Hidden = 8;
        var table = new float[Vocab * Hidden];
        for (var i = 0; i < table.Length; i++)
        {
            table[i] = i * 0.25f;
        }

        int[] ids = [3, -1, 99, 0, 50];
        using var tableBuf = device.CreateBuffer<float>(table);
        using var idBuf = device.CreateBuffer<int>(ids);
        using var gathered = device.CreateBuffer<float>(ids.Length * Hidden);

        const int M = 5, N = 6, K = 7;
        var a = new byte[M * K];
        var b = new sbyte[K * N];
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = (byte)((i * 29) % 251);
        }

        for (var i = 0; i < b.Length; i++)
        {
            b[i] = (sbyte)((i * 13 % 255) - 127);
        }

        byte[] zeroA = [17];
        using var aBuf = device.CreateBuffer<byte>(a);
        using var bBuf = device.CreateBuffer<sbyte>(b);
        using var zBuf = device.CreateBuffer<byte>(zeroA);
        using var cBuf = device.CreateBuffer<int>(M * N);

        var q = new float[300];
        for (var i = 0; i < q.Length; i++)
        {
            q[i] = ((i * 37) % 101 - 40) / 9f;
        }

        using var qIn = device.CreateBuffer<float>(q);
        using var qOut = device.CreateBuffer<byte>(q.Length);
        using var qScale = device.CreateBuffer<float>(1);
        using var qZero = device.CreateBuffer<byte>(1);
        using var gather = new GatherKernels(device);
        using var gemm = new Int8Gemm(device);
        using var dq = new DynamicQuantizeKernels(device);

        var commands = device.BeginCommands();
        gather.EncodeGather(commands, tableBuf, 0, [Vocab, Hidden], 0, idBuf, 0, ids.Length, gathered, 0);
        var problem = new Int8GemmProblem(M, N, K) { Lda = K, Ldb = N };
        gemm.EncodeInt32(commands, problem, aBuf, bBuf, cBuf, zeroPointA: new Int8Operand<byte>(zBuf, 0));
        dq.Encode(commands, qIn, 0, q.Length, qOut, 0, qScale, 0, qZero, 0);
        device.Submit(commands);

        var gotG = new float[ids.Length * Hidden];
        var gotC = new int[M * N];
        var gotQ = new byte[q.Length];
        var gotScale = new float[1];
        var gotZero = new byte[1];
        await gathered.ReadAsync(gotG);
        await cBuf.ReadAsync(gotC);
        await qOut.ReadAsync(gotQ);
        await qScale.ReadAsync(gotScale);
        await qZero.ReadAsync(gotZero);

        for (var r = 0; r < ids.Length; r++)
        {
            var row = ids[r] < 0 ? ids[r] + Vocab : ids[r];
            for (var h = 0; h < Hidden; h++)
            {
                wrong += gotG[r * Hidden + h] == table[row * Hidden + h] ? 0 : 1;
            }
        }

        for (var m = 0; m < M; m++)
        {
            for (var n = 0; n < N; n++)
            {
                var sum = 0;
                for (var k = 0; k < K; k++)
                {
                    sum += (a[m * K + k] - zeroA[0]) * b[k * N + n];
                }

                wrong += gotC[m * N + n] == sum ? 0 : 1;
            }
        }

        // The scale and zero point follow DynamicQuantizeLinear; the quantized values are checked against them.
        var lo = Math.Min(0f, q.Min());
        var hi = Math.Max(0f, q.Max());
        var scale = (hi - lo) / 255f;
        wrong += gotScale[0] == scale ? 0 : 1;
        wrong += gotZero[0] == (byte)Math.Clamp(Math.Round(-(lo / scale), MidpointRounding.ToEven), 0, 255) ? 0 : 1;
        for (var i = 0; i < q.Length; i++)
        {
            var expect = (byte)Math.Clamp(Math.Round(q[i] / scale, MidpointRounding.ToEven) + gotZero[0], 0, 255);
            wrong += gotQ[i] == expect ? 0 : 1;
        }

        return wrong;
    }

    // A product with a column bias and GELU in one dispatch, an 8-bit gather and DequantizeLinear, against plain C#.
    static async Task<int> EpilogueAndDequantizeAsync(GpuDevice device)
    {
        var wrong = 0;
        const int M = 5, N = 6, K = 4;
        var a = new float[M * K];
        var b = new float[K * N];
        var bias = new float[N];
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = ((i * 7) % 11 - 5) / 4f;
        }

        for (var i = 0; i < b.Length; i++)
        {
            b[i] = ((i * 5) % 13 - 6) / 8f;
        }

        for (var i = 0; i < N; i++)
        {
            bias[i] = (i - 2) / 4f;
        }

        byte[] table = Enumerable.Range(0, 64).Select(i => (byte)(i * 7)).ToArray();
        int[] ids = [3, 0, 7];
        byte[] quantized = Enumerable.Range(0, 20).Select(i => (byte)(i * 13)).ToArray();
        float[] scales = [0.5f, 0.25f];
        using var ab = device.CreateBuffer<float>(a);
        using var bb = device.CreateBuffer<float>(b);
        using var biasBuf = device.CreateBuffer<float>(bias);
        using var c = device.CreateBuffer<float>(M * N);
        using var tableBuf = device.CreateBuffer<byte>(table);
        using var idBuf = device.CreateBuffer<int>(ids);
        using var gathered = device.CreateBuffer<byte>(GpuKernelBuffers.PaddedBoolLength(ids.Length * 8));
        using var qBuf = device.CreateBuffer<byte>([.. quantized, 0, 0, 0, 0]);
        using var scaleBuf = device.CreateBuffer<float>(scales);
        using var dequantized = device.CreateBuffer<float>(quantized.Length);
        using var gemm = new Gemm(device, GemmElementType.F32);
        using var gather = new GatherKernels(device);
        using var dequantize = new DequantizeKernels(device);

        var commands = device.BeginCommands();
        gemm.Encode(commands, new GemmProblem(M, N, K), ab, bb, c, new GemmEpilogue(biasBuf, 0, GemmActivation.GeluErf));
        gather.EncodeGather(commands, tableBuf, 0, [8, 8], 0, idBuf, 0, ids.Length, gathered, 0);
        dequantize.Encode<byte, float>(commands, qBuf, 0, [2, 10], 0, true, scaleBuf, 0, null, 0, dequantized, 0);
        device.Submit(commands);

        var gotC = new float[M * N];
        var gotG = new byte[GpuKernelBuffers.PaddedBoolLength(ids.Length * 8)];
        var gotD = new float[quantized.Length];
        await c.ReadAsync(gotC);
        await gathered.ReadAsync(gotG);
        await dequantized.ReadAsync(gotD);
        for (var m = 0; m < M; m++)
        {
            for (var n = 0; n < N; n++)
            {
                double sum = bias[n];
                for (var k = 0; k < K; k++)
                {
                    sum += (double)a[m * K + k] * b[k * N + n];
                }

                var want = 0.5 * sum * (1 + Erf(sum / Math.Sqrt(2)));
                wrong += Math.Abs(gotC[m * N + n] - want) < 1e-4 ? 0 : 1;
            }
        }

        for (var r = 0; r < ids.Length; r++)
        {
            for (var h = 0; h < 8; h++)
            {
                wrong += gotG[r * 8 + h] == table[ids[r] * 8 + h] ? 0 : 1;
            }
        }

        for (var i = 0; i < quantized.Length; i++)
        {
            wrong += gotD[i] == (float)quantized[i] * scales[i / 10] ? 0 : 1;
        }

        return wrong;
    }

    // A padded 3x3 convolution with bias and ReLU, a 2x2 max pooling and a nearest 2x resize, against plain C#.
    static async Task<int> VisionAsync(GpuDevice device)
    {
        var wrong = 0;
        const int Cin = 2, Cout = 3, H = 5, W = 5;
        var x = new float[Cin * H * W];
        var w = new float[Cout * Cin * 9];
        var bias = new float[Cout];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = ((i * 31) % 17 - 8) / 8f;
        }

        for (var i = 0; i < w.Length; i++)
        {
            w[i] = ((i * 7) % 11 - 5) / 10f;
        }

        for (var i = 0; i < Cout; i++)
        {
            bias[i] = (i - 1) * 0.1f;
        }

        var problem = new ConvProblem(1, Cin, H, W, Cout, 3, 3)
        {
            StrideHeight = 1,
            StrideWidth = 1,
            DilationHeight = 1,
            DilationWidth = 1,
            PadTop = 1,
            PadLeft = 1,
            PadBottom = 1,
            PadRight = 1,
            Activation = new GpuActivation(GpuUnaryOp.Relu),
        };
        using var xb = device.CreateBuffer<float>(x);
        using var wb = device.CreateBuffer<float>(w);
        using var bb = device.CreateBuffer<float>(bias);
        using var yb = device.CreateBuffer<float>(Cout * H * W);
        using var pooled = device.CreateBuffer<float>(Cout * 2 * 2);
        using var resized = device.CreateBuffer<float>(Cout * 10 * 10);
        using var conv = new ConvKernels(device);
        using var pool = new PoolKernels(device);
        using var resize = new ResizeKernels(device);
        var pp = new PoolProblem(1, Cout, H, W, 2, 2) { StrideHeight = 2, StrideWidth = 2, DilationHeight = 1, DilationWidth = 1 };
        using var plan = resize.CreatePlan(new ResizeProblem(Cout, H, W, 10, 10, GpuResizeMode.Nearest));
        var commands = device.BeginCommands();
        conv.Encode(commands, problem, xb, wb, bb, null, yb);
        pool.Encode(commands, pp, GpuPoolKind.Max, yb, pooled);
        resize.Encode(commands, plan, yb, 0, resized, 0);
        device.Submit(commands);
        var y = new float[Cout * H * W];
        var gotPooled = new float[Cout * 4];
        var gotResized = new float[Cout * 100];
        await yb.ReadAsync(y);
        await pooled.ReadAsync(gotPooled);
        await resized.ReadAsync(gotResized);
        for (var o = 0; o < Cout; o++)
        {
            for (var r = 0; r < H; r++)
            {
                for (var c = 0; c < W; c++)
                {
                    double acc = bias[o];
                    for (var i = 0; i < Cin; i++)
                    {
                        for (var kr = 0; kr < 3; kr++)
                        {
                            for (var kc = 0; kc < 3; kc++)
                            {
                                int ir = r + kr - 1, ic = c + kc - 1;
                                if (ir >= 0 && ir < H && ic >= 0 && ic < W)
                                {
                                    acc += (double)x[(i * H + ir) * W + ic] * w[((o * Cin + i) * 3 + kr) * 3 + kc];
                                }
                            }
                        }
                    }

                    wrong += Math.Abs(y[(o * H + r) * W + c] - Math.Max(0, acc)) < 1e-4 ? 0 : 1;
                }
            }

            for (var r = 0; r < 2; r++)
            {
                for (var c = 0; c < 2; c++)
                {
                    var m = float.MinValue;
                    for (var a = 0; a < 2; a++)
                    {
                        for (var b = 0; b < 2; b++)
                        {
                            m = Math.Max(m, y[(o * H + 2 * r + a) * W + 2 * c + b]);
                        }
                    }

                    wrong += gotPooled[(o * 2 + r) * 2 + c] == m ? 0 : 1;
                }
            }

            // Scale 2 with half-pixel coordinates repeats every source element twice.
            for (var r = 0; r < 10; r++)
            {
                for (var c = 0; c < 10; c++)
                {
                    wrong += gotResized[(o * 10 + r) * 10 + c] == y[(o * H + r / 2) * W + c / 2] ? 0 : 1;
                }
            }
        }

        return wrong;
    }

    // A bidirectional LSTM with a sequence length per row and QuantizeLinear (a tie, a saturating value and NaN), against plain C#.
    static async Task<int> RecurrentAndQuantizeAsync(GpuDevice device)
    {
        var wrong = 0;
        const int S = 4, B = 2, I = 3, H = 5;
        var x = new float[S * B * I];
        var w = new float[2 * 4 * H * I];
        var r = new float[2 * 4 * H * H];
        var bias = new float[2 * 8 * H];
        int[] lens = [4, 2];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = ((i * 7) % 11 - 5) / 5f;
        }

        for (var i = 0; i < w.Length; i++)
        {
            w[i] = ((i * 5) % 13 - 6) / 12f;
        }

        for (var i = 0; i < r.Length; i++)
        {
            r[i] = ((i * 3) % 17 - 8) / 20f;
        }

        for (var i = 0; i < bias.Length; i++)
        {
            bias[i] = ((i * 11) % 7 - 3) / 10f;
        }

        using var xb = device.CreateBuffer<float>(x);
        using var wb = device.CreateBuffer<float>(w);
        using var rb = device.CreateBuffer<float>(r);
        using var bb = device.CreateBuffer<float>(bias);
        using var lb = device.CreateBuffer<int>(lens);
        using var yb = device.CreateBuffer<float>(S * 2 * B * H);
        using var yhb = device.CreateBuffer<float>(2 * B * H);
        using var lstm = new RecurrentKernels(device);
        float[] q = [1.25f, 2.5f, 1000f, float.NaN, -3.5f, 0.75f];
        using var qIn = device.CreateBuffer<float>(q);
        using var qScale = device.CreateBuffer<float>([0.5f]);
        using var qOut = device.CreateBuffer<sbyte>(GpuKernelBuffers.PaddedBoolLength(q.Length));
        using var quantize = new DequantizeKernels(device);

        var commands = device.BeginCommands();
        var problem = new RecurrentProblem(GpuRecurrentCell.Lstm, S, B, I, H) { Direction = GpuRecurrentDirection.Bidirectional };
        lstm.Encode(commands, problem, new RecurrentOperands<float> { X = new(xb), W = new(wb), R = new(rb), B = new(bb), SequenceLengths = new(lb), Y = new(yb), YH = new(yhb) });
        quantize.EncodeQuantize<float, sbyte>(commands, qIn, 0, [q.Length], 0, false, qScale, 0, null, 0, qOut, 0);
        device.Submit(commands);

        var gotY = new float[S * 2 * B * H];
        var gotYh = new float[2 * B * H];
        var gotQ = new sbyte[GpuKernelBuffers.PaddedBoolLength(q.Length)];
        await yb.ReadAsync(gotY);
        await yhb.ReadAsync(gotYh);
        await qOut.ReadAsync(gotQ);

        // The reference: ONNX LSTM, gates i, o, f, c, in double precision.
        var wantY = new double[S * 2 * B * H];
        var wantYh = new double[2 * B * H];
        static double Sig(double v) => 1 / (1 + Math.Exp(-v));
        for (var d = 0; d < 2; d++)
        {
            for (var b = 0; b < B; b++)
            {
                var h = new double[H];
                var c = new double[H];
                for (var s = 0; s < lens[b]; s++)
                {
                    var t = d == 1 ? lens[b] - 1 - s : s;
                    var g = new double[4 * H];
                    for (var row = 0; row < 4 * H; row++)
                    {
                        var acc = (double)bias[d * 8 * H + row] + bias[d * 8 * H + 4 * H + row];
                        for (var k = 0; k < I; k++)
                        {
                            acc += (double)x[(t * B + b) * I + k] * w[(d * 4 * H + row) * I + k];
                        }

                        for (var k = 0; k < H; k++)
                        {
                            acc += h[k] * r[(d * 4 * H + row) * H + k];
                        }

                        g[row] = acc;
                    }

                    for (var j = 0; j < H; j++)
                    {
                        c[j] = Sig(g[2 * H + j]) * c[j] + Sig(g[j]) * Math.Tanh(g[3 * H + j]);
                        h[j] = Sig(g[H + j]) * Math.Tanh(c[j]);
                        wantY[((t * 2 + d) * B + b) * H + j] = h[j];
                    }
                }

                for (var j = 0; j < H; j++)
                {
                    wantYh[(d * B + b) * H + j] = h[j];
                }
            }
        }

        for (var i = 0; i < wantY.Length; i++)
        {
            wrong += Math.Abs(gotY[i] - wantY[i]) < 1e-5 ? 0 : 1;
        }

        for (var i = 0; i < wantYh.Length; i++)
        {
            wrong += Math.Abs(gotYh[i] - wantYh[i]) < 1e-5 ? 0 : 1;
        }

        sbyte[] wantQ = [2, 5, 127, -128, -7, 2];
        for (var i = 0; i < q.Length; i++)
        {
            wrong += gotQ[i] == wantQ[i] ? 0 : 1;
        }

        return wrong;
    }


    // Abramowitz and Stegun 7.1.26, enough for a 1e-4 smoke check.
    static double Erf(double x)
    {
        var t = 1 / (1 + 0.3275911 * Math.Abs(x));
        var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return x >= 0 ? y : -y;
    }
}
