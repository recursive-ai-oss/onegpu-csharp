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

        // A reduction over one long axis runs in two passes through the family's scratch buffer.
        var big = new float[20000];
        for (var i = 0; i < big.Length; i++)
        {
            big[i] = (i % 13) / 4f;
        }

        using var bigInput = device.CreateBuffer<float>(big);
        using var bigSum = device.CreateBuffer<float>(1);

        var commands = device.BeginCommands();
        softmax.Encode(commands, input, 0, [Rows, Len], 1, soft, 0, mode: GpuKernelMode.Deterministic);
        reduce.EncodeReduce(commands, GpuReduceOp.Sum, input, 0, [Rows, Len], [1], sums, 0);
        elementwise.EncodeBinary(commands, GpuBinaryOp.Add, input, 0, [Rows, Len], input, 0, [Rows, Len], added, 0, new GpuActivation(GpuUnaryOp.Relu));
        transpose.Encode(commands, input, 0, [Rows, Len], [1, 0], transposed, 0);
        reduce.EncodeReduce(commands, GpuReduceOp.Sum, bigInput, 0, [big.Length], [0], bigSum, 0, GpuKernelMode.Deterministic);
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

        var wrong = 0;
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

        wrong += Math.Abs(gotBig[0] - big.Sum(v => (double)v)) < 1e-2 ? 0 : 1;
        return wrong + await GatherAndInt8Async(device);
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
}
