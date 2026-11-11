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

        var commands = device.BeginCommands();
        attention.Encode(commands, attnProblem, attnQ, attnK, attnV, attnO, mode: GpuKernelMode.Deterministic);
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

        var gotAttn = new float[aq.Length];
        await attnO.ReadAsync(gotAttn);

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

        wrong += Math.Abs(gotBig[0] - big.Sum(v => (double)v)) < 1e-2 ? 0 : 1;
        return wrong;
    }
}
