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

        var commands = device.BeginCommands();
        softmax.Encode(commands, input, 0, [Rows, Len], 1, soft, 0, mode: GpuKernelMode.Deterministic);
        reduce.EncodeReduce(commands, GpuReduceOp.Sum, input, 0, [Rows, Len], [1], sums, 0);
        elementwise.EncodeBinary(commands, GpuBinaryOp.Add, input, 0, [Rows, Len], input, 0, [Rows, Len], added, 0, new GpuActivation(GpuUnaryOp.Relu));
        transpose.Encode(commands, input, 0, [Rows, Len], [1, 0], transposed, 0);
        device.Submit(commands);

        var gotSoft = new float[x.Length];
        var gotSums = new float[Rows];
        var gotAdded = new float[x.Length];
        var gotT = new float[x.Length];
        await soft.ReadAsync(gotSoft);
        await sums.ReadAsync(gotSums);
        await added.ReadAsync(gotAdded);
        await transposed.ReadAsync(gotT);

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

        return wrong;
    }
}
