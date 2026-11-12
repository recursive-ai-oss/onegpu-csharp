namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>The program in the README, so the example it shows keeps compiling and giving the right answer.</summary>
public sealed class ReadmeExampleTests
{
    [Fact]
    public async Task ReadmeGemmExampleMultipliesMatrices()
    {
        using var gpu = new GpuFixture();
        var device = gpu.Device;

        // --- as in README.md ---
        const int M = 4, N = 3, K = 2;
        float[] a = [1, 2, 3, 4, 5, 6, 7, 8];     // 4 x 2, row-major
        float[] b = [1, 0, 2, 0, 1, 3];           // 2 x 3, row-major
        using var bufferA = device.CreateBuffer<float>(a);
        using var bufferB = device.CreateBuffer<float>(b);
        using var bufferC = device.CreateBuffer<float>(M * N);

        using var gemm = new Gemm(device, GemmElementType.F32);   // GpuKernelMode.Deterministic for bit-repeatable sums
        var encoder = device.BeginCommands();
        gemm.Encode(encoder, new GemmProblem(M, N, K), bufferA, bufferB, bufferC);
        device.Submit(encoder);

        var c = new float[M * N];
        await bufferC.ReadAsync(c);                               // completes when the GPU is done, without blocking a thread
        // --- end ---

        float[] expected = [1, 2, 8, 3, 4, 18, 5, 6, 28, 7, 8, 38];
        Assert.Equal(expected, c);
    }
}
