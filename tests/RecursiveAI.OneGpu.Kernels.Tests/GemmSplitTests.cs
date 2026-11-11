namespace RecursiveAI.OneGpu.Kernels.Tests;

/// <summary>
/// A product larger than a submission should carry is cut into dispatches over groups of batches or blocks of rows. The cut must not change
/// a single bit of the result: each output element is still summed by one thread in the same order.
/// </summary>
public sealed class GemmSplitTests
{
    static async Task<float[]> Run(GpuDevice d, GemmElementType type, GemmProblem p, ulong? budget, bool allowSplit, int seed)
    {
        using var gemm = new Gemm(d, type, GpuKernelMode.Deterministic, new GemmOptions { SubmissionWorkBudget = budget, AllowSplitting = allowSplit });
        var rowsA = p.TransposeA ? p.K : p.M;
        var colsA = p.TransposeA ? p.M : p.K;
        var rowsB = p.TransposeB ? p.N : p.K;
        var colsB = p.TransposeB ? p.K : p.N;
        var lenA = p.OffsetA + ((p.Batch - 1) * p.StrideA) + (rowsA * colsA);
        var lenB = p.OffsetB + ((p.Batch - 1) * p.StrideB) + (rowsB * colsB);
        var lenC = p.OffsetC + ((p.Batch - 1) * p.StrideC) + (p.M * p.N);
        var a = Rig.Random(lenA, seed, -1f, 1f);
        var b = Rig.Random(lenB, seed + 1, -1f, 1f);
        var c = Rig.Random(lenC, seed + 2, -1f, 1f);
        using var bc = d.CreateBuffer<float>(c);
        var encoder = d.BeginCommands();
        if (type == GemmElementType.F32)
        {
            using var ba = d.CreateBuffer<float>(a);
            using var bb = d.CreateBuffer<float>(b);
            gemm.Encode(encoder, p, ba, bb, bc);
            d.Submit(encoder);
            return await Rig.DownloadAsync(bc);
        }

        using var ha = d.CreateBuffer<Half>(Rig.ToHalf(a));
        using var hb = d.CreateBuffer<Half>(Rig.ToHalf(b));
        gemm.Encode(encoder, p, ha, hb, bc);
        d.Submit(encoder);
        return await Rig.DownloadAsync(bc);
    }

    [Fact]
    public async Task SplittingChangesNoBits()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        var problems = new List<(string Name, GemmProblem P)>();
        problems.Add(("one product, rows cut", new GemmProblem(300, 130, 70) { Lda = 70, Ldb = 130, Ldc = 130, Alpha = 1.5f, Beta = 0.5f }));
        problems.Add(("one product, A transposed", new GemmProblem(200, 90, 33) { TransposeA = true, Lda = 200, Ldb = 90, Ldc = 90, OffsetA = 3, OffsetB = 1, OffsetC = 5 }));
        problems.Add(("one product, B transposed", new GemmProblem(129, 66, 65) { TransposeB = true, Lda = 65, Ldb = 65, Ldc = 66 }));
        problems.Add(("batch 7, products grouped", new GemmProblem(40, 50, 60) { Batch = 7, Lda = 60, Ldb = 50, Ldc = 50, StrideA = 40 * 60, StrideB = 60 * 50, StrideC = 40 * 50 }));
        problems.Add(("batch 5 shared B", new GemmProblem(70, 45, 20) { Batch = 5, Lda = 20, Ldb = 45, Ldc = 45, StrideA = 70 * 20, StrideB = 0, StrideC = 70 * 45, Beta = 1f }));
        problems.Add(("batch 3, rows cut", new GemmProblem(260, 64, 64) { Batch = 3, Lda = 64, Ldb = 64, Ldc = 64, StrideA = 260 * 64, StrideB = 64 * 64, StrideC = 260 * 64 }));
        long diff = 0;
        var seed = 1;
        var rows = 0;
        foreach (var (name, p) in problems)
        {
            var ops = p.Flops;
            // Budgets from a hundredth of one product to more than the whole batch.
            foreach (var budget in new ulong[] { (ulong)(ops / 100) + 1, (ulong)(ops / 7) + 1, (ulong)(ops / (p.Batch * 2.0)) + 1, (ulong)(ops * 2) })
            {
                foreach (var type in new[] { GemmElementType.F32, GemmElementType.F16 })
                {
                    var whole = await Run(d, type, p, null, false, seed);
                    var cut = await Run(d, type, p, budget, true, seed);
                    diff += Rig.BitMismatches(cut, whole);
                    rows++;
                    seed++;
                }
            }

            Rig.Row(d, "gemm split", "f32/f16", name, $"{4 * 2} budgets/types compared bit for bit with the unsplit product, hash {Rig.Hash<float>(await Run(d, GemmElementType.F32, p, null, false, 99))}");
        }

        Assert.Equal(0, diff);
        _ = rows;
    }

    [Fact]
    public async Task TheDefaultPolicyOnTheLargestShapeKeepsItsBits()
    {
        using var gpu = new GpuFixture();
        var d = gpu.Device;
        // The largest deterministic shape of the model set is a 1024 x 3072 x 768 product; a 4096 square product is the stress case.
        var shapes = OperatingSystem.IsBrowser() ? new[] { 1024 } : new[] { 1024, 4096 };
        foreach (var n in shapes)
        {
            var p = new GemmProblem(n, n, n == 4096 ? 1024 : 768) { Lda = n == 4096 ? 1024 : 768, Ldb = n, Ldc = n };
            var split = await Run(d, GemmElementType.F32, p, null, true, 21);
            var whole = await Run(d, GemmElementType.F32, p, null, false, 21);
            Assert.Equal(0, Rig.BitMismatches(split, whole));
            var budget = d.Capabilities.Compute.SubmissionWorkBudget;
            Rig.Row(d, "gemm split", "f32", $"{n} x {n} x {p.K} deterministic, device budget {budget} ops ({p.Flops / 1e9:F1} GFLOP in {(budget == 0 ? 1 : Math.Ceiling(p.Flops / budget))} pieces)", $"default policy and unsplit hash {Rig.Hash<float>(split)} / {Rig.Hash<float>(whole)}");
        }
    }
}
