namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Element type of the A and B operands of a <see cref="Gemm"/>. C is always float32.</summary>
public enum GemmElementType
{
    /// <summary>IEEE binary32. Buffers are <c>GpuBuffer&lt;float&gt;</c>.</summary>
    F32 = 0,

    /// <summary>IEEE binary16. Buffers are <c>GpuBuffer&lt;Half&gt;</c>. Products and sums are float32.</summary>
    F16 = 1,

    /// <summary>
    /// bfloat16: the high 16 bits of a binary32. Buffers are <c>GpuBuffer&lt;ushort&gt;</c> holding those bits. Products and sums are float32.
    /// </summary>
    BF16 = 2,
}

/// <summary>
/// The shape of one matrix multiplication <c>C = Alpha * op(A) * op(B) + Beta * C</c> over <see cref="Batch"/> independent products, where
/// <c>op(A)</c> is M x K and <c>op(B)</c> is K x N. All matrices are dense row-major with a leading dimension (the distance in elements
/// between rows of the stored matrix). All sizes, strides and offsets count elements, not bytes.
/// </summary>
public struct GemmProblem
{
    /// <summary>Creates a problem with dense leading dimensions, a single batch, <see cref="Alpha"/> 1 and <see cref="Beta"/> 0.</summary>
    /// <param name="m">Rows of <c>op(A)</c> and of C.</param>
    /// <param name="n">Columns of <c>op(B)</c> and of C.</param>
    /// <param name="k">Columns of <c>op(A)</c>, rows of <c>op(B)</c>.</param>
    public GemmProblem(int m, int n, int k)
    {
        M = m;
        N = n;
        K = k;
        Batch = 1;
        Alpha = 1f;
    }

    /// <summary>Rows of <c>op(A)</c> and of C. At least 0.</summary>
    public int M { get; set; }

    /// <summary>Columns of <c>op(B)</c> and of C. At least 0.</summary>
    public int N { get; set; }

    /// <summary>Inner dimension. At least 0; with 0 the product is empty and C becomes <c>Beta * C</c>.</summary>
    public int K { get; set; }

    /// <summary>Number of independent products, at least 1.</summary>
    public int Batch { get; set; }

    /// <summary>A is stored K x M (so <c>op(A)</c> is its transpose) instead of M x K.</summary>
    public bool TransposeA { get; set; }

    /// <summary>B is stored N x K (so <c>op(B)</c> is its transpose) instead of K x N.</summary>
    public bool TransposeB { get; set; }

    /// <summary>Scale of the product.</summary>
    public float Alpha { get; set; }

    /// <summary>Scale of the old C. With 0, C is not read, so it may hold anything (NaN included).</summary>
    public float Beta { get; set; }

    /// <summary>Row stride of the stored A; 0 means dense (K, or M when <see cref="TransposeA"/>).</summary>
    public int Lda { get; set; }

    /// <summary>Row stride of the stored B; 0 means dense (N, or K when <see cref="TransposeB"/>).</summary>
    public int Ldb { get; set; }

    /// <summary>Row stride of C; 0 means dense (N).</summary>
    public int Ldc { get; set; }

    /// <summary>Index of A's first element in its buffer.</summary>
    public int OffsetA { get; set; }

    /// <summary>Index of B's first element in its buffer.</summary>
    public int OffsetB { get; set; }

    /// <summary>Index of C's first element in its buffer.</summary>
    public int OffsetC { get; set; }

    /// <summary>Elements between the A of one product and the next. 0 shares one A across the batch.</summary>
    public int StrideA { get; set; }

    /// <summary>Elements between the B of one product and the next. 0 shares one B across the batch, as a weight matrix is.</summary>
    public int StrideB { get; set; }

    /// <summary>Elements between the C of one product and the next. Required when <see cref="Batch"/> is above 1, and must keep the outputs apart.</summary>
    public int StrideC { get; set; }

    /// <summary>Floating-point operations of the product, counting a multiply-add as two: <c>2 * Batch * M * N * K</c>.</summary>
    public readonly double Flops => 2.0 * Batch * M * N * K;
}
