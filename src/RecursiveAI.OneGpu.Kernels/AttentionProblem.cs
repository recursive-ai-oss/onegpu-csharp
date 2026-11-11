namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// Where a rank-4 attention tensor sits in its buffer: an element offset and the distances, in elements, between batch entries, heads and
/// rows (sequence positions). The elements of one row are contiguous. Strides may be 0 to broadcast (masks) but must not be negative.
/// </summary>
public readonly record struct GpuAttentionLayout(int Offset, int BatchStride, int HeadStride, int RowStride)
{
    /// <summary>Dense <c>[batch, heads, rows, headDim]</c>.</summary>
    /// <param name="heads">Number of heads.</param>
    /// <param name="rows">Sequence positions per head.</param>
    /// <param name="headDim">Elements per row.</param>
    /// <param name="offset">Element offset of the tensor in its buffer.</param>
    /// <exception cref="GpuKernelShapeException">A size is negative or the strides do not fit 32 bits.</exception>
    public static GpuAttentionLayout HeadsFirst(int heads, int rows, int headDim, int offset = 0)
    {
        long head = checked((long)rows * headDim);
        return Make(offset, checked(heads * head), head, headDim);
    }

    /// <summary>
    /// Dense <c>[batch, rows, heads, headDim]</c>, which is also the packed <c>[batch, rows, heads * headDim]</c> form: what a projection
    /// leaves before the heads are split, and what the heads are merged into after attention.
    /// </summary>
    /// <param name="heads">Number of heads.</param>
    /// <param name="rows">Sequence positions.</param>
    /// <param name="headDim">Elements per head and row.</param>
    /// <param name="offset">Element offset of the tensor in its buffer.</param>
    /// <exception cref="GpuKernelShapeException">A size is negative or the strides do not fit 32 bits.</exception>
    public static GpuAttentionLayout PositionsFirst(int heads, int rows, int headDim, int offset = 0)
    {
        long row = checked((long)heads * headDim);
        return Make(offset, checked(rows * row), headDim, row);
    }

    /// <summary>
    /// The layout of an additive mask of the given shape that broadcasts, right-aligned, against <c>[batch, heads, qLen, kLen]</c>: a
    /// dimension of extent 1 gets stride 0. The last dimension must be <paramref name="kLen"/> (a mask with one value per row has to be
    /// widened by the caller).
    /// </summary>
    /// <param name="shape">The mask's shape, at most four dimensions.</param>
    /// <param name="batch">Batch size of the attention.</param>
    /// <param name="heads">Query heads of the attention.</param>
    /// <param name="qLen">Query length.</param>
    /// <param name="kLen">Key length.</param>
    /// <param name="offset">Element offset of the mask in its buffer.</param>
    /// <exception cref="GpuKernelShapeException">The shape cannot broadcast to the attention scores, or its last dimension is not <paramref name="kLen"/>.</exception>
    public static GpuAttentionLayout ForMask(ReadOnlySpan<int> shape, int batch, int heads, int qLen, int kLen, int offset = 0)
    {
        const string Name = "Attention";
        if (shape.Length is < 1 or > 4)
        {
            throw new GpuKernelShapeException(Name, "mask", $"a mask has 1 to 4 dimensions, not {shape.Length}");
        }

        Span<int> full = [batch, heads, qLen, kLen];
        Span<int> dims = [1, 1, 1, 1];
        for (var i = 0; i < shape.Length; i++)
        {
            dims[4 - shape.Length + i] = shape[i];
        }

        for (var i = 0; i < 4; i++)
        {
            if (dims[i] != 1 && dims[i] != full[i])
            {
                throw new GpuKernelShapeException(Name, $"mask[{i}]", $"extent {dims[i]} cannot broadcast to {full[i]}");
            }
        }

        if (dims[3] != kLen)
        {
            throw new GpuKernelShapeException(Name, "mask[3]", $"the last dimension must hold the {kLen} keys, not {dims[3]}");
        }

        long row = dims[2] == 1 ? 0 : kLen;
        long head = dims[1] == 1 ? 0 : checked((long)dims[2] * kLen);
        long bat = dims[0] == 1 ? 0 : checked(dims[1] * (long)dims[2] * kLen);
        return Make(offset, bat, head, row);
    }

    static GpuAttentionLayout Make(int offset, long batch, long head, long row)
    {
        if (batch > int.MaxValue || head > int.MaxValue || row > int.MaxValue)
        {
            throw new GpuKernelShapeException("Attention", "layout", "a stride does not fit 32 bits");
        }

        return new GpuAttentionLayout(offset, (int)batch, (int)head, (int)row);
    }
}

/// <summary>What an attention row with no visible key (everything masked, or no keys at all) produces.</summary>
public enum GpuAttentionMaskedRows
{
    /// <summary>NaN in every output of the row, as the textbook formula gives (0 / 0).</summary>
    Nan = 0,

    /// <summary>Zeros: a row that attends to nothing contributes nothing.</summary>
    Zero = 1,
}

/// <summary>
/// One scaled dot-product attention: <c>O = softmax(Scale * Q Kᵀ + Mask) V</c> for every batch entry and head, where the keys and values
/// have <see cref="KLen"/> positions and the queries <see cref="QLen"/>. Q, K, V and O have <see cref="HeadDim"/> elements per row and
/// are described by a <see cref="GpuAttentionLayout"/> each, so any mix of head-first and position-first tensors, and slices of a packed
/// query/key/value projection, need no copy. K and V may have fewer heads than Q (<see cref="KvHeads"/>): query head <c>h</c> reads
/// key/value head <c>h / (Heads / KvHeads)</c>.
/// </summary>
public struct AttentionProblem
{
    /// <summary>Creates a problem with head-first dense layouts, <see cref="Scale"/> = 1 / sqrt(headDim), equal query and key/value heads, no causal mask.</summary>
    /// <param name="batch">Batch entries.</param>
    /// <param name="heads">Query heads.</param>
    /// <param name="qLen">Query positions.</param>
    /// <param name="kLen">Key/value positions (the cache length plus the new positions in a decoder).</param>
    /// <param name="headDim">Elements per head and position, 1 to 256.</param>
    public AttentionProblem(int batch, int heads, int qLen, int kLen, int headDim)
    {
        Batch = batch;
        Heads = heads;
        KvHeads = heads;
        QLen = qLen;
        KLen = kLen;
        HeadDim = headDim;
        Scale = headDim > 0 ? 1f / MathF.Sqrt(headDim) : 1f;
        Q = GpuAttentionLayout.HeadsFirst(heads, qLen, headDim);
        K = GpuAttentionLayout.HeadsFirst(heads, kLen, headDim);
        V = K;
        O = Q;
    }

    /// <summary>Batch entries, at least 0.</summary>
    public int Batch { get; set; }

    /// <summary>Query heads, at least 0.</summary>
    public int Heads { get; set; }

    /// <summary>Key/value heads; <see cref="Heads"/> must be a multiple of it. Grouped-query attention uses fewer than <see cref="Heads"/>.</summary>
    public int KvHeads { get; set; }

    /// <summary>Query positions, at least 0.</summary>
    public int QLen { get; set; }

    /// <summary>Key/value positions, at least 0. With 0 every row is "masked" (see <see cref="MaskedRows"/>).</summary>
    public int KLen { get; set; }

    /// <summary>Elements per head and position, 1 to 256. Half precision needs it even.</summary>
    public int HeadDim { get; set; }

    /// <summary>Multiplier of the dot products, applied before the mask is added. The constructor sets <c>1 / sqrt(HeadDim)</c>.</summary>
    public float Scale { get; set; }

    /// <summary>Hide keys after the query's own position: key <c>j</c> is visible to query <c>i</c> when <c>j &lt;= i + CausalOffset</c>.</summary>
    public bool Causal { get; set; }

    /// <summary>
    /// With <see cref="Causal"/>, how many cached keys precede the queries. When null, <c>KLen - QLen</c>, which is right when the queries are
    /// the last <see cref="QLen"/> positions of the keys (a decoder step with a key/value cache, or 0 for self-attention without a cache).
    /// </summary>
    public int? CausalOffset { get; set; }

    /// <summary>Result of a row without any visible key. The default matches a plain softmax (NaN).</summary>
    public GpuAttentionMaskedRows MaskedRows { get; set; }

    /// <summary>Query tensor, <c>[Batch, Heads, QLen, HeadDim]</c> in the layout given.</summary>
    public GpuAttentionLayout Q { get; set; }

    /// <summary>Key tensor, <c>[Batch, KvHeads, KLen, HeadDim]</c> in the layout given.</summary>
    public GpuAttentionLayout K { get; set; }

    /// <summary>Value tensor, <c>[Batch, KvHeads, KLen, HeadDim]</c> in the layout given.</summary>
    public GpuAttentionLayout V { get; set; }

    /// <summary>Output tensor, <c>[Batch, Heads, QLen, HeadDim]</c> in the layout given. It must not overlap Q, K or V.</summary>
    public GpuAttentionLayout O { get; set; }

    /// <summary>Layout of the additive mask passed to <c>Encode</c> (see <see cref="GpuAttentionLayout.ForMask"/>); ignored without a mask.</summary>
    public GpuAttentionLayout Mask { get; set; }

    /// <summary>Offset in the key-length buffer passed to <c>Encode</c>; entry <c>b</c> is read at <c>KeyLengthsOffset + b</c>.</summary>
    public int KeyLengthsOffset { get; set; }
}
