# Attention

`AttentionKernels` computes `O = softmax(Scale · Q Kᵀ + Mask) V` for every batch entry and head in one kernel. The score matrix is never written to
memory: a workgroup keeps a block of 64 query rows and walks over the keys one tile at a time, keeping for each row a running maximum, a running
sum of `exp(s − max)` and the accumulated `exp(s − max) · V`, and rescales them whenever a tile raises the maximum (Dao et al., *FlashAttention*,
2022, for the tiling; Milakov and Gimelshein, *Online normalizer calculation for softmax*, 2018, for the normaliser).

```csharp
using var attention = new AttentionKernels(device);
var p = new AttentionProblem(batch, heads, qLen, kLen, headDim)
{
    Q = GpuAttentionLayout.PositionsFirst(heads, qLen, headDim),   // [batch, qLen, heads, headDim], as a projection leaves it
    K = GpuAttentionLayout.PositionsFirst(heads, kLen, headDim),
    V = GpuAttentionLayout.PositionsFirst(heads, kLen, headDim),
    O = GpuAttentionLayout.PositionsFirst(heads, qLen, headDim),
    Mask = GpuAttentionLayout.ForMask([batch, 1, 1, kLen], batch, heads, qLen, kLen),   // optional additive mask
};
var encoder = device.BeginCommands();
attention.Encode(encoder, p, q, k, v, o, mask: maskBuffer, mode: GpuKernelMode.Deterministic);
device.Submit(encoder);
```

## What it covers

| | |
|---|---|
| Element types | float32; `Half` Q, K, V and O with float32 accumulation (the mask is always float32) |
| Layouts | any `GpuAttentionLayout` (offset, batch, head and row strides; elements of a row contiguous): head-first, position-first, slices of a packed QKV projection |
| Lengths | query and key/value lengths differ (cross-attention, decoder steps with a key/value cache) |
| Heads | equal query and key/value heads, or fewer key/value heads (grouped and multi-query attention) |
| Head dimension | 1 to 256; 32 and 64 are tuned, others are padded up to 32, 64, 128 or 256 (above 64 the registers spill: correct, slow). Half needs it even |
| Masks | additive float mask broadcast over batch, heads and rows (`-∞` hides a key); per-batch key lengths (`GpuBuffer<int>`); causal with an offset (default `KLen − QLen`, the cached-decoder case) |

## Special values

- A key hidden by `-∞` in the mask, by its batch entry's key length or by the causal rule has weight 0.
- A query row with no visible key gives **NaN** in every output (what `0 / 0` gives, and what a plain softmax over an all-`-∞` row gives), or zeros with
  `MaskedRows = GpuAttentionMaskedRows.Zero` (a row that attends to nothing contributes nothing). `KLen = 0` is such a row.
- A row whose score is NaN or `+∞` gives NaN. V is not inspected.
- In a browser, WGSL does not promise infinities and NaN behave, so use a large finite negative mask value there (`-1e30`); the tests do.

## Variants and ordering

One tiled kernel serves both modes. A thread owns a query row, so every sum (the dot products over the head dimension, the normaliser, the `P V`
accumulation over keys) is a sequential loop in increasing index order; there are no atomics and no subgroup operations. `GpuKernelMode.Fast` and
`GpuKernelMode.Deterministic` therefore give identical bits, on one device and driver, run to run, and the forced-off instance
(`allowSubgroups: false`) too. Results may differ in the last bits between devices because the compilers contract multiply-adds differently.

When a problem has too few query blocks to fill a GPU, the keys are cut into at most 16 contiguous ranges of a multiple of 32 keys (at least 128 keys
each, enough ranges for about 256 workgroups); each range stores its partial (maximum, sum, accumulator) in a scratch buffer and a second pass merges them
in range order. The cut depends only on the shape. `AttentionKernels.ScratchLength(problem)` says how many floats are needed (0 without a cut); pass a
buffer as `scratch`, or leave it null and the family keeps one of its own.

A cooperative-matrix variant (`Q Kᵀ` and `P V` on matrix hardware) would be chosen next to the tiled one in `AttentionKernels` when the device reports the
shapes; it waits for matrix types in the shader compiler.

## Tolerances

Scaled error (max |gpu − reference| / max |reference| per output, reference in double on the CPU, inputs rounded to half first for half rows): float32
≤ 1e-5, half with float32 accumulation ≤ 2e-3. The tests assert these on the shapes of the reference models (MiniLM, BERT-base, Whisper-small encoder,
decoder self- and cross-attention) and on tile edges, remainders, size 1, non-power-of-two sizes, masks with `-∞`, fully masked rows, key lengths of 0,
grouped heads and odd head dimensions; they also compare the fused kernel with GEMM + add + softmax + GEMM.
