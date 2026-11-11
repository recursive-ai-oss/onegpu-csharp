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
| Element types | float32; `Half` Q, K, V and O with float32 accumulation; bfloat16 (`ushort` bit patterns, `EncodeBFloat16`) on matrix hardware only, head dimension up to 128 (the mask is always float32) |
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

Keys are folded into each query row one at a time, in increasing key order, with the online normaliser, from key and value tiles held in workgroup
memory. Two kernels share that structure (`AttentionKernels.Variant`, chosen once in the constructor):

- **`rows`**: one thread owns a query row (64 rows per workgroup). Needs nothing from the device.
- **`pairs`**: two neighbouring threads own a row, each the half of the head dimension, and add their halves of a dot product after one subgroup shuffle
  (32 rows per workgroup). This halves the registers per thread. Chosen automatically on Apple GPUs, which spill past about 128 registers per thread
  (a 64-wide head needs 128 for the query and the accumulator); on the RTX 4070 it was slower for large problems and faster for small ones, so `Auto`
  picks `rows` there. `GpuAttentionVariant` forces either; `allowSubgroups: false` rules out `pairs`.

The compiler keeps loops as loops and a thread-local array indexed by a loop variable can live in slow memory (Apple's compiler does that), so the
SPIR-V and Metal builds write every access to the query and accumulator registers as a separate statement with a literal index (macros in
`Attention.hlsl`). The WebGPU build (`AttentionLoops.hlsl`) keeps plain loops instead: the software adapter's compiler does not finish compiling the unrolled
form, and the browsers' real GPU back ends unroll small loops themselves.

### Long calls

A single submission that keeps an Apple GPU busy for a long time (the one-thread-per-row kernel on a batch of 8 Whisper encoder inputs was the example) is
stopped by the system and leaves the output half written. `Encode` therefore cuts a call whose estimated work exceeds the device's `SubmissionWorkBudget`
(`submissionWorkBudget` in the constructor overrides it; 0 never cuts) into dispatches of whole workgroups and tells the encoder the work after each
(`ReportWork`), so the encoder can end the submission between them. Every workgroup computes its own rows from the same inputs in the same order, so a cut
call equals the whole call bit for bit; the tests compare them on every device. The estimate is `4 · batch · heads · QLen · KLen · HeadDim` operations,
times 1.5 for the matrix kernel (it multiplies `Q Kᵀ` twice) and times 4 for the scalar kernels (several times slower per operation, measured on the M2).

`Fast` and `Deterministic` run the same kernel: every sum has an order the code fixes, there are no atomics and no subgroup arithmetic, and the one
shuffle adds two numbers (which does not depend on operand order). Two runs on one device and driver give identical bits. `rows` and `pairs` add a
dot product in a different order, so they may differ from each other, and devices from each other, in the last bits.

When a problem has too few query blocks to fill a GPU, the keys are cut into at most 16 contiguous ranges of a multiple of 32 keys (at least 128 keys
each, enough ranges for about 256 workgroups); each range stores its partial (maximum, sum, accumulator) in a scratch buffer and a second pass merges them
in range order. The cut depends only on the shape. `ScratchLength(problem)` says how many floats are needed (0 without a cut); pass a
buffer as `scratch`, or leave it null and the family keeps one of its own and releases it on `Dispose`.

### Matrix hardware (`cooperative-matrix`)

For half and bfloat16 tensors with a head dimension up to 128, `Auto` picks a kernel that runs `Q Kᵀ` and `P V` as 16 x 16 x 16 cooperative matrices
(`VK_KHR_cooperative_matrix` on Vulkan, simdgroup matrices on Metal), when the device lists such matrices of the element type with a float32 accumulator,
runs 32-wide subgroups (pinned where the device can run several widths) with full subgroups, and offers the kernel's shader features. Everything else (float32,
head dimensions 129 to 256, devices without matrix hardware such as lavapipe and the RADV iGPU, WebGPU) keeps `rows` or `pairs`; `GpuAttentionVariant.Cooperative`
forces the matrix kernel (and throws where it cannot run) and `allowMatrixHardware: false` rules it out. `Variant` names the kernel used for half tensors,
`FloatVariant` the one for float32.

A workgroup owns 64 query rows as four subgroups of 16 rows and visits the keys in tiles of 32 (16 for 128-wide heads), staged through workgroup memory
(at most 12.8 KB, so it also fits a Metal limit that shader validation counts twice). The elements of a cooperative matrix have no specified layout, so an
accumulator cannot be scaled per row without a round trip through workgroup memory; the kernel avoids the rescaling of the online normaliser instead. A first pass over the
keys computes only the scores and finds the exact maximum of every row; a second pass recomputes them, forms `p = exp(score − max)` and accumulates `P V` in
matrix registers. The price is a second `Q Kᵀ`, 1.5x the matrix work. The probabilities are rounded to the element type before the product with `V`, and
the normaliser sums the rounded values, so numerator and denominator see the same `p`. Scores and probabilities pass between the matrix units and the
lanes through workgroup memory, where lane `l` of a subgroup owns row `l & 15` and every other column of each 16-wide block; the two lanes of a row meet once per pass in a
shuffle that maximises or adds two numbers (operand order does not matter). The key-range split and the merge pass are those of the scalar kernels.

This kernel serves `Fast` and `Deterministic` alike: every order is fixed by the code (tiles in increasing key order, the hardware's matrix product depends only on
its operands), there are no atomics and no subgroup arithmetic, and two runs on one device and driver give identical bits (checked per device by the tests). Devices differ
from each other in the last bits, and so do the matrix and scalar kernels.

## Tolerances

Scaled error (max |gpu − reference| / max |reference| per output, reference in double on the CPU, inputs rounded to half first for half rows): float32
≤ 1e-5, half with float32 accumulation ≤ 2e-3, bfloat16 with float32 accumulation ≤ 8e-3 (the probabilities and the output are rounded to bfloat16's eight bits of precision, which costs about 4e-3). The tests assert these on the shapes of the reference models (MiniLM, BERT-base, Whisper-small encoder,
decoder self- and cross-attention) and on tile edges, remainders, size 1, non-power-of-two sizes, masks with `-∞`, fully masked rows, key lengths of 0,
grouped heads and odd head dimensions; they also compare the fused kernel with GEMM + add + softmax + GEMM.
