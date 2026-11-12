# Matrix multiplication (`Gemm`)

`RecursiveAI.OneGpu.Kernels.Gemm` computes `C = alpha * op(A) * op(B) + beta * C` for `Batch` independent products. `op(X)` is `X` or its
transpose. A and B are float32, float16 or bfloat16; products and sums are float32 and C is float32. Matrices are dense row-major with a
leading dimension; sizes, strides and offsets count elements. The class is created once per element type and mode, picks its variant
from the device's capabilities at creation, and records into a caller's `GpuCommandEncoder`.

## Variants

| Variant (`Gemm.Variant`) | Used when | Notes |
|---|---|---|
| `tiled-64x64x16` | everything else, and always in `Deterministic` mode | HLSL through the shader compiler: SPIR-V, MSL and WGSL. |
| `cooperative-matrix-16x16x16-tile64x64x16-8sg` (`-4sg`) | Vulkan, `Fast`, f16 or bf16 inputs; subgroups of 32 (fixed, or selectable and pinned to 32); pipelines with full subgroups; the device lists a 16x16x16 combination of the input type with float32 accumulation; the kernel's shader features (16-bit storage, `VK_KHR_cooperative_matrix`, the Vulkan memory model, and for bf16 `VK_KHR_shader_bfloat16`) | HLSL `CooperativeMatrix` through the shader compiler. Eight subgroups per workgroup; four if the device allows fewer than 256 invocations or too little workgroup memory. |
| `simdgroup-matrix-16x16x16-tile64x64x16-8sg` (`-4sg`) | Metal, `Fast`, the device lists an 8x8x8 combination of the input type with float32 accumulation, 32-wide simdgroups | The same HLSL; the shader compiler lowers a 16x16 matrix to 2x2 `simdgroup_matrix` 8x8 tiles. Float32 inputs use it too. The eight-subgroup float32 shape declares 18944 bytes of threadgroup memory; the constructor takes the shape from the length the compiled pipeline reports (`GpuComputePipeline.ReportedWorkgroupMemoryBytes`), so with Metal shader validation on, which doubles the reported figure to 37888 against the 32768-byte limit, it falls back to four simdgroups, and without validation it keeps eight. A three-component vector in workgroup memory takes 16 bytes in MSL, which a size formula would also miss. |

On Metal the float32 matrix kernel reads its fragments straight from the operands for every tile that lies inside C and every whole 16 steps of k, and stages only edge tiles and the ragged k tail
(`GEMM_DIRECT`), with four subgroups: 1024 x 3072 x 768 ran at 1488 GFLOP/s against 1065 for the staged eight-subgroup kernel, and 1024 x 768 x 3072 at 1470 against 869 (Apple M2, host clock, indicative).

`GemmOptions.AllowMatrixHardware = false` forces the portable variant. WebGPU has no matrix hardware in the browser today and always runs
the portable variant, whose half-precision inputs are read as pairs inside 32-bit words, so it needs no 16-bit storage feature and runs in
Chrome without `shader-f16`.

**Which devices take the matrix kernel.** The choice is made once, in the constructor, from `GpuCapabilities.Compute` and the kernel's own
reflection (`EntryLayout.RequiredFeatures` against `ShaderFeatures`); nothing is guessed from the vendor. The kernel is written for 32-wide
subgroups: its pipeline is created with full subgroups and, where the device can run several widths, with the subgroup size pinned to 32
(`ComputePipelineDesc.RequireFullSubgroups`, `RequiredSubgroupSize`). A device that only has 64-wide subgroups, such as an AMD GPU in wave64
mode with no wave32 option, does not qualify and runs the portable kernel; a wave64 variant has not been written because no device that
offers one was available to test it on. The AMD GPU in the RADV test machine (Raphael iGPU, subgroups 32 to 64, size selectable) reports no
cooperative-matrix shapes at all, so it takes the portable kernel as well, as does lavapipe.

### Portable kernel

One 64 x 64 tile of C per workgroup of 256 invocations; the k loop runs in steps of 16 through two workgroup-memory tiles (Goto and van
de Geijn blocking); every invocation owns a 4 x 4 register tile with interleaved rows and columns. Tiles are staged with bounds checks, so
any M, N, K, leading dimension and offset works. A grid of 64 x 64 tiles has `ceil(M/64) * ceil(N/64) * Batch` workgroups.

### Matrix-hardware kernels

The same 64 x 64 output tile and 16-wide k step as the portable kernel. Eight subgroups (2 x 4) each own 32 x 16 of C as two 16 x 16 float32
accumulators; the four-subgroup shape (2 x 2) owns 32 x 32 as four. Each k step copies the A and B tiles from the operand buffers, in their
own element type, into padded workgroup memory (this is where the bounds, the transposes and odd leading dimensions are handled), then every
subgroup loads its fragments and multiplies them with cooperative matrices. The accumulators leave through workgroup memory so alpha, beta and
the edges are plain code. The copy involves no conversion: reading bfloat16 through float32 first cost a third of the throughput. The 16-bit
elements are read from storage buffers as 16-bit values, which needs 16-bit storage access; every device with cooperative matrices has it.

## Deterministic mode

`GpuKernelMode.Deterministic` always uses the portable kernel. Each element of C is summed by one invocation, in increasing k, in float32
with fused multiply-add, or, where the grid is small (see "Small grids and few rows"), as range sums added in range order, in both modes; there is
no atomic, no subgroup arithmetic and no dependence on how workgroups are scheduled, so two runs on one device and driver give the same bits. Different devices and drivers may differ in the last bits (fused multiply-add
contraction and rounding of the compiler's output are theirs). `Fast` may use another summation order and the matrix-hardware variants, so
`Fast` and `Deterministic` results differ by rounding.

## Small grids and few rows (portable kernel, Vulkan and WebGPU)

Two cases leave a large GPU idle with the 64 x 64 tile: a small grid over a long k, and a product with only a few rows. Both are chosen from the shape alone and serve
`Fast` and `Deterministic` alike, because both fix the order in which an element's terms are added.

- **Split k.** With fewer than 384 workgroups of 64 x 64 (a 128 x 768 product has 24) and at least 128 terms of k, k is cut into up to 16 ranges of a multiple of 32 terms (at least 64
  terms each), aiming at about 384 workgroups. Each range writes its raw sums to its own plane of an instance-owned 16 MiB float32 scratch buffer (allocated on first use; fewer ranges when
  the planes would not fit), and one more pass adds the planes in range order, then applies alpha, beta, the bias and the activation. A product cut into row or batch pieces for the
  submission budget uses the ranges of the whole product, so its bits do not change. Not used on matrix hardware, whose kernels are untouched.
- **Few rows.** A float32 product with at most four rows of A (decoding one to four tokens), A not transposed, runs a kernel that streams B once: for B stored K x N a workgroup
  owns 16 columns (32 from N = 8192, so a warp reads 128 bytes of a row) and its 256 threads split k into 16 (8) interleaved slices; for B stored N x K a workgroup owns 8 columns and 32
  threads per column read consecutive k. Every thread chains its terms in increasing k, and one thread adds the slice (lane) sums in order through workgroup memory.
  Not on Metal. Float16 and bfloat16 inputs keep the tile.

Measured on the RTX 4070 (GPU time from timestamps, float32, indicative; before is the 64 x 64 tile alone):

| Product (M x N x K) | Before | After | |
|---|---|---|---|
| 128 x 768 x 768 (BERT-base projection, batch 1) | 58 us | 29 us | 2.0x |
| 128 x 2304 x 768 | 94 us | 76 us | 1.2x |
| 128 x 3072 x 768 (feed-forward up) | 135 us | 100 us | 1.35x |
| 128 x 768 x 3072 (feed-forward down) | 222 us | 100 us | 2.2x |
| 1024 x 768 x 3072 (BERT batch 8, feed-forward down) | 925 us | 758 us | 1.2x |
| 1 x 768 x 768 | 57 us | 5.9 us | 9.7x |
| 1 x 3072 x 768 | 93 us | 11 us | 8.4x |
| 1 x 768 x 3072 | 221 us | 15 us | 15x |
| 1 x 51865 x 768, B stored N x K (decoder logits) | 771 us | 330 us | 2.3x (159 MB in 330 us: 480 GB/s) |
| 1 x 51865 x 768, B stored K x N | 768 us | 337 us | 2.3x |
| 4 x 3072 x 768 | 93 us | 11 us | 8.6x |

After a split the 128-row products run at 6.0 TFLOP/s, the tile's rate on large shapes (5.1-6.9), so the next step for them is a faster float32 tile, not more splitting.

## Accumulation order

Rounding in one chain over k grows roughly in proportion to k, because every addition rounds a partial sum that keeps growing (Higham,
*Accuracy and Stability of Numerical Algorithms*, chapter 4). The float32 kernels sum in blocks and add the block sums, so each rounding
happens on a smaller partial sum. The order depends only on the shape and the variant: no atomics, no subgroup operations, nothing that
varies with scheduling, so two runs on one device and driver are bit-identical in both modes.

- Portable kernel (all element types, and the convolution kernel that shares its tile): the 16 products of one staged step chain in a
  register; two staged steps (32 terms) form a block sum; four block sums (128 terms) form a group sum; group sums join the total in
  increasing k. Float16 and bfloat16 inputs read through the same kernel use the same order.
- Float32 matrix hardware (Apple, and the implicit-GEMM convolution on it): each subgroup keeps a second accumulator that takes eight
  staged steps (128 terms) and then joins the total, in increasing k.
- Float16 and bfloat16 on matrix hardware keep a single chain: the hardware sums each 16-term step before it reaches the accumulator, the
  inputs' own rounding dominates the error at these types' tolerances, and the second accumulator cost about half the throughput on NVIDIA.
- Convolution split along k (Apple small grids) adds its partial results in range order, as before.

## Bias and activation epilogue

`Encode` takes an optional `GemmEpilogue`: a float32 bias vector (element `j` is added to column `j` of every row of every product in the batch, from an element offset) and an activation
(`Relu`, `GeluErf`, `GeluTanh`), applied to each result element after `Alpha * product + Beta * C`, in the same dispatch and by every variant (portable, cooperative-matrix, simdgroup-matrix).
The activations are the functions `ElementwiseKernels` runs (`GpuUnaryOp.Relu`, `GeluErf`, `GeluTanh`), and the bias is one float32 add, so the fused result is the result of the product followed by those kernels:
the tests compare the two bit for bit (every dtype, variant, mode, activation, with and without bias) and the comparison is exact on every device tested. Where a compiler fuses
`Alpha * product` with the bias add into one fused multiply-add, the last bit can differ for an `Alpha` that is neither 1 nor a power of two; none of the devices tested did. Bias and activation are push
constants plus one more binding, not pipeline variants, so the deterministic variant stays deterministic (the epilogue is per element). The bias is float32 whatever the input type, because C is float32.

## Accuracy

Metric: `max |gpu - ref| / max |ref|` over the output, with the reference computed in double from the inputs after rounding to the input
type. The tests report it together with the largest element-wise relative error over elements whose reference is above 1e-3 of the largest
(smaller ones are cancellation noise).

| Inputs | Bound |
|---|---|
| float32 | 1e-5 |
| float16 with float32 accumulation | 2e-3 |
| bfloat16 with float32 accumulation | 8e-3 |

All variants measured far below these, because the inputs are exactly the rounded values and only the float32 accumulation order differs.
The float32 accumulation order is described under "Accumulation order" below; with it the scaled error against a double-precision sum is
about 1.5e-7 to 4e-7 at K = 768 to 4096 (a single chain over k gave 1e-6 to 4.5e-6 there). Elements outside the written region of C are checked to be untouched. With `Beta = 0` C is never
read.

## Limits and errors

- Every shape, stride, offset and buffer range is checked in 64-bit arithmetic before anything is recorded; a bad one throws
  `GpuKernelShapeException` naming the operation and the dimension. The grid must fit `MaxWorkgroupCount` per axis.
- Each operand must fit one storage-buffer binding (`MaxStorageBufferBindingBytes`); a larger one throws `GpuUnsupportedException`. Software
  Vulkan drivers report 128 MB, which a 51865 x 768 float32 matrix exceeds.
- Buffers made by `CreateBuffer<T>` are padded to a whole 32-bit word, so a 16-bit element at the end of an odd-length buffer can be read
  as part of a word.
- Recording a call allocates nothing once the bind group for its three buffers is cached (64 triples; `ReleaseBindings` drops them).
- A batch with `StrideB = 0` shares one B (a weight matrix) across the batch; `StrideC` must keep the outputs apart.

## Throughput (indicative)

GPU time from timestamp queries (median of 30 dispatches) on the RTX 4070, driver 595; host time between submit and idle on the Apple M2
(Metal here reports no timestamp queries). Peaks used: RTX 4070 float32 29.2 TFLOP/s (5888 lanes x 2 x 2.475 GHz); float16 tensor with
float32 accumulation about 58 TFLOP/s dense (derived from the Ada figures, so approximate); M2 float32 3.6 TFLOP/s (Apple's figure for the
10-core GPU). These are indicative numbers, measured while other jobs shared the machines.

| Device, variant, inputs | Large shapes (GFLOP/s) | Fraction of peak |
|---|---|---|
| RTX 4070, tiled, float32 | 5100 to 6900 (square 4096: 5500) | 18 to 24 % |
| RTX 4070, cooperative matrix, float16, 8 subgroups | 25500 to 34200 (square 4096: 36500) | 44 to 59 % (square: 63 %) |
| RTX 4070, cooperative matrix, bfloat16, 8 subgroups | 25100 to 33900 (square 4096: 36300) | 43 to 58 % (square: 63 %) |
| RTX 4070, cooperative matrix, float16 or bfloat16, 4 subgroups | 19800 to 29700 (square 4096: 30900) | 34 to 51 % |
| Apple M2, simdgroup matrix, float16, 8 simdgroups | 436 to 456 (square 4096: 402 to 407) | 12 % |
| Apple M2, simdgroup matrix, bfloat16, 8 simdgroups | 419 to 440 (square 4096: 406 to 435) | 12 % |
| Apple M2, simdgroup matrix, float32, 4 simdgroups | 308 to 358 (square 4096: 328) | 9 to 10 % |
| Apple M2, tiled | 100 to 139 | 3 to 4 % |

The Apple rows were measured on a machine that other jobs were using, so treat them as lower bounds. The float32 matrix kernel on the M2 is
about 20 % slower than the 8x8 kernel written by hand that it replaced (308 to 358 against 398 to 452), because the Metal compiler counts the
8-simdgroup float32 shape at twice its declared threadgroup memory and so it keeps four simdgroups; float16 is 2 to 9 % below the old kernel
and bfloat16 is three to five times faster (the old kernel converted each element). On the RTX 4070 the float16 kernel is 1.25 to 2.0 times
the hand-written SPIR-V it replaced and bfloat16 is 1.3 to 2.2 times (the smallest gain is the batched 128 x 128 scores product, the largest the
square 4096 product), because the operands are now copied into workgroup memory without a detour through float32 and eight subgroups hide more
latency than four.

Single-row products (decode, M = 1) are memory bound. On Vulkan and WebGPU, float32 products of at most four rows have their own kernel (see "Small grids and few rows"):
242 GFLOP/s for the vocabulary projection on the RTX 4070 at 480 GB/s of weights, against 103 with the tile. On the M2 they still use one row of a 64-row tile (2 to 9 GFLOP/s);
a few-rows kernel and larger tiles for the Apple GPU are the next tuning steps.

## 8-bit products on Apple matrix hardware

`Int8Gemm` on Metal stages each tile as float16 after subtracting the zero points (a value in -255..255 is exact in float16), multiplies on the simdgroup matrices (products exact in float32) and adds float32 sums of at most 2^24 in magnitude. A chunk of 16 steps of 16 (256 terms) is exact for two operands up to 255 in magnitude; 32 or 64 steps when an operand is a plain int8. After each chunk the sums become int32 and add into int32 registers, which wrap like the definition. The result is the same integer for every summation order, so one kernel serves both modes. Tests include K = 4096 at the extremes.
`Int8GemmSkinny` takes M <= 16 with a non-transposed B (decoder steps): 32 columns x 4 rows per workgroup, k split 16 ways, integer sums.
Indicative on an M2 (GPU timestamps): the matrix tile reaches 0.67-0.77 TOPS at BERT, MiniLM and Whisper-encoder shapes against 0.52-0.58 for the portable tile (1.3x, staging bound); decoder-step products with M = 1 take 28 us (projection), 80 us (ffn up), 97 us (ffn down) and 2.97 ms (vocabulary) against 271, 619, 1099 and 10020 us.
A transposed B with M = 1 still takes the portable path.
