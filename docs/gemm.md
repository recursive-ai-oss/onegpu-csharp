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
with fused multiply-add; there is no split of k, no atomic, no subgroup arithmetic and no dependence on how workgroups are scheduled, so two
runs on one device and driver give the same bits. Different devices and drivers may differ in the last bits (fused multiply-add
contraction and rounding of the compiler's output are theirs). `Fast` may use another summation order and the matrix-hardware variants, so
`Fast` and `Deterministic` results differ by rounding.

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

All variants measured far below these (about 1e-6 to 8e-6 for K up to 3072), because the inputs are exactly the rounded values and only
the float32 accumulation order differs. Elements outside the written region of C are checked to be untouched. With `Beta = 0` C is never
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

Single-row products (decode, M = 1) are memory bound and use one row of a 64-row tile: 100 to 150 GFLOP/s on the RTX 4070, 2 to 9 on the M2.
A skinny-M variant and larger tiles for the Apple GPU are the next tuning steps.
