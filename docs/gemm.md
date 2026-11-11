# Matrix multiplication (`Gemm`)

`RecursiveAI.OneGpu.Kernels.Gemm` computes `C = alpha * op(A) * op(B) + beta * C` for `Batch` independent products. `op(X)` is `X` or its
transpose. A and B are float32, float16 or bfloat16; products and sums are float32 and C is float32. Matrices are dense row-major with a
leading dimension; sizes, strides and offsets count elements. The class is created once per element type and mode, picks its variant
from the device's capabilities at creation, and records into a caller's `GpuCommandEncoder`.

## Variants

| Variant (`Gemm.Variant`) | Used when | Notes |
|---|---|---|
| `tiled-64x64x16` | everything else, and always in `Deterministic` mode | HLSL through the shader compiler: SPIR-V, MSL and WGSL. |
| `cooperative-matrix-16x16x16-tile64x64x16` | Vulkan, `Fast`, f16 or bf16 inputs, 32-wide subgroups, the device lists a 16x16x16 combination of the input type with float32 accumulation, 12 KB of workgroup memory | Hand-written SPIR-V (`tools/kernels`) until the shader compiler has cooperative-matrix types. Needs the device features `cooperativeMatrix` and `vulkanMemoryModel` (the device enables both when it has them), and for bf16 `shaderBFloat16CooperativeMatrix`. |
| `simdgroup-matrix-8x8x8-tile64x64x16` | Metal, `Fast`, the device lists an 8x8x8 combination of the input type with float32 accumulation, 32-wide simdgroups, 256 invocations, 16 KB of threadgroup memory | Metal Shading Language text (`src/RecursiveAI.OneGpu.Kernels/Raw`). Also used for float32 inputs. |

`GemmOptions.AllowMatrixHardware = false` forces the portable variant. WebGPU has no matrix hardware in the browser today and always runs
the portable variant. Half-precision inputs are read as pairs inside 32-bit words, so the portable variant needs no 16-bit storage
feature and runs in Chrome without `shader-f16`.

### Portable kernel

One 64 x 64 tile of C per workgroup of 256 invocations; the k loop runs in steps of 16 through two workgroup-memory tiles (Goto and van
de Geijn blocking); every invocation owns a 4 x 4 register tile with interleaved rows and columns. Tiles are staged with bounds checks, so
any M, N, K, leading dimension and offset works. A grid of 64 x 64 tiles has `ceil(M/64) * ceil(N/64) * Batch` workgroups.

### Matrix-hardware kernels

The same 64 x 64 output tile and 16-wide k step. Vulkan: four subgroups (2 x 2), each with four 16 x 16 float32 accumulators. Metal: eight
simdgroups (2 x 4), each with eight 8 x 8 accumulators. Operands are converted into workgroup memory in the matrix element type
(including the bounds handling and the transposes), the accumulators leave through workgroup memory so alpha, beta and the edges are plain
code. bf16 uses the native bfloat16 matrix type on both APIs.

## Deterministic mode

`GpuKernelMode.Deterministic` always uses the portable kernel. Each element of C is summed by one invocation, in increasing k, in float32
with fused multiply-add; there is no split of k, no atomic, no subgroup arithmetic and no dependence on how workgroups are scheduled, so two
runs on one device and driver give the same bits. Different devices and drivers may differ in the last bits (fused multiply-add
contraction and rounding of the compiler's output are theirs). `Fast` may use another summation order and the matrix-hardware variants, so
`Fast` and `Deterministic` results differ by rounding.

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
10-core GPU). These are first-version numbers, measured while other jobs shared the machines.

| Device, variant, inputs | Large shapes (GFLOP/s) | Fraction of peak |
|---|---|---|
| RTX 4070, tiled, float32 | 5500 to 7100 (square 4096: 7130) | 19 to 24 % |
| RTX 4070, cooperative matrix, float16 | 17700 to 25400 (square 4096: 25400) | 30 to 44 % |
| RTX 4070, cooperative matrix, bfloat16 | 16200 to 28400 (square 4096: 22400 to 28400) | 28 to 49 % |
| Apple M2, simdgroup matrix, float32 and float16 | 420 to 475 (square 4096: 406 to 442) | 12 % |
| Apple M2, simdgroup matrix, bfloat16 | 64 to 133 | 2 to 4 % (bfloat16 matrices are not native on this GPU) |
| Apple M2, tiled | 100 to 139 | 3 to 4 % |

Single-row products (decode, M = 1) are memory bound and use one row of a 64-row tile: 100 to 150 GFLOP/s on the RTX 4070, 2 to 9 on the M2.
A skinny-M variant and larger tiles for the Apple GPU are the next tuning steps.
