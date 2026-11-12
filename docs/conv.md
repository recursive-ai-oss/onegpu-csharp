# Convolution, pooling and resize

`ConvKernels`, `PoolKernels` and `ResizeKernels` in `RecursiveAI.OneGpu.Kernels` cover ONNX `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
`GlobalMaxPool` and `Resize` for the vision models (ResNet-50, MobileNetV3, DBNet, a recurrent-recognizer's convolutional stem). They follow the binding shape in
[kernels.md](kernels.md): one class per family, created from a `GpuDevice`, operands as `GpuBuffer<T>` plus element offsets, shape errors as `GpuKernelShapeException`.

## Layout and operands

- Tensors are **NCHW**, weights **OIHW** (`ConvTranspose`: `[Cin, Cout/groups, KH, KW]`). One-dimensional operators are two-dimensional with height 1. NCHW is what ONNX graphs
  carry, so a graph needs no layout pass; NHWC would only pay with one.
- Element types: `float` and `Half` (half in and out, float32 accumulation). Pooling and resize also move any 32-bit or 16-bit element bit-for-bit where the operation copies.
- A problem struct (`ConvProblem`, `ConvTransposeProblem`, `PoolProblem`, `ResizeProblem`) carries the shape, explicit pads, dilation, groups, the fused activation and element
  offsets. `WithAutoPad` and `GpuWindow.OutputExtent` / `TransposeOutputExtent` turn ONNX `auto_pad`, `output_padding` and `output_shape` into pads and output sizes. Sizes are
  checked with `checked` arithmetic and the device's limits before anything is recorded.
- Fused into the last pass, in this order, each step rounded in float32: sum, bias, residual addition, activation. Activations fused into the matrix kernel: identity, tanh, sigmoid,
  relu, silu, hard-sigmoid, hard-swish, clip, leaky-relu; any other `GpuUnaryOp` runs as one extra element-wise pass.
- Scratch (the column matrix of the unfold variants and, for half, float32 accumulators) is supplied by the caller as a buffer **and an element offset** (`columns`/`columnsOffset`,
  `accumulators`/`accumulatorsOffset`, `EncodeTranspose` likewise) or kept by the instance up to 64 MiB. `ConvKernels.Scratch(problem, half)` says how much this device needs for a shape
  (nothing for the variants that gather windows while they multiply), for the whole batch and for one image; a smaller buffer makes the batch run in chunks of whole images, which changes
  no bit of the result. The scratch may live in the buffer that holds the operands (an arena) as long as its range overlaps no operand's (`GpuKernelShapeException` otherwise). A browser
  cannot read and write one buffer in a dispatch, so there the scratch needs a buffer of its own, and so do the output and the inputs.

## Variants

`ConvKernels.DescribeVariant(problem, half)` names what a device runs.

| Layer class | Variant |
|---|---|
| depthwise (one input channel per group, any channel multiplier) | direct kernel, one thread per output element, window order fixed |
| every other convolution on the portable matrix kernel (float32 everywhere; half on devices without matrix hardware) | **implicit GEMM**: a 64x64x16 tile kernel that gathers the input window while it loads the tile; float32 fuses bias, residual and activation into the store, half writes float32 accumulators and an epilogue pass finishes. A float32 grid of fewer than 384 workgroups **splits k** (below) |
| float32 on Metal, fast mode, windows and strided or small-image pointwise layers (channels per group a multiple of 16) | **taps-innermost implicit GEMM** (`ConvTap.hlsli`, 64 x 64 or, for small grids, 64 x 32 tiles), described under "Apple M2, float32, taps innermost" below; the pointwise layers of a normal image size keep the kernel of the next row |
| float32 on Metal (simdgroup matrices) | **implicit GEMM on matrix hardware** (`ConvCoopF32.hlsl`): the 4-subgroup 64x64x16 simdgroup tile of `Gemm`, the window gathered while the input tile is staged (no column matrix), the bias, residual and activation applied as the result is stored, so the output is written once. A pointwise convolution (1x1, stride 1, no padding) whose tile lies inside the output reads weights and input straight from memory with no staging. Grids under 40 workgroups use a 64x32 tile. Deterministic mode, and fast mode where the row above does not apply (stem, depthwise-like channel counts, 1x1 layers of 64 or more pixels): both modes (see Determinism) |
| float16 on Metal (simdgroup matrices) | the same implicit kernel with half operands (`ConvCoop.hlsli` with `CONV_F16`): float32 accumulation on one chain, bias, residual and activation applied as the halves are stored; with split k or an activation the kernel does not fuse, it leaves float32 planes in the accumulator scratch for the epilogue pass |
| 1x1, stride 1, no padding, with matrix hardware (half) | `Gemm` on the input as it lies (no unfolding) |
| any other shape with matrix hardware (half) | im2col (Chellapilla et al., 2006) into the column scratch, then `Gemm` (cooperative or simdgroup matrices) and the epilogue pass; groups loop |
| transposed convolution | `Gemm` of the transposed weights with the input, then a gathering col2im (one thread per output element, no atomics) with the same epilogue |

Variants are chosen once per instance from `GpuCapabilities.Compute`; a device without cooperative matrices (lavapipe, a forced-off run) runs the portable kernel, and the tests
run every layer set in four configurations (fast, fast with matrix hardware off, deterministic, fast with the implicit kernel off).

### Not built, with the numbers

- **Accumulation order.** The implicit-GEMM kernels sum k in blocks exactly as `Gemm` does (portable tile: 32-term block sums in 128-term groups; Apple matrix tile: a second accumulator over 128 terms); the im2col path inherits it from `Gemm`. Depthwise and direct kernels sum at most kernel-area terms in one chain. Scaled error against double at the Whisper stems and ResNet-50 3x3 layers is 1.4e-7 to 3.3e-7 (a single chain gave 4.6e-7 to 3.4e-6).
- **Cooperative-matrix implicit GEMM** (window gather in the matrix tile load). On the RTX 4070, f16 ResNet-50 at b8 the im2col+`Gemm` layers already run at 6.6-13 TFLOP/s
  for 3x3 and 11-21 TFLOP/s for the 1x1 layers that use `Gemm` directly; the unfold pass is a bandwidth-bound write and read of at most a few tens of megabytes per layer. The
  ceiling for the change is the gap between the 3x3 and the 1x1 rate, about 1.3-1.6x on 3x3 layers, which are 2.4 of about 5.8 ms. Dropped as not worth a second coop kernel family now.
- **Winograd F(2x2,3x3)** (Lavin and Gray, 2016); not needed once small grids split k (see above). `WinogradBound` in the tests times the sixteen batched products it needs with the real `Gemm` plus a copy moving the bytes its
  transforms must move: a floor no implementation can beat. RTX 4070, ResNet-50 stride-1 3x3 layers at b8 summed (host clock, indicative): **float32 direct 6.3 ms, floor 2.7 ms;
  half direct 2.4 ms, floor 1.0 ms.** The floor leaves room (the transforms run as three irregular passes, realistically 1.5x the copy floor, and half Winograd carries
  larger rounding error through the transformed data than the 2e-3 half tolerance leaves comfortably), but it is a speed-up for one model's 3x3 layers only, MobileNetV3 and DBNet
  have almost none. Left until ResNet-50 speed needs it; the variant would be a fourth `ConvVariant` behind the same `Encode`.

### Apple M2, float32 (indicative, host clock, 10 calls per command buffer; Metal has no timestamp queries)

Convolution layers summed over the model's layer list, before and after the matrix-hardware implicit kernel (and the direct-load 4-subgroup `Gemm` it shares):

| Model | batch | im2col + `Gemm` + epilogue pass | implicit simdgroup kernel |
|---|---|---|---|
| ResNet-50 | 1 | 26.0 ms | 24.4 ms |
| ResNet-50 | 8 | 117.8 ms | 89.9 ms |
| MobileNetV3-large | 1 | 5.8 ms | 4.3 ms |
| DBNet 512 | 1 | 9.9 ms | 7.2 ms |

The late layers of a classifier at batch 1 (14 x 14 and 7 x 7 images, 4608 deep) ran at 130-300 GFLOP/s: the grid had too few workgroups and each ran its 16-wide k steps one after another.
They now **split k**: when the grid has fewer than 128 workgroups and k has at least 32 steps of 16, k is cut into up to 8 equal ranges (at least 16 steps each; the cut depends only on the shape).
Each range writes a raw partial plane to the scratch, and the epilogue pass adds the planes in range order, then bias, residual and activation (so any activation works). No atomics. `Scratch()` reports
the planes. Batch is part of the shape, so chunking an oversized batch does not change a bit.

| Model (float32, fast, indicative) | batch | split-k off | split-k on |
|---|---|---|---|
| ResNet-50 | 1 | 24-26 ms | 20.3-22.3 ms (earlier session: 19 to 16.9) |
| MobileNetV3-large | 1 | 4.24 ms | 3.50 ms |
| DBNet 512 | 1 | no layer splits (k under 32 steps) | no layer splits; sums differ by run order on the shared GPU clock, not by kernel |
| ResNet-50 | 8 | 90.0 ms | 90.4 ms |
| MobileNetV3-large | 8 | 17.9 ms | 17.6 ms |

The M2's clock state moves ResNet-50 b1 by about 20 % between sessions, so the table pairs runs made alternately in one session (five pairs; split-k won every one). Deterministic mode, ResNet-50 b1 20.4 ms, b8 90.5 ms.

A version that prefetched the next tile into registers ran 40 % slower (register pressure), and wider narrow-tile thresholds ran slower at batch 8.

### RTX 4070, float32 portable kernel: split k for small grids

At batch 1 the late layers of a classifier have few tiles (a 7 x 7 image is one tile of pixels, 512 output channels eight tiles) and each workgroup walks 4608 terms of k alone, so the 4070's 46 multiprocessors
mostly idle: those layers ran at 0.4 to 1.5 TFLOP/s. The portable implicit kernel therefore splits k the way the Apple kernel does: when the grid has fewer than 384 workgroups and k has at least
eight 32-term blocks, k is cut into up to 16 ranges (at least 128 terms each, aiming at about 512 workgroups) that run as workgroups of their own, each into its own plane of raw sums in the column scratch,
and the epilogue pass adds the planes in range order, then bias, residual and activation (any activation works). Ranges are a multiple of 32 terms, so each starts on a block-sum boundary of the kernel;
the cut depends only on the shape, so both modes use it and two runs are bit-identical. `Scratch()` reports the planes (`ranges * output size` per image; the static `ColumnElements` counts the unfolded columns only).
A grid of 392 workgroups (56 x 56 at batch 8) stays whole: two ranges measured 4 to 30 % slower there. Used on every device whose float32 product is the portable tile: NVIDIA, AMD and
Intel Vulkan, lavapipe, WebGPU (Metal has its own split above).

Winograd, which this document priced at a floor of 2.3-2.4x, is not needed: ResNet-50's convolutions at batch 1 went from 6.48 to 2.57 ms.

GPU time from timestamps, RTX 4070, driver 595.99, float32, fast mode, indicative (convolution layers summed over the model's layer list, each shape counted as often as the model uses it; whole-k is the same binary
with the split turned off):

| Model | batch | whole k | split k |
|---|---|---|---|
| ResNet-50 | 1 | 6.48 ms | 2.57 ms |
| ResNet-50 | 8 | 16.67 ms | 15.52 ms |
| MobileNetV3-large | 1 | 1.51 ms | 0.92 ms |
| MobileNetV3-large | 8 | 2.87 ms | 2.57 ms |
| DBNet 512 | 1 | 0.95 ms | 0.95 ms |
| DBNet 512 | 8 | 6.77 ms | 5.75 ms |

ResNet-50 batch 1, layers (microseconds, whole k then split k): 3x3 512 -> 512 at 7 x 7 520 -> 73; 3x3 512 -> 512 stride 2 on 14 x 14 532 -> 74; 3x3 256 -> 256 at 14 x 14 264 -> 70;
1x1 2048 -> 512 at 7 x 7 209 -> 36; 1x1 1024 -> 256 at 14 x 14 104 -> 36; 3x3 128 -> 128 at 28 x 28 138 -> 67. Scaled error against double over the 32 accumulation-test rows is at most 2.7e-7 (2.1e-7 to 3.3e-7 before).

### Apple M2, float32, taps innermost (indicative, GPU timestamps, 30 calls per command buffer)

`ConvCoop.hlsli` numbers k as (input channel, kernel row, kernel column), so each staged element needs a division chain, two bounds tests and a tap update, and the float32 wide tile staged without
prefetch. Measured with a register-only kernel the M2's simdgroup matrices do 3.25 TFLOP/s in float32 (90 % of the ALU peak) and a plain 64 x 64 matrix product with the blocked sums above does 1.9 TFLOP/s,
so the convolutions' 0.4-0.8 TFLOP/s were staging cost, not matrix throughput. `ConvTap.hlsli` runs k in chunks of 16 input channels and, inside a chunk, over the taps of the window: a step of 16 is one tap of
16 channels, so the tap's row, column and padding test are the same for every element a thread stages, the channels' addresses differ by one input plane, and a thread's step is one test and eight independent
loads (plus eight weight loads, next step in flight). Bias, residual and activation are still applied as the result is stored. It is used for float32 in fast mode when the channels per group are a multiple of 16,
for every layer except pointwise ones with at least 64 (32 for the narrow tile) pixels, whose direct fragment loads measured 1.4-1.5x faster than staging. Deterministic mode and every other shape keep the previous kernel,
so deterministic results are unchanged bit for bit (the two-run hashes of the conv tests equal those of the previous build); the fast sums differ in order only, with the same block structure.

Convolution layers of ResNet-50 summed by class (ms; each shape counted as often as the model uses it; the stem, 7 x 7 with 3 input channels, is untouched):

| Class | b1 before | b1 now | b8 before | b8 now |
|---|---|---|---|---|
| 3x3 stride 1 | 5.08 | 4.50 | 45.2 | 27.2 |
| 3x3 stride 2 | 1.14 | 1.06 | 11.5 | 6.4 |
| 1x1 stride 2 (downsample) | 1.42 | 0.90 | 11.7 | 6.3 |
| 1x1 stride 1 (reduce, expand) | 4.36 | 4.57 | 32.3 | 27.9 |
| stem | 0.50 | 0.50 | 3.7 | 3.7 |
| **sum** | **12.4** | **11.5** | **104** | **71.5** |

Scaled error against float64 on the 23 ResNet-50 layers at batch 1 and 8, the MobileNetV3 and DBNet layers and the 35 edge shapes: at most 5.9e-7 (3.8e-7 for ResNet-50), the same as the previous kernel. Whole model through
onemodel's `.Gpu` (fast mode, 10 runs, same session, alternating builds): ResNet-50 float32 b1 14.55 -> 13.15 ms, b8 110.9 -> 78.1 ms, scaled error against onemodel's CPU exact mode 1.28e-6 (before: 1.3e-6).

What is left between this and a Core ML run of the same model on the same GPU (6.1 ms at b1, 35.3 ms at b8; its float32 output differs from ours by 1e-6 as ours does):
- Ours is 0.9 TFLOP/s over the convolutions at b8 (0.7 at b1), against 1.9 TFLOP/s for a plain matrix product, and a run at 35 ms means 1.9 TFLOP/s over the whole 65 GFLOP model: more than a direct convolution can
  reach on this GPU, so the reference must also save arithmetic (Winograd or similar) on the 3 x 3 layers (45 % of the model's flops).
- The weights of a 3 x 3 layer are read as 16 words at a stride of 9 per row and step; with the weights re-laid out so those reads are contiguous (a timing-only change, results wrong) the 3 x 3 layers ran 27 % faster.
  A real layout needs a per-model weight preparation pass, which this API does not have.
- Table-driven window decode with the weights in their own order (contiguous weight reads) cost more per staged element than it saved: b8 71.5 -> 89.9 ms.
- At batch 1 the layers are latency bound (12 to 140 workgroups, 4 to 150 steps each): moving the narrow-tile and split-k thresholds in either direction measured slower (best 11.8 against 12.6-14.0 ms).
- Pointwise layers with 64 or more pixels still use the direct-load kernel at 0.6-1.6 TFLOP/s; those with few input channels are bound by the epilogue (tile through workgroup memory, scalar stores).

## Pooling and resize

- `MaxPool` / `AveragePool`: padding, dilation, `ceil_mode` (a last window that starts inside the end padding is dropped, as in ONNX), `count_include_pad` (padding counts, taps
  beyond it do not). Max skips NaN taps unless all are NaN; an empty max window gives the lowest finite value of the type (-3.4028235e38; -65504 for half); an empty average window
  gives NaN. `MaxPool`'s indices output is not provided. `GlobalAveragePool` / `GlobalMaxPool` reduce a plane with a fixed halving tree.
- `Resize`: nearest (all four rounding modes) and linear along each axis (bilinear in two), coordinate transformations half_pixel, half_pixel_symmetric, pytorch_half_pixel,
  align_corners and asymmetric. The float32 coordinate arithmetic runs on the host once per shape into tap tables (`ResizePlan`, create once, encode many times) so every device gets
  the same taps. Not provided: cubic, antialias, `exclude_outside`, `tf_crop_and_resize`.
- `BatchNormalization` has no kernel: inference batch norm folds into the convolution weights in the graph, and the DBNet and reference exports are already folded.

## ConvInteger

`ConvIntegerKernels` is ONNX `ConvInteger`: `y = sum (x - xZero) * (w - wZero[m])` in int32 with wrap-around, for uint8 and int8 `x` and `w` in any mix, the geometry of `ConvProblem` (strides, dilations,
asymmetric padding, groups, 1-D as height 1). The zero point of `x` is one value and that of `w` one value or one per output channel, in device buffers of their operand's own type. A padded tap is
the real value zero, which is the quantized value `xZero`, so it adds nothing: the unfolding pass writes `xZero` into padded positions and the product subtracts it again, which is
how onemodel's CPU kernel treats padding.

It is lowered to `Int8Gemm` as the float convolution is to `Gemm` (Chellapilla et al., 2006): one pass unfolds the windows of each image and group into a byte matrix (`K = InChannels / Groups * KH * KW` rows, rows padded to
a multiple of four bytes so the product reads whole words), then the weights of the group multiply it. One group makes one product for the whole batch. Several groups make one product per image and group,
which is slow for depthwise convolutions with many channels (no direct 8-bit depthwise kernel yet). Sums are exact integers, so the packed-dot and portable products, every device and both `GpuKernelMode`
values give the same bits. Column scratch is the instance's own (up to 64 MiB, then the batch runs in chunks of images) or caller-placed (`GpuBuffer<TX>` and a byte offset that is a multiple of four,
`ConvIntegerKernels.ScratchBytesPerImage`). Tested bit for bit against a plain C# definition of the operator on the Whisper-small front end (80 to 768 channels over 3000 frames, kernel 3, and 768 to 768 with stride 2),
ResNet-50 and MobileNet shapes, dilation, asymmetric padding, groups, offsets, extreme values and all four type combinations, with the packed-dot product and with it ruled out.

## Tolerances (what the tests assert; scaled error as in [kernels.md](kernels.md))

| Operation | float32 | half in, float32 accumulation |
|---|---|---|
| Conv, ConvTranspose (fast and deterministic) | 1e-5 (measured <= 2.1e-6 over 23 ResNet-50, 45 MobileNetV3-large and 34 DBNet layer shapes, b1 and b8) | 2e-3 (measured <= 4.4e-4) |
| ConvInteger | bit-exact (int32) | |
| MaxPool, GlobalMaxPool | bit-exact | bit-exact |
| AveragePool, GlobalAveragePool | 1e-5 (measured 2.3e-7) | 2e-3 |
| Resize nearest | bit-exact | bit-exact |
| Resize linear | 1e-5 (measured 9.8e-8) | 2e-3 (measured 2.8e-4) |

## Determinism

No kernel here uses an atomic or a subgroup operation. Each output is summed in an order fixed by the shape: the matrix kernel's, the window order of the direct kernels, or, for split-k, shape-determined
ranges of k added in range order. On Apple GPUs `GpuKernelMode.Deterministic` runs the float32 simdgroup implicit kernel too (a simdgroup matrix product is a fixed function of its operands, and each
output tile is summed by one simdgroup in k order), which is 1.6-1.7x faster than the scalar tile it used before (ResNet-50 b1 27-29 ms to 16.4-17.1 ms, b8 145 to 90 ms, MobileNetV3 b1 5.2 to 3.8 ms); elsewhere it selects the portable
matrix kernel. Results are bit-identical run to run on one device and driver in that mode; `Fast` may use matrix hardware on other devices, whose summation order is the hardware's. The tests run two passes of seven layer shapes
(two split) per element type and compare hashes.

### Apple M2, float16 and the pointwise edge tile (indicative, GPU timestamps, 100 calls per command buffer)

The last pixel tile of a pointwise layer whose pixel count is not a multiple of 64 (14 x 14 = 196, 28 x 28 = 784) is moved back to end at the last pixel, so it is a whole tile and reads its operands directly. It recomputes pixels the previous tile also writes, from the same inputs in the same order, so the stores agree.
Float32 with the wide tile stages straight into the tiles (register staging cost occupancy there: ResNet-50 b8 127 ms against 110); the narrow tile keeps it.

| Convolution layers of | batch | float32 before | float32 now | float16 before | float16 now |
|---|---|---|---|---|---|
| ResNet-50 | 1 | 17.0 ms | 12.4 ms | 18.0 ms | 9.7 ms |
| ResNet-50 | 8 | 110 ms | 105 ms | 114 ms | 62 ms |

Where the float16 time goes at batch 1 (9.7 ms): 3x3 layers 4.8 ms at about 800 GFLOP/s, 1x1 stride-1 layers about 3.5 ms (600-1400), stride-2 1x1 projections about 1 ms, the stem 0.3 ms; the measured matrix peak of the M2 is about 1.7 TFLOP/s.
The 3x3 layers are bound by the gathering of the input tile (one scattered load per element); loading the weights straight from memory (no change) and compile-time 3 x 3 window arithmetic (9% on those layers) do not close it.
A kernel that reads padded input rows as matrix operands would need a padded copy of the input and a weight layout with the taps outermost.
