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
| every other convolution on the portable matrix kernel (float32 everywhere; half on devices without matrix hardware) | **implicit GEMM**: a 64x64x16 tile kernel that gathers the input window while it loads the tile; float32 fuses bias, residual and activation into the store, half writes float32 accumulators and an epilogue pass finishes |
| float32 on Metal (simdgroup matrices) | **implicit GEMM on matrix hardware** (`ConvCoopF32.hlsl`): the 4-subgroup 64x64x16 simdgroup tile of `Gemm`, the window gathered while the input tile is staged (no column matrix), the bias, residual and activation applied as the result is stored, so the output is written once. A pointwise convolution (1x1, stride 1, no padding) whose tile lies inside the output reads weights and input straight from memory with no staging. Grids under 40 workgroups use a 64x32 tile. Fast mode only |
| 1x1, stride 1, no padding, with matrix hardware (half) | `Gemm` on the input as it lies (no unfolding) |
| any other shape with matrix hardware (half) | im2col (Chellapilla et al., 2006) into the column scratch, then `Gemm` (cooperative or simdgroup matrices) and the epilogue pass; groups loop |
| transposed convolution | `Gemm` of the transposed weights with the input, then a gathering col2im (one thread per output element, no atomics) with the same epilogue |

Variants are chosen once per instance from `GpuCapabilities.Compute`; a device without cooperative matrices (lavapipe, a forced-off run) runs the portable kernel, and the tests
run every layer set in four configurations (fast, fast with matrix hardware off, deterministic, fast with the implicit kernel off).

### Not built, with the numbers

- **Cooperative-matrix implicit GEMM** (window gather in the matrix tile load). On the RTX 4070, f16 ResNet-50 at b8 the im2col+`Gemm` layers already run at 6.6-13 TFLOP/s
  for 3x3 and 11-21 TFLOP/s for the 1x1 layers that use `Gemm` directly; the unfold pass is a bandwidth-bound write and read of at most a few tens of megabytes per layer. The
  ceiling for the change is the gap between the 3x3 and the 1x1 rate, about 1.3-1.6x on 3x3 layers, which are 2.4 of about 5.8 ms. Dropped as not worth a second coop kernel family now.
- **Winograd F(2x2,3x3)** (Lavin and Gray, 2016). `WinogradBound` in the tests times the sixteen batched products it needs with the real `Gemm` plus a copy moving the bytes its
  transforms must move: a floor no implementation can beat. RTX 4070, ResNet-50 stride-1 3x3 layers at b8 summed (host clock, indicative): **float32 direct 6.3 ms, floor 2.7 ms;
  half direct 2.4 ms, floor 1.0 ms.** The floor leaves room (the transforms run as three irregular passes, realistically 1.5x the copy floor, and half Winograd carries
  larger rounding error through the transformed data than the 2e-3 half tolerance leaves comfortably), but it is a speed-up for one model's 3x3 layers only, MobileNetV3 and DBNet
  have almost none. Left for the speed milestone if ResNet-50 misses the bar; the variant would be a fourth `ConvVariant` behind the same `Encode`.

### Apple M2, float32 (indicative, host clock, 10 calls per command buffer; Metal has no timestamp queries)

Convolution layers summed over the model's layer list, before and after the matrix-hardware implicit kernel (and the direct-load 4-subgroup `Gemm` it shares):

| Model | batch | im2col + `Gemm` + epilogue pass | implicit simdgroup kernel |
|---|---|---|---|
| ResNet-50 | 1 | 26.0 ms | 24.4 ms |
| ResNet-50 | 8 | 117.8 ms | 89.9 ms |
| MobileNetV3-large | 1 | 5.8 ms | 4.3 ms |
| DBNet 512 | 1 | 9.9 ms | 7.2 ms |

Float16 and the deterministic mode are unchanged by this kernel. The late layers of a classifier at batch 1 (14 x 14 and 7 x 7 images, 4608 deep) run at 130-300 GFLOP/s: the grid has too few
workgroups and each runs its 16-wide k steps one after another; split-k (a second pass or atomics) or folding the batch into the pixel axis would help and are not built. A version that
prefetched the next tile into registers ran 40 % slower (register pressure), and wider narrow-tile thresholds ran slower at batch 8.

## Pooling and resize

- `MaxPool` / `AveragePool`: padding, dilation, `ceil_mode` (a last window that starts inside the end padding is dropped, as in ONNX), `count_include_pad` (padding counts, taps
  beyond it do not). Max skips NaN taps unless all are NaN; an empty max window gives the lowest finite value of the type (-3.4028235e38; -65504 for half); an empty average window
  gives NaN. `MaxPool`'s indices output is not provided. `GlobalAveragePool` / `GlobalMaxPool` reduce a plane with a fixed halving tree.
- `Resize`: nearest (all four rounding modes) and linear along each axis (bilinear in two), coordinate transformations half_pixel, half_pixel_symmetric, pytorch_half_pixel,
  align_corners and asymmetric. The float32 coordinate arithmetic runs on the host once per shape into tap tables (`ResizePlan`, create once, encode many times) so every device gets
  the same taps. Not provided: cubic, antialias, `exclude_outside`, `tf_crop_and_resize`.
- `BatchNormalization` has no kernel: inference batch norm folds into the convolution weights in the graph, and the DBNet and reference exports are already folded.

## Tolerances (what the tests assert; scaled error as in [kernels.md](kernels.md))

| Operation | float32 | half in, float32 accumulation |
|---|---|---|
| Conv, ConvTranspose (fast and deterministic) | 1e-5 (measured <= 2.1e-6 over 23 ResNet-50, 45 MobileNetV3-large and 34 DBNet layer shapes, b1 and b8) | 2e-3 (measured <= 4.4e-4) |
| MaxPool, GlobalMaxPool | bit-exact | bit-exact |
| AveragePool, GlobalAveragePool | 1e-5 (measured 2.3e-7) | 2e-3 |
| Resize nearest | bit-exact | bit-exact |
| Resize linear | 1e-5 (measured 9.8e-8) | 2e-3 (measured 2.8e-4) |

## Determinism

No kernel here uses an atomic, a subgroup operation or a split of the accumulation: each output is summed in an order fixed by the shape (the matrix kernel's, or the window order of
the direct kernels). `GpuKernelMode.Deterministic` selects the portable matrix kernel, so results are bit-identical run to run on one device and driver; fast mode may use matrix
hardware, whose summation order is the hardware's. The tests run two passes of five layer shapes per element type and compare hashes.
