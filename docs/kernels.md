# Kernels

`RecursiveAI.OneGpu.Kernels` holds the compute kernels that machine-learning inference needs, written in HLSL and compiled by the shader
build for SPIR-V (Vulkan), Metal Shading Language and WGSL (browser). This page describes the element-wise, reduction, softmax, normalisation
and layout families; each is a public class created from a `GpuDevice`.

## Using a family

```csharp
using var elementwise = new ElementwiseKernels(device);
var encoder = device.BeginCommands();
elementwise.EncodeBinary(encoder, GpuBinaryOp.Add, x, 0, [rows, cols], bias, 0, [cols], y, 0, new GpuActivation(GpuUnaryOp.GeluErf));
device.Submit(encoder);
```

- A family is created once; pipelines are built the first time a variant is used and kept. Calling `Encode…` records one compute pass into the
  caller's encoder. With the same buffers, a call allocates nothing (bind groups are cached by buffer identity; offsets and sizes travel in push constants).
- Operands are a `GpuBuffer<T>`, an element offset and a shape. Tensors are dense and row-major. `T` is `float` or `Half` for arithmetic; `int`, `uint`,
  `long` where an operation only moves or converts bits. Comparison results and conditions are bool tensors (see below).
- Every size, shape, axis, permutation and offset is checked in safe code before anything is recorded. A bad argument throws
  `GpuKernelShapeException`, which names the operation and the argument or dimension (`Reduce: axes: axis 3 is outside [-3, 3)`).
- Families are not thread-safe. Disposing a family releases its pipelines once the GPU has finished with them.
- `GpuKernelMode.Fast` is the default. `GpuKernelMode.Deterministic` is accepted by every family; for the element-wise and layout families there is no reduction order, so
  both modes give the same bits.

## Families

| Family | Operations |
|---|---|
| `ElementwiseKernels` | binary Add, Sub, Mul, Div, Pow, Min, Max with ONNX broadcasting (up to 6 dimensions after merging, inputs up to rank 8) and an optional fused unary activation; unary Identity, Neg, Abs, Sqrt, Reciprocal, Exp, Log, Tanh, Erf, Sigmoid, Relu, Gelu (erf and tanh forms), Silu, HardSigmoid, HardSwish, Clip, LeakyRelu; Equal, Less, Greater, LessOrEqual, GreaterOrEqual over float and Half; Where over float, Half, int, uint and long; Cast between float, Half, int, long and bool |
| `ReduceKernels` | Sum, Mean, Max, Min, Prod, L1, L2, SumSquare, LogSum over any set of axes; ArgMax and ArgMin (64-bit result) |
| `SoftmaxKernels` | Softmax and LogSoftmax along any axis |
| `NormKernels` | layer normalisation and RMS normalisation over the last axis, optional scale and bias |
| `IntegerKernels` | int32 and int64 Add, Sub, Mul, Div, Mod, Fmod, Min, Max, Neg, Abs; Equal, Less, Greater, LessOrEqual, GreaterOrEqual over int32, int64 and bool; And, Or, Xor, Not on bool; Fill (`ConstantOfShape`) and Range |
| `TransposeKernels` | N-d permutation, NCHW↔NHWC, split and merge of attention heads |

Adding an operation to the element-wise family is a case in `Shaders/Functions.hlsl` and an enum member: the operation is a push constant, not a pipeline.

## Integer and bool tensors

- **int32** is one 32-bit word per element. **int64** is two words (low, then high) handled with 32-bit arithmetic, so a device needs no 64-bit integer support;
  every value from `long.MinValue` to `long.MaxValue` is exact, and so is division (a shift-and-subtract loop runs only when an operand does not fit 32 bits).
- **bool** is a `GpuBuffer<byte>`: one byte per element, 0 or 1, four to a 32-bit word. It is the same layout as a CPU `bool[]`, so a model's bool tensor uploads as it is.
  Allocate `GpuKernelBuffers.PaddedBoolLength(n)` elements; a buffer that ends inside a word throws `GpuKernelShapeException`. A thread owns whole output words, so a kernel
  that writes a bool range at an odd offset keeps the neighbouring bytes of the words it shares, and no atomics are needed. Inputs treat any nonzero byte as true; outputs are 0 or 1.
- Arithmetic wraps (two's complement). `Div` truncates toward zero, is 0 for a zero divisor, and the minimum value divided by -1 is itself. `Mod` takes the divisor's sign, `Fmod` the dividend's; both give 0 for a divisor of 0 or -1.
  These are onemodel's CPU semantics, which the tests check bit for bit, as they do every operation of the family: integer, bool and cast results have no tolerance.
- `Range` takes its start and delta as host values (a graph's `Range` inputs are known when a model's shapes are planned) and its length from `IntegerKernels.RangeLength`, which follows the ONNX formula in
  exact integer arithmetic. The float range is `start + i * delta` as two separately rounded float32 operations.
- `Cast` to bool is "not equal to zero" (NaN counts, negative zero does not); from bool it gives 0 or 1.
- One variant serves `Fast` and `Deterministic`: integer arithmetic has no order to fix and no atomics are used.
- An int64 `Where` moves two words per element and so needs one more dimension than the other types: its shapes may have one fewer dimension after merging.

## Binding caches

A family builds a bind group the first time it sees a set of buffers and reuses it on every later call with the same set, so a warm call allocates nothing. Every family constructor takes
`maxCachedBindGroups` (default `GpuKernelBindings.DefaultMaxCachedBindGroups`, 1024). When a family has seen more sets than that it drops all of them and starts again, so the bound should exceed the number of
distinct buffer sets one run records: a 72-weight transformer touches 72 sets in `Gemm` per pass, and with a bound of 64 every pass rebuilt them all (72, then 144 bind groups over two passes) where a bound of 72 or more builds
72 once. `ReleaseBindings()` on every family (and `Gemm`) drops the cached groups immediately; call it after disposing buffers that will not be used again, or after replacing a model's buffers, so stale entries do not keep them reachable.
The pipelines stay, and the next call with a set of buffers builds its group again.

## Half precision

Half tensors are stored as pairs in 32-bit words and converted in the shader with integer arithmetic, so they need no 16-bit storage feature and run on
every backend, including a browser without `shader-f16`. Arithmetic is in single precision and the result is rounded to nearest even once; the rounding
is written out in the shader because the drivers' own float-to-half conversion does not promise a rounding mode.

- A buffer the kernels touch must end on a word boundary: allocate `GpuKernelBuffers.PaddedHalfLength(n)` halves. Otherwise `GpuKernelShapeException`.
- An output may overlap an input only with the same offset and extent.
- Softmax and normalisation write whole rows as words, so half tensors need the softmax axis last, an even row length and even element offsets.
- Index arithmetic avoids integer division where it can (same-shape and bias-style broadcasts, a single run of reduced dimensions): on Apple GPUs a chain of divisions per element made binary add and reductions two to three times slower, so those cases are compiled as separate variants.
- Reduction outputs are written with two integer atomic AND/OR operations on disjoint bits, so neighbouring outputs of different workgroups never clash. There are no floating-point atomics anywhere.

## Tolerances

The error measure is max|gpu − reference| / max|reference| over an output, with the reference computed on the CPU in double precision (inputs first rounded to half for half rows).
The limits below are what the tests assert; the measured values are far inside them (see the parity table the tests print).

| Operation | f32 | f16 (f32 accumulation) |
|---|---|---|
| Add, Sub, Mul, Min, Max, Where, compare, Cast, Transpose, every `IntegerKernels` operation, Fill, Range (float: bit-exact against `start + i * delta`), Identity, Neg, Abs, Relu, ArgMax/ArgMin | bit-exact (Cast and Transpose also for int and long) | bit-exact vs. `(Half)((float)a op (float)b)` |
| Div | 1e-5 (measured 3e-8) | 2e-3 |
| Pow, Exp, Log, Tanh, Erf, Sigmoid, Gelu, Silu, Hard*, LeakyRelu, Clip, Sqrt, Reciprocal | 1e-5 | 2e-3 |
| Sum, Mean, L1, L2, SumSquare, LogSum | 1e-5 | 2e-3 |
| Max, Min | exact | exact |
| Prod | max(1e-5, 1e-6·√n) for n factors: every multiplication rounds | 2e-3 |
| Softmax, LogSoftmax | 1e-5 | 2e-3 |
| Layer and RMS normalisation | 1e-5 | 2e-3 |

Erf is the Abramowitz–Stegun 7.1.26 approximation (absolute error below 1.5e-7); the Gelu erf form uses its complementary function so the negative tail keeps its relative accuracy.
A float sum of n values has an error of about ε·√n·(Σ|x|/|Σx|) in the worst conditioning: a sum that cancels to almost nothing cannot be accurate in relative terms in any
float implementation. The tests use mostly positive data for sums.

## Fast and deterministic

The reductions, softmax and normalisation have both modes.

- **Fast**: each output (a reduction element, a softmax row, a norm row) is computed by one 256-thread workgroup. Every thread folds a strided slice in index order (reductions add short runs of 32 first, which keeps the rounding error near that of a pairwise sum), and the 256 partial results are combined with the device's subgroup arithmetic when `GpuCapabilities.Compute` reports basic, vote and arithmetic subgroup operations and no subgroup is smaller than 4 lanes; otherwise with the shared-memory tree. `Variant` on each family says which. Constructing a family with `allowSubgroups: false` forces the shared-memory path.
- **Deterministic**: the same slices, then a fixed halving tree in shared memory (thread *i* combines with *i + s* for *s* = 128, 64, … 1). No subgroup operation, no floating-point atomic, no dependence on occupancy, scheduling or subgroup size. Two runs on the same device and driver give identical bits, and because nothing depends on the subgroup size the result is also the same for the fast variant without subgroups.
- A reduction with fewer than 32 elements per output, or between 32 and 255 along a non-contiguous axis with thousands of outputs, and a softmax row shorter than 16, is done by one thread per output in index order; that path is deterministic by construction and used by both modes.
- A softmax row of 16 to 1024 elements gets a power-of-two group of threads (about four elements each), so one workgroup handles 256 / group rows. The statistics are merged by the fixed halving tree in both modes.
- A reduction with fewer than 128 outputs and at least 16384 reduced elements (a sum over a whole tensor) runs in two passes: the reduced elements are cut into at most 256 contiguous chunks of at least 4096, a workgroup reduces each chunk to a partial result in a scratch buffer of 128 x 256 floats (`ReduceKernels.ScratchLength`; the family keeps its own unless the caller passes one), and one workgroup per output combines the partials in index order. The cut is a function of the shape, so the deterministic mode stays bit-identical run to run.
- Otherwise a workgroup takes each output. The choice depends on the shape only, never on the device.

## Special values

NaN and infinity follow IEEE 754 on Vulkan and Metal: Min, Max and the reductions Max/Min propagate NaN; ArgMax/ArgMin treat NaN as the extreme value (first NaN wins); softmax of a row with a NaN, a +infinity, or only −infinity gives NaN for the row, while −infinity beside finite values gives 0 (so additive masks work). The kernels test NaN with bit patterns, because Metal compiles with fast-math.
**In a browser, WGSL does not require NaN or infinity to behave** (implementations may assume they never occur), so those cases are not tested on WebGPU and masks should use a large negative finite value there.

## Limits

Element counts are below 2³¹ and every index is 32-bit. Broadcasting merges neighbouring dimensions that every operand walks contiguously, then needs at most 6; a reduction needs at most 4 runs of kept and 4 runs of reduced dimensions after merging; a transpose at most 6 unmergeable dimensions. Workgroups are 256 threads, which every Vulkan, Metal and WebGPU device offers; dispatches are two-dimensional so more than 65535 workgroups work everywhere.
