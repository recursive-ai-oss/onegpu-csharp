# onegpu

GPU compute and rendering for .NET from pure C#, on Vulkan, Metal and browser WebGPU. There are no native binaries of ours and no CUDA
install: the backends talk to the platform's own GPU API through the driver already on the machine (the Vulkan loader, the Objective-C
runtime and Metal framework, the browser's WebGPU).

It has two jobs. It is a small, compute-first device layer (typed buffers, async readback, compute pipelines, command encoders, timestamps,
capabilities as data) with the surfaces, textures and graphics pipelines a renderer needs. And it ships the compute kernels that
machine-learning inference needs, written in HLSL and compiled at build time for all three backends.

onemodel's `RecursiveAI.OneModel.Gpu` runs ONNX models on these kernels, and onegame, the family's game engine, takes its GPU layer from
these packages (its render graph and 2D/3D rendering stay in onegame).

| Package | What it is |
|---|---|
| `RecursiveAI.OneGpu` | Devices, adapters and capabilities, typed buffers, uploads and async readback, compute pipelines, command encoders, dispatch, timestamps, memory statistics, plus surfaces, textures and graphics pipelines. BCL only; no interop. |
| `RecursiveAI.OneGpu.Vulkan` | Vulkan 1.3 backend (Windows, Linux). |
| `RecursiveAI.OneGpu.Metal` | Metal backend (macOS, Apple silicon). |
| `RecursiveAI.OneGpu.WebGpu` | WebGPU backend for the browser (`[JSImport]`, publishes `onegpu.js`). |
| `RecursiveAI.OneGpu.Backends` | `GpuBackends`: picks Metal on Apple platforms, WebGPU in a browser and Vulkan elsewhere. Reference it to get all three. |
| `RecursiveAI.OneGpu.Kernels` | HLSL compute kernels with generated bindings: GEMM, attention, convolution, reductions, softmax, norms, gather and scatter, int8, recurrent layers, element-wise, integer and bool ops. |

## Platforms and requirements

- **.NET 10.** Every package is trimmable and Native AOT compatible.
- **Linux and Windows:** a Vulkan 1.3 driver (NVIDIA, AMD, Intel; the Mesa software driver works too). Tested on NVIDIA and AMD hardware and on
  llvmpipe; Intel is not tested. The loader library (`libvulkan.so.1`, `vulkan-1.dll`) must be installed.
- **macOS:** Apple silicon with Metal. Tested on an M2.
- **Browser:** Chrome with WebGPU. `shader-f16` and `subgroups` are used when the adapter offers them; a kernel that needs a missing feature
  falls back to a portable variant or reports it by name.
- No OpenGL, no CUDA, no mobile.

## Install

```
dotnet add package RecursiveAI.OneGpu.Backends
dotnet add package RecursiveAI.OneGpu.Kernels
```

`Backends` brings the three backends; reference `RecursiveAI.OneGpu` and one backend alone to be smaller.

## A compute example

```csharp
using RecursiveAI.OneGpu;
using RecursiveAI.OneGpu.Kernels;

using var device = GpuBackends.CreateDevice();            // Metal on Apple silicon, Vulkan elsewhere
Console.WriteLine($"{device.Adapter.Name}: {device.Capabilities.Compute.Describe()}");

const int M = 4, N = 3, K = 2;
float[] a = [1, 2, 3, 4, 5, 6, 7, 8];     // 4 x 2, row-major
float[] b = [1, 0, 2, 0, 1, 3];           // 2 x 3, row-major
using var bufferA = device.CreateBuffer<float>(a);
using var bufferB = device.CreateBuffer<float>(b);
using var bufferC = device.CreateBuffer<float>(M * N);

using var gemm = new Gemm(device, GemmElementType.F32);   // GpuKernelMode.Deterministic for bit-repeatable sums
var encoder = device.BeginCommands();
gemm.Encode(encoder, new GemmProblem(M, N, K), bufferA, bufferB, bufferC);
device.Submit(encoder);

var c = new float[M * N];
await bufferC.ReadAsync(c);                               // completes when the GPU is done, without blocking a thread
```

A family such as `Gemm` is created once and keeps its pipelines; `Encode` records one or more compute passes into the encoder you pass and
allocates nothing once its bind groups are cached. Your own shaders work the same way: compile an HLSL file with
`RecursiveAI.OneShader.Build` and pass its generated entry point to `GpuDevice.CreateComputeKernel`. [`docs/compute.md`](docs/compute.md) has
the details: capabilities, timestamps, indirect dispatch, memory statistics.

Bad input is checked in safe code before anything reaches the driver: a wrong shape, stride, size or offset throws
`GpuKernelShapeException` naming the operation and the argument, and a missing capability or exceeded limit throws `GpuUnsupportedException`
with `Capability`, `Expected` and `Actual`.

## Kernel families

| Family | Operations |
|---|---|
| `Gemm`, `Int8Gemm` | batched matrix products in float32, float16 and bfloat16 (cooperative-matrix variants where the device has them), uint8/int8 with int32 sums |
| `AttentionKernels` | fused scaled dot-product attention with masks, causal rule and grouped heads |
| `ConvKernels`, `ConvIntegerKernels`, `PoolKernels`, `ResizeKernels` | Conv, ConvTranspose, ConvInteger, pooling, resize |
| `RecurrentKernels` | RNN, GRU, LSTM |
| `ElementwiseKernels`, `IntegerKernels`, `ReduceKernels`, `SoftmaxKernels`, `NormKernels` | broadcasting binary and unary ops, integer and bool ops, reductions, softmax, layer and RMS normalisation |
| `GatherKernels`, `ScatterKernels`, `LayoutKernels`, `TransposeKernels` | gather and scatter, slice, concat, expand, transpose |
| `DequantizeKernels`, `DynamicQuantizeKernels` | QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear |

[`docs/kernels.md`](docs/kernels.md) lists every operation; the other pages in [`docs/`](docs/) cover GEMM, attention, convolution and
recurrent layers in detail.

## Determinism and tolerances

- **Fast is the default.** A fast kernel may sum in any order and use fused multiply-add.
- **`GpuKernelMode.Deterministic`** exists for GEMM, reductions, softmax and attention: a fixed reduction and accumulation order, no float
  atomics, nothing that depends on occupancy or on how the driver schedules subgroups. Two runs on the same GPU and driver give identical
  bits. Results are not bit-identical across vendors or drivers. Other families give the same bits in both modes.
- **Accuracy.** The tests compare every family with a plain C# reference in double precision. Integer, bool, copy and cast operations,
  float32 add, sub, mul, min and max, and int8 GEMM are bit-exact. Float32 GEMM, reductions, softmax, norms and transcendentals are within
  1e-5 of the reference, scaled by the largest output value (measured values are far inside that); float16 inputs with float32 accumulation within
  2e-3; bfloat16 within 8e-3. The limit per operation and dtype is in [`docs/kernels.md`](docs/kernels.md).
- **Capabilities are queried, never assumed.** Each capability-gated variant has a portable fallback that the tests also run.

## Browser

Reference `RecursiveAI.OneGpu.Backends` (or `.WebGpu`) from a .NET WebAssembly browser app. The package publishes `onegpu.js` beside the
app. Create the device with `await GpuBackends.CreateDeviceAsync()` and read results with `ReadAsync`, because the browser cannot block.
The browser records each submission as one byte stream and runs it in a single call into the script. Float16 and bfloat16 data work
without the `shader-f16` feature (the kernels read them as packed 32-bit words). The kernel families' bind-group caches (`maxCachedBindGroups`)
must cover everything one command buffer binds, because the browser keeps those groups alive until the submission runs. Fused attention
has browser-specific variants; see [`docs/attention.md`](docs/attention.md).

Environment variables, the JavaScript module's surface and how the public API is frozen are in [`docs/environment.md`](docs/environment.md).

## Build and test

```
dotnet build OneGpu.slnx
tools/ci/test-vulkan.sh      # Vulkan, validation on
tools/ci/test-metal.sh       # macOS
tools/ci/test-web.sh         # headless Chrome on WebGPU
tools/ci/test-web-package.sh # a browser app consuming the packed WebGPU backend, in headless Chrome
tools/ci/aot-smoke.sh        # Native AOT publish of a small program, run on a device
```

Output goes to `artifacts/` at the repository root, or to `$ONEGPU_ARTIFACTS` when set. Device tests skip, and print why, when no
adapter is present. On Vulkan, run with the Khronos validation layer (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, or
`ONEGPU_VALIDATION=1`); a run that prints validation messages fails. On Metal the scripts turn on API and shader validation.

## Licence

`MIT OR Apache-2.0`, © Recursive AI, LLC. See [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
