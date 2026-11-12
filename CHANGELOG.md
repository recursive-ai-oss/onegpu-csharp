# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-12

First stable release. Everything below is new in 1.0; the public API of every package is recorded in its `PublicAPI.Shipped.txt`, and
names (package ids, namespaces, `ONEGPU_*` variables, the `onegpu` JavaScript module) are listed in [`docs/environment.md`](docs/environment.md).

### Device layer

- `RecursiveAI.OneGpu`: `GpuDevice`, adapters and `GpuCapabilities` as data (subgroup sizes and operations, float16, bfloat16, int8 and
  integer dot product, cooperative-matrix shapes, device limits), typed `GpuBuffer<T>`, uploads, non-blocking `ReadAsync` readback,
  compute pipelines checked against the shader's reflection, command encoders with dispatch, indirect dispatch, pass timestamps and
  `ComputePassCount` / `DispatchCount`, memory statistics, typed errors that name the capability, what was expected and what the device has.
  Surfaces, textures, graphics pipelines and render passes for renderers.
- Backends: Vulkan 1.3 (own memory allocator, timeline-semaphore readback, validation routing per device), Metal on Apple silicon (typed
  `objc_msgSend`, shared compute encoders), WebGPU in the browser (`[JSImport]`, several devices per page). `GpuBackends` picks one;
  `ONEGPU_BACKEND` overrides.
- No native binaries of ours. The core and the kernels have no interop. All packages are trimmable and Native AOT compatible.

### Kernels

- GEMM in float32, float16, bfloat16 (cooperative-matrix variants on Vulkan and Metal where the device has them) and int8, with fused
  bias and activation epilogues; fused attention with masks, causal rule, grouped heads and Apple simdgroup variants; Conv, ConvTranspose,
  ConvInteger, pooling and resize; RNN, GRU and LSTM; broadcasting element-wise, integer and bool operations; reductions, softmax, layer and
  RMS normalisation; gather, scatter, slice, concat, expand and transpose; quantize, dequantize and dynamic quantize.
- Float32 sums accumulate in blocks, so their error against a double-precision sum is roughly ten times lower than a single chain's.
- `GpuKernelMode.Deterministic` for GEMM, reductions, softmax and attention: a fixed order, no float atomics, bit-identical runs on one
  GPU and driver. Tolerances per operation and dtype are in [`docs/kernels.md`](docs/kernels.md).
- Shaders are HLSL compiled at build time for SPIR-V, Metal Shading Language and WGSL, so a layout mismatch is a build error.

### Tooling

- Tests run with the Khronos validation layer required and clean on Vulkan, with API and shader validation fatal on Metal, and in headless
  Chrome (SwiftShader and a hardware adapter) on WebGPU. `tools/ci` holds the scripts.
- `PublicAPI.Shipped.txt` per package makes any public change a build error until the file changes in the same diff.

[1.0.0]: https://github.com/recursive-ai-oss/onegpu-csharp/releases/tag/v1.0.0
