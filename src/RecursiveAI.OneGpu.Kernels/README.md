# RecursiveAI.OneGpu.Kernels

HLSL compute kernels for machine-learning inference on Vulkan, Metal and WebGPU: GEMM, attention, convolution and more.

Compute kernels written in HLSL and compiled at build time by `RecursiveAI.OneShader.Build` for Vulkan (SPIR-V), Metal and WebGPU. A
family class (`Gemm`, `AttentionKernels`, `ConvKernels`, `ReduceKernels`, `SoftmaxKernels`, `NormKernels`, `ElementwiseKernels`,
`GatherKernels`, `RecurrentKernels`, `Int8Gemm` and others) is created once from a `GpuDevice` and records passes into your command encoder. It
chooses its variant from the device's capabilities (cooperative matrices, subgroup sizes, float16) and keeps a portable fallback.
`GpuKernelMode.Deterministic` gives bit-repeatable GEMM, reductions, softmax and attention on one GPU and driver.

Shapes, strides and offsets are checked in safe code first; a bad one throws `GpuKernelShapeException` naming the operation and argument.

```csharp
using var gemm = new Gemm(device, GemmElementType.F32);
gemm.Encode(encoder, new GemmProblem(m, n, k), a, b, c);
```

Operations, tolerances and determinism rules per family are in the repository's `docs/`.

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
