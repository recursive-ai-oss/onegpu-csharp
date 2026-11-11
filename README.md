# onegpu

GPU compute and rendering for .NET from pure C#, on Vulkan, Metal and browser WebGPU. No native binaries of ours and no CUDA install: the
backends talk to the platform's own GPU API through the installed driver (the Vulkan loader, the Objective-C runtime and the Metal
framework, the browser's WebGPU).

| Package | What it is |
|---|---|
| `RecursiveAI.OneGpu` | Devices, adapters and capabilities, typed buffers, uploads and async readback, compute pipelines, command encoders, dispatch, timestamps, memory statistics, plus surfaces, textures and graphics pipelines. BCL only. |
| `RecursiveAI.OneGpu.Vulkan` | Vulkan backend (Windows, Linux). |
| `RecursiveAI.OneGpu.Metal` | Metal backend (macOS, Apple silicon). |
| `RecursiveAI.OneGpu.WebGpu` | WebGPU backend for the browser (`[JSImport]`). |
| `RecursiveAI.OneGpu.Backends` | `GpuBackends`: picks Metal on Apple platforms, WebGPU in a browser and Vulkan elsewhere. |
| `RecursiveAI.OneGpu.Kernels` | HLSL compute kernels (GEMM, attention, reductions, softmax, norms, gather and scatter, int8, integer and bool ops) built with oneshader, with generated bindings. |

Shaders are HLSL, compiled at build time by [`RecursiveAI.OneShader.Build`](https://github.com/recursive-ai-oss/oneshader-csharp) to SPIR-V,
Metal Shading Language and WGSL; the generated C# binding carries each kernel's resource layout, so a mismatch between the shader and
the host fails at build or pipeline creation.

Docs: [`docs/`](docs/) (compute, kernels, GEMM, attention).

## Build and test

```
dotnet build OneGpu.slnx
tools/ci/test-vulkan.sh      # Vulkan, validation on
tools/ci/test-metal.sh       # macOS
tools/ci/test-web.sh         # headless Chrome on WebGPU
tools/ci/test-web-package.sh # a browser app consuming the packed WebGPU backend, in headless Chrome
```

Output goes to `artifacts/` at the repository root, or to `$ONEGPU_ARTIFACTS` when set. Device tests skip, and print why, when no
adapter is present. For Vulkan, run with the Khronos validation layer (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, or
`ONEGPU_VALIDATION=1`); a run that prints validation messages fails.

Environment variables: `ONEGPU_BACKEND` (`vulkan`, `metal`, `webgpu`) overrides backend selection, `ONEGPU_ADAPTER` picks an adapter
by name or kind, `ONEGPU_VALIDATION=1` turns validation on.

## Licence

`MIT OR Apache-2.0`, © Recursive AI, LLC.
