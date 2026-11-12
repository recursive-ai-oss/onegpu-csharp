# Third-party notices

onegpu is licensed `MIT OR Apache-2.0` (see `LICENSE-MIT` and `LICENSE-APACHE`).

The shipped packages contain only this repository's code and the other `RecursiveAI.One*` packages they reference. Nothing else is
linked, copied or bundled.

## Specifications

`src/RecursiveAI.OneGpu.Vulkan/Interop/VulkanApi.g.cs` is generated from the Khronos Vulkan API registry (`vk.xml`), which is
Copyright 2015-2025 The Khronos Group Inc. and licensed under Apache-2.0. The generated declarations of types, enums and
function names follow it. `tools/RecursiveAI.OneGpu.VkBindGen` produces it from the registry release named in `tools/fetch-vkxml.sh`
(a pinned tag, checked by hash); `tools/gen-vulkan-bindings.sh` runs it.

## Test-time tools

The tests and the build use these only as separate programs or build-time packages. None ships:

- The Khronos Vulkan validation layer, which checks Vulkan usage during test runs (Apache-2.0).
- Headless Chrome, which runs the WebGPU tests, and Node.js, which runs the small script that drives it (`tools/web`, Node's own modules only).
- xUnit v3 (`xunit.v3`, Apache-2.0), the test framework.
- `Microsoft.CodeAnalysis.PublicApiAnalyzers` (MIT), a build-time analyzer that checks the public API against `PublicAPI.Shipped.txt`.
- `RecursiveAI.OneShader.Build`, the family's shader compiler, at build time. The runtime packages also reference `RecursiveAI.OneShader`, its
  small support library of resource-binding types.

## Test data and models

`testdata/gpu/*.png` are reference images that this repository's own render tests produced (`ONEGPU_UPDATE_GOLDENS=1` rewrites them);
they contain no third-party material. No model files are stored here. The kernel tests compare against plain C# reference code written
for the tests, with reference values computed in the test run itself; no other implementation's output or data is bundled.
