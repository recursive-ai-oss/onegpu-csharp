# Third-party notices

onegpu is licensed `MIT OR Apache-2.0` (see `LICENSE-MIT` and `LICENSE-APACHE`).

The shipped packages contain only this repository's code and the other `RecursiveAI.One*` packages they reference. Nothing else is
linked, copied or bundled.

## Specifications

`src/RecursiveAI.OneGpu.Vulkan/Interop/VulkanApi.g.cs` is generated from the Khronos Vulkan API registry (`vk.xml`), which is
Copyright 2015-2025 The Khronos Group Inc. and licensed under Apache-2.0. The generated declarations of types, enums and
function names follow it.

## Test-time tools

The tests and the build use these only as separate programs or build-time packages. None ships:

- The Khronos Vulkan validation layer, which checks Vulkan usage during test runs.
- Headless Chrome, which runs the WebGPU tests.
