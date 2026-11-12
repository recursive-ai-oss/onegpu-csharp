# RecursiveAI.OneGpu.Vulkan

Vulkan 1.3 backend for onegpu on Windows and Linux, with no native binaries.

The Vulkan backend of onegpu. It loads the system's Vulkan loader (`libvulkan.so.1` or `vulkan-1.dll`) at run time and has its own memory
allocator, synchronisation tracking and validation-message routing. Needs a Vulkan 1.3 driver. `VulkanGpu.CreateDevice()` creates a
device directly; `RecursiveAI.OneGpu.Backends` chooses a backend for the platform.

Set `ONEGPU_VALIDATION=1` to turn on the Khronos validation layer when it is installed.

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
