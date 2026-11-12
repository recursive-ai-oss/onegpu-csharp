# RecursiveAI.OneGpu.Backends

Picks the onegpu backend for the platform: Metal on Apple, WebGPU in a browser, Vulkan elsewhere.

`GpuBackends` creates a `GpuDevice` on the backend that fits the platform and brings all three backend packages with it. Set
`ONEGPU_BACKEND` to `vulkan` or `metal` to override, and `ONEGPU_ADAPTER` to pick an adapter by name or kind.

```csharp
using var device = GpuBackends.CreateDevice();
```

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
