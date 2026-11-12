# RecursiveAI.OneGpu

Devices, buffers, compute pipelines and command encoders for .NET, on any backend.

The backend-independent core of onegpu: `GpuDevice`, adapters and `GpuCapabilities` (subgroup sizes, float16, int8 dot product,
cooperative-matrix shapes, limits), typed `GpuBuffer<T>` over blittable types, uploads and non-blocking readback, compute pipelines built
from shaders compiled by `RecursiveAI.OneShader.Build`, command encoders with dispatch, indirect dispatch, pass timestamps and pass counts,
memory statistics, and the surfaces, textures and graphics pipelines a renderer needs. BCL only; it contains no interop. Add a backend
package (or `RecursiveAI.OneGpu.Backends`) to get a device.

```csharp
using var device = GpuBackends.CreateDevice();     // from RecursiveAI.OneGpu.Backends
using var buffer = device.CreateBuffer<float>(new float[1024]);
```

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
