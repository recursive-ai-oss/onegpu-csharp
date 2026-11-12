# RecursiveAI.OneGpu.WebGpu

WebGPU backend for onegpu in the browser (.NET WebAssembly).

The WebGPU backend of onegpu for .NET apps running in a browser. The managed side records work into one byte stream; the package's
`onegpu.js` (published at `_content/RecursiveAI.OneGpu.WebGpu/onegpu.js`) runs it against the browser's WebGPU API through `[JSImport]`.
Create the device with `await WebGpu.CreateDeviceAsync()` (or `GpuBackends.CreateDeviceAsync()`) and use asynchronous readback; the browser
cannot block.

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
