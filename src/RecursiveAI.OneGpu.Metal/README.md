# RecursiveAI.OneGpu.Metal

Metal backend for onegpu on macOS (Apple silicon), with no native binaries.

The Metal backend of onegpu. It reaches Metal through the Objective-C runtime with typed `objc_msgSend` calls, so there is no native
library to ship. Apple silicon only. `MetalGpu.CreateDevice()` creates a device directly; `RecursiveAI.OneGpu.Backends` chooses a backend for
the platform.

Run with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` to make API misuse fatal while testing.

Part of [onegpu](https://github.com/recursive-ai-oss/onegpu-csharp). `MIT OR Apache-2.0`, © Recursive AI, LLC.
