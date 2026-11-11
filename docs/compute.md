# Compute surface

What `RecursiveAI.OneGpu` offers for compute, how each backend fills it in, and the rules that hold on every backend.

## Capabilities

`GpuDevice.Capabilities.Compute` is a `GpuComputeCapabilities`. Every value comes from the graphics API's own query on the adapter,
never from the vendor name or the API version. Zero means the API did not report the value.

| Value | Vulkan | Metal | WebGPU |
|---|---|---|---|
| Subgroup size, min, max | `VkPhysicalDeviceSubgroupProperties`, `SubgroupSizeControlProperties` | `threadExecutionWidth` of a compiled probe pipeline | `GPUAdapterInfo.subgroupMinSize/MaxSize` with the `subgroups` feature; size itself is not reported |
| Subgroup operations | `supportedOperations`, when the compute stage is listed | one probe kernel per operation class must compile | all but clustered, when the `subgroups` feature is on |
| Shader features (`DeviceFeatures`) | core 1.1/1.2 feature structs, `shaderInt16/64`, `shaderFloat64` | half, 16-bit and 64-bit integers are available on every Apple GPU | `shader-f16` |
| 8-bit integers | `shaderInt8` and `storageBuffer8BitAccess` | yes | no |
| Packed int8 dot product | `shaderIntegerDotProduct` | no (Metal has no packed dot instruction) | WGSL language feature `packed_4x8_integer_dot_product` |
| bfloat16 | `VK_KHR_shader_bfloat16` | a `bfloat` kernel compiles (Metal Shading Language 3.1) | no |
| Cooperative matrices | `VK_KHR_cooperative_matrix` properties, subgroup scope | `simdgroup_matrix` 8×8×8 combinations whose probe kernel compiles | none |
| Workgroup size, invocations, count, memory | `VkPhysicalDeviceLimits` | `maxThreadsPerThreadgroup`, `maxTotalThreadsPerThreadgroup` of the probe, `maxThreadgroupMemoryLength` | `maxCompute*` limits of the device |
| Buffer limits | `maxStorageBufferRange`, `maxMemoryAllocationSize` | `maxBufferLength` | `maxStorageBufferBindingSize`, `maxBufferSize` |

Metal has no feature list for its shader language, so a feature counts when a kernel that uses it compiles on the device. The probes run
once, the first time `Capabilities` is read.

Vulkan devices enable the features they report when they are created (16-bit and 8-bit storage, `shaderFloat16`, `shaderInt8`, integer dot
product, subgroup size control, cooperative matrix, bfloat16), so a kernel that was checked against the capabilities can use them.
A WebGPU device requests `shader-f16` and `subgroups` when the adapter has them.

## Kernels

`GpuDevice.CreateComputeKernel(in ShaderSource, label)` builds the shader, the pipeline layout and the pipeline from the entry point's
reflection. Before it creates anything it checks the kernel against the device and throws `GpuUnsupportedException` naming the missing
capability (`Capability`, `Expected`, `Actual`): the workgroup size against the per-axis and total limits, the required scalar types
against `ShaderFeatures`, the subgroup operations against `SubgroupOperations`. The bindings and push-constant block come from the
reflection, so the host cannot declare a layout that differs from the shader's.

Push constants work on every backend. On WebGPU they are a uniform buffer in the shader's highest bind group, written into a ring and
bound with a dynamic offset; a bind group that the application makes for that group carries the binding.

## Buffers and readback

`GpuBuffer<T>` wraps a `GpuBuffer` of `T` elements (`T` unmanaged; its layout must match the shader's). `Write` queues an upload that runs
before the next submission. `Read` waits for the device. `ReadAsync` (and `GpuDevice.ReadBufferAsync` for bytes) records a copy into a
host-visible buffer behind the work submitted so far, and a helper thread waits for that submission alone; the calling thread and the
device keep working, later submissions are not delayed, and the returned `ValueTask` completes when the bytes are in the destination. On
WebGPU the browser's own asynchronous map does the same. Completions arrive in submission order. Disposing the device finishes every
pending readback first.

## Timestamps

`Capabilities.Timestamps` says whether passes can be timed. `BeginPassTiming` names a `GpuTimestampSet` and the next passes write their
start and end into consecutive slots; `TryReadTimestamps` returns nanoseconds once the submission has finished.

- **Vulkan** writes a timestamp query at each end of the pass; the resolution is the device's `timestampPeriod`.
- **Metal** samples the GPU clock at the boundaries of a compute pass or render pass (counter sampling at stage boundaries, which Apple
  GPUs support; they cannot sample between draws or dispatches, so one pass is the finest unit). The samples are ticks of the GPU clock;
  the device converts them to nanoseconds with the rate it measures between the CPU and GPU clocks. On an Apple M2 one tick is 1 ns, and
  the compute pass times of a test run add up to within 2 % of the wall time of the run that held them. Metal skips the sample of a
  stage that did no work, so a render pass without draws or a compute pass without dispatches reads as zero length at the time of the pass
  before it.
- **WebGPU** does not report timestamps.

## Errors

| Exception | Raised when | Extra data |
|---|---|---|
| `GpuUnsupportedException` | a capability is missing or a limit is exceeded | `Capability`, `Expected`, `Actual` |
| `GpuOutOfMemoryException` | the device or host has no memory left | `RequestedBytes` |
| `GpuDeviceLostException` | the driver reset or the GPU was removed | `Operation`, the API call that saw it |

## Memory statistics

`GpuDevice.MemoryStats` covers compute buffers like any other buffer. Vulkan reports the blocks held from the driver, the bytes live
resources occupy and the staging memory; resources that were disposed count until the GPU has finished with them. Metal reports the
device's allocated size as reserved and the live buffers (count and bytes) as used; textures are not itemised. WebGPU keeps no tally and
reports zeros.

## Devices and instances

The Vulkan backend shares one instance per process (one more when validation is requested) and keeps it for the life of the process; each
device is its own logical device on it. Some drivers refuse new instances after a few dozen create and destroy pairs in one process
(NVIDIA 595 stopped after about 80), and a long-running tool or test run must not hit that. Validation messages reach every device that is alive
when the layer reports them, so devices created at the same time in one process may see each other's messages.

## Browser

`RecursiveAI.OneGpu.WebGpu` ships `onegpu.js` as a static web asset. An app that references the package publishes it at
`_content/RecursiveAI.OneGpu.WebGpu/onegpu.js`, and `WebGpu.CreateDeviceAsync` loads it and registers it as the module `onegpu`; nothing is
copied by hand. A page that registers its own copy (`setModuleImports('onegpu', ...)`) after importing it keeps working: the script marks the
page, and the library then does not import a second one. `tools/ci/test-web-package.sh` checks the whole path with
`tests/RecursiveAI.OneGpu.WebConsumer`, an app that restores the packed libraries from a scratch feed.

## Known environment issue

With the NVIDIA Vulkan driver of the 595 series on Linux, loading the validation layer first leaves too little static thread-local storage
for the driver, and instance creation fails with "cannot allocate memory in static TLS block". Setting
`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096` for the process fixes it; `tools/ci/env.sh` does.
