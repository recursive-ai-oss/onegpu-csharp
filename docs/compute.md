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
| Full subgroups (`FullSubgroups`) | Vulkan 1.3 `computeFullSubgroups` | yes (a threadgroup that is a multiple of the SIMD-group width runs whole SIMD groups) | no |
| Selectable subgroup size (`SubgroupSizeControl`) | Vulkan 1.3 `subgroupSizeControl` and the compute stage in `requiredSubgroupSizeStages` | no (fixed at 32 on Apple GPUs) | no |
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

**Workgroup memory a pipeline uses.** `GpuComputePipeline.ReportedWorkgroupMemoryBytes` is what the compiled pipeline reports (Metal's
`staticThreadgroupMemoryLength`; 0 on backends that do not report it). Metal enforces its limit against that figure, which is larger than the declared size
while shader validation is on (it doubles it) and when a three-component vector takes 16 bytes, so a kernel with several shapes picks one from this number
after compiling (the float32 GEMM does).

**Subgroup requirements.** `ComputePipelineDesc.RequireFullSubgroups` makes every subgroup of every workgroup full (the workgroup size must
be a multiple of the subgroup size) and `RequiredSubgroupSize` pins the width. Cooperative-matrix kernels need the first below SPIR-V 1.6,
so the shader compiler lists `FullSubgroups` in the reflection of every entry point that uses them and `CreateComputeKernel` sets it for any kernel whose reflection lists that feature (devices report it in `ShaderFeatures` when `GpuComputeCapabilities.FullSubgroups` is true), and takes an optional `requiredSubgroupSize` for
kernels written for one width. Vulkan passes `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` and chains
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (Vulkan 1.3 core, `VK_EXT_subgroup_size_control`). A request the device cannot meet
throws `GpuUnsupportedException` whose `Capability` is `FullSubgroups` or `RequiredSubgroupSize`, before anything is created, on every
backend: a device with one fixed subgroup size accepts that size and no other; Metal accepts full subgroups (its SIMD groups are whole when the
threadgroup is a multiple of the SIMD width, which `CreateComputeKernel` checks against the workgroup size of a cooperative-matrix kernel); WebGPU has no control over either and
rejects full subgroups and any pinned size unless the adapter reports a single size.

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
(NVIDIA 595 stopped after about 80), and a long-running tool or test run must not hit that.

## Adapter order

The Vulkan backend keeps adapters in one total order, whatever order the drivers are enumerated in: kind (discrete, integrated, virtual,
software, other), then PCI vendor id, device id, name, API version (newest first) and the driver's pipeline-cache UUID. The default is the
first adapter in that order, and `GpuDeviceOptions.Adapter` or `ONEGPU_ADAPTER` (a name substring or a kind) picks the first match in it,
so a multi-GPU machine chooses the same adapter every run.

## Validation messages

`GpuDevice.TakeValidationMessages` returns what the layer said about this device. Devices share one instance and one message callback, so
the Vulkan backend records which device created each object and sends a message to the owner of the first object it names; messages raised
while a `VkDevice` is created or destroyed go to that device. Two devices alive at once never see each other's messages. A message that
names no object any device owns goes to every live device, because dropping it would hide a real problem.

Call `TakeValidationMessages` again after `Dispose`: destroying a Vulkan device is where the layer lists the objects that were never
disposed (`vkDestroyDevice(): ... has N leaked objects`), and the Metal backend adds a line when buffers were still alive. The test
fixtures do this, so every suite also checks for leaks. A command encoder that was begun and never submitted does not leak: the device
destroys every encoder it made.

## Browser

`RecursiveAI.OneGpu.WebGpu` ships `onegpu.js` as a static web asset. An app that references the package publishes it at
`_content/RecursiveAI.OneGpu.WebGpu/onegpu.js`, and `WebGpu.CreateDeviceAsync` loads it and registers it as the module `onegpu`; nothing is
copied by hand. A page that registers its own copy (`setModuleImports('onegpu', ...)`) after importing it keeps working: the script marks the
page, and the library then does not import a second one. `tools/ci/test-web-package.sh` checks the whole path with
`tests/RecursiveAI.OneGpu.WebConsumer`, an app that restores the packed libraries from a scratch feed.

Several devices can share a page: each `CreateDeviceAsync` makes a context of its own in `onegpu.js` (its WebGPU device, handle table, push-constant ring and pass state), and every call from a device names its context. Buffers and bind groups belong to the device that made them, as in WebGPU.

Disposing a bind group, buffer-backed or not, while a command encoder that recorded it has not been submitted is safe on every backend. On the browser backend an
encoder pins each bind group it records and a group disposed meanwhile is destroyed in the stream right after that encoder's commands, because WebGPU forbids destroying an object queued commands still name.
An encoder that is never submitted keeps its groups alive.

## Known environment issue

With the NVIDIA Vulkan driver of the 595 series on Linux, loading the validation layer first leaves too little static thread-local storage
for the driver, and instance creation fails with "cannot allocate memory in static TLS block". Setting
`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096` for the process fixes it; `tools/ci/env.sh` does.

With the RADV driver on a Ryzen 7000 integrated GPU (Mesa 26.0.3, kernel 7.0.0-42), a dispatch that starts after the GPU has idled for a few
milliseconds sometimes writes one wrong value: about 3% of dispatches of a 1536-workgroup shared-memory reduction, none while another process
keeps the GPU at its high clock. A plain Vulkan program (no library code) reproduces it with every barrier in place, and the same GPU model passes on an
older Mesa, kernel and firmware, so it is not a synchronisation fault in this library. `tools/gpu-idle-repro` holds the program and the
numbers. Results from that GPU on that stack are not trusted until it is updated.
