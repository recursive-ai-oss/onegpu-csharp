# Names that are part of the interface

Everything on this page is frozen for 1.x: a rename is a breaking change.

## Environment variables

Read by the packages:

| Variable | Read by | Meaning |
|---|---|---|
| `ONEGPU_BACKEND` | `GpuBackends` | `vulkan` or `metal` (any case) overrides the platform's choice. Anything else, or unset: Metal on macOS and iOS, Vulkan elsewhere. In a browser the backend is always WebGPU. |
| `ONEGPU_ADAPTER` | Vulkan, Metal | Picks the adapter whose name contains this text (case-insensitive) or whose kind is `software`, `discrete` or `integrated`. Wins over `GpuDeviceOptions.Adapter`. The first match in the backend's adapter order is used; no match throws `GpuUnsupportedException`. |
| `ONEGPU_VALIDATION` | Vulkan | `1` or `true` turns on the Khronos validation layer, as `GpuDeviceOptions.Validation` does. Messages are collected per device (`GpuDevice.TakeValidationMessages`). |
| `ONEGPU_VALIDATION_OPTIONAL` | Vulkan | `1` or `true` makes `GpuDeviceOptions.RequireValidation` fall back to running without the layer, with a notice, when the layer is not installed. Otherwise a missing layer throws. |
| `ONEGPU_VULKAN_LIBRARY` | Vulkan | A path or name of a Vulkan loader to try before the system's (`libvulkan.so.1`, `vulkan-1.dll`, `libvulkan.1.dylib`). |

Read by the tests, tools and benchmarks only (never by a package):

| Variable | Meaning |
|---|---|
| `ONEGPU_ARTIFACTS` | Build output folder (default `artifacts/` in the repository). Read by `Directory.Build.props`. |
| `ONEGPU_REQUIRE_DEVICE` | `1` makes a missing GPU fail a device test instead of skipping it. `tools/ci/env.sh` sets it. |
| `ONEGPU_CHROME` | The Chrome executable `tools/web/run-page.mjs` starts. |
| `ONEGPU_PARITY_REPORT`, `ONEGPU_REPORT`, `ONEGPU_GOLDEN_REPORT` | A file that the parity, throughput and golden-image tests append their result rows to. |
| `ONEGPU_UPDATE_GOLDENS`, `ONEGPU_CAPTURE_DIR` | `1` rewrites the golden images in `testdata/`; the folder the render tests save captures to. |
| `ONEGPU_GEMM_FULL`, `ONEGPU_CONV_FULL`, `ONEGPU_ATTN_LARGE` | `1` adds the large shape sets, which take long on the CPU reference. |
| `ONEGPU_GEMM_NO_HARDWARE`, `ONEGPU_GEMM_SUBGROUPS`, `ONEGPU_CONV_BENCH_NOCOOP`, `ONEGPU_CONV_BENCH_NOSPLIT` | Force a variant off, to run the fallback on a device that has the feature, or to compare. |
| `ONEGPU_BENCH_VALIDATION` | `0` times without the validation layer. |
| `ONEGPU_GEMM_BENCH`, `ONEGPU_CONV_BENCH`, `ONEGPU_CONV_BENCH_BATCHES`, `ONEGPU_CONV_BENCH_LAYERS`, `ONEGPU_CONV_BENCH_MODE`, `ONEGPU_CONV_BENCH_ONLY`, `ONEGPU_ATTN_BENCH`, `ONEGPU_RECURRENT_BENCH`, `ONEGPU_DISPATCH_BENCH` | Select and shape the throughput tests (they are skipped unless set). |

The Vulkan loader's and drivers' own variables also apply: `VK_ICD_FILENAMES` picks a driver, `VK_INSTANCE_LAYERS` and `VK_ADD_LAYER_PATH`
control layers. `tools/ci/fetch-validation-layer.sh [folder]` downloads the validation layer the hosted CI runs (Ubuntu 24.04's
`vulkan-validationlayers`, version pinned in that script and read from it by `ci.yml`) into a folder; point `VK_ADD_LAYER_PATH` at it to
reproduce CI locally. A second run against the newest Khronos release catches what the older layer does not model. On Metal, `MTL_DEBUG_LAYER=1`, `MTL_DEBUG_LAYER_ERROR_MODE=assert` and `MTL_SHADER_VALIDATION=1` make misuse fatal.

## The JavaScript module `onegpu`

The WebGPU backend imports a script as the module named `onegpu` (`[JSImport("...", "onegpu")]`). The package publishes it at
`_content/RecursiveAI.OneGpu.WebGpu/onegpu.js`; `WebGpu.CreateDeviceAsync` imports it from there unless the page already imported it and
registered it under that name, which it signals by setting `globalThis.__onegpuLoaded`.

The script's exports are an implementation interface between the package's managed code and the browser, not something an app calls.
They are named here because a page that registers its own copy (for example to bundle it) must provide them:

| Export | Purpose |
|---|---|
| `init(requestJson)` | Asynchronously requests an adapter and device and creates a context. Returns JSON with the context number, adapter name, limits, features and subgroup sizes. `{"fallback":true}` asks for the software adapter. |
| `exec(ctx, bytes)` | Runs one recorded stream of resource creation, uploads and commands in the context. |
| `takeMessages(ctx)` | Returns and clears the validation and uncaptured errors the context saw. |
| `whenIdle(ctx)` | Resolves when the device's queue has finished all submitted work. |
| `readBuffer(ctx, id, offset, size)`, `readTexture(ctx, id, mip, layer, width, height, depth, bytesPerTexel)` | Start a readback and resolve to a ticket. |
| `takeRead(ticket, destination)` | Copies the bytes of a finished readback out. |
| `configureCanvas(ctx, canvasId, width, height)`, `setCanvasSize(canvasId, width, height)` | Attach and resize a canvas for a surface. |

Every device in a page is a context of its own, so several devices can share one page; each function takes the context number first
(the canvas functions after the first take it implicitly through the canvas id).

## The public API baseline

Each package has `PublicAPI.Shipped.txt` beside its project: every public type and member, written as the compiler sees it. The
`Microsoft.CodeAnalysis.PublicApiAnalyzers` analyzer (build time only, `PrivateAssets="all"`) makes a build fail when the code and the
file disagree, so a public change cannot happen without a change to the file in the same diff. To change the API, edit the code and
add or remove the matching lines; the analyzer's messages (`RS0016`, `RS0017`) give the exact text of each missing line, and a code fix
in an IDE writes them. A removal or a changed signature is a breaking change and needs a major version.

`RecursiveAI.OneGpu.Kernels` also contains one generated class per HLSL file (namespace `RecursiveAI.OneGpu.Kernels.Shaders`, produced
by the shader build), which is why its file is large: a changed kernel entry point or variant axis changes that surface and shows in
the diff. Those classes are the shader blobs and layouts the hand-written families use; apps should use the families.
