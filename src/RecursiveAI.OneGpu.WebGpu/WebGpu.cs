using System.Runtime.InteropServices.JavaScript;
using System.Text.Json;
using OneShader.Reflection;

namespace RecursiveAI.OneGpu.WebGpu;

/// <summary>Creates WebGPU devices in the browser.</summary>
public static class WebGpu
{
    /// <summary>True when the page has a WebGPU implementation (navigator.gpu exists).</summary>
    public static bool IsAvailable => OperatingSystem.IsBrowser();

    /// <summary>Runs everything recorded so far and waits until the GPU has finished it, so errors it caused have been reported.</summary>
    public static async Task SettleAsync(GpuDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        device.WaitIdle();
        await Js.WhenIdle(device is WebGpuDevice web ? web.Context : 0).ConfigureAwait(true);
        await Task.Delay(5).ConfigureAwait(true);
    }

    // The package publishes the script at _content/<package id>/onegpu.js. The runtime resolves a module URL against its own folder
    // (_framework), and a host page that serves the app from a sub-path still reaches _content through the same relative step.
    static readonly string[] ScriptUrls = ["../_content/RecursiveAI.OneGpu.WebGpu/onegpu.js", "./_content/RecursiveAI.OneGpu.WebGpu/onegpu.js"];
    static bool _scriptReady;

    // A page that imported onegpu.js itself and registered it as the module "onegpu" marks the global scope (see the script); anything else
    // gets the copy the package publishes. Calling an import of a module that was never registered ends the runtime, so this must come first.
    static async Task EnsureScriptAsync()
    {
        if (_scriptReady || JSHost.GlobalThis.GetPropertyAsBoolean("__onegpuLoaded"))
        {
            _scriptReady = true;
            return;
        }

        Exception? last = null;
        foreach (var url in ScriptUrls)
        {
            try
            {
                await JSHost.ImportAsync("onegpu", url).ConfigureAwait(true);
                _scriptReady = true;
                return;
            }
            catch (JSException ex)
            {
                last = ex;
            }
        }

        throw new GpuUnsupportedException("The page did not register onegpu.js and the copy published with the package could not be loaded: " + last?.Message);
    }

    /// <summary>
    /// Asks the browser for an adapter and a device. The browser answers asynchronously, so a game awaits this before it creates its
    /// instance and hands the device to it.
    /// </summary>
    /// <remarks>An app that references the package gets <c>onegpu.js</c> published beside it, and this call loads it unless the page already registered the script as the module "onegpu".</remarks>
    /// <exception cref="GpuUnsupportedException">The browser has no WebGPU, or no adapter, or the script could not be loaded.</exception>
    public static async Task<GpuDevice> CreateDeviceAsync(GpuDeviceOptions? options = null)
    {
        if (!OperatingSystem.IsBrowser())
        {
            throw new GpuUnsupportedException("The WebGPU backend runs inside a browser.");
        }

        options ??= new GpuDeviceOptions();
        await EnsureScriptAsync().ConfigureAwait(true);
        string json;
        try
        {
            json = await Js.Init("{\"fallback\":" + (options.Adapter == "software" ? "true" : "false") + "}").ConfigureAwait(true);
        }
        catch (Exception ex) when (ex is System.Runtime.InteropServices.JavaScript.JSException)
        {
            throw new GpuUnsupportedException(ex.Message);
        }

        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        var features = new HashSet<string>(StringComparer.Ordinal);
        foreach (var f in r.GetProperty("features").EnumerateArray())
        {
            features.Add(f.GetString() ?? "");
        }

        var info = new GpuAdapterInfo(
            r.GetProperty("name").GetString() ?? "WebGPU",
            r.GetProperty("software").GetBoolean() ? AdapterKind.Software : AdapterKind.Other,
            0,
            0,
            "WebGPU");
        var lim = r.GetProperty("limits");
        uint Limit(string name) => lim.TryGetProperty(name, out var v) ? (uint)Math.Min(v.GetDouble(), uint.MaxValue) : 0;
        ulong Limit64(string name) => lim.TryGetProperty(name, out var v) ? (ulong)v.GetDouble() : 0;
        var f16 = features.Contains("shader-f16");
        var subgroups = features.Contains("subgroups");
        var sgMin = r.TryGetProperty("subgroupMin", out var smin) ? (uint)smin.GetInt32() : 0;
        var sgMax = r.TryGetProperty("subgroupMax", out var smax) ? (uint)smax.GetInt32() : 0;
        var dot4 = r.TryGetProperty("dot4", out var dot) && dot.GetBoolean();
        var compute = new GpuComputeCapabilities
        {
            SubgroupSize = 0,
            SubgroupMinSize = subgroups ? sgMin : 0,
            SubgroupMaxSize = subgroups ? sgMax : 0,
            SubgroupOperations = subgroups
                ? GpuSubgroupOperations.Basic | GpuSubgroupOperations.Vote | GpuSubgroupOperations.Arithmetic | GpuSubgroupOperations.Ballot
                    | GpuSubgroupOperations.Shuffle | GpuSubgroupOperations.ShuffleRelative | GpuSubgroupOperations.Quad
                : GpuSubgroupOperations.None,

            // f16 in WGSL covers storage, uniform and the uniform buffer that stands in for push constants.
            // The packed dot product is a WGSL language feature rather than a device feature; kernels that use it are checked against this flag all the same.
            ShaderFeatures = (f16 ? DeviceFeatures.Float16 | DeviceFeatures.StorageBuffer16 | DeviceFeatures.UniformBuffer16 | DeviceFeatures.PushConstant16 : DeviceFeatures.None)
                | (dot4 ? DeviceFeatures.IntegerDotProduct : DeviceFeatures.None),
            Integer8 = false,
            Int8DotProduct = dot4,
            BFloat16 = false,
            MaxWorkgroupSizeX = Limit("maxComputeWorkgroupSizeX"),
            MaxWorkgroupSizeY = Limit("maxComputeWorkgroupSizeY"),
            MaxWorkgroupSizeZ = Limit("maxComputeWorkgroupSizeZ"),
            MaxWorkgroupInvocations = Limit("maxComputeInvocationsPerWorkgroup"),
            MaxWorkgroupCount = Limit("maxComputeWorkgroupsPerDimension"),
            MaxWorkgroupMemoryBytes = Limit("maxComputeWorkgroupStorageSize"),
            MaxStorageBufferBindingBytes = Limit64("maxStorageBufferBindingSize"),
            MaxBufferBytes = Limit64("maxBufferSize"),
            MaxStorageBuffersPerStage = Limit("maxStorageBuffersPerShaderStage"),
            MaxBindGroups = Limit("maxBindGroups"),
            StorageBufferOffsetAlignment = Limit("minStorageBufferOffsetAlignment"),
        };
        var caps = new GpuCapabilities(
            Timestamps: false,
            MultiDrawIndirect: false,
            IndirectFirstInstance: features.Contains("indirect-first-instance"),
            DepthClamp: features.Contains("depth-clip-control"),
            DepthBiasClamp: true,
            NonUniformArrayIndexing: false,
            MaxSampleCount: 4,
            MaxTextureSize: (uint)r.GetProperty("maxTexture2D").GetInt32(),
            MaxTextureSize3D: (uint)r.GetProperty("maxTexture3D").GetInt32(),
            MaxArrayLayers: (uint)r.GetProperty("maxLayers").GetInt32(),
            MaxResourceArrayCount: 1)
        {
            Compute = compute,
        };
        return new WebGpuDevice(info, caps, r.GetProperty("ctx").GetInt32());
    }
}
