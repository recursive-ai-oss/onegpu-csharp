using OneShader.Layouts;
using OneShader.Reflection;

namespace RecursiveAI.OneGpu;

/// <summary>
/// A compute entry point ready to dispatch: the shader, its pipeline layout and the pipeline, all derived from the entry point's reflection
/// so the host cannot disagree with the shader about bindings, push constants or workgroup size.
/// </summary>
/// <remarks>Create one per kernel and reuse it; creating a pipeline is the expensive step. Not thread-safe.</remarks>
public sealed class GpuComputeKernel : IDisposable
{
    readonly GpuShader _shader;

    internal GpuComputeKernel(GpuShader shader, GpuPipelineLayout layout, GpuComputePipeline pipeline, EntryLayout reflection, ResourceBinding[] bindings, PushConstantRange pushConstants)
    {
        _shader = shader;
        PipelineLayout = layout;
        Pipeline = pipeline;
        Reflection = reflection;
        Bindings = bindings;
        PushConstants = pushConstants;
    }

    /// <summary>The compiled pipeline; pass it to <see cref="GpuComputePass.SetPipeline(GpuComputePipeline)"/>.</summary>
    public GpuComputePipeline Pipeline { get; }

    /// <summary>The pipeline layout, shared by the bind groups made with <see cref="CreateBindGroup"/>.</summary>
    public GpuPipelineLayout PipelineLayout { get; }

    /// <summary>The entry point's reflection from the shader build.</summary>
    public EntryLayout Reflection { get; }

    /// <summary>The resource bindings the shader declares, ordered by set then binding.</summary>
    public IReadOnlyList<ResourceBinding> Bindings { get; }

    /// <summary>The push-constant block the shader declares; <see cref="PushConstantRange.Size"/> is 0 when it has none.</summary>
    public PushConstantRange PushConstants { get; }

    /// <summary>Threads per workgroup along x, as declared by <c>numthreads</c>.</summary>
    public uint WorkgroupSizeX => Reflection.WorkgroupSize.X;

    /// <summary>Threads per workgroup along y.</summary>
    public uint WorkgroupSizeY => Reflection.WorkgroupSize.Y;

    /// <summary>Threads per workgroup along z.</summary>
    public uint WorkgroupSizeZ => Reflection.WorkgroupSize.Z;

    /// <summary>Creates a bind group for one set of the kernel's layout. Entries must cover every binding of that set.</summary>
    /// <exception cref="GpuException">An entry does not fit the binding it names.</exception>
    public GpuBindGroup CreateBindGroup(GpuDevice device, uint setIndex, ReadOnlySpan<BindGroupEntry> entries) =>
        device.CreateBindGroup(PipelineLayout, setIndex, entries);

    /// <summary>How many workgroups cover <paramref name="threadsX"/> × <paramref name="threadsY"/> × <paramref name="threadsZ"/> threads (each axis rounded up).</summary>
    public (uint X, uint Y, uint Z) GroupsFor(uint threadsX, uint threadsY = 1, uint threadsZ = 1) =>
        (CeilDiv(threadsX, WorkgroupSizeX), CeilDiv(threadsY, WorkgroupSizeY), CeilDiv(threadsZ, WorkgroupSizeZ));

    /// <summary>Releases the pipeline, layout and shader once the GPU has finished with them.</summary>
    public void Dispose()
    {
        Pipeline.Dispose();
        PipelineLayout.Dispose();
        _shader.Dispose();
    }

    static uint CeilDiv(uint a, uint b) => b == 0 ? 0 : (uint)(((ulong)a + b - 1) / b);
}

public abstract partial class GpuDevice
{
    /// <summary>The usage <c>CreateBuffer&lt;T&gt;</c> picks when none is given: a storage buffer that can be uploaded to and read back.</summary>
    public const BufferUsage DefaultComputeUsage = BufferUsage.Storage | BufferUsage.CopySource | BufferUsage.CopyDestination;

    /// <summary>Creates a typed buffer of <paramref name="length"/> elements. Contents are undefined until written.</summary>
    /// <typeparam name="T">A blittable element type.</typeparam>
    /// <param name="length">Number of elements, at least 1.</param>
    /// <param name="usage">Every use the buffer will have.</param>
    /// <param name="label">Optional name shown in debugging tools.</param>
    /// <exception cref="ArgumentOutOfRangeException">The length is not positive.</exception>
    /// <exception cref="GpuUnsupportedException">The size exceeds <see cref="GpuComputeCapabilities.MaxBufferBytes"/>.</exception>
    /// <exception cref="GpuOutOfMemoryException">The device has no memory left.</exception>
    public GpuBuffer<T> CreateBuffer<T>(int length, BufferUsage usage = DefaultComputeUsage, string? label = null)
        where T : unmanaged
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var bytes = checked((ulong)length * (ulong)GpuBuffer<T>.ElementSizeOf);
        var max = Capabilities.Compute.MaxBufferBytes;
        if (max != 0 && bytes > max)
        {
            throw new GpuUnsupportedException($"a buffer of {bytes} bytes is larger than {Adapter.Name} allows", nameof(GpuComputeCapabilities.MaxBufferBytes), $"at most {max} bytes", $"{bytes} bytes requested");
        }

        // Shaders read 16-bit elements as pairs inside 32-bit words, so the allocation is padded to a whole word: the last element of an
        // odd-length buffer then still has its word inside the binding.
        return new GpuBuffer<T>(this, CreateBuffer(new BufferDesc((bytes + 3) & ~3UL, usage) { Label = label }), length);
    }

    /// <summary>Creates a typed buffer holding <paramref name="data"/>; the upload runs before the next submitted commands.</summary>
    /// <typeparam name="T">A blittable element type.</typeparam>
    /// <param name="data">Initial contents; at least one element.</param>
    /// <param name="usage">Every use the buffer will have; it must include <see cref="BufferUsage.CopyDestination"/>.</param>
    /// <param name="label">Optional name shown in debugging tools.</param>
    public GpuBuffer<T> CreateBuffer<T>(ReadOnlySpan<T> data, BufferUsage usage = DefaultComputeUsage, string? label = null)
        where T : unmanaged
    {
        var buffer = CreateBuffer<T>(data.Length, usage, label);
        buffer.Write(data);
        return buffer;
    }

    /// <summary>
    /// Creates a compute kernel from a compiled entry point: shader, pipeline layout (from the entry point's reflection) and pipeline. Checks the
    /// kernel against this device first, so a missing feature or an oversized workgroup fails here and names what was needed.
    /// </summary>
    /// <param name="source">The entry point, filled from the members the shader build generates; its <see cref="ShaderSource.Layout"/> is required.</param>
    /// <param name="label">Optional name shown in debugging tools.</param>
    /// <param name="requiredSubgroupSize">Runs the kernel with exactly this subgroup size in lanes (see <see cref="ComputePipelineDesc.RequiredSubgroupSize"/>); 0 leaves it to the driver.</param>
    /// <exception cref="ArgumentException">The source is not a compute entry point or has no layout.</exception>
    /// <exception cref="GpuUnsupportedException">The source has no code for this backend, the device lacks a feature the kernel needs (<see cref="GpuUnsupportedException.Capability"/> names it), or the workgroup exceeds a device limit.</exception>
    public GpuComputeKernel CreateComputeKernel(in ShaderSource source, string? label = null, uint requiredSubgroupSize = 0)
    {
        if (source.Stage != ShaderStage.Compute)
        {
            throw new ArgumentException($"entry point {source.EntryPoint} is a {source.Stage} shader, not a compute shader", nameof(source));
        }

        var reflection = source.Layout ?? throw new ArgumentException($"entry point {source.EntryPoint} has no Layout; fill it from the generated class's Layout member so the pipeline layout comes from the shader", nameof(source));
        CheckKernelFits(reflection);

        var bindings = ShaderLayouts.Bindings([reflection]);
        var push = ShaderLayouts.PushConstants([reflection]);
        var shader = CreateShader(source);
        GpuPipelineLayout? layout = null;
        try
        {
            layout = push.Size > 0 ? CreatePipelineLayout(bindings, push) : CreatePipelineLayout(bindings);
            // The shader compiler lists FullSubgroups for entry points that use cooperative matrices (they need full subgroups below SPIR-V 1.6).
            var full = (reflection.RequiredFeatures & DeviceFeatures.FullSubgroups) != 0;
            var pipeline = CreateComputePipeline(new ComputePipelineDesc { Layout = layout, Compute = shader, RequireFullSubgroups = full, RequiredSubgroupSize = requiredSubgroupSize, Label = label ?? source.EntryPoint });
            return new GpuComputeKernel(shader, layout, pipeline, reflection, bindings, push);
        }
        catch
        {
            layout?.Dispose();
            shader.Dispose();
            throw;
        }
    }

    void CheckKernelFits(EntryLayout kernel)
    {
        var c = Capabilities.Compute;
        var (x, y, z) = kernel.WorkgroupSize;
        var name = kernel.Name;
        if (c.MaxWorkgroupSizeX != 0 && (x > c.MaxWorkgroupSizeX || y > c.MaxWorkgroupSizeY || z > c.MaxWorkgroupSizeZ))
        {
            throw new GpuUnsupportedException($"kernel {name} uses workgroups of {x}x{y}x{z}, more than {Adapter.Name} allows", "MaxWorkgroupSize",
                $"at most {c.MaxWorkgroupSizeX}x{c.MaxWorkgroupSizeY}x{c.MaxWorkgroupSizeZ}", $"{x}x{y}x{z}");
        }

        if (c.MaxWorkgroupInvocations != 0 && (ulong)x * y * z > c.MaxWorkgroupInvocations)
        {
            throw new GpuUnsupportedException($"kernel {name} uses {(ulong)x * y * z} invocations per workgroup, more than {Adapter.Name} allows", nameof(GpuComputeCapabilities.MaxWorkgroupInvocations),
                $"at most {c.MaxWorkgroupInvocations}", $"{(ulong)x * y * z}");
        }

        if ((kernel.RequiredFeatures & DeviceFeatures.FullSubgroups) != 0 && c.SubgroupMinSize != 0 && (ulong)x * y * z % c.SubgroupMinSize != 0 && (ulong)x * y * z % c.SubgroupMaxSize != 0)
        {
            throw new GpuUnsupportedException($"kernel {name} uses cooperative matrices with {(ulong)x * y * z} invocations per workgroup, not a whole number of subgroups on {Adapter.Name}", "FullSubgroups",
                $"a multiple of {c.SubgroupMinSize}", $"{(ulong)x * y * z}");
        }

        var missing = kernel.RequiredFeatures & ~c.ShaderFeatures;
        if (missing != DeviceFeatures.None)
        {
            throw new GpuUnsupportedException($"kernel {name} needs shader features {missing} that {Adapter.Name} ({Backend}) does not offer", nameof(GpuComputeCapabilities.ShaderFeatures),
                kernel.RequiredFeatures.ToString(), c.ShaderFeatures.ToString());
        }

        var ops = (GpuSubgroupOperations)(int)kernel.SubgroupOperations;
        var missingOps = ops & ~c.SubgroupOperations;
        if (missingOps != GpuSubgroupOperations.None)
        {
            throw new GpuUnsupportedException($"kernel {name} uses subgroup operations {missingOps} that {Adapter.Name} ({Backend}) does not offer", nameof(GpuComputeCapabilities.SubgroupOperations),
                ops.ToString(), c.SubgroupOperations.ToString());
        }
    }
}
