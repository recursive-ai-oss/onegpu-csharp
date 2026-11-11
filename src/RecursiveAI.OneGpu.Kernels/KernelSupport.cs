using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Element types the kernels understand.</summary>
internal enum ElementKind
{
    F32 = 0,
    F16 = 1,
    I32 = 2,
    I64 = 3,
    U32 = 4,
}

internal static class KernelTypes
{
    public static ElementKind Of<T>(string operation)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return ElementKind.F32;
        }

        if (typeof(T) == typeof(Half))
        {
            return ElementKind.F16;
        }

        if (typeof(T) == typeof(int))
        {
            return ElementKind.I32;
        }

        if (typeof(T) == typeof(long))
        {
            return ElementKind.I64;
        }

        if (typeof(T) == typeof(uint))
        {
            return ElementKind.U32;
        }

        throw new GpuKernelShapeException(operation, "T", $"element type {typeof(T).Name} is not supported; use float, Half, int, long or uint as the operation allows");
    }

    /// <summary>The element type of a float tensor: float or Half.</summary>
    public static bool IsHalf<T>(string operation)
        where T : unmanaged
    {
        var kind = Of<T>(operation);
        if (kind is ElementKind.F32)
        {
            return false;
        }

        if (kind is ElementKind.F16)
        {
            return true;
        }

        throw new GpuKernelShapeException(operation, "T", $"element type {typeof(T).Name} is not supported; use float or Half");
    }
}

/// <summary>Argument checks shared by the families. Everything that comes from a model is validated here, in safe arithmetic, before work is recorded.</summary>
internal static class Check
{
    /// <summary>The kernels address elements with 32-bit unsigned integers; sizes stay below this.</summary>
    public const long MaxElements = int.MaxValue;

    /// <summary>Checks a range of a buffer; half-precision ranges also need their last 32-bit word inside the buffer, because the kernels read and write halves in pairs.</summary>
    public static void Range(string operation, string name, int bufferLength, int offset, long extent, bool half = false)
    {
        if (offset < 0)
        {
            throw new GpuKernelShapeException(operation, name + "Offset", $"offset {offset} is negative");
        }

        if (extent < 0 || (long)offset + extent > bufferLength)
        {
            throw new GpuKernelShapeException(operation, name, $"elements {offset}..{(long)offset + extent} reach past the buffer of {bufferLength}");
        }

        if (half && extent > 0 && ((((long)offset + extent + 1) >> 1) << 1) > bufferLength)
        {
            throw new GpuKernelShapeException(operation, name, $"the buffer of {bufferLength} halves ends inside a 32-bit word; allocate GpuKernelBuffers.PaddedHalfLength(n) elements");
        }
    }

    public static long Product(string operation, string name, ReadOnlySpan<int> shape)
    {
        long n = 1;
        for (var i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
            {
                throw new GpuKernelShapeException(operation, $"{name}[{i}]", $"dimension {shape[i]} is negative");
            }

            n = checked(n * shape[i]);
            if (n > MaxElements)
            {
                throw new GpuKernelShapeException(operation, $"{name}[{i}]", $"the tensor holds more than {MaxElements} elements");
            }
        }

        return n;
    }
}

/// <summary>Picks kernel variants from the device's capabilities, once, when a family is created.</summary>
internal static class Variants
{
    /// <summary>
    /// Subgroup arithmetic is used by the fast variants when the device offers it and no subgroup is smaller than four lanes (the
    /// kernels keep one shared-memory slot per subgroup of a 256-thread workgroup, 64 at most).
    /// </summary>
    public static bool UseSubgroups(GpuDevice device, bool allowed)
    {
        if (!allowed)
        {
            return false;
        }

        var c = device.Capabilities.Compute;
        var min = c.SubgroupMinSize != 0 ? c.SubgroupMinSize : c.SubgroupSize;
        return (c.SubgroupOperations & GpuSubgroupOperations.Arithmetic) != 0 && (c.SubgroupOperations & GpuSubgroupOperations.Basic) != 0 && min >= 4;
    }
}

/// <summary>Grid arithmetic. Dispatches are two-dimensional so that more than 65535 workgroups fit on every device.</summary>
internal static class Grid
{
    public static (uint X, uint Y) For(GpuDevice device, long groups)
    {
        if (groups <= 0)
        {
            return (0, 0);
        }

        var limit = device.Capabilities.Compute.MaxWorkgroupCount;
        var maxX = limit == 0 ? 65535u : Math.Min(limit, 65535u);
        var x = (uint)Math.Min(groups, maxX);
        var y = (uint)((groups + x - 1) / x);
        return (x, y);
    }
}

/// <summary>
/// One compute pipeline plus the bind groups made for it. A bind group depends only on which buffers are bound (offsets and sizes travel
/// in push constants), so repeated calls with the same buffers allocate nothing.
/// </summary>
internal sealed class KernelPipeline : IDisposable
{
    const int MaxCachedGroups = 512;

    readonly GpuDevice _device;
    readonly GpuComputeKernel _kernel;
    readonly Dictionary<BindKey, GpuBindGroup> _groups = [];
    readonly BindGroupEntry[] _entries;

    public KernelPipeline(GpuDevice device, in ShaderSource source, int bindings, int pushWords, string label)
    {
        _device = device;
        _kernel = device.CreateComputeKernel(source, label);
        if (_kernel.Bindings.Count != bindings || _kernel.PushConstants.Size != pushWords * 4)
        {
            var kernel = _kernel;
            kernel.Dispose();
            throw new InvalidOperationException($"kernel {label}: the shader declares {_kernel.Bindings.Count} bindings and {_kernel.PushConstants.Size} bytes of push constants, the host expects {bindings} and {pushWords * 4}");
        }

        _entries = new BindGroupEntry[bindings];
    }

    public GpuComputeKernel Kernel => _kernel;

    public void Record(GpuCommandEncoder encoder, ReadOnlySpan<uint> push, uint groupsX, uint groupsY, GpuBuffer b0, GpuBuffer? b1 = null, GpuBuffer? b2 = null, GpuBuffer? b3 = null)
    {
        if (groupsX == 0 || groupsY == 0)
        {
            return;
        }

        var key = new BindKey(b0, b1, b2, b3);
        if (!_groups.TryGetValue(key, out var group))
        {
            if (_groups.Count >= MaxCachedGroups)
            {
                foreach (var g in _groups.Values)
                {
                    g.Dispose();
                }

                _groups.Clear();
            }

            var e = _entries;
            e[0] = BindGroupEntry.ForBuffer(0, b0);
            if (e.Length > 1)
            {
                e[1] = BindGroupEntry.ForBuffer(1, b1!);
            }

            if (e.Length > 2)
            {
                e[2] = BindGroupEntry.ForBuffer(2, b2!);
            }

            if (e.Length > 3)
            {
                e[3] = BindGroupEntry.ForBuffer(3, b3!);
            }

            group = _kernel.CreateBindGroup(_device, 0, e);
            _groups.Add(key, group);
        }

        var pass = encoder.BeginComputePass();
        pass.SetPipeline(_kernel.Pipeline);
        pass.SetBindGroup(0, group);
        pass.SetPushConstants(0, MemoryMarshal.AsBytes(push));
        pass.Dispatch(groupsX, groupsY);
        pass.EndPass();
    }

    public void Dispose()
    {
        foreach (var g in _groups.Values)
        {
            g.Dispose();
        }

        _groups.Clear();
        _kernel.Dispose();
    }

    readonly struct BindKey(GpuBuffer b0, GpuBuffer? b1, GpuBuffer? b2, GpuBuffer? b3) : IEquatable<BindKey>
    {
        readonly GpuBuffer _b0 = b0;
        readonly GpuBuffer? _b1 = b1;
        readonly GpuBuffer? _b2 = b2;
        readonly GpuBuffer? _b3 = b3;

        public bool Equals(BindKey o) => ReferenceEquals(_b0, o._b0) && ReferenceEquals(_b1, o._b1) && ReferenceEquals(_b2, o._b2) && ReferenceEquals(_b3, o._b3);

        public override bool Equals(object? obj) => obj is BindKey k && Equals(k);

        public override int GetHashCode() =>
            HashCode.Combine(RuntimeHelpers.GetHashCode(_b0), RuntimeHelpers.GetHashCode(_b1), RuntimeHelpers.GetHashCode(_b2), RuntimeHelpers.GetHashCode(_b3));
    }
}
