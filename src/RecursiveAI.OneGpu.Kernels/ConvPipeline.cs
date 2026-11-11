using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// The compute pipeline of the implicit convolution kernels and the bind groups made for it: five storage buffers, a three-dimensional grid (the third
/// axis counts images times groups), push constants, and optionally a pinned subgroup size for the matrix-hardware variants. Like the other pipeline
/// wrappers, offsets and sizes travel in push constants, so repeated calls with the same buffers allocate nothing.
/// </summary>
internal sealed class ConvPipeline : IDisposable
{
    const int Buffers = 5;

    readonly GpuDevice _device;
    readonly GpuComputeKernel _kernel;
    readonly Dictionary<(GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer), GpuBindGroup> _groups = [];
    readonly BindGroupEntry[] _entries = new BindGroupEntry[Buffers];
    readonly int _maxGroups;

    public ConvPipeline(GpuDevice device, in ShaderSource source, int pushWords, string label, int maxGroups, uint requiredSubgroupSize = 0)
    {
        _device = device;
        _maxGroups = maxGroups;
        _kernel = requiredSubgroupSize == 0 ? device.CreateComputeKernel(source, label) : device.CreateComputeKernel(source, label, requiredSubgroupSize);
        if (_kernel.Bindings.Count != Buffers || _kernel.PushConstants.Size != pushWords * 4)
        {
            var kernel = _kernel;
            kernel.Dispose();
            throw new InvalidOperationException($"kernel {label}: the shader declares {_kernel.Bindings.Count} bindings and {_kernel.PushConstants.Size} bytes of push constants, the host expects {Buffers} and {pushWords * 4}");
        }
    }

    /// <summary>How many bind groups this pipeline has built, for tests of the cache.</summary>
    public int Created { get; private set; }

    public static int Total(params ConvPipeline?[][] sets)
    {
        var n = 0;
        foreach (var set in sets)
        {
            foreach (var p in set)
            {
                n += p?.Created ?? 0;
            }
        }

        return n;
    }

    public void Record(GpuCommandEncoder encoder, ReadOnlySpan<uint> push, uint groupsX, uint groupsY, uint groupsZ, GpuBuffer b0, GpuBuffer b1, GpuBuffer b2, GpuBuffer b3, GpuBuffer b4)
    {
        if (groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            return;
        }

        var key = (b0, b1, b2, b3, b4);
        if (!_groups.TryGetValue(key, out var group))
        {
            if (_groups.Count >= _maxGroups)
            {
                ReleaseBindings();
            }

            _entries[0] = BindGroupEntry.ForBuffer(0, b0);
            _entries[1] = BindGroupEntry.ForBuffer(1, b1);
            _entries[2] = BindGroupEntry.ForBuffer(2, b2);
            _entries[3] = BindGroupEntry.ForBuffer(3, b3);
            _entries[4] = BindGroupEntry.ForBuffer(4, b4);
            group = _kernel.CreateBindGroup(_device, 0, _entries);
            _groups.Add(key, group);
            Created++;
        }

        var pass = encoder.BeginComputePass();
        pass.SetPipeline(_kernel.Pipeline);
        pass.SetBindGroup(0, group);
        pass.SetPushConstants(0, MemoryMarshal.AsBytes(push));
        pass.Dispatch(groupsX, groupsY, groupsZ);
        pass.EndPass();
    }

    /// <summary>Drops the cached bind groups; the next call with a buffer set builds its group again.</summary>
    public void ReleaseBindings()
    {
        foreach (var g in _groups.Values)
        {
            g.Dispose();
        }

        _groups.Clear();
    }

    public void Dispose()
    {
        ReleaseBindings();
        _kernel.Dispose();
    }
}
