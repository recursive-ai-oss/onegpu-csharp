using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// A compute pipeline with up to eight storage-buffer bindings and a three-dimensional grid, plus the bind groups made for it. Like
/// <see cref="KernelPipeline"/>, offsets and sizes travel in push constants, so repeated calls with the same buffers allocate nothing.
/// </summary>
internal sealed class WideKernelPipeline : IDisposable
{
    const int MaxBuffers = 8;

    readonly GpuDevice _device;
    readonly GpuComputeKernel _kernel;
    readonly Dictionary<BindKey, GpuBindGroup> _groups = [];
    readonly BindGroupEntry[] _entries;
    readonly int _span;
    readonly int _maxGroups;

    public WideKernelPipeline(GpuDevice device, in ShaderSource source, int pushWords, string label, int maxGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        _device = device;
        _maxGroups = maxGroups;
        _kernel = device.CreateComputeKernel(source, label);
        if (_kernel.PushConstants.Size != pushWords * 4 || _kernel.Bindings.Count > MaxBuffers)
        {
            var kernel = _kernel;
            kernel.Dispose();
            throw new InvalidOperationException($"kernel {label}: the shader declares {_kernel.Bindings.Count} bindings and {_kernel.PushConstants.Size} bytes of push constants, the host expects at most {MaxBuffers} and {pushWords * 4}");
        }

        _entries = new BindGroupEntry[_kernel.Bindings.Count];
        _span = 0;
        foreach (var b in _kernel.Bindings)
        {
            _span = Math.Max(_span, (int)b.Binding + 1);
        }
    }

    public GpuComputeKernel Kernel => _kernel;

    /// <summary>How many bind groups this pipeline has built, for tests of the cache.</summary>
    public int Created { get; private set; }

    /// <summary>Sums <see cref="Created"/> over pipelines that may not exist yet.</summary>
    public static int Total(params WideKernelPipeline?[][] sets)
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

    /// <summary>Drops the cached bind groups; the next call with a buffer set builds its group again.</summary>
    public void ReleaseBindings()
    {
        foreach (var g in _groups.Values)
        {
            g.Dispose();
        }

        _groups.Clear();
    }

    /// <summary>Records one pass. <paramref name="buffers"/> is indexed by the shader's binding number; slots the shader does not declare are ignored.</summary>
    public void Record(GpuCommandEncoder encoder, ReadOnlySpan<uint> push, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<GpuBuffer> buffers)
    {
        if (groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            return;
        }

        if (buffers.Length < _span)
        {
            throw new ArgumentException($"the kernel needs {_span} buffers, got {buffers.Length}", nameof(buffers));
        }

        var key = new BindKey(buffers);
        if (!_groups.TryGetValue(key, out var group))
        {
            if (_groups.Count >= _maxGroups)
            {
                foreach (var g in _groups.Values)
                {
                    g.Dispose();
                }

                _groups.Clear();
            }

            var i = 0;
            foreach (var b in _kernel.Bindings)
            {
                _entries[i++] = BindGroupEntry.ForBuffer(b.Binding, buffers[(int)b.Binding]);
            }

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

    public void Dispose()
    {
        ReleaseBindings();
        _kernel.Dispose();
    }

    [InlineArray(MaxBuffers)]
    struct Slots
    {
        GpuBuffer? _first;
    }

    readonly struct BindKey : IEquatable<BindKey>
    {
        readonly Slots _slots;
        readonly int _count;

        public BindKey(ReadOnlySpan<GpuBuffer> buffers)
        {
            _count = Math.Min(buffers.Length, MaxBuffers);
            for (var i = 0; i < _count; i++)
            {
                _slots[i] = buffers[i];
            }
        }

        public bool Equals(BindKey o)
        {
            if (_count != o._count)
            {
                return false;
            }

            for (var i = 0; i < _count; i++)
            {
                if (!ReferenceEquals(_slots[i], o._slots[i]))
                {
                    return false;
                }
            }

            return true;
        }

        public override bool Equals(object? obj) => obj is BindKey k && Equals(k);

        public override int GetHashCode()
        {
            var h = new HashCode();
            for (var i = 0; i < _count; i++)
            {
                h.Add(RuntimeHelpers.GetHashCode(_slots[i]));
            }

            return h.ToHashCode();
        }
    }
}
