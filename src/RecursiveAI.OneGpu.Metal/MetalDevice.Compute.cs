using System.Collections.Concurrent;
using OneShader.Reflection;
using RecursiveAI.OneGpu.Metal.Interop;

namespace RecursiveAI.OneGpu.Metal;

internal sealed unsafe partial class MetalDevice
{
    // MTLLanguageVersion values: major in the high 16 bits, minor in the low 16.
    const nuint Msl30 = 3u << 16;
    const nuint Msl31 = (3u << 16) | 1;

    ReadbackWaiter? _waiter;
    readonly ConcurrentStack<MetalReadback> _readbackPool = new();
    long _liveBufferBytes;
    int _liveBuffers;

    // ---- memory tally ----

    internal void BufferCreated(ulong size)
    {
        Interlocked.Add(ref _liveBufferBytes, (long)size);
        Interlocked.Increment(ref _liveBuffers);
    }

    internal void BufferReleased(ulong size)
    {
        Interlocked.Add(ref _liveBufferBytes, -(long)size);
        Interlocked.Decrement(ref _liveBuffers);
    }

    /// <summary>
    /// Reserved is what the device reports as allocated to this process (all resources, including textures and staging); used and allocations count the
    /// live buffers this backend created, so compute buffers are covered exactly and textures are not itemised.
    /// </summary>
    public override GpuMemoryStats MemoryStats => new(Msg.UL(_device, Sel.currentAllocatedSize), (ulong)Math.Max(0, Interlocked.Read(ref _liveBufferBytes)), Volatile.Read(ref _liveBuffers));

    // ---- capabilities ----

    const string Preamble = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n";

    /// <summary>Compiles a one-kernel library; returns the compute pipeline state (owned by the caller) or 0 when this device's compiler rejects the source.</summary>
    nint TryProbe(string body, nuint languageVersion)
    {
        var source = Preamble + "kernel void probe(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n" + body + "\n}\n";
        return TryCompile(source, languageVersion);
    }

    nint TryCompile(string source, nuint languageVersion)
    {
        using var pool = new Pool();
        var options = Msg.Obj(Cls.MTLCompileOptions, Sel.@new);
        Msg.Void(options, Sel.setFastMathEnabled_, (byte)0);
        Msg.Void(options, Sel.setLanguageVersion_, languageVersion);
        nint error = 0;
        var library = Msg.Obj(_device, Sel.newLibraryWithSource_options_error_, Objc.NSString(source), options, (nint)(&error));
        Objc.objc_release(options);
        if (library == 0)
        {
            return 0;
        }

        var function = Msg.Obj(library, Sel.newFunctionWithName_, Objc.NSString("probe"));
        nint pipeline = 0;
        if (function != 0)
        {
            error = 0;
            pipeline = Msg.Obj(_device, Sel.newComputePipelineStateWithFunction_error_, function, (nint)(&error));
            Objc.objc_release(function);
        }

        Objc.objc_release(library);
        return pipeline;
    }

    bool Probes(string body, nuint languageVersion = Msl30)
    {
        var pipeline = TryProbe(body, languageVersion);
        if (pipeline == 0)
        {
            return false;
        }

        Objc.objc_release(pipeline);
        return true;
    }

    /// <summary>
    /// Reads what Metal reports (limits, SIMD-group width of a compiled pipeline) and asks the shader compiler whether each optional language feature
    /// builds on this device, because Metal has no feature list for them: a feature counts when a kernel using it compiles.
    /// </summary>
    GpuComputeCapabilities QueryCompute()
    {
        using var pool = new Pool();
        var maxThreads = Msg.Ret<MTLSize>(_device, Sel.maxThreadsPerThreadgroup);
        uint simdWidth = 0;
        uint invocations = 0;
        var basic = TryProbe("out[i] = lane + i;", Msl30);
        if (basic != 0)
        {
            simdWidth = (uint)Msg.UL(basic, Sel.threadExecutionWidth);
            invocations = (uint)Msg.UL(basic, Sel.maxTotalThreadsPerThreadgroup);
            Objc.objc_release(basic);
        }

        var ops = GpuSubgroupOperations.None;
        if (simdWidth != 0)
        {
            ops |= GpuSubgroupOperations.Basic;
            ops |= Probes("out[i] = (uint)simd_all(i > 3) + (uint)simd_any(i > 3);") ? GpuSubgroupOperations.Vote : 0;
            ops |= Probes("out[i] = simd_sum(i) + simd_prefix_exclusive_sum(i) + simd_max(i);") ? GpuSubgroupOperations.Arithmetic : 0;
            ops |= Probes("out[i] = (uint)static_cast<ulong>(simd_ballot(i > 1));") ? GpuSubgroupOperations.Ballot : 0;
            ops |= Probes("out[i] = simd_shuffle(i, 1);") ? GpuSubgroupOperations.Shuffle : 0;
            ops |= Probes("out[i] = simd_shuffle_up(i, 1) + simd_shuffle_down(i, 1);") ? GpuSubgroupOperations.ShuffleRelative : 0;
            ops |= Probes("out[i] = quad_shuffle(i, 1);") ? GpuSubgroupOperations.Quad : 0;
        }

        // Every Apple GPU computes in half precision and in 8-, 16- and 64-bit integers; the 16-bit storage and push-constant forms are plain bytes to Metal.
        var features = DeviceFeatures.Float16 | DeviceFeatures.Int16 | DeviceFeatures.Int64 | DeviceFeatures.StorageBuffer16 | DeviceFeatures.UniformBuffer16
            | DeviceFeatures.PushConstant16 | DeviceFeatures.NonUniformIndexing;

        var matrices = new List<GpuCooperativeMatrixShape>();
        foreach (var (a, b, c, sa, sb, sc) in new[]
        {
            ("half", "half", "half", GpuScalarType.F16, GpuScalarType.F16, GpuScalarType.F16),
            ("half", "half", "float", GpuScalarType.F16, GpuScalarType.F16, GpuScalarType.F32),
            ("float", "float", "float", GpuScalarType.F32, GpuScalarType.F32, GpuScalarType.F32),
            ("bfloat", "bfloat", "float", GpuScalarType.BF16, GpuScalarType.BF16, GpuScalarType.F32),
        })
        {
            var body = $"simdgroup_matrix<{a},8,8> ma = simdgroup_matrix<{a},8,8>(1); simdgroup_matrix<{b},8,8> mb = simdgroup_matrix<{b},8,8>(1); " +
                $"simdgroup_matrix<{c},8,8> mc = simdgroup_matrix<{c},8,8>(0); simdgroup_multiply_accumulate(mc, ma, mb, mc); " +
                $"simdgroup_store(mc, (device {c}*)out, 8); (void)lane;";
            if (Probes(body, a == "bfloat" ? Msl31 : Msl30))
            {
                matrices.Add(new GpuCooperativeMatrixShape(8, 8, 8, sa, sb, sc, sc));
            }
        }

        var bfloat16 = Probes("bfloat v = bfloat(i) * bfloat(2.0); out[i] = (uint)float(v);", Msl31);

        // The names are the shader compiler's: a kernel's reflection lists these bits and CreateComputeKernel checks them against this set.
        // Metal has no separate memory model to enable, so the Vulkan memory model bit that comes with cooperative matrices is met.
        if (matrices.Count > 0)
        {
            features |= DeviceFeatures.CooperativeMatrix | DeviceFeatures.VulkanMemoryModel;
        }

        if (bfloat16)
        {
            features |= DeviceFeatures.BFloat16;
            if (matrices.Exists(m => m.AType == GpuScalarType.BF16))
            {
                features |= DeviceFeatures.BFloat16CooperativeMatrix;
            }
        }

        var invocationLimit = invocations != 0 ? invocations : 1024;
        return new GpuComputeCapabilities
        {
            SubgroupSize = simdWidth,
            SubgroupMinSize = simdWidth,
            SubgroupMaxSize = simdWidth,
            SubgroupOperations = ops,
            FullSubgroups = simdWidth != 0,
            SubgroupSizeControl = false,
            ShaderFeatures = features,
            Integer8 = true,
            Int8DotProduct = false,
            BFloat16 = bfloat16,
            CooperativeMatrices = matrices,
            MaxWorkgroupSizeX = (uint)Math.Min(maxThreads.Width, uint.MaxValue),
            MaxWorkgroupSizeY = (uint)Math.Min(maxThreads.Height, uint.MaxValue),
            MaxWorkgroupSizeZ = (uint)Math.Min(maxThreads.Depth, uint.MaxValue),
            MaxWorkgroupInvocations = invocationLimit,
            MaxWorkgroupCount = uint.MaxValue,
            MaxWorkgroupMemoryBytes = (uint)Msg.UL(_device, Sel.maxThreadgroupMemoryLength),
            MaxStorageBufferBindingBytes = Msg.UL(_device, Sel.maxBufferLength),
            MaxBufferBytes = Msg.UL(_device, Sel.maxBufferLength),
            MaxStorageBuffersPerStage = 31,
            SubmissionWorkBudget = SubmissionWorkBudget,
            MaxBindGroups = 4,
            StorageBufferOffsetAlignment = 256,
        };
    }

    // ---- asynchronous readback ----

    /// <inheritdoc />
    public override ValueTask ReadBufferAsync(GpuBuffer buffer, ulong offset, Memory<byte> destination)
    {
        var b = buffer as MetalBuffer ?? throw new ArgumentException("buffer belongs to another backend", nameof(buffer));
        if ((b.Usage & BufferUsage.CopySource) == 0)
        {
            throw new ArgumentException("buffer was not created with BufferUsage.CopySource", nameof(buffer));
        }

        if (offset + (ulong)destination.Length > b.Size)
        {
            throw new ArgumentOutOfRangeException(nameof(destination), "read extends past the end of the buffer");
        }

        if (destination.IsEmpty)
        {
            return ValueTask.CompletedTask;
        }

        using var pool = new Pool();
        FlushUploads();
        var staging = Msg.Obj(_device, Sel.newBufferWithLength_options_, (nuint)destination.Length, MT.StorageShared);
        if (staging == 0)
        {
            throw new GpuOutOfMemoryException("could not allocate a readback buffer of " + destination.Length + " bytes", (ulong)destination.Length);
        }

        var cb = Msg.Obj(_queue, Sel.commandBuffer);
        Objc.objc_retain(cb);
        var blit = Msg.Obj(cb, Sel.blitCommandEncoder);
        Msg.Void(blit, Sel.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_, b.Handle, (nuint)offset, staging, (nuint)0, (nuint)destination.Length);
        Msg.Void(blit, Sel.endEncoding);

        // The device's own list of in-flight command buffers holds one reference and releases it when the buffer finishes; the waiter keeps another.
        Objc.objc_retain(cb);
        Commit(cb, frame: false);

        if (!_readbackPool.TryPop(out var op))
        {
            op = new MetalReadback(this);
        }

        op.Start(cb, staging, destination);
        (_waiter ??= new ReadbackWaiter()).Enqueue(op);
        return op.AsValueTask();
    }

    sealed class MetalReadback(MetalDevice device) : ReadbackOperation
    {
        nint _commandBuffer;
        nint _staging;
        int _length;

        public void Start(nint commandBuffer, nint staging, Memory<byte> destination)
        {
            _commandBuffer = commandBuffer;
            _staging = staging;
            _length = destination.Length;
            Begin(destination);
        }

        protected override void WaitForGpu()
        {
            using var pool = new Pool();
            Msg.Void(_commandBuffer, Sel.waitUntilCompleted);
            if (Msg.UL(_commandBuffer, Sel.status) == MT.StatusError)
            {
                throw Describe(_commandBuffer, "a readback");
            }
        }

        protected override void CopyOut(Span<byte> destination) =>
            new ReadOnlySpan<byte>((void*)Msg.Obj(_staging, Sel.contents), _length).CopyTo(destination);

        protected override void Release()
        {
            Objc.objc_release(_staging);
            Objc.objc_release(_commandBuffer);
            _staging = 0;
            _commandBuffer = 0;
        }

        protected override void Recycle() => device._readbackPool.Push(this);
    }
}
