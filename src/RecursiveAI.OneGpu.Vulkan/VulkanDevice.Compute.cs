using System.Collections.Concurrent;
using OneShader.Reflection;
using RecursiveAI.OneGpu.Vulkan.Interop;

namespace RecursiveAI.OneGpu.Vulkan;

internal sealed unsafe partial class VulkanDevice
{
    ReadbackWaiter? _waiter;
    readonly ConcurrentQueue<Staging> _readbackDone = new();
    readonly ConcurrentStack<VulkanReadback> _readbackPool = new();

    // ---- capabilities ----

    /// <summary>Reads the compute capabilities from the adapter's own limits, features and extension properties.</summary>
    GpuComputeCapabilities QueryCompute(VulkanInstance.AdapterEntry info, in VkPhysicalDeviceLimits limits, in VkPhysicalDeviceFeatures features,
        in VkPhysicalDeviceVulkan11Features f11, in VkPhysicalDeviceVulkan12Features f12, in VkPhysicalDeviceVulkan13Features f13,
        bool nonUniformIndexing, bool cooperativeMatrix, bool bfloat16, bool bfloat16CooperativeMatrix)
    {
        var sub = new VkPhysicalDeviceSubgroupProperties();
        var size = new VkPhysicalDeviceSubgroupSizeControlProperties { pNext = &sub };
        var m3 = new VkPhysicalDeviceMaintenance3Properties { pNext = &size };
        var p2 = new VkPhysicalDeviceProperties2 { pNext = &m3 };
        _instanceFns.vkGetPhysicalDeviceProperties2(_physical, &p2);

        // Subgroup operations count for compute only when the adapter lists the compute stage as supporting them.
        var ops = (sub.supportedStages & VkShaderStageFlagBits.Compute) != 0 ? (GpuSubgroupOperations)((uint)sub.supportedOperations & 0xFF) : GpuSubgroupOperations.None;

        var shader = DeviceFeatures.None;
        if (f12.shaderFloat16 != 0)
        {
            shader |= DeviceFeatures.Float16;
        }

        if (features.shaderInt16 != 0)
        {
            shader |= DeviceFeatures.Int16;
        }

        if (features.shaderFloat64 != 0)
        {
            shader |= DeviceFeatures.Float64;
        }

        if (features.shaderInt64 != 0)
        {
            shader |= DeviceFeatures.Int64;
        }

        if (f11.storageBuffer16BitAccess != 0)
        {
            shader |= DeviceFeatures.StorageBuffer16;
        }

        if (f11.uniformAndStorageBuffer16BitAccess != 0)
        {
            shader |= DeviceFeatures.UniformBuffer16;
        }

        if (f11.storagePushConstant16 != 0)
        {
            shader |= DeviceFeatures.PushConstant16;
        }

        if (nonUniformIndexing)
        {
            shader |= DeviceFeatures.NonUniformIndexing;
        }

        var matrices = new List<GpuCooperativeMatrixShape>();
        if (cooperativeMatrix)
        {
            var fns = _instanceFns;
            uint n = 0;
            if (fns.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR != null && fns.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(_physical, &n, null) >= 0 && n > 0)
            {
                var list = new VkCooperativeMatrixPropertiesKHR[n];
                for (var i = 0; i < list.Length; i++)
                {
                    list[i] = new VkCooperativeMatrixPropertiesKHR();
                }

                fixed (VkCooperativeMatrixPropertiesKHR* lp = list)
                {
                    if (fns.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(_physical, &n, lp) >= 0)
                    {
                        for (var i = 0; i < n; i++)
                        {
                            // Subgroup scope: the matrices a compute shader's subgroup multiplies cooperatively.
                            if (lp[i].scope == VkScopeKHR.Subgroup)
                            {
                                matrices.Add(new GpuCooperativeMatrixShape(lp[i].MSize, lp[i].NSize, lp[i].KSize, (GpuScalarType)(int)lp[i].AType, (GpuScalarType)(int)lp[i].BType,
                                    (GpuScalarType)(int)lp[i].CType, (GpuScalarType)(int)lp[i].ResultType, lp[i].saturatingAccumulation != 0));
                            }
                        }
                    }
                }
            }
        }

        // The names are the shader compiler's: a kernel's reflection lists these bits and CreateComputeKernel checks them against this set.
        if (matrices.Count > 0)
        {
            shader |= DeviceFeatures.CooperativeMatrix;
            if (f12.vulkanMemoryModel != 0)
            {
                shader |= DeviceFeatures.VulkanMemoryModel;
            }
        }

        if (bfloat16)
        {
            shader |= DeviceFeatures.BFloat16;
            if (bfloat16CooperativeMatrix)
            {
                shader |= DeviceFeatures.BFloat16CooperativeMatrix;
            }
        }

        if (f13.shaderIntegerDotProduct != 0)
        {
            shader |= DeviceFeatures.IntegerDotProduct;
        }

        var counts = limits.maxComputeWorkGroupCount;
        return new GpuComputeCapabilities
        {
            SubgroupSize = sub.subgroupSize,
            SubgroupMinSize = size.minSubgroupSize != 0 ? size.minSubgroupSize : sub.subgroupSize,
            SubgroupMaxSize = size.maxSubgroupSize != 0 ? size.maxSubgroupSize : sub.subgroupSize,
            SubgroupOperations = ops,
            FullSubgroups = f13.computeFullSubgroups != 0,
            SubgroupSizeControl = f13.subgroupSizeControl != 0 && (size.requiredSubgroupSizeStages & VkShaderStageFlagBits.Compute) != 0,
            ShaderFeatures = shader,
            Integer8 = f12.shaderInt8 != 0 && f12.storageBuffer8BitAccess != 0,
            Int8DotProduct = f13.shaderIntegerDotProduct != 0,
            BFloat16 = bfloat16,
            CooperativeMatrices = matrices,
            MaxWorkgroupSizeX = limits.maxComputeWorkGroupSize[0],
            MaxWorkgroupSizeY = limits.maxComputeWorkGroupSize[1],
            MaxWorkgroupSizeZ = limits.maxComputeWorkGroupSize[2],
            MaxWorkgroupInvocations = limits.maxComputeWorkGroupInvocations,
            MaxWorkgroupCount = Math.Min(counts[0], Math.Min(counts[1], counts[2])),
            MaxWorkgroupMemoryBytes = limits.maxComputeSharedMemorySize,
            MaxStorageBufferBindingBytes = limits.maxStorageBufferRange,
            MaxBufferBytes = m3.maxMemoryAllocationSize,
            MaxStorageBuffersPerStage = limits.maxPerStageDescriptorStorageBuffers,
            MaxBindGroups = limits.maxBoundDescriptorSets,
            StorageBufferOffsetAlignment = (uint)limits.minStorageBufferOffsetAlignment,
        };
    }

    // ---- asynchronous readback ----

    /// <summary>Copies a buffer range out once the GPU has finished the work submitted so far; only a helper thread waits for the GPU.</summary>
    public override ValueTask ReadBufferAsync(GpuBuffer buffer, ulong offset, Memory<byte> destination)
    {
        var b = (VulkanBuffer)buffer;
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

        var st = AcquireStaging((ulong)destination.Length, true);
        var enc = (VulkanEncoder)BeginCommands();
        enc.CopyBufferToRaw(b, offset, st.Buffer, (ulong)destination.Length);
        Submit(enc);

        if (!_readbackPool.TryPop(out var op))
        {
            op = new VulkanReadback(this);
        }

        op.Start(st, _serial, destination);
        (_waiter ??= new ReadbackWaiter()).Enqueue(op);
        return op.AsValueTask();
    }

    /// <summary>Takes back the readback memory the waiter thread has finished with.</summary>
    void CollectReadbacks()
    {
        while (_readbackDone.TryDequeue(out var st))
        {
            ReleaseStaging(st);
        }
    }

    void StopReadbackWaiter()
    {
        _waiter?.Dispose();
        _waiter = null;
        CollectReadbacks();
    }

    sealed class VulkanReadback(VulkanDevice device) : ReadbackOperation
    {
        Staging? _staging;
        ulong _serial;
        int _length;

        public void Start(Staging staging, ulong serial, Memory<byte> destination)
        {
            _staging = staging;
            _serial = serial;
            _length = destination.Length;
            Begin(destination);
        }

        protected override void WaitForGpu()
        {
            var sem = device._timeline;
            var serial = _serial;
            var wi = new VkSemaphoreWaitInfo { semaphoreCount = 1, pSemaphores = &sem, pValues = &serial };
            device.Check(device._deviceFns.vkWaitSemaphores(device._device, &wi, ulong.MaxValue), "vkWaitSemaphores");
        }

        protected override void CopyOut(Span<byte> destination) =>
            new ReadOnlySpan<byte>(_staging!.Alloc.Mapped, _length).CopyTo(destination);

        protected override void Release()
        {
            var st = _staging;
            _staging = null;
            if (st is not null)
            {
                device._readbackDone.Enqueue(st);
            }
        }

        protected override void Recycle() => device._readbackPool.Push(this);
    }
}
