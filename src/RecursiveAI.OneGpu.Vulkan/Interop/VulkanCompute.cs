using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu.Vulkan.Interop;

// Declarations for the compute-related queries, written by hand from the Vulkan registry (vk.xml) because the generated file
// holds only what the graphics path uses. Field order and types follow the registry; VkBool32 is a uint here as in the generated file.

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupProperties
{
    public VkStructureType sType;
    public void* pNext;
    public uint subgroupSize;
    public uint supportedStages;
    public uint supportedOperations;
    public uint quadOperationsInAllStages;
    public VkPhysicalDeviceSubgroupProperties() { sType = VkStructureType.PhysicalDeviceSubgroupProperties; }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupSizeControlProperties
{
    public VkStructureType sType;
    public void* pNext;
    public uint minSubgroupSize;
    public uint maxSubgroupSize;
    public uint maxComputeWorkgroupSubgroups;
    public uint requiredSubgroupSizeStages;
    public VkPhysicalDeviceSubgroupSizeControlProperties() { sType = VkStructureType.PhysicalDeviceSubgroupSizeControlProperties; }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMaintenance3Properties
{
    public VkStructureType sType;
    public void* pNext;
    public uint maxPerSetDescriptors;
    public ulong maxMemoryAllocationSize;
    public VkPhysicalDeviceMaintenance3Properties() { sType = VkStructureType.PhysicalDeviceMaintenance3Properties; }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR
{
    public VkStructureType sType;
    public void* pNext;
    public uint cooperativeMatrix;
    public uint cooperativeMatrixRobustBufferAccess;
    public VkPhysicalDeviceCooperativeMatrixFeaturesKHR() { sType = VkStructureType.PhysicalDeviceCooperativeMatrixFeatures; }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceShaderBfloat16FeaturesKHR
{
    public VkStructureType sType;
    public void* pNext;
    public uint shaderBFloat16Type;
    public uint shaderBFloat16DotProduct;
    public uint shaderBFloat16CooperativeMatrix;
    public VkPhysicalDeviceShaderBfloat16FeaturesKHR() { sType = VkStructureType.PhysicalDeviceShaderBfloat16Features; }
}

/// <summary>One entry of <c>vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR</c>. The component types are VkComponentTypeKHR, the scope VkScopeKHR.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCooperativeMatrixPropertiesKHR
{
    public VkStructureType sType;
    public void* pNext;
    public uint MSize;
    public uint NSize;
    public uint KSize;
    public int AType;
    public int BType;
    public int CType;
    public int ResultType;
    public uint saturatingAccumulation;
    public int scope;
    public VkCooperativeMatrixPropertiesKHR() { sType = VkStructureType.CooperativeMatrixProperties; }
}

/// <summary>Instance-level functions for the extensions above, loaded by name because the generated tables do not carry them.</summary>
internal sealed unsafe class VkComputeFns
{
    public delegate* unmanaged<VkPhysicalDevice, uint*, VkCooperativeMatrixPropertiesKHR*, VkResult> vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;

    public static VkComputeFns Load(nint instance)
    {
        var fns = new VkComputeFns();
        fns.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = (delegate* unmanaged<VkPhysicalDevice, uint*, VkCooperativeMatrixPropertiesKHR*, VkResult>)
            VulkanLoader.GetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0"u8.GetPointer());
        return fns;
    }
}
