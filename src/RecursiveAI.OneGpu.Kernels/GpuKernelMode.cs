namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Which variant of a kernel family to run.</summary>
public enum GpuKernelMode
{
    /// <summary>The quickest variant for the device. Reduction order and the use of subgroup arithmetic are the kernel's choice, so results can differ in the last bits between devices and drivers.</summary>
    Fast = 0,

    /// <summary>
    /// A fixed reduction order that does not depend on occupancy, scheduling or subgroup size, and no floating-point atomics or subgroup
    /// arithmetic: two runs on the same device and driver give identical bits.
    /// </summary>
    Deterministic = 1,
}
