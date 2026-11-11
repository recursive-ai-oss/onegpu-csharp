namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Which guarantee a kernel gives about its result.</summary>
public enum GpuKernelMode
{
    /// <summary>
    /// The fastest variant the device supports. The order of floating-point operations may depend on the variant, the device and the
    /// scheduling, so results can differ in the last bits between variants and devices.
    /// </summary>
    Fast = 0,

    /// <summary>
    /// A variant whose floating-point accumulation order is fixed by the kernel and does not depend on scheduling, occupancy or the
    /// subgroup size: no floating-point atomics, no data-dependent split of a reduction, no subgroup arithmetic whose order the driver
    /// may choose. Two runs on one device and driver give identical bits. Results may differ between devices and drivers.
    /// </summary>
    Deterministic = 1,
}
