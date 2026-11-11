namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// How many bind groups a kernel family keeps. A bind group is built the first time a family sees a set of buffers and reused every time
/// the same buffers come back, so a warm call allocates nothing. When a family has seen more sets than its limit it drops all of them and
/// starts again, which costs one bind group per set on the next pass: size the limit above the number of distinct buffer sets one run touches.
/// </summary>
public static class GpuKernelBindings
{
    /// <summary>
    /// The default limit of every family. A transformer with a dozen layers and a handful of weights each touches under a hundred
    /// buffer sets per matrix-multiply family; 1024 leaves room for several models sharing a family, at a few hundred bytes a set.
    /// </summary>
    public const int DefaultMaxCachedBindGroups = 1024;

    internal static int Check(int maxCachedBindGroups)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxCachedBindGroups, 1);
        return maxCachedBindGroups;
    }
}
