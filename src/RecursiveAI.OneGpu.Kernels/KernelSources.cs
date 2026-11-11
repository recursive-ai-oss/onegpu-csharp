using OneShader.Layouts;

namespace RecursiveAI.OneGpu.Kernels;

internal static class KernelSources
{
    /// <summary>Gathers the members the shader build generates for one entry point into a <see cref="ShaderSource"/>.</summary>
    public static ShaderSource Make(string entryPoint, ReadOnlySpan<byte> spirV, ReadOnlySpan<byte> msl, string? mslFunction, ReadOnlySpan<byte> wgsl, string? wgslEntry, EntryLayout layout) =>
        new(ShaderStage.Compute, entryPoint)
        {
            SpirV = spirV,
            Msl = msl,
            MslFunctionName = mslFunction,
            Wgsl = wgsl,
            WgslEntryPoint = wgslEntry,
            Layout = layout,
        };
}
