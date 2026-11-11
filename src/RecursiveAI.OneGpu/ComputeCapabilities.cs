using System.Text;
using OneShader.Reflection;

namespace RecursiveAI.OneGpu;

/// <summary>Subgroup operations a device supports in compute shaders. The values match Vulkan's <c>VkSubgroupFeatureFlagBits</c>.</summary>
[Flags]
public enum GpuSubgroupOperations
{
    /// <summary>No subgroup operation.</summary>
    None = 0,
    /// <summary>Lane index, lane count and first-lane election.</summary>
    Basic = 1,
    /// <summary>Any, all and all-equal votes.</summary>
    Vote = 2,
    /// <summary>Sums, minimums, maximums, bit operations and prefix scans.</summary>
    Arithmetic = 4,
    /// <summary>Ballots and bit counting.</summary>
    Ballot = 8,
    /// <summary>Reading a value from any lane.</summary>
    Shuffle = 16,
    /// <summary>Reading a value from a lane at a fixed distance.</summary>
    ShuffleRelative = 32,
    /// <summary>Reductions over clusters of lanes.</summary>
    Clustered = 64,
    /// <summary>Reads across a 2x2 quad of lanes.</summary>
    Quad = 128,
}

/// <summary>The element types a cooperative matrix may hold. The values match Vulkan's <c>VkComponentTypeKHR</c> for the ones it defines.</summary>
public enum GpuScalarType
{
    /// <summary>16-bit IEEE float.</summary>
    F16 = 0,
    /// <summary>32-bit IEEE float.</summary>
    F32 = 1,
    /// <summary>64-bit IEEE float.</summary>
    F64 = 2,
    /// <summary>Signed 8-bit integer.</summary>
    I8 = 3,
    /// <summary>Signed 16-bit integer.</summary>
    I16 = 4,
    /// <summary>Signed 32-bit integer.</summary>
    I32 = 5,
    /// <summary>Signed 64-bit integer.</summary>
    I64 = 6,
    /// <summary>Unsigned 8-bit integer.</summary>
    U8 = 7,
    /// <summary>Unsigned 16-bit integer.</summary>
    U16 = 8,
    /// <summary>Unsigned 32-bit integer.</summary>
    U32 = 9,
    /// <summary>Unsigned 64-bit integer.</summary>
    U64 = 10,
    /// <summary>16-bit brain float (8 exponent bits, 7 mantissa bits).</summary>
    BF16 = 1000141000,
    /// <summary>Signed 8-bit integers packed four to a 32-bit word.</summary>
    I8Packed = 1000491000,
    /// <summary>Unsigned 8-bit integers packed four to a 32-bit word.</summary>
    U8Packed = 1000491001,
    /// <summary>8-bit float with 4 exponent bits and 3 mantissa bits (E4M3).</summary>
    F8E4M3 = 1000491002,
    /// <summary>8-bit float with 5 exponent bits and 2 mantissa bits (E5M2).</summary>
    F8E5M2 = 1000491003,
}

/// <summary>One matrix multiply-accumulate shape a device runs on dedicated matrix hardware: <c>Result = A × B + C</c>, with A of M×K, B of K×N and C and Result of M×N.</summary>
/// <param name="M">Rows of A, C and the result.</param>
/// <param name="N">Columns of B, C and the result.</param>
/// <param name="K">Columns of A and rows of B.</param>
/// <param name="AType">Element type of A.</param>
/// <param name="BType">Element type of B.</param>
/// <param name="CType">Element type of C.</param>
/// <param name="ResultType">Element type of the result.</param>
/// <param name="Saturating">The accumulation saturates instead of wrapping (integer types).</param>
public readonly record struct GpuCooperativeMatrixShape(uint M, uint N, uint K, GpuScalarType AType, GpuScalarType BType, GpuScalarType CType, GpuScalarType ResultType, bool Saturating = false)
{
    /// <inheritdoc />
    public override string ToString() => $"{M}x{N}x{K} {AType}*{BType}+{CType}->{ResultType}{(Saturating ? " saturating" : "")}";
}

/// <summary>
/// What a device offers compute shaders beyond the portable baseline, and its compute limits. Every value comes from the graphics API's own
/// query on the adapter; nothing is assumed from the vendor or the API version. Zero means the API did not report the value.
/// </summary>
public sealed record GpuComputeCapabilities
{
    /// <summary>Nothing beyond the baseline and no limits known. Backends replace it with what they query.</summary>
    public static GpuComputeCapabilities None { get; } = new();

    /// <summary>The subgroup (SIMD group, wave, warp) size shaders usually run with, in lanes; 0 when unknown.</summary>
    public uint SubgroupSize { get; init; }

    /// <summary>Smallest subgroup size the device can run a compute shader with, in lanes; 0 when unknown.</summary>
    public uint SubgroupMinSize { get; init; }

    /// <summary>Largest subgroup size the device can run a compute shader with, in lanes; 0 when unknown.</summary>
    public uint SubgroupMaxSize { get; init; }

    /// <summary>Subgroup operation classes compute shaders may use.</summary>
    public GpuSubgroupOperations SubgroupOperations { get; init; }

    /// <summary>
    /// Scalar types and storage widths shaders may use, in the vocabulary of the shader compiler's reflection
    /// (<c>EntryLayout.RequiredFeatures</c>). A kernel whose features are not a subset of this cannot run here.
    /// </summary>
    public DeviceFeatures ShaderFeatures { get; init; }

    /// <summary>8-bit integers in storage buffers and in shader arithmetic.</summary>
    public bool Integer8 { get; init; }

    /// <summary>Packed 4×8-bit integer dot products with 32-bit accumulation (<c>dot4add</c>).</summary>
    public bool Int8DotProduct { get; init; }

    /// <summary>The 16-bit brain float type in shaders.</summary>
    public bool BFloat16 { get; init; }

    /// <summary>Matrix multiply-accumulate shapes the device runs on matrix hardware (tensor cores, simdgroup matrices); empty when it has none.</summary>
    public IReadOnlyList<GpuCooperativeMatrixShape> CooperativeMatrices { get; init; } = [];

    /// <summary>The device has matrix hardware reachable from shaders (<see cref="CooperativeMatrices"/> is not empty).</summary>
    public bool CooperativeMatrix => CooperativeMatrices.Count > 0;

    /// <summary>Largest workgroup size in x.</summary>
    public uint MaxWorkgroupSizeX { get; init; }

    /// <summary>Largest workgroup size in y.</summary>
    public uint MaxWorkgroupSizeY { get; init; }

    /// <summary>Largest workgroup size in z.</summary>
    public uint MaxWorkgroupSizeZ { get; init; }

    /// <summary>Most invocations (the product of the workgroup's three sizes) in one workgroup.</summary>
    public uint MaxWorkgroupInvocations { get; init; }

    /// <summary>Most workgroups a dispatch may launch along one axis.</summary>
    public uint MaxWorkgroupCount { get; init; }

    /// <summary>Most bytes of workgroup (shared, threadgroup) memory one workgroup may declare.</summary>
    public uint MaxWorkgroupMemoryBytes { get; init; }

    /// <summary>Most bytes a single storage buffer binding may expose to a shader.</summary>
    public ulong MaxStorageBufferBindingBytes { get; init; }

    /// <summary>Most bytes a single buffer may hold.</summary>
    public ulong MaxBufferBytes { get; init; }

    /// <summary>Most storage buffers one compute stage may bind.</summary>
    public uint MaxStorageBuffersPerStage { get; init; }

    /// <summary>Most bind group sets a pipeline layout may use.</summary>
    public uint MaxBindGroups { get; init; }

    /// <summary>A storage buffer binding that starts inside a buffer (<see cref="BindGroupEntry.Offset"/>) must start at a multiple of this many bytes; the backend reports the hardware's own requirement, or a safe bound where the API gives none. <see cref="GpuDevice.DynamicOffsetAlignment"/> still applies to dynamic offsets.</summary>
    public uint StorageBufferOffsetAlignment { get; init; }

    /// <summary>Describes the capabilities in a few lines of plain text, for logs and test output.</summary>
    public string Describe()
    {
        var sb = new StringBuilder();
        sb.Append("subgroup: size ").Append(SubgroupSize).Append(", range ").Append(SubgroupMinSize).Append('-').Append(SubgroupMaxSize).Append(", ops ").Append(SubgroupOperations).AppendLine();
        sb.Append("shader features: ").Append(ShaderFeatures).Append("; int8 ").Append(Integer8).Append(", int8 dot ").Append(Int8DotProduct).Append(", bfloat16 ").Append(BFloat16).AppendLine();
        sb.Append("cooperative matrix: ");
        if (CooperativeMatrices.Count == 0)
        {
            sb.Append("none");
        }
        else
        {
            for (var i = 0; i < CooperativeMatrices.Count; i++)
            {
                sb.Append(i == 0 ? "" : "; ").Append(CooperativeMatrices[i]);
            }
        }

        sb.AppendLine();
        sb.Append("workgroup: size ").Append(MaxWorkgroupSizeX).Append('x').Append(MaxWorkgroupSizeY).Append('x').Append(MaxWorkgroupSizeZ)
          .Append(", invocations ").Append(MaxWorkgroupInvocations).Append(", count ").Append(MaxWorkgroupCount).Append(", memory ").Append(MaxWorkgroupMemoryBytes).AppendLine(" B");
        sb.Append("buffers: binding ").Append(MaxStorageBufferBindingBytes).Append(" B, buffer ").Append(MaxBufferBytes).Append(" B, storage per stage ").Append(MaxStorageBuffersPerStage)
          .Append(", bind groups ").Append(MaxBindGroups).Append(", offset alignment ").Append(StorageBufferOffsetAlignment);
        return sb.ToString();
    }
}
