using System.Runtime.InteropServices;

namespace RecursiveAI.OneGpu;

/// <summary>
/// A device buffer holding <see cref="Length"/> elements of an unmanaged type, with typed uploads and readbacks. It owns the underlying
/// <see cref="GpuBuffer"/>; bind <see cref="Buffer"/> to shaders.
/// </summary>
/// <typeparam name="T">A blittable element type. The layout must match the shader's: scalars, vectors and structs laid out like the HLSL structured buffer.</typeparam>
/// <remarks>Not thread-safe, like the device that made it. Disposing releases the buffer once the GPU has finished with it.</remarks>
public sealed class GpuBuffer<T> : IDisposable
    where T : unmanaged
{
    // One probe element: the element size of an unmanaged type is read from a one-element array, which needs no pointer code.
    static readonly int ElementSize = MemoryMarshal.AsBytes<T>(new T[1]).Length;

    readonly GpuDevice _device;

    internal GpuBuffer(GpuDevice device, GpuBuffer buffer, int length)
    {
        _device = device;
        Buffer = buffer;
        Length = length;
    }

    /// <summary>The untyped buffer to bind in a <see cref="BindGroupEntry"/> or copy with an encoder.</summary>
    public GpuBuffer Buffer { get; }

    /// <summary>Number of elements.</summary>
    public int Length { get; }

    /// <summary>Size in bytes.</summary>
    public ulong SizeInBytes => Buffer.Size;

    /// <summary>Size of one element in bytes.</summary>
    public int ElementBytes => ElementSize;

    internal static int ElementSizeOf => ElementSize;

    /// <summary>The device that owns the buffer.</summary>
    public GpuDevice Device => _device;

    /// <summary>Queues an upload that runs before the next submitted commands. Needs <see cref="BufferUsage.CopyDestination"/>.</summary>
    /// <param name="data">The elements to write.</param>
    /// <param name="elementOffset">Index of the first element written.</param>
    /// <exception cref="ArgumentOutOfRangeException">The write extends past the end of the buffer.</exception>
    public void Write(ReadOnlySpan<T> data, int elementOffset = 0)
    {
        CheckRange(elementOffset, data.Length);
        _device.WriteBuffer(Buffer, ByteOffset(elementOffset), MemoryMarshal.AsBytes(data));
    }

    /// <summary>Waits for all submitted work, then copies elements out. Blocks; use <see cref="ReadAsync"/> in a render loop or a browser. Needs <see cref="BufferUsage.CopySource"/>.</summary>
    /// <param name="destination">Receives the elements; its length is the number read.</param>
    /// <param name="elementOffset">Index of the first element read.</param>
    /// <exception cref="ArgumentOutOfRangeException">The read extends past the end of the buffer.</exception>
    public void Read(Span<T> destination, int elementOffset = 0)
    {
        CheckRange(elementOffset, destination.Length);
        _device.ReadBuffer(Buffer, ByteOffset(elementOffset), MemoryMarshal.AsBytes(destination));
    }

    /// <summary>
    /// Copies elements out once the GPU has finished the work submitted so far, without waiting for the device: the task completes when the
    /// bytes are in <paramref name="destination"/>, and other submissions keep running meanwhile. Needs <see cref="BufferUsage.CopySource"/>.
    /// </summary>
    /// <param name="destination">Receives the elements. It must stay valid and unused until the task completes.</param>
    /// <param name="elementOffset">Index of the first element read.</param>
    /// <exception cref="ArgumentOutOfRangeException">The read extends past the end of the buffer.</exception>
    /// <remarks>Allocates one small object per call; read into a reused <c>Memory&lt;byte&gt;</c> through <see cref="GpuDevice.ReadBufferAsync"/> to avoid that.</remarks>
    public ValueTask ReadAsync(Memory<T> destination, int elementOffset = 0)
    {
        CheckRange(elementOffset, destination.Length);
        return _device.ReadBufferAsync(Buffer, ByteOffset(elementOffset), new TypedMemory<T>(destination).Memory);
    }

    /// <summary>Releases the buffer once the GPU has finished with it.</summary>
    public void Dispose() => Buffer.Dispose();

    static ulong ByteOffset(int elementOffset) => (ulong)elementOffset * (ulong)ElementSize;

    void CheckRange(int elementOffset, int count)
    {
        if (elementOffset < 0 || count < 0 || (long)elementOffset + count > Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count), $"elements {elementOffset}..{(long)elementOffset + count} lie outside the buffer of {Length}");
        }
    }
}
