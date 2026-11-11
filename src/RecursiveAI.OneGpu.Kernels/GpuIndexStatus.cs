namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// A one-word buffer in which the gather and scatter kernels record that an index was out of range. A kernel cannot throw from the
/// GPU, so the caller creates a status, passes it to the kernels, and asks it after the work has finished.
/// </summary>
/// <remarks>
/// An invalid index never touches memory outside the operand: a gather writes zero for that element and a scatter skips it. The status
/// holds the smallest position (index of the element in the indices tensor, as an integer minimum, so the answer does not depend on
/// scheduling) of any invalid index recorded since <see cref="Reset"/>. One status may serve several kernels; then <see cref="GpuKernelIndexException.Operation"/>
/// names the last operation that was recorded with it. Not thread-safe.
/// </remarks>
public sealed class GpuIndexStatus : IDisposable
{
    const uint None = uint.MaxValue;

    readonly GpuBuffer<uint> _buffer;
    string? _operation;

    /// <summary>Creates a status, ready to record.</summary>
    /// <param name="device">The device the kernels run on.</param>
    public GpuIndexStatus(GpuDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _buffer = device.CreateBuffer<uint>(1, label: "index status");
        Reset();
    }

    internal GpuBuffer<uint> Buffer => _buffer;

    internal void Note(string operation) => _operation = operation;

    /// <summary>Clears the record. The write happens before the next submitted commands, so call it before submitting the kernels it should watch.</summary>
    public void Reset() => _buffer.Write([None]);

    /// <summary>Waits for the GPU to finish the commands submitted so far, reads the status and throws if an index was invalid.</summary>
    /// <exception cref="GpuKernelIndexException">An index was outside its axis.</exception>
    public async ValueTask ThrowIfFailedAsync()
    {
        var word = new uint[1];
        await _buffer.ReadAsync(word);
        Throw(word[0]);
    }

    /// <summary>The blocking form of <see cref="ThrowIfFailedAsync"/>; not available in a browser.</summary>
    /// <exception cref="GpuKernelIndexException">An index was outside its axis.</exception>
    public void ThrowIfFailed()
    {
        Span<uint> word = stackalloc uint[1];
        _buffer.Read(word);
        Throw(word[0]);
    }

    void Throw(uint position)
    {
        if (position != None)
        {
            throw new GpuKernelIndexException(_operation ?? "index operation", position);
        }
    }

    /// <summary>Releases the buffer.</summary>
    public void Dispose() => _buffer.Dispose();
}

/// <summary>An index given to a gather or scatter was outside its axis (after a negative index counted from the end).</summary>
public sealed class GpuKernelIndexException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="operation">The operation, for example <c>Gather</c>.</param>
    /// <param name="position">Position of the offending index in the indices tensor, counting its elements in row-major order.</param>
    public GpuKernelIndexException(string operation, long position)
        : base($"{operation}: the index at position {position} of the indices is outside its axis")
    {
        Operation = operation;
        Position = position;
    }

    /// <summary>Creates the exception for a check made on the host, with the index value and the axis length.</summary>
    /// <param name="operation">The operation.</param>
    /// <param name="position">Position of the offending index.</param>
    /// <param name="index">The index value.</param>
    /// <param name="axisLength">The length of the axis it addresses.</param>
    public GpuKernelIndexException(string operation, long position, long index, long axisLength)
        : base($"{operation}: the index {index} at position {position} is outside [{-axisLength}, {axisLength})")
    {
        Operation = operation;
        Position = position;
        Index = index;
        AxisLength = axisLength;
    }

    /// <summary>The operation that met the index.</summary>
    public string Operation { get; }

    /// <summary>Position of the offending index in the indices tensor.</summary>
    public long Position { get; }

    /// <summary>The index value when it is known (host-side checks), otherwise null.</summary>
    public long? Index { get; }

    /// <summary>The length of the axis the index addresses when known, otherwise null.</summary>
    public long? AxisLength { get; }

    /// <summary>
    /// Checks indices on the host, in safe code, with the same rule the kernels apply: each index must lie in [-axisLength, axisLength).
    /// Use it when the indices are known on the host before the upload, so a bad value fails there with its value instead of after the GPU ran.
    /// </summary>
    /// <param name="operation">The operation name for the message.</param>
    /// <param name="indices">The indices.</param>
    /// <param name="axisLength">Length of the axis they address.</param>
    /// <exception cref="GpuKernelIndexException">An index is out of range.</exception>
    public static void Validate(string operation, ReadOnlySpan<long> indices, long axisLength)
    {
        for (var i = 0; i < indices.Length; i++)
        {
            if (indices[i] < -axisLength || indices[i] >= axisLength)
            {
                throw new GpuKernelIndexException(operation, i, indices[i], axisLength);
            }
        }
    }

    /// <summary>The <see cref="int"/> form of <see cref="Validate(string, ReadOnlySpan{long}, long)"/>.</summary>
    /// <param name="operation">The operation name for the message.</param>
    /// <param name="indices">The indices.</param>
    /// <param name="axisLength">Length of the axis they address.</param>
    /// <exception cref="GpuKernelIndexException">An index is out of range.</exception>
    public static void Validate(string operation, ReadOnlySpan<int> indices, long axisLength)
    {
        for (var i = 0; i < indices.Length; i++)
        {
            if (indices[i] < -axisLength || indices[i] >= axisLength)
            {
                throw new GpuKernelIndexException(operation, i, indices[i], axisLength);
            }
        }
    }
}
