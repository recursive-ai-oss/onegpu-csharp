using System.Buffers;
using System.Collections.Concurrent;
using System.Threading.Tasks.Sources;

namespace RecursiveAI.OneGpu;

/// <summary>
/// One asynchronous readback in flight. A backend subclasses it to say how to wait for its submission and how to copy the bytes out; the
/// shared waiter thread runs it, so the thread that submitted work is never blocked and the device keeps running other submissions.
/// </summary>
internal abstract class ReadbackOperation : IValueTaskSource
{
    ManualResetValueTaskSourceCore<bool> _core = new() { RunContinuationsAsynchronously = true };

    /// <summary>Where the bytes go. Set by the backend before the operation is queued.</summary>
    protected Memory<byte> Destination { get; private set; }

    /// <summary>Prepares a (possibly reused) operation for a new readback.</summary>
    protected void Begin(Memory<byte> destination) => Destination = destination;

    /// <summary>Blocks the waiter thread until the GPU has finished the copy into the backend's readback memory.</summary>
    protected abstract void WaitForGpu();

    /// <summary>Copies the finished bytes into <paramref name="destination"/> (on the waiter thread).</summary>
    protected abstract void CopyOut(Span<byte> destination);

    /// <summary>Hands the backend's readback memory back (on the waiter thread, after the copy or a failure). Must only touch thread-safe state.</summary>
    protected abstract void Release();

    /// <summary>Returns the operation to the backend's pool once its result has been observed.</summary>
    protected virtual void Recycle()
    {
    }

    /// <summary>Starts waiting; the awaitable completes when the bytes are in the destination.</summary>
    public ValueTask AsValueTask() => new(this, _core.Version);

    /// <summary>Runs on the waiter thread.</summary>
    public void Run()
    {
        Exception? failure = null;
        try
        {
            WaitForGpu();
            CopyOut(Destination.Span);
        }
        catch (Exception e)
        {
            failure = e;
        }
        finally
        {
            try
            {
                Release();
            }
            catch (Exception e)
            {
                failure ??= e;
            }
        }

        if (failure is null)
        {
            _core.SetResult(true);
        }
        else
        {
            _core.SetException(failure);
        }
    }

    void IValueTaskSource.GetResult(short token)
    {
        try
        {
            _core.GetResult(token);
        }
        finally
        {
            _core.Reset();
            Destination = default;
            Recycle();
        }
    }

    ValueTaskSourceStatus IValueTaskSource.GetStatus(short token) => _core.GetStatus(token);

    void IValueTaskSource.OnCompleted(Action<object?> continuation, object? state, short token, ValueTaskSourceOnCompletedFlags flags) =>
        _core.OnCompleted(continuation, state, token, flags);
}

/// <summary>A background thread that runs readback operations in the order they were queued. Submissions finish in order on every backend, so one thread suffices.</summary>
internal sealed class ReadbackWaiter : IDisposable
{
    readonly ConcurrentQueue<ReadbackOperation> _queue = new();
    readonly SemaphoreSlim _signal = new(0);
    Thread? _thread;
    volatile bool _stop;

    /// <summary>Queues an operation; starts the thread on first use.</summary>
    public void Enqueue(ReadbackOperation operation)
    {
        if (_thread is null)
        {
            _thread = new Thread(Loop) { IsBackground = true, Name = "onegpu readback" };
            _thread.Start();
        }

        _queue.Enqueue(operation);
        _signal.Release();
    }

    void Loop()
    {
        while (true)
        {
            _signal.Wait();
            if (_queue.TryDequeue(out var op))
            {
                op.Run();
            }
            else if (_stop)
            {
                return;
            }
        }
    }

    /// <summary>Finishes the queued operations and stops the thread. The device must be idle first, or this waits for the GPU.</summary>
    public void Dispose()
    {
        _stop = true;
        _signal.Release();
        _thread?.Join();
        _signal.Dispose();
    }
}

/// <summary>Presents a typed memory as bytes so the byte-oriented readback can fill it.</summary>
internal sealed class TypedMemory<T> : MemoryManager<byte>
    where T : unmanaged
{
    readonly Memory<T> _memory;

    public TypedMemory(Memory<T> memory) => _memory = memory;

    public override Span<byte> GetSpan() => System.Runtime.InteropServices.MemoryMarshal.AsBytes(_memory.Span);

    // The readback only needs spans; handing out raw pointers would need unsafe code in a package that has none.
    public override MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException("pinning is not supported for typed readback destinations");

    public override void Unpin()
    {
    }

    protected override void Dispose(bool disposing)
    {
    }
}
