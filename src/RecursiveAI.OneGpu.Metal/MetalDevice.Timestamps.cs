using RecursiveAI.OneGpu.Metal.Interop;

namespace RecursiveAI.OneGpu.Metal;

// GPU timestamps come from a counter sample buffer: a pass names two slots of it in its pass descriptor and the GPU writes its own clock there when
// the pass's work starts and ends (Metal's "stage boundary" sampling, which Apple GPUs support; they cannot sample between draws or dispatches).
// The samples are ticks of the GPU clock. The device reports a CPU and a GPU timestamp taken together (sampleTimestamps:gpuTimestamp:), and two
// such pairs a few milliseconds apart give the number of CPU-clock nanoseconds per GPU tick.
internal sealed unsafe partial class MetalDevice
{
    /// <summary>Largest timestamp set, as on the other backends.</summary>
    const uint MaxTimestampQueries = 8192;

    nint _timestampCounterSet;
    long _tickNumerator;
    long _tickDenominator;

    /// <summary>Looks for the timestamp counter set and for stage-boundary sampling; true when passes can be timed on this device.</summary>
    bool FindTimestampCounterSet()
    {
        if (!Msg.Flag(_device, Sel.supportsCounterSampling_, MT.SamplingAtStageBoundary))
        {
            return false;
        }

        using var pool = new Pool();
        var sets = Msg.Obj(_device, Sel.counterSets);
        var n = sets == 0 ? 0 : (int)Msg.UL(sets, Sel.count);
        for (var i = 0; i < n; i++)
        {
            var set = Msg.Obj(sets, Sel.objectAtIndex_, (nuint)i);
            // MTLCommonCounterSetTimestamp
            if (Objc.ToManaged(Msg.Obj(set, Sel.name)) == "timestamp")
            {
                _timestampCounterSet = Objc.objc_retain(set);
                return true;
            }
        }

        return false;
    }

    /// <summary>Measures how many CPU-clock nanoseconds one GPU tick lasts, once.</summary>
    void CalibrateTimestamps()
    {
        if (_tickDenominator != 0)
        {
            return;
        }

        ulong cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
        Msg.Void(_device, Sel.sampleTimestamps_gpuTimestamp_, (nint)(&cpu0), (nint)(&gpu0));
        Thread.Sleep(10);
        Msg.Void(_device, Sel.sampleTimestamps_gpuTimestamp_, (nint)(&cpu1), (nint)(&gpu1));
        var dCpu = (long)(cpu1 - cpu0);
        var dGpu = (long)(gpu1 - gpu0);
        if (dCpu <= 0 || dGpu <= 0)
        {
            // A clock that did not advance cannot be scaled; report raw ticks rather than invent a rate.
            dCpu = dGpu = 1;
        }

        _tickNumerator = dCpu;
        _tickDenominator = dGpu;
    }

    /// <summary>Nanoseconds per tick of the GPU clock, as measured against the CPU clock; 0 until the first timestamp set is created.</summary>
    internal double NanosecondsPerTick => _tickDenominator == 0 ? 0 : (double)_tickNumerator / _tickDenominator;

    /// <inheritdoc />
    public override GpuTimestampSet CreateTimestampSet(uint count)
    {
        ThrowIfDisposed();
        if (!_caps.Timestamps)
        {
            throw new GpuUnsupportedException($"{Adapter.Name} cannot sample GPU timestamps at pass boundaries");
        }

        if (count is 0 or > MaxTimestampQueries)
        {
            throw new ArgumentOutOfRangeException(nameof(count), $"a timestamp set holds 1 to {MaxTimestampQueries} queries");
        }

        CalibrateTimestamps();
        using var pool = new Pool();
        var desc = Msg.Obj(Cls.MTLCounterSampleBufferDescriptor, Sel.@new);
        try
        {
            Msg.Void(desc, Sel.setCounterSet_, _timestampCounterSet);
            Msg.Void(desc, Sel.setStorageMode_, MT.StorageShared);
            Msg.Void(desc, Sel.setSampleCount_, (nuint)count);
            nint error = 0;
            var buffer = Msg.Obj(_device, Sel.newCounterSampleBufferWithDescriptor_error_, desc, (nint)(&error));
            if (buffer == 0)
            {
                throw new GpuException("Metal could not create a timestamp sample buffer: " + Objc.ErrorText(error));
            }

            return new MetalTimestampSet(buffer, count);
        }
        finally
        {
            Objc.objc_release(desc);
        }
    }

    /// <inheritdoc />
    public override bool TryReadTimestamps(GpuTimestampSet timestamps, uint first, Span<ulong> nanoseconds)
    {
        ThrowIfDisposed();
        var set = timestamps as MetalTimestampSet ?? throw new ArgumentException("timestamp set belongs to another backend", nameof(timestamps));
        if ((ulong)first + (ulong)nanoseconds.Length > set.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(nanoseconds), "read extends past the end of the timestamp set");
        }

        if (nanoseconds.Length == 0)
        {
            return true;
        }

        for (var i = 0; i < nanoseconds.Length; i++)
        {
            if (!set.IsWritten(first + (uint)i))
            {
                return false;
            }
        }

        // The samples are final once the last submission that records into the set has completed.
        Poll();
        if (_inflight.Count > 0 && _inflight.Peek().Serial <= set.PendingSerial)
        {
            return false;
        }

        // Read the set from its start: a pass that did no GPU work leaves its slots empty, and filling them in needs the slots before and beside them.
        var extent = (int)set.Extent;
        if (_timestampScratch.Length < extent)
        {
            _timestampScratch = new ulong[extent];
        }

        using var pool = new Pool();
        var data = Msg.Obj(set.Buffer, Sel.resolveCounterRange_, new NSRange(0, (nuint)extent));
        if (data == 0 || Msg.UL(data, Sel.length) < (nuint)(extent * sizeof(ulong)))
        {
            return false;
        }

        var ticks = (ulong*)Msg.Obj(data, Sel.bytes);
        ulong lastSeen = 0;
        for (var i = 0; i < extent; i++)
        {
            if (!set.IsWritten((uint)i))
            {
                _timestampScratch[i] = 0;
                continue;
            }

            var raw = ticks[i];
            if (raw == MT.CounterErrorValue)
            {
                return false;
            }

            // Metal skips the sample of a stage that did no work (a render pass with no draws has no fragment end, an empty compute pass has
            // neither slot). Such a pass took no time: its empty slot takes the value of the other end of the pass, or else of the pass before.
            if (raw == 0)
            {
                var partner = ticks[set.PartnerOf((uint)i)];
                raw = partner != 0 && partner != MT.CounterErrorValue ? partner : lastSeen;
            }

            lastSeen = raw;
            _timestampScratch[i] = raw;
        }

        for (var i = 0; i < nanoseconds.Length; i++)
        {
            nanoseconds[i] = (ulong)((UInt128)_timestampScratch[first + (uint)i] * (ulong)_tickNumerator / (ulong)_tickDenominator);
        }

        return true;
    }

    ulong[] _timestampScratch = new ulong[64];
}
