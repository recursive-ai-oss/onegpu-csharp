using RecursiveAI.OneGpu.Vulkan.Interop;

namespace RecursiveAI.OneGpu.Vulkan;

/// <summary>
/// Sends each validation-layer message to the device it concerns. Devices share one Vulkan instance and so one debug messenger, and the
/// layer's callback says only which objects a message names (<c>VkDebugUtilsMessengerCallbackDataEXT.pObjects</c>), so a device records the
/// handles it creates here, and a message goes to the owner of the first object it names. The layer calls back on the thread that made the
/// Vulkan call, so a device also marks the thread while it creates or destroys its <c>VkDevice</c>: messages raised there (the report of
/// objects still alive at <c>vkDestroyDevice</c>, for one) belong to it whatever objects they name. A message that names nothing known
/// goes to every live device, because dropping it would hide a real problem.
/// </summary>
internal static class ValidationRouter
{
    // Open-addressing table of (type, handle) -> owner, so that recording ownership of a new object allocates nothing until the table grows
    // (a warm session creates command buffers, which must not allocate managed memory). Entries are never removed one at a time: a device
    // leaving rebuilds the table without its entries.
    static readonly object TableLock = new();
    static VkObjectType[] _types = new VkObjectType[4096];
    static ulong[] _handles = new ulong[4096];
    static ValidationSink?[] _sinks = new ValidationSink?[4096];
    static int _count;

    static readonly List<ValidationSink> Live = [];

    [ThreadStatic]
    static ValidationSink? _scope;

    /// <summary>Starts routing messages for a device.</summary>
    public static void Register(ValidationSink sink)
    {
        lock (Live)
        {
            Live.Add(sink);
        }
    }

    /// <summary>Records that <paramref name="handle"/> belongs to the device with this sink. A handle the driver reuses simply changes owner.</summary>
    public static void Own(VkObjectType type, ulong handle, ValidationSink sink)
    {
        lock (TableLock)
        {
            if ((_count + 1) * 2 > _sinks.Length)
            {
                Rebuild(_sinks.Length * 2, null);
            }

            Insert(type, handle, sink);
        }
    }

    static int Slot(VkObjectType type, ulong handle, int length)
    {
        var h = (handle ^ ((ulong)type << 40)) * 0x9E3779B97F4A7C15UL;
        return (int)(h >> 40) & (length - 1);
    }

    static void Insert(VkObjectType type, ulong handle, ValidationSink sink)
    {
        var i = Slot(type, handle, _sinks.Length);
        while (_sinks[i] is not null && !(_handles[i] == handle && _types[i] == type))
        {
            i = (i + 1) & (_sinks.Length - 1);
        }

        if (_sinks[i] is null)
        {
            _count++;
        }

        _types[i] = type;
        _handles[i] = handle;
        _sinks[i] = sink;
    }

    static ValidationSink? Find(VkObjectType type, ulong handle)
    {
        lock (TableLock)
        {
            var i = Slot(type, handle, _sinks.Length);
            while (_sinks[i] is not null)
            {
                if (_handles[i] == handle && _types[i] == type)
                {
                    return _sinks[i];
                }

                i = (i + 1) & (_sinks.Length - 1);
            }

            return null;
        }
    }

    static void Rebuild(int length, ValidationSink? without)
    {
        var types = _types;
        var handles = _handles;
        var sinks = _sinks;
        _types = new VkObjectType[length];
        _handles = new ulong[length];
        _sinks = new ValidationSink?[length];
        _count = 0;
        for (var i = 0; i < sinks.Length; i++)
        {
            if (sinks[i] is { } sink && !ReferenceEquals(sink, without))
            {
                Insert(types[i], handles[i], sink);
            }
        }
    }

    /// <summary>Stops routing to a device and forgets its handles.</summary>
    public static void Forget(ValidationSink sink)
    {
        lock (Live)
        {
            Live.Remove(sink);
        }

        lock (TableLock)
        {
            Rebuild(_sinks.Length, sink);
        }
    }

    /// <summary>Marks the calling thread as working for a device until the scope is disposed.</summary>
    public static Scope Enter(ValidationSink? sink)
    {
        var previous = _scope;
        _scope = sink;
        return new Scope(previous);
    }

    internal readonly struct Scope(ValidationSink? previous) : IDisposable
    {
        public void Dispose() => _scope = previous;
    }

    /// <summary>Delivers one message.</summary>
    public static unsafe void Route(VkDebugUtilsMessengerCallbackDataEXT* data, string text)
    {
        var target = _scope;
        if (target is null)
        {
            for (var i = 0; i < data->objectCount && target is null; i++)
            {
                var o = data->pObjects[i];
                target = Find(o.objectType, o.objectHandle);
            }
        }

        if (target is not null)
        {
            target.Add(text);
            return;
        }

        lock (Live)
        {
            foreach (var s in Live)
            {
                s.Add(text);
            }
        }
    }
}
