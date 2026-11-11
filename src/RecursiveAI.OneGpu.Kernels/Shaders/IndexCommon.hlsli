// Index handling shared by the gather and scatter kernels (ONNX Gather, GatherElements, GatherND, ScatterND, ScatterElements).
//
// An index tensor holds int32 or int64 values, read as 32-bit words (an int64 is a low and a high word). A negative index counts from the
// end of its axis. An index outside [-len, len) does not touch memory: the thread records the position of the offending index in a
// status word with an integer minimum (so the smallest position wins whatever the scheduling) and the host turns it into an error.

#include "Common.hlsl"

#define STATUS_NONE 0xFFFFFFFFu

// Reads index i. wide selects int64 storage. False when an int64 does not fit an int32, which no axis can be that long.
bool ReadIndex(StructuredBuffer<uint> idx, uint i, bool wide, out int v)
{
    if (wide)
    {
        uint lo = idx[2u * i];
        uint hi = idx[2u * i + 1u];
        v = (int)lo;
        return hi == (((int)lo < 0) ? 0xFFFFFFFFu : 0u);
    }

    v = (int)idx[i];
    return true;
}

// Maps a possibly negative index into [0, len); false when it is outside [-len, len).
bool Normalize(int v, uint len, out uint r)
{
    r = 0u;
    if (v >= 0)
    {
        r = (uint)v;
        return r < len;
    }

    uint a = 0u - (uint)v;
    if (a > len)
    {
        return false;
    }

    r = len - a;
    return true;
}
