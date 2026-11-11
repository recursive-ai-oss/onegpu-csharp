// Output loop shared by the gather-style kernels: output element j is a copy of source element Src(j), or zero when Src fails (an index
// out of range, already recorded in the status word). The including file defines Data, Out, P (with Count, OffOut and GridX) and Src.
// ELEM selects the element size: W8 bytes and W16 halves (one thread owns one 32-bit word, so two threads never write the same word), W32 or W64.

void RunGather(uint2 gid, uint tid)
{
    uint t = (gid.y * P.GridX + gid.x) * WG + tid;
#if ELEM_W8
    // Bytes (uint8 and int8) are packed four to a 32-bit word like bool tensors; a thread owns one output word and fills only the bytes
    // that belong to the output, so neighbours never write the same word. Surplus threads stay off the buffer, as for halves.
    if (t >= BoolWords(P.OffOut, P.Count))
    {
        return;
    }

    uint word = (P.OffOut >> 2) + t;
    uint res = Out[word];
    [unroll]
    for (uint k = 0u; k < 4u; k++)
    {
        uint b = word * 4u + k;
        if (b >= P.OffOut && b < P.OffOut + P.Count)
        {
            uint s;
            uint v = 0u;
            if (Src(b - P.OffOut, s))
            {
                v = RawB(Data, s);
            }

            res = WithByte(res, k, v);
        }
    }

    Out[word] = res;
#elif ELEM_W16
    // Surplus threads must not touch the buffer: a browser clamps an out-of-range index to the last word, and a surplus thread would then
    // write that word back with stale contents.
    if (t >= HalfWords(P.OffOut, P.Count))
    {
        return;
    }

    uint word = (P.OffOut >> 1) + t;
    uint res = Out[word];
    [unroll]
    for (uint k = 0u; k < 2u; k++)
    {
        uint h = word * 2u + k;
        if (h >= P.OffOut && h < P.OffOut + P.Count)
        {
            uint s;
            uint v = 0u;
            if (Src(h - P.OffOut, s))
            {
                v = RawH(Data, s);
            }

            res = (res & ~(0xFFFFu << (k * 16u))) | (v << (k * 16u));
        }
    }

    Out[word] = res;
#elif ELEM_W64
    if (t < P.Count)
    {
        uint s;
        uint lo = 0u;
        uint hi = 0u;
        if (Src(t, s))
        {
            lo = Data[2u * s];
            hi = Data[2u * s + 1u];
        }

        Out[2u * (P.OffOut + t)] = lo;
        Out[2u * (P.OffOut + t) + 1u] = hi;
    }
#else
    if (t < P.Count)
    {
        uint s;
        uint v = 0u;
        if (Src(t, s))
        {
            v = Data[s];
        }

        Out[P.OffOut + t] = v;
    }
#endif
}
