// Applies one update to the output. Included after the including file declares Upd (StructuredBuffer<uint>) and Out (RWStructuredBuffer<uint>).

// out[o] = combine(out[o], upd[u]); o and u are element indices. Declared by the including file: Upd (StructuredBuffer<uint>), Out
// (RWStructuredBuffer<uint>), the op and the float flag.
void Apply(uint o, uint u, uint op, bool isFloat)
{
#if ELEM_W16
    uint sh = (o & 1u) << 4;
    uint curH = (Out[o >> 1] >> sh) & 0xFFFFu;
    uint updH = RawH(Upd, u);
    uint res = updH;
    if (op != 0u)
    {
        float a = H2F(curH);
        float b = H2F(updH);
        float r = a;
        if (op == 1u) { r = a + b; }
        else if (op == 2u) { r = a * b; }
        else if (op == 3u) { r = (b < a || IsNaN(b)) ? b : a; }
        else { r = (b > a || IsNaN(b)) ? b : a; }
        res = F2H(r);
    }

    InterlockedAnd(Out[o >> 1], ~(0xFFFFu << sh));
    InterlockedOr(Out[o >> 1], res << sh);
#elif ELEM_W64
    uint lo = Out[2u * o];
    uint hi = Out[2u * o + 1u];
    Combine64(lo, hi, Upd[2u * u], Upd[2u * u + 1u], op);
    Out[2u * o] = lo;
    Out[2u * o + 1u] = hi;
#else
    Out[o] = Combine32(Out[o], Upd[u], op, isFloat);
#endif
}
