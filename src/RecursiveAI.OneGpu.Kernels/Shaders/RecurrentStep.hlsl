// One time step of an ONNX RNN, GRU or LSTM layer for every batch row and for both directions of a bidirectional layer.
//
// The input projection of all steps (Xw = X * W^T, one matrix product for the whole sequence) is done beforehand; this kernel forms the
// recurrent product Rec = H * R^T for the step and the gate arithmetic that follows, so a step is one dispatch and the dispatches are
// ordered by the barriers between passes. A workgroup owns U hidden units of one direction (U = ROWS / gates computed in the pass, so ROWS
// rows of R per workgroup) and all batch rows. For each row of R, LANES lanes (consecutive k, so loads coalesce) each sum k = lane, lane + LANES, ...
// in increasing order, and the partial sums are combined by a fixed halving tree in shared memory. Nothing depends on the subgroup size, the
// scheduling or the occupancy, and there are no atomics, so a result is bit-identical from run to run on one device.
//
// State lives in one float buffer: H (two copies, swapped every step so no workgroup reads what another writes), C, r*H and z of a GRU
// whose reset gate comes first, and the hidden states of all steps (Ys). Each element has one writer per step. Rows whose sequence has ended
// copy their state through unchanged.
//
// Operation order follows the ONNX definitions: pre = ((xw + rec) + Wb) + Rb, then the peephole terms, then the clip, then the activation;
// gate order in W, R and B is i, o, f, c (LSTM) and z, r, h (GRU). F16 reads R, B and P as packed halves (element 2k in the low half of word k).

#include "Functions.hlsl"

#define BT 8u
// Lanes that share one row of R, and so the rows a workgroup covers (256 / LANES). Fewer lanes per row means fewer workgroups per pass, each with more
// work, which is what hides the latency of a pass that is a few microseconds of arithmetic.
#define LANES 4u
#define ROWS (256u / LANES)

struct StepParams
{
    uint Hid;
    uint Batch;
    uint Seq;
    uint Step;
    uint Flags;      // bits 0-1 cell (0 RNN, 1 GRU, 2 LSTM), 2 linear_before_reset, 3 input_forget, 4 bias, 5 peepholes, 6 sequence lengths,
                     // 7 layout 1, 9 second GRU phase, 10 two directions, 12-13 direction d runs backwards
    float Clip;      // 0 for no clip
    uint Kinds;      // activation kinds, 4 bits each: direction 0 F, G, H in bits 0-11, direction 1 in bits 12-23
    uint OffXw;
    uint OffR;
    uint OffBias;
    uint OffPeep;
    uint OffLens;
    uint OffState;
    uint Gates;      // bits 0-3: gates formed by this pass, bits 4-7: the first of them
    uint V0;         // activation parameters (alpha, beta) for each activation of each direction, as float bits
    uint V1;
    uint V2;
    uint V3;
    uint V4;
    uint V5;
    uint V6;
    uint V7;
    uint V8;
    uint V9;
    uint V10;
    uint V11;
};

[[vk::push_constant]] ConstantBuffer<StepParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<float> Xw;
[[vk::binding(1, 0)]] StructuredBuffer<uint> R;
[[vk::binding(2, 0)]] StructuredBuffer<uint> Bias;
[[vk::binding(3, 0)]] StructuredBuffer<uint> Peep;
[[vk::binding(4, 0)]] StructuredBuffer<uint> Lens;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> State;

groupshared float Sh[256 * BT];

float Param(uint i)
{
    switch (i)
    {
        case 0u: return asfloat(P.V0);
        case 1u: return asfloat(P.V1);
        case 2u: return asfloat(P.V2);
        case 3u: return asfloat(P.V3);
        case 4u: return asfloat(P.V4);
        case 5u: return asfloat(P.V5);
        case 6u: return asfloat(P.V6);
        case 7u: return asfloat(P.V7);
        case 8u: return asfloat(P.V8);
        case 9u: return asfloat(P.V9);
        case 10u: return asfloat(P.V10);
        default: return asfloat(P.V11);
    }
}

// The activation a (0 f, 1 g, 2 h) of direction d. Definitions as in the ONNX operator documentation.
float Act(uint d, uint a, float x)
{
    uint kind = (P.Kinds >> (d * 12u + a * 4u)) & 15u;
    uint pi = (d * 3u + a) * 2u;
    float alpha = Param(pi);
    float beta = Param(pi + 1u);
    switch (kind)
    {
        case 0u: return Sigmoid(x);
        case 1u: return Tanh(x);
        case 2u: return ReluF(x);
        case 3u: return alpha * x + beta;
        case 4u: return x < 0.0 ? alpha * x : x;
        case 5u: return (x > alpha || IsNaN(x)) ? x : 0.0;
        case 6u: return alpha * Tanh(beta * x);
        case 7u:
        {
            float v = alpha * x + beta;
            return v > 1.0 ? 1.0 : (v < 0.0 ? 0.0 : v);
        }
        case 8u: return (x >= 0.0 || IsNaN(x)) ? x : alpha * (exp(x) - 1.0);
        case 9u: return x / (1.0 + abs(x));
        default: return (x > 20.0 || IsNaN(x)) ? x : log(1.0 + exp(x));
    }
}

float LoadR(uint i)
{
#if F16
    return LoadH(R, i);
#else
    return asfloat(R[i]);
#endif
}

float LoadBias(uint i)
{
#if F16
    return LoadH(Bias, i);
#else
    return asfloat(Bias[i]);
#endif
}

float LoadPeep(uint i)
{
#if F16
    return LoadH(Peep, i);
#else
    return asfloat(Peep[i]);
#endif
}

float ClipV(float v)
{
    if (P.Clip > 0.0)
    {
        return v > P.Clip ? P.Clip : (v < -P.Clip ? -P.Clip : v);
    }

    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void StepOp(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint d = gid.y;
    uint hid = P.Hid;
    uint batch = P.Batch;
    uint cell = P.Flags & 3u;
    bool lbr = (P.Flags & 4u) != 0u;
    bool inputForget = (P.Flags & 8u) != 0u;
    bool hasBias = (P.Flags & 16u) != 0u;
    bool hasPeep = (P.Flags & 32u) != 0u;
    bool hasLens = (P.Flags & 64u) != 0u;
    bool layout1 = (P.Flags & 128u) != 0u;
    bool phase1 = (P.Flags & 512u) != 0u;
    uint dirs = (P.Flags & 1024u) != 0u ? 2u : 1u;
    bool rev = ((P.Flags >> (12u + d)) & 1u) != 0u;
    uint totalGates = cell == 2u ? 4u : (cell == 1u ? 3u : 1u);
    uint gh = totalGates * hid;
    uint dbh = dirs * batch * hid;
    uint ng = P.Gates & 15u;
    uint gate0 = (P.Gates >> 4) & 15u;
    // ROWS rows of R per workgroup: U hidden units times the gates formed in this pass.
    uint U = ROWS / ng;
    uint j0 = gid.x * U;
    uint row = tid / LANES;
    uint lane = tid % LANES;
    uint rowUnit = row % U;
    uint rowGate = row / U;
    bool rowLive = row < U * ng && j0 + rowUnit < hid;

    uint hIn = P.OffState + (P.Step & 1u) * dbh;
    uint hOut = P.OffState + (1u - (P.Step & 1u)) * dbh;
    uint cOff = P.OffState + 2u * dbh;
    uint rhOff = P.OffState + 3u * dbh;
    uint zOff = P.OffState + 4u * dbh;
    uint yOff = P.OffState + 5u * dbh;
    // The second pass of a GRU with the reset gate applied before the product multiplies r * H instead of H.
    uint vecOff = phase1 ? rhOff : hIn;
    uint rRow = P.OffR + d * gh * hid + ((gate0 + rowGate) * hid + j0 + rowUnit) * hid;
    bool finishes = !(cell == 1u && !lbr && !phase1);
    uint xBase = P.OffXw + d * P.Seq * batch * gh;

    for (uint b0 = 0u; b0 < batch; b0 += BT)
    {
        uint nb = min(BT, batch - b0);
        float acc[BT];
        [unroll]
        for (uint i = 0u; i < BT; i++)
        {
            acc[i] = 0.0;
        }

        if (rowLive)
        {
            uint vBase = vecOff + (d * batch + b0) * hid;
            uint k = lane;
            // Four independent loads per lane in flight at a time (the sums still go in increasing k: k, k + LANES, k + 2 LANES, k + 3 LANES, then
            // the next block), which hides most of the memory latency that a one-load loop pays for every k.
            for (; k + 3u * LANES < hid; k += 4u * LANES)
            {
                float r0 = LoadR(rRow + k);
                float r1 = LoadR(rRow + k + LANES);
                float r2 = LoadR(rRow + k + 2u * LANES);
                float r3 = LoadR(rRow + k + 3u * LANES);
                [unroll]
                for (uint bi = 0u; bi < BT; bi++)
                {
                    if (bi < nb)
                    {
                        uint v = vBase + bi * hid + k;
                        float h0 = State[v];
                        float h1 = State[v + LANES];
                        float h2 = State[v + 2u * LANES];
                        float h3 = State[v + 3u * LANES];
                        acc[bi] += r0 * h0;
                        acc[bi] += r1 * h1;
                        acc[bi] += r2 * h2;
                        acc[bi] += r3 * h3;
                    }
                }
            }

            for (; k < hid; k += LANES)
            {
                float rv = LoadR(rRow + k);
                [unroll]
                for (uint bi = 0u; bi < BT; bi++)
                {
                    if (bi < nb)
                    {
                        acc[bi] += rv * State[vBase + bi * hid + k];
                    }
                }
            }
        }

        [unroll]
        for (uint i2 = 0u; i2 < BT; i2++)
        {
            Sh[tid * BT + i2] = acc[i2];
        }

        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint s = LANES / 2u; s > 0u; s >>= 1)
        {
            if (lane < s)
            {
                [unroll]
                for (uint i3 = 0u; i3 < BT; i3++)
                {
                    Sh[tid * BT + i3] += Sh[(tid + s) * BT + i3];
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Gate arithmetic: one thread per (unit, batch row); a workgroup with more of them than threads loops.
        for (uint item = tid; item < U * BT; item += 256u)
        {
            uint bi = item % BT;
            uint u = item / BT;
            uint j = j0 + u;
            if (bi < nb && j < hid)
            {
                uint b = b0 + bi;
                uint len = hasLens ? min(Lens[P.OffLens + b], P.Seq) : P.Seq;
                uint vi = (d * batch + b) * hid + j;
                float hPrev = State[hIn + vi];
                if (P.Step >= len)
                {
                    if (finishes)
                    {
                        State[hOut + vi] = hPrev;
                    }
                }
                else
                {
                    uint t = rev ? len - 1u - P.Step : P.Step;
                    uint xRow = layout1 ? b * P.Seq + t : t * batch + b;
                    uint xw = xBase + xRow * gh + j;
                    uint bB = P.OffBias + d * 2u * gh + j;
                    uint rB = bB + gh;
                    // Recurrent product of gate g (relative to this pass) for this unit and row.
                    #define REC(g) Sh[((g * U + u) * LANES) * BT + bi]
                    float hNew = 0.0;
                    if (cell == 2u)
                    {
                        float pi = Xw[xw] + REC(0u);
                        float po = Xw[xw + hid] + REC(1u);
                        float pf = Xw[xw + 2u * hid] + REC(2u);
                        float pc = Xw[xw + 3u * hid] + REC(3u);
                        if (hasBias)
                        {
                            pi = (pi + LoadBias(bB)) + LoadBias(rB);
                            po = (po + LoadBias(bB + hid)) + LoadBias(rB + hid);
                            pf = (pf + LoadBias(bB + 2u * hid)) + LoadBias(rB + 2u * hid);
                            pc = (pc + LoadBias(bB + 3u * hid)) + LoadBias(rB + 3u * hid);
                        }

                        float cPrev = State[cOff + vi];
                        uint pp = P.OffPeep + d * 3u * hid + j;
                        if (hasPeep)
                        {
                            pi += LoadPeep(pp) * cPrev;
                            pf += LoadPeep(pp + 2u * hid) * cPrev;
                        }

                        float gi = Act(d, 0u, ClipV(pi));
                        float gc = Act(d, 1u, ClipV(pc));
                        float gf = inputForget ? 1.0 - gi : Act(d, 0u, ClipV(pf));
                        float cNew = gf * cPrev + gi * gc;
                        if (hasPeep)
                        {
                            po += LoadPeep(pp + hid) * cNew;
                        }

                        float go = Act(d, 0u, ClipV(po));
                        hNew = go * Act(d, 2u, cNew);
                        State[cOff + vi] = cNew;
                    }
                    else if (cell == 1u)
                    {
                        if (!phase1)
                        {
                            float pz = Xw[xw] + REC(0u);
                            float pr = Xw[xw + hid] + REC(1u);
                            if (hasBias)
                            {
                                pz = (pz + LoadBias(bB)) + LoadBias(rB);
                                pr = (pr + LoadBias(bB + hid)) + LoadBias(rB + hid);
                            }

                            float gz = Act(d, 0u, ClipV(pz));
                            float gr = Act(d, 0u, ClipV(pr));
                            if (lbr)
                            {
                                // The candidate's recurrent product came from the same pass: ht = g(xw_h + r * (rec_h + Rb_h) + Wb_h).
                                float rec = REC(2u);
                                if (hasBias)
                                {
                                    rec += LoadBias(rB + 2u * hid);
                                }

                                float v = Xw[xw + 2u * hid] + gr * rec;
                                if (hasBias)
                                {
                                    v += LoadBias(bB + 2u * hid);
                                }

                                float cand = Act(d, 1u, ClipV(v));
                                hNew = (1.0 - gz) * cand + gz * hPrev;
                            }
                            else
                            {
                                // Keep z and r * H for the second pass, which needs r * H of every unit before it can form its product.
                                State[zOff + vi] = gz;
                                State[rhOff + vi] = gr * hPrev;
                            }
                        }
                        else
                        {
                            float v = Xw[xw + 2u * hid] + REC(0u);
                            if (hasBias)
                            {
                                v = (v + LoadBias(bB + 2u * hid)) + LoadBias(rB + 2u * hid);
                            }

                            float cand = Act(d, 1u, ClipV(v));
                            float gz = State[zOff + vi];
                            hNew = (1.0 - gz) * cand + gz * hPrev;
                        }
                    }
                    else
                    {
                        float v = Xw[xw] + REC(0u);
                        if (hasBias)
                        {
                            v = (v + LoadBias(bB)) + LoadBias(rB);
                        }

                        hNew = Act(d, 0u, ClipV(v));
                    }

                    if (finishes)
                    {
                        State[hOut + vi] = hNew;
                        State[yOff + ((d * P.Seq + t) * batch + b) * hid + j] = hNew;
                    }
                }
            }
        }

        // The next chunk of batch rows reuses the shared memory.
        GroupMemoryBarrierWithGroupSync();
    }
}
