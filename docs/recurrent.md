# Recurrent layers

`RecurrentKernels` runs ONNX `RNN`, `GRU` and `LSTM` layers on the GPU: forward, reverse and bidirectional, layouts 0 and 1, `sequence_lens`, initial states, peepholes, `clip`,
`input_forget`, `linear_before_reset`, and every activation ONNX names (Sigmoid, Tanh, Relu, Affine, LeakyRelu, ThresholdedRelu, ScaledTanh, HardSigmoid, Elu, Softsign, Softplus, with
`activation_alpha` and `activation_beta` already resolved by the caller into `GpuRecurrentActivation`). Tensors are float32 or `Half` (all of them the same type) with float32 accumulation and float32 state.
The semantics are those of onemodel's CPU kernels, which settled what the specification leaves open.

```csharp
using var rnn = new RecurrentKernels(device);                    // GpuKernelMode.Fast by default
var problem = new RecurrentProblem(GpuRecurrentCell.Lstm, seq: 80, batch: 1, input: 256, hidden: 256)
    { Direction = GpuRecurrentDirection.Bidirectional };
var tensors = new RecurrentOperands<float> { X = new(x), W = new(w), R = new(r), B = new(b), Y = new(y), YH = new(yh) };
rnn.Encode(encoder, problem, tensors);
```

W, R, B, P are used as stored (`[dirs, gates * hidden, input]` and so on, gate order i, o, f, c for LSTM and z, r, h for GRU); there is no repacking. Every tensor is a buffer plus an element offset (`GpuTensorRef<T>`).
Outputs that are null are not written. `Half` tensors follow the usual packing rule (buffers end on a 32-bit word, `GpuKernelBuffers.PaddedHalfLength`); an output that shares a word with a neighbour keeps the neighbour's half.

## How a layer runs

1. **Input projection** of every step and both directions: one batched `Gemm` (`Xw = X · Wᵀ`, `Batch` = directions, W through the transposed-B form). The family creates the `Gemm` for its mode, so `Fast` may use matrix hardware for `Half` and `Deterministic` uses the portable tile.
2. **State setup**: one pass that writes the first hidden state and the cell state from `InitialH`/`InitialC` or zeros.
3. **One pass per step** (two for a GRU with `linear_before_reset = 0`, because the candidate's product needs `r ⊙ H` of every unit). A pass covers both directions and all batch rows. A workgroup owns 32 rows of R (hidden units × gates of the pass) and, for each, 8 lanes
   sum `k = lane, lane + 32, …` in increasing order; the 32 partial sums are combined by a halving tree in shared memory; then one thread per (unit, batch row) does the gate arithmetic. The hidden state is two buffers that swap every step, so no workgroup reads what another writes
   in the same pass; the barrier between passes orders the steps. Rows whose sequence has ended copy their state through. The thread that forms a unit's gates adds the 8 lane sums of each row itself, in the order of the halving tree
   (`((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))`), after one barrier. The kernel is compiled per cell (LSTM, GRU, RNN), per precision, for one batch row or up to eight at a time, and for the ONNX default activations (Sigmoid, Tanh) or all eleven.
4. **One pass per requested output** (Y, Y_h, Y_c): layout, zeroing past the sequence length, rounding to `Half`.

Why a pass per step: a kernel that loops over all steps in one dispatch needs every workgroup of the dispatch to reach a grid-wide barrier, which Vulkan, Metal and WebGPU do not guarantee (workgroups may not all be resident).
The alternative that needs no grid barrier, one workgroup per direction that walks every step itself ordered by workgroup barriers, was built and measured and is not shipped: a workgroup runs on one compute unit, so each step's 1 MB of weights
comes through one unit's memory path. Bidirectional LSTM, sequence 80, batch 1, input and hidden 256, float32, untimed, same bits as the pass per step: Apple M2 9.9 ms against 2.4 ms for a pass per step, RTX 4070 5.1 ms against 0.7 ms.

What a step costs is mostly the kernel's start, not its arithmetic. A step kernel that held all eleven activation functions inline (five call sites, a switch each) cost about 20 µs more per step on an Apple M2 than the same kernel without them (removing the activations alone took 1.5 ms off an 80-step layer; the size of the kernel, not its arithmetic, is the likely cause); the default activations get a kernel without them, and each cell gets a kernel without the other cells. Carrying eight batch rows through the product when the batch is one cost another 2x
(registers and shared memory limit how many workgroups are resident); a batch of one gets a one-row kernel. The model-level numbers are in the report of the unit that wrote this page and are indicative.
`RecurrentProblem.MaxLength` (a host-side bound on the sequence lengths) shortens the loop when `sequence_lens` is given.

Scratch: the input projection (`RecurrentKernels.XwLength` floats) and the state with every step's output (`StateLength` floats) are kept by the family, grown on demand, or supplied by the caller (`xwScratch`, `stateScratch`, each with an element offset `xwScratchOffset`, `stateScratchOffset`, so both regions may sit in one arena buffer; a browser refuses one buffer bound read-only and read-write in a dispatch, so there they stay in two buffers) so a memory planner can place them.
Both must fit one storage binding; larger layers throw `GpuUnsupportedException`. Sizes, offsets and shapes are checked in safe arithmetic before anything is recorded (`GpuKernelShapeException` naming `LSTM`/`GRU`/`RNN` and the argument).

## Arithmetic and determinism

The order is fixed by the shapes: the projection is `Gemm`'s; each recurrent dot product is the 32-lane strided sums then the halving tree; the gate arithmetic is ONNX's `pre = ((xw + rec) + Wb) + Rb`, then the peephole terms
(`i`, `f` with the previous cell state, `o` with the new one), then `clip`, then the activation, then `c = f·c + i·c̃`, `h = o·h_act(c)` (GRU: `h = (1 − z)·h̃ + z·h`). No atomics, no subgroup arithmetic, nothing depends on the subgroup size or scheduling.
One kernel therefore serves both `GpuKernelMode` values and a run is bit-identical to the next on one device and driver (checked: every test case runs twice; the recognizer shapes also in `Deterministic` mode). Whether a compiler fuses a multiply with the following
add is the device's, so results differ in the last bits between devices; they are compared with the reference below, not with each other.

## Semantics

- `sequence_lens`: row `b` runs its first `len` steps, in reverse from step `len − 1`; Y is zero at the steps past `len`; Y_h and Y_c are the state after the last valid step; **a row of length 0 has Y_h = Y_c = 0** (onemodel's CPU kernel; the specification is silent).
  Lengths are int32 on the device and not validated there (a value above `Seq` is treated as `Seq`); validate them on the host.
- `Seq = 0`: nothing runs; Y_h and Y_c are the initial states or zeros. `Batch = 0`: nothing is recorded.
- `clip` clamps every gate's pre-activation to `[−clip, clip]` when positive (NaN passes through).
- `input_forget` couples the forget gate: `f = 1 − i`. `linear_before_reset = 1`: `h̃ = g(xw_h + r·(rec_h + Rb_h) + Wb_h)`.
- Peepholes, `input_forget`, `InitialC` and `YC` exist only for an LSTM, `linear_before_reset` only for a GRU; anything else throws.

## Tolerance and evidence

Scaled error `max|gpu − ref| / max|ref|` against a double-precision C# reference written from the operator definitions: float32 ≤ 1e-5, `Half` (float32 accumulation and state, outputs rounded to half) ≤ 2e-3.
The tests cover every cell × direction × layout with option sets (bias, lengths including 0, initial state, peepholes, clip, `input_forget`, `linear_before_reset`), every activation in every slot of both directions, odd offsets with sentinels around every output,
empty sequence and batch, hidden sizes that do not divide the workgroup, batches above eight, and the shapes of a text-line recognizer (bidirectional LSTM, sequence 80 and 48, batch 1 and 8, input 256 and 192, hidden 256 and 96).
