using RecursiveAI.OneGpu.Kernels.Shaders;

namespace RecursiveAI.OneGpu.Kernels;

/// <summary>
/// ONNX <c>RNN</c>, <c>GRU</c> and <c>LSTM</c>: forward, reverse and bidirectional layers, layouts 0 and 1, sequence lengths, initial states, peepholes,
/// <c>clip</c>, <c>input_forget</c>, <c>linear_before_reset</c> and every activation ONNX names, over float32 or <see cref="Half"/> tensors with float32
/// accumulation and float32 state.
/// </summary>
/// <remarks>
/// <para>
/// A layer is recorded as: one batched matrix product for the input projection of every step and both directions (<see cref="Gemm"/>, W used as stored
/// through its transposed-operand form), one pass that sets up the state, one pass per step (two for a GRU whose reset gate is applied before the product,
/// because the candidate's product needs <c>r * H</c> of every unit), and one pass per requested output. A step pass runs both directions of a bidirectional
/// layer side by side, all batch rows, and walks each row of R with 32 lanes whose partial sums are combined by a fixed halving tree. The recurrent
/// state is a hidden-state pair that swaps every step, so a workgroup never reads what another writes in the same pass; ordering between steps comes from the
/// barriers between passes. A persistent kernel that loops over the steps inside one dispatch would need a barrier across workgroups, which neither Vulkan, Metal
/// nor WebGPU guarantee to be safe (workgroups of one dispatch may not all be resident), so the step is the unit of dispatch.
/// </para>
/// <para>
/// The order of arithmetic is fixed by the shapes: the input projection is the matrix product of <see cref="Gemm"/> (its order for the mode this family was created for),
/// each recurrent dot product sums k in four-byte strides of 32 lanes in increasing order and then a halving tree, and the gate arithmetic is ONNX's
/// <c>((xw + rec) + Wb) + Rb</c> followed by the peephole terms, the clip and the activation. There are no atomics and no subgroup arithmetic, so one variant serves
/// both <see cref="GpuKernelMode"/> values and results are bit-identical from run to run on one device and driver. (Which instructions a compiler fuses into multiply-adds
/// is the device's, so results differ in the last bits between devices.)
/// </para>
/// <para>
/// Semantics follow onemodel's CPU kernels: with <c>sequence_lens</c> a row runs its first <c>len</c> steps, in reverse from step <c>len - 1</c>; Y is zero at steps
/// past the length; Y_h and Y_c are the state after the row's last step; a row of length 0 has Y_h = Y_c = 0. Sequence lengths are read on the device and are not checked
/// there; a length above <see cref="RecurrentProblem.Seq"/> is clamped, a negative one (as unsigned) is clamped the same way. Rows past their length copy their state through.
/// </para>
/// <para>
/// The family keeps two scratch buffers (the input projection, <see cref="XwLength"/> floats, and the state with every step's output, <see cref="StateLength"/> floats),
/// grown when a larger layer comes and kept until <see cref="Dispose"/>; a caller that plans memory can pass its own. Consecutive calls reuse them; passes are ordered, so
/// that is safe within one encoder. Bind groups for buffer sets already seen are cached, so a warm call allocates nothing. Not thread-safe.
/// </para>
/// </remarks>
public sealed class RecurrentKernels : IDisposable
{
    const int StepWords = 26;
    const int InitWords = 9;
    const int OutWords = 12;
    const int RowsPerGroup = 64;   // 256 threads / 4 lanes per row; matches ROWS in RecurrentStep.hlsl

    readonly GpuDevice _device;
    readonly GpuKernelMode _mode;
    readonly GemmOptions? _gemmOptions;
    readonly int _maxBindGroups;
    readonly KernelPipeline?[] _step = new KernelPipeline?[2];
    readonly KernelPipeline?[] _init = new KernelPipeline?[2];
    readonly KernelPipeline?[] _outputs = new KernelPipeline?[2];
    readonly Gemm?[] _gemm = new Gemm?[2];
    readonly GpuBuffer[] _buffers = new GpuBuffer[6];
    GpuBuffer<uint>? _dummy;
    GpuBuffer<float>? _xw;
    GpuBuffer<float>? _state;

    /// <summary>Creates the family for a device.</summary>
    /// <param name="device">The device the buffers belong to.</param>
    /// <param name="mode">The mode of the input projection; the step passes are the same in both modes. See the remarks of the class.</param>
    /// <param name="gemmOptions">Options of the input projection's <see cref="Gemm"/>; null for the defaults.</param>
    /// <param name="maxCachedBindGroups">How many sets of buffers the family remembers bind groups for; see <see cref="GpuKernelBindings"/>.</param>
    public RecurrentKernels(GpuDevice device, GpuKernelMode mode = GpuKernelMode.Fast, GemmOptions? gemmOptions = null, int maxCachedBindGroups = GpuKernelBindings.DefaultMaxCachedBindGroups)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _mode = mode;
        _gemmOptions = gemmOptions;
        _maxBindGroups = GpuKernelBindings.Check(maxCachedBindGroups);
    }

    /// <summary>Floats of input-projection scratch a layer needs: <c>dirs * Seq * Batch * gates * Hidden</c>.</summary>
    /// <param name="problem">The layer.</param>
    public static long XwLength(in RecurrentProblem problem) => (long)problem.Directions * problem.Seq * problem.Batch * problem.Gates * problem.Hidden;

    /// <summary>Floats of state scratch a layer needs: <c>(5 + Seq) * dirs * Batch * Hidden</c> (two hidden states, the cell state, two GRU temporaries and every step's output).</summary>
    /// <param name="problem">The layer.</param>
    public static long StateLength(in RecurrentProblem problem) => (5L + problem.Seq) * problem.Directions * problem.Batch * problem.Hidden;

    /// <summary>Records one layer.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>.</typeparam>
    /// <param name="encoder">The encoder to record into; no other pass may be open on it.</param>
    /// <param name="problem">The shape and attributes.</param>
    /// <param name="operands">The tensors.</param>
    /// <param name="xwScratch">Optional input-projection scratch of at least <see cref="XwLength"/> floats; null uses the family's own.</param>
    /// <param name="stateScratch">Optional state scratch of at least <see cref="StateLength"/> floats; null uses the family's own.</param>
    /// <exception cref="GpuKernelShapeException">A size, offset, range, attribute or operand is invalid for the cell (for example peepholes on a GRU); names the operation and the argument.</exception>
    /// <exception cref="GpuUnsupportedException">A buffer is larger than one storage binding can cover on this device.</exception>
    public void Encode<T>(GpuCommandEncoder encoder, in RecurrentProblem problem, RecurrentOperands<T> operands, GpuBuffer<float>? xwScratch = null, GpuBuffer<float>? stateScratch = null)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(operands);
        var name = problem.Cell switch { GpuRecurrentCell.Lstm => "LSTM", GpuRecurrentCell.Gru => "GRU", GpuRecurrentCell.Rnn => "RNN", _ => throw new GpuKernelShapeException("Recurrent", "Cell", $"cell {problem.Cell} is not defined") };
        var half = KernelTypes.IsHalf<T>(name);
        if (problem.Direction is < GpuRecurrentDirection.Forward or > GpuRecurrentDirection.Bidirectional)
        {
            throw new GpuKernelShapeException(name, nameof(problem.Direction), $"direction {problem.Direction} is not defined");
        }

        Dimension(name, nameof(problem.Seq), problem.Seq, 0);
        Dimension(name, nameof(problem.Batch), problem.Batch, 0);
        Dimension(name, nameof(problem.Input), problem.Input, 1);
        Dimension(name, nameof(problem.Hidden), problem.Hidden, 1);
        if (!(problem.Clip >= 0f))
        {
            throw new GpuKernelShapeException(name, nameof(problem.Clip), "clip must not be negative or NaN");
        }

        var isLstm = problem.Cell == GpuRecurrentCell.Lstm;
        var isGru = problem.Cell == GpuRecurrentCell.Gru;
        var dirs = problem.Directions;
        int seq = problem.Seq, batch = problem.Batch, input = problem.Input, hid = problem.Hidden, gates = problem.Gates;
        var gh = Product(name, "Hidden", gates, hid);
        var o = operands;

        // Sizes of every operand, in elements; each product is checked against the 32-bit index space the kernels use.
        var xCount = Product(name, "X", seq, batch, input);
        var wCount = Product(name, "W", dirs, gh, input);
        var rCount = Product(name, "R", dirs, gh, hid);
        var stateCount = Product(name, "state", dirs, batch, hid);
        var yCount = Product(name, "Y", seq, dirs, batch, hid);
        Check.Range(name, "X", o.X.Buffer.Length, o.X.Offset, xCount, half);
        Check.Range(name, "W", o.W.Buffer.Length, o.W.Offset, wCount, half);
        Check.Range(name, "R", o.R.Buffer.Length, o.R.Offset, rCount, half);
        Bindable(name, "X", o.X.Buffer.Buffer);
        Bindable(name, "W", o.W.Buffer.Buffer);
        Bindable(name, "R", o.R.Buffer.Buffer);
        if (o.B is { } bias)
        {
            Check.Range(name, "B", bias.Buffer.Length, bias.Offset, Product(name, "B", dirs, 2, gh), half);
            Bindable(name, "B", bias.Buffer.Buffer);
        }

        if (o.P is { } peep)
        {
            if (!isLstm)
            {
                throw new GpuKernelShapeException(name, "P", "peephole weights exist only for an LSTM");
            }

            Check.Range(name, "P", peep.Buffer.Length, peep.Offset, Product(name, "P", dirs, 3, hid), half);
            Bindable(name, "P", peep.Buffer.Buffer);
        }

        if (o.InitialC is not null && !isLstm)
        {
            throw new GpuKernelShapeException(name, "InitialC", "an initial cell state exists only for an LSTM");
        }

        if (o.YC is not null && !isLstm)
        {
            throw new GpuKernelShapeException(name, "YC", "a final cell state exists only for an LSTM");
        }

        if (problem.InputForget && !isLstm)
        {
            throw new GpuKernelShapeException(name, nameof(problem.InputForget), "input_forget exists only for an LSTM");
        }

        if (problem.LinearBeforeReset && !isGru)
        {
            throw new GpuKernelShapeException(name, nameof(problem.LinearBeforeReset), "linear_before_reset exists only for a GRU");
        }

        if (o.SequenceLengths is { } lens)
        {
            Check.Range(name, "SequenceLengths", lens.Buffer.Length, lens.Offset, batch);
        }

        if (o.InitialH is { } h0)
        {
            Check.Range(name, "InitialH", h0.Buffer.Length, h0.Offset, stateCount, half);
        }

        if (o.InitialC is { } c0)
        {
            Check.Range(name, "InitialC", c0.Buffer.Length, c0.Offset, stateCount, half);
        }

        if (o.Y is { } y)
        {
            Check.Range(name, "Y", y.Buffer.Length, y.Offset, yCount, half);
        }

        if (o.YH is { } yh)
        {
            Check.Range(name, "YH", yh.Buffer.Length, yh.Offset, stateCount, half);
        }

        if (o.YC is { } yc)
        {
            Check.Range(name, "YC", yc.Buffer.Length, yc.Offset, stateCount, half);
        }

        var steps = seq;
        if (o.SequenceLengths is not null && problem.MaxLength >= 0)
        {
            if (problem.MaxLength > seq)
            {
                throw new GpuKernelShapeException(name, nameof(problem.MaxLength), $"MaxLength {problem.MaxLength} is above Seq {seq}");
            }

            steps = problem.MaxLength;
        }

        var xwLength = Product(name, "Xw", dirs, seq, batch, gh);
        var stateLength = Product(name, "scratch", 5L + seq, dirs, batch, hid);
        var maxGroups = _device.Capabilities.Compute.MaxWorkgroupCount;
        var limitX = maxGroups == 0 ? 65535u : Math.Min(maxGroups, 65535u);
        if ((hid + 1) / 2 > limitX)
        {
            throw new GpuKernelShapeException(name, "Hidden", $"hidden size {hid} needs more workgroups per step than the device dispatches ({limitX})");
        }

        // Nothing is stored for an empty batch, and without a requested output nothing observable would change.
        if (batch == 0 || (o.Y is null && o.YH is null && o.YC is null))
        {
            return;
        }

        var xw = Scratch(name, "xwScratch", xwScratch, xwLength, ref _xw, "recurrent input projection");
        var state = Scratch(name, "stateScratch", stateScratch, stateLength, ref _state, "recurrent state");
        var dummy = (_dummy ??= _device.CreateBuffer<uint>(4, label: "recurrent unused operand")).Buffer;

        // 1. Input projection of all steps: Xw[d] = X * W[d]^T, M = Seq * Batch rows. W is stored [G*H, Input], the N x K form of the right operand.
        if (seq > 0)
        {
            var m = checked(seq * batch);
            var gemm = Projection(half);
            var p = new GemmProblem(m, (int)gh, input)
            {
                Batch = dirs,
                TransposeB = true,
                Lda = input,
                Ldb = input,
                Ldc = (int)gh,
                OffsetA = o.X.Offset,
                OffsetB = o.W.Offset,
                OffsetC = 0,
                StrideA = 0,
                StrideB = checked((int)gh * input),
                StrideC = checked(m * (int)gh),
            };
            if (half)
            {
                gemm.Encode(encoder, p, (GpuBuffer<Half>)(object)o.X.Buffer, (GpuBuffer<Half>)(object)o.W.Buffer, xw);
            }
            else
            {
                gemm.Encode(encoder, p, (GpuBuffer<float>)(object)o.X.Buffer, (GpuBuffer<float>)(object)o.W.Buffer, xw);
            }
        }

        // 2. State: H copy 0 and C from the initial tensors, or zeros.
        {
            Span<uint> w = stackalloc uint[InitWords];
            w[0] = (uint)stateCount;
            w[1] = (uint)hid;
            w[2] = (uint)batch;
            w[3] = (uint)dirs;
            w[4] = (o.InitialH is null ? 0u : 1u) | (o.InitialC is null ? 0u : 2u) | (problem.Layout1 ? 4u : 0u);
            w[5] = (uint)(o.InitialH?.Offset ?? 0);
            w[6] = (uint)(o.InitialC?.Offset ?? 0);
            w[7] = 0;
            var pipeline = InitPipeline(half);
            Launch.Record(_device, encoder, pipeline, w, stateCount, 0, 0, 8, o.InitialH?.Buffer.Buffer ?? dummy, o.InitialC?.Buffer.Buffer ?? dummy, state.Buffer);
        }

        // 3. The steps.
        var stepPipeline = StepPipeline(half);
        var flags = problem.Cell switch { GpuRecurrentCell.Lstm => 2u, GpuRecurrentCell.Gru => 1u, _ => 0u };
        flags |= problem.LinearBeforeReset ? 4u : 0u;
        flags |= problem.InputForget ? 8u : 0u;
        flags |= o.B is null ? 0u : 16u;
        flags |= o.P is null ? 0u : 32u;
        flags |= o.SequenceLengths is null ? 0u : 64u;
        flags |= problem.Layout1 ? 128u : 0u;
        flags |= dirs == 2 ? 1024u : 0u;
        flags |= problem.Direction switch { GpuRecurrentDirection.Reverse => 1u << 12, GpuRecurrentDirection.Bidirectional => 1u << 13, _ => 0u };
        var forward = problem.Activations ?? GpuRecurrentActivations.Default(problem.Cell);
        var reverse = problem.ReverseActivations ?? forward;

        Span<uint> sw = stackalloc uint[StepWords];
        sw[0] = (uint)hid;
        sw[1] = (uint)batch;
        sw[2] = (uint)seq;
        sw[5] = BitConverter.SingleToUInt32Bits(problem.Clip);
        sw[6] = Activations(name, forward, reverse, dirs, sw[14..]);
        sw[7] = 0;
        sw[8] = (uint)o.R.Offset;
        sw[9] = (uint)(o.B?.Offset ?? 0);
        sw[10] = (uint)(o.P?.Offset ?? 0);
        sw[11] = (uint)(o.SequenceLengths?.Offset ?? 0);
        sw[12] = 0;
        var buffers = _buffers;
        buffers[0] = xw.Buffer;
        buffers[1] = o.R.Buffer.Buffer;
        buffers[2] = o.B?.Buffer.Buffer ?? dummy;
        buffers[3] = o.P?.Buffer.Buffer ?? dummy;
        buffers[4] = o.SequenceLengths?.Buffer.Buffer ?? dummy;
        buffers[5] = state.Buffer;
        var split = isGru && !problem.LinearBeforeReset;
        var perStep = 2.0 * dirs * batch * gh * hid + 10.0 * dirs * batch * hid;
        for (var s = 0; s < steps; s++)
        {
            sw[3] = (uint)s;
            if (split)
            {
                RecordStep(encoder, stepPipeline, sw, flags, hid, dirs, 2, 0);
                RecordStep(encoder, stepPipeline, sw, flags | 512u, hid, dirs, 1, 2);
            }
            else
            {
                RecordStep(encoder, stepPipeline, sw, flags, hid, dirs, gates, 0);
            }

            encoder.ReportWork(perStep);
        }

        // 4. Outputs.
        var outPipeline = OutputsPipeline(half);
        var lensBuffer = o.SequenceLengths?.Buffer.Buffer ?? dummy;
        if (o.Y is { } yOut)
        {
            Output(encoder, outPipeline, 0, yCount, problem, steps, o.SequenceLengths, yOut.Offset, state.Buffer, lensBuffer, yOut.Buffer.Buffer, half);
        }

        if (o.YH is { } yhOut)
        {
            Output(encoder, outPipeline, 1, stateCount, problem, steps, o.SequenceLengths, yhOut.Offset, state.Buffer, lensBuffer, yhOut.Buffer.Buffer, half);
        }

        if (o.YC is { } ycOut)
        {
            Output(encoder, outPipeline, 2, stateCount, problem, steps, o.SequenceLengths, ycOut.Offset, state.Buffer, lensBuffer, ycOut.Buffer.Buffer, half);
        }
    }

    /// <summary>How many bind groups this family has built since it was created. A warm call with buffers it has already seen builds none, so a consumer can record a run twice and check that the second adds nothing. Read-only; not reset by <c>ReleaseBindings</c>.</summary>
    public int BindGroupsCreated => KernelPipeline.Total(_step, _init, _outputs) + (_gemm[0]?.BindGroupsCreated ?? 0) + (_gemm[1]?.BindGroupsCreated ?? 0);

    /// <summary>The input projection's variant, for logs (for example "portable-tile64x64x16"), or null before the first call.</summary>
    public string? ProjectionVariant => (_gemm[1] ?? _gemm[0])?.Variant;

    /// <summary>Drops the cached bind groups (the pipelines stay); the next call with a buffer set builds its group again. Call it once the GPU has finished with the buffers they refer to.</summary>
    public void ReleaseBindings()
    {
        foreach (var set in new[] { _step, _init, _outputs })
        {
            foreach (var p in set)
            {
                p?.ReleaseBindings();
            }
        }

        foreach (var g in _gemm)
        {
            g?.ReleaseBindings();
        }
    }

    /// <summary>Releases the pipelines, the scratch buffers and the cached bind groups once the GPU has finished with them.</summary>
    public void Dispose()
    {
        foreach (var set in new[] { _step, _init, _outputs })
        {
            for (var i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }

        for (var i = 0; i < _gemm.Length; i++)
        {
            _gemm[i]?.Dispose();
            _gemm[i] = null;
        }

        _xw?.Dispose();
        _state?.Dispose();
        _dummy?.Dispose();
        _xw = null;
        _state = null;
        _dummy = null;
    }

    void RecordStep(GpuCommandEncoder encoder, KernelPipeline pipeline, Span<uint> sw, uint flags, int hid, int dirs, int gates, int firstGate)
    {
        // 256 / lanes rows of R per workgroup (the shader's ROWS): that many divided by the gates formed in the pass hidden units.
        var units = RowsPerGroup / gates;
        sw[4] = flags;
        sw[13] = (uint)gates | ((uint)firstGate << 4);
        pipeline.Record(encoder, sw, (uint)((hid + units - 1) / units), (uint)dirs, _buffers);
    }

    void Output(GpuCommandEncoder encoder, KernelPipeline pipeline, uint mode, long count, in RecurrentProblem problem, int steps, GpuTensorRef<int>? lens, int offset, GpuBuffer state, GpuBuffer lensBuffer, GpuBuffer target, bool half)
    {
        Span<uint> w = stackalloc uint[OutWords];
        w[0] = (uint)count;
        w[1] = mode;
        w[2] = (uint)problem.Hidden;
        w[3] = (uint)problem.Batch;
        w[4] = (uint)problem.Seq;
        w[5] = (uint)problem.Directions;
        w[6] = (problem.Layout1 ? 1u : 0u) | (lens is null ? 0u : 2u);
        w[7] = (uint)offset;
        w[8] = (uint)(lens?.Offset ?? 0);
        w[9] = 0;
        w[10] = (uint)(steps & 1);
        Launch.Record(_device, encoder, pipeline, w, count, offset, half ? 1 : 0, 11, state, lensBuffer, target);
    }

    // Packs the activation kinds into one word and their parameters into the twelve float words that follow.
    static uint Activations(string name, GpuRecurrentActivations forward, GpuRecurrentActivations reverse, int dirs, Span<uint> values)
    {
        uint kinds = 0;
        for (var d = 0; d < dirs; d++)
        {
            var set = d == 0 ? forward : reverse;
            for (var a = 0; a < 3; a++)
            {
                var act = a == 0 ? set.F : a == 1 ? set.G : set.H;
                if (act.Kind is < GpuRecurrentActivationKind.Sigmoid or > GpuRecurrentActivationKind.Softplus)
                {
                    throw new GpuKernelShapeException(name, "Activations", $"activation {act.Kind} is not defined");
                }

                kinds |= (uint)act.Kind << ((d * 12) + (a * 4));
                values[(((d * 3) + a) * 2)] = BitConverter.SingleToUInt32Bits(act.Alpha);
                values[(((d * 3) + a) * 2) + 1] = BitConverter.SingleToUInt32Bits(act.Beta);
            }
        }

        return kinds;
    }

    Gemm Projection(bool half) => _gemm[half ? 1 : 0] ??= new Gemm(_device, half ? GemmElementType.F16 : GemmElementType.F32, _mode, _gemmOptions, _maxBindGroups);

    KernelPipeline InitPipeline(bool half)
    {
        var i = half ? 1 : 0;
        if (_init[i] is { } ready)
        {
            return ready;
        }

        var e = RecurrentInit.Variants.Get(new RecurrentInit.Key(f16: half)).InitOp;
        return _init[i] = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, InitWords, "recurrent init", _maxBindGroups);
    }

    KernelPipeline StepPipeline(bool half)
    {
        var i = half ? 1 : 0;
        if (_step[i] is { } ready)
        {
            return ready;
        }

        var e = RecurrentStep.Variants.Get(new RecurrentStep.Key(f16: half)).StepOp;
        return _step[i] = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 6, StepWords, "recurrent step", _maxBindGroups);
    }

    KernelPipeline OutputsPipeline(bool half)
    {
        var i = half ? 1 : 0;
        if (_outputs[i] is { } ready)
        {
            return ready;
        }

        var e = RecurrentOutputs.Variants.Get(new RecurrentOutputs.Key(f16: half)).OutputsOp;
        return _outputs[i] = new KernelPipeline(_device, KernelSources.Make(e.EntryPoint, e.SpirV, e.Msl, e.MslFunctionName, e.Wgsl, e.WgslEntryPoint, e.Layout), 3, OutWords, "recurrent outputs", _maxBindGroups);
    }

    GpuBuffer<float> Scratch(string name, string argument, GpuBuffer<float>? given, long length, ref GpuBuffer<float>? owned, string label)
    {
        if (given is not null)
        {
            if (given.Length < length)
            {
                throw new GpuKernelShapeException(name, argument, $"the scratch holds {given.Length} floats, {length} are needed");
            }

            Bindable(name, argument, given.Buffer);
            return given;
        }

        if (owned is null || owned.Length < length)
        {
            owned?.Dispose();
            owned = null;
            ReleaseBindings();
            owned = _device.CreateBuffer<float>((int)Math.Max(length, 4), label: label);
        }

        Bindable(name, argument, owned.Buffer);
        return owned;
    }

    void Bindable(string name, string operand, GpuBuffer buffer)
    {
        var maxBinding = _device.Capabilities.Compute.MaxStorageBufferBindingBytes;
        if (maxBinding != 0 && buffer.Size > maxBinding)
        {
            throw new GpuUnsupportedException($"{name}: {operand} is {buffer.Size} bytes, more than one storage binding can cover on {_device.Adapter.Name}", nameof(GpuComputeCapabilities.MaxStorageBufferBindingBytes), $"at most {maxBinding} bytes", $"{buffer.Size} bytes");
        }
    }

    static void Dimension(string name, string argument, int value, int minimum)
    {
        if (value < minimum)
        {
            throw new GpuKernelShapeException(name, argument, $"{argument} {value} must be at least {minimum}");
        }
    }

    static long Product(string name, string argument, params ReadOnlySpan<long> factors)
    {
        long n = 1;
        foreach (var f in factors)
        {
            n = checked(n * f);
            if (n > Check.MaxElements)
            {
                throw new GpuKernelShapeException(name, argument, $"the tensor holds more than {Check.MaxElements} elements");
            }
        }

        return n;
    }
}
