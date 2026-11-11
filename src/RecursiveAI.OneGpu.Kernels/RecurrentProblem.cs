namespace RecursiveAI.OneGpu.Kernels;

/// <summary>The recurrent cell of a layer.</summary>
public enum GpuRecurrentCell
{
    /// <summary>ONNX <c>RNN</c>: one gate, <c>H = f(X Wi + H Ri + Wbi + Rbi)</c>.</summary>
    Rnn = 0,

    /// <summary>ONNX <c>GRU</c> (Cho et al.): gates z, r, h in that order in W, R and B.</summary>
    Gru = 1,

    /// <summary>ONNX <c>LSTM</c> (Hochreiter and Schmidhuber, with the peepholes of Gers and Schmidhuber): gates i, o, f, c in that order in W, R and B.</summary>
    Lstm = 2,
}

/// <summary>The direction a layer runs in (the ONNX <c>direction</c> attribute).</summary>
public enum GpuRecurrentDirection
{
    /// <summary>First step to last.</summary>
    Forward = 0,

    /// <summary>Last step to first; each row starts at its own last valid step.</summary>
    Reverse = 1,

    /// <summary>Both; the weights, biases, peepholes and states carry a leading dimension of 2 (forward, then reverse).</summary>
    Bidirectional = 2,
}

/// <summary>The activation functions ONNX lets a recurrent operator name.</summary>
public enum GpuRecurrentActivationKind
{
    /// <summary><c>1 / (1 + e^-x)</c>.</summary>
    Sigmoid = 0,

    /// <summary>Hyperbolic tangent.</summary>
    Tanh = 1,

    /// <summary><c>max(x, 0)</c>; NaN passes through.</summary>
    Relu = 2,

    /// <summary><c>alpha * x + beta</c>.</summary>
    Affine = 3,

    /// <summary><c>x &lt; 0 ? alpha * x : x</c>.</summary>
    LeakyRelu = 4,

    /// <summary><c>x &gt; alpha ? x : 0</c>.</summary>
    ThresholdedRelu = 5,

    /// <summary><c>alpha * tanh(beta * x)</c>.</summary>
    ScaledTanh = 6,

    /// <summary><c>min(1, max(0, alpha * x + beta))</c>.</summary>
    HardSigmoid = 7,

    /// <summary><c>x &gt;= 0 ? x : alpha * (e^x - 1)</c>.</summary>
    Elu = 8,

    /// <summary><c>x / (1 + |x|)</c>.</summary>
    Softsign = 9,

    /// <summary><c>log(1 + e^x)</c>, and <c>x</c> above 20.</summary>
    Softplus = 10,
}

/// <summary>One activation with its parameters, which the caller has already defaulted (ONNX gives each activation its own defaults).</summary>
/// <param name="Kind">The function.</param>
/// <param name="Alpha">The first parameter, used by Affine, LeakyRelu, ThresholdedRelu, ScaledTanh, HardSigmoid and Elu.</param>
/// <param name="Beta">The second parameter, used by Affine, ScaledTanh and HardSigmoid.</param>
public readonly record struct GpuRecurrentActivation(GpuRecurrentActivationKind Kind, float Alpha = 0f, float Beta = 0f)
{
    /// <summary>The logistic function.</summary>
    public static GpuRecurrentActivation Sigmoid => new(GpuRecurrentActivationKind.Sigmoid);

    /// <summary>Hyperbolic tangent.</summary>
    public static GpuRecurrentActivation Tanh => new(GpuRecurrentActivationKind.Tanh);

    /// <summary>The rectifier.</summary>
    public static GpuRecurrentActivation Relu => new(GpuRecurrentActivationKind.Relu);
}

/// <summary>The three activations of one direction: <c>F</c> for the gates (LSTM i, o, f; GRU z, r; RNN the output), <c>G</c> for the candidate (LSTM c; GRU h), <c>H</c> for the LSTM's output.</summary>
/// <param name="F">The gate activation.</param>
/// <param name="G">The candidate activation. Unused by an RNN.</param>
/// <param name="H">The cell-output activation. Used by an LSTM only.</param>
public readonly record struct GpuRecurrentActivations(GpuRecurrentActivation F, GpuRecurrentActivation G, GpuRecurrentActivation H)
{
    /// <summary>The ONNX defaults of a cell: Sigmoid, Tanh, Tanh for an LSTM; Sigmoid, Tanh for a GRU; Tanh for an RNN.</summary>
    /// <param name="cell">The cell.</param>
    public static GpuRecurrentActivations Default(GpuRecurrentCell cell) =>
        cell == GpuRecurrentCell.Rnn
            ? new(GpuRecurrentActivation.Tanh, GpuRecurrentActivation.Tanh, GpuRecurrentActivation.Tanh)
            : new(GpuRecurrentActivation.Sigmoid, GpuRecurrentActivation.Tanh, GpuRecurrentActivation.Tanh);
}

/// <summary>
/// The shape and attributes of one recurrent layer (ONNX <c>RNN</c>, <c>GRU</c>, <c>LSTM</c>). Sizes count elements.
/// X is <c>[Seq, Batch, Input]</c> (layout 0) or <c>[Batch, Seq, Input]</c> (layout 1); W is <c>[dirs, gates * Hidden, Input]</c>, R is
/// <c>[dirs, gates * Hidden, Hidden]</c>, B is <c>[dirs, 2 * gates * Hidden]</c> (the W biases, then the R biases), P is <c>[dirs, 3 * Hidden]</c> (i, o, f),
/// sequence lengths <c>[Batch]</c>, initial and final states <c>[dirs, Batch, Hidden]</c> (layout 1: <c>[Batch, dirs, Hidden]</c>), Y is <c>[Seq, dirs, Batch, Hidden]</c> (layout 1: <c>[Batch, Seq, dirs, Hidden]</c>).
/// <c>dirs</c> is 2 for a bidirectional layer, else 1; gates is 1 (RNN), 3 (GRU) or 4 (LSTM).
/// </summary>
public struct RecurrentProblem
{
    /// <summary>Creates a forward layer with the cell's default activations and no clip.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="seq">Time steps, at least 0.</param>
    /// <param name="batch">Batch rows, at least 0.</param>
    /// <param name="input">Input features per step, at least 1.</param>
    /// <param name="hidden">Hidden units, at least 1.</param>
    public RecurrentProblem(GpuRecurrentCell cell, int seq, int batch, int input, int hidden)
    {
        Cell = cell;
        Seq = seq;
        Batch = batch;
        Input = input;
        Hidden = hidden;
        MaxLength = -1;
    }

    /// <summary>The cell.</summary>
    public GpuRecurrentCell Cell { get; set; }

    /// <summary>The direction.</summary>
    public GpuRecurrentDirection Direction { get; set; }

    /// <summary>Time steps (X's sequence dimension), at least 0.</summary>
    public int Seq { get; set; }

    /// <summary>Batch rows, at least 0.</summary>
    public int Batch { get; set; }

    /// <summary>Input features per step, at least 1.</summary>
    public int Input { get; set; }

    /// <summary>Hidden units, at least 1.</summary>
    public int Hidden { get; set; }

    /// <summary>True for the batch-major layout (ONNX <c>layout = 1</c>).</summary>
    public bool Layout1 { get; set; }

    /// <summary>The ONNX <c>clip</c> attribute: pre-activations are clamped to <c>[-Clip, Clip]</c> when it is above 0 (NaN passes through).</summary>
    public float Clip { get; set; }

    /// <summary>LSTM only: couple the forget gate to the input gate, <c>f = 1 - i</c>.</summary>
    public bool InputForget { get; set; }

    /// <summary>GRU only: apply the reset gate after the recurrent product (<c>linear_before_reset = 1</c>) instead of before it.</summary>
    public bool LinearBeforeReset { get; set; }

    /// <summary>
    /// A bound on the sequence lengths known on the host: the number of steps recorded, when sequence lengths are given. Negative (the default) means <see cref="Seq"/>.
    /// Every length must be at most this; rows are not checked on the device.
    /// </summary>
    public int MaxLength { get; set; }

    /// <summary>The activations of the forward direction; null means the cell's defaults.</summary>
    public GpuRecurrentActivations? Activations { get; set; }

    /// <summary>The activations of the reverse direction of a bidirectional layer; null means <see cref="Activations"/>.</summary>
    public GpuRecurrentActivations? ReverseActivations { get; set; }

    /// <summary>Directions: 2 for a bidirectional layer, else 1.</summary>
    public readonly int Directions => Direction == GpuRecurrentDirection.Bidirectional ? 2 : 1;

    /// <summary>Gates per cell: 1, 3 or 4.</summary>
    public readonly int Gates => Cell switch { GpuRecurrentCell.Lstm => 4, GpuRecurrentCell.Gru => 3, _ => 1 };
}

/// <summary>A tensor in a buffer: the buffer and the element offset of its first element.</summary>
/// <typeparam name="T">The buffer's element type.</typeparam>
/// <param name="Buffer">The buffer.</param>
/// <param name="Offset">Element offset of the tensor.</param>
public readonly record struct GpuTensorRef<T>(GpuBuffer<T> Buffer, int Offset = 0)
    where T : unmanaged;

/// <summary>The tensors of one <see cref="RecurrentKernels"/> call. Required ones must be set; optional ones are null when the ONNX input or output is absent.</summary>
/// <typeparam name="T"><see cref="float"/> or <see cref="Half"/>: the type of every floating-point tensor.</typeparam>
public sealed class RecurrentOperands<T>
    where T : unmanaged
{
    /// <summary>The input sequence.</summary>
    public required GpuTensorRef<T> X { get; init; }

    /// <summary>The input weights; used as stored, with no repacking.</summary>
    public required GpuTensorRef<T> W { get; init; }

    /// <summary>The recurrence weights.</summary>
    public required GpuTensorRef<T> R { get; init; }

    /// <summary>The biases, or null for zero.</summary>
    public GpuTensorRef<T>? B { get; init; }

    /// <summary>The valid length of each batch row (int32, values in [0, Seq]), or null for all steps.</summary>
    public GpuTensorRef<int>? SequenceLengths { get; init; }

    /// <summary>The initial hidden state, or null for zeros.</summary>
    public GpuTensorRef<T>? InitialH { get; init; }

    /// <summary>The initial cell state (LSTM), or null for zeros.</summary>
    public GpuTensorRef<T>? InitialC { get; init; }

    /// <summary>The peephole weights (LSTM), or null for none.</summary>
    public GpuTensorRef<T>? P { get; init; }

    /// <summary>Receives the hidden state of every step, or null when not wanted. Zero at steps past a row's length.</summary>
    public GpuTensorRef<T>? Y { get; init; }

    /// <summary>Receives the final hidden state, or null. Zero for a row of length 0.</summary>
    public GpuTensorRef<T>? YH { get; init; }

    /// <summary>Receives the final cell state (LSTM), or null. Zero for a row of length 0.</summary>
    public GpuTensorRef<T>? YC { get; init; }
}
