namespace RecursiveAI.OneGpu.Kernels;

/// <summary>Binary arithmetic with ONNX broadcasting. The numbers are the shader's operation codes.</summary>
public enum GpuBinaryOp
{
    /// <summary>a + b.</summary>
    Add = 0,
    /// <summary>a - b.</summary>
    Sub = 1,
    /// <summary>a * b.</summary>
    Mul = 2,
    /// <summary>a / b (IEEE division: x/0 is infinity, 0/0 is NaN).</summary>
    Div = 3,
    /// <summary>a raised to b; a negative base needs an integral exponent, otherwise NaN.</summary>
    Pow = 4,
    /// <summary>The smaller of a and b; NaN if either is NaN.</summary>
    Min = 5,
    /// <summary>The larger of a and b; NaN if either is NaN.</summary>
    Max = 6,
}

/// <summary>Unary functions. The numbers are the shader's operation codes.</summary>
public enum GpuUnaryOp
{
    /// <summary>x.</summary>
    Identity = 0,
    /// <summary>-x.</summary>
    Neg = 1,
    /// <summary>|x|.</summary>
    Abs = 2,
    /// <summary>Square root.</summary>
    Sqrt = 3,
    /// <summary>1 / x.</summary>
    Reciprocal = 4,
    /// <summary>e^x.</summary>
    Exp = 5,
    /// <summary>Natural logarithm.</summary>
    Log = 6,
    /// <summary>Hyperbolic tangent.</summary>
    Tanh = 7,
    /// <summary>Error function (absolute error below 1.5e-7).</summary>
    Erf = 8,
    /// <summary>1 / (1 + e^-x).</summary>
    Sigmoid = 9,
    /// <summary>max(x, 0); NaN passes through.</summary>
    Relu = 10,
    /// <summary>x * Phi(x) with the exact normal CDF (the ONNX <c>Gelu</c> with approximate "none").</summary>
    GeluErf = 11,
    /// <summary>The tanh approximation of GELU (the ONNX <c>Gelu</c> with approximate "tanh").</summary>
    GeluTanh = 12,
    /// <summary>x * sigmoid(x), also called Swish.</summary>
    Silu = 13,
    /// <summary>max(0, min(1, alpha * x + beta)) with <c>alpha</c> = p0 and <c>beta</c> = p1 (ONNX defaults 0.2 and 0.5).</summary>
    HardSigmoid = 14,
    /// <summary>x * max(0, min(1, x / 6 + 0.5)).</summary>
    HardSwish = 15,
    /// <summary>min(max(x, p0), p1).</summary>
    Clip = 16,
    /// <summary>x for x >= 0, else p0 * x.</summary>
    LeakyRelu = 17,
}

/// <summary>Comparisons whose result is a 32-bit 0 or 1 per element. The numbers are the shader's operation codes.</summary>
public enum GpuCompareOp
{
    /// <summary>a == b (NaN is never equal).</summary>
    Equal = 0,
    /// <summary>a &lt; b.</summary>
    Less = 1,
    /// <summary>a &gt; b.</summary>
    Greater = 2,
    /// <summary>a &lt;= b.</summary>
    LessOrEqual = 3,
    /// <summary>a &gt;= b.</summary>
    GreaterOrEqual = 4,
}

/// <summary>A unary function applied to the result of a binary operation inside the same kernel, for example the bias add and activation of a feed-forward layer.</summary>
/// <param name="Op">The function.</param>
/// <param name="P0">First parameter of the function, where it has one (see <see cref="GpuUnaryOp"/>).</param>
/// <param name="P1">Second parameter of the function, where it has one.</param>
public readonly record struct GpuActivation(GpuUnaryOp Op, float P0 = 0f, float P1 = 0f);

/// <summary>Arithmetic on int32 and int64 tensors. Add, Sub, Mul and Neg wrap on overflow, as two's complement does.</summary>
public enum GpuIntegerOp
{
    /// <summary>a + b.</summary>
    Add = 0,
    /// <summary>a - b.</summary>
    Sub = 1,
    /// <summary>a * b.</summary>
    Mul = 2,
    /// <summary>a / b truncated toward zero; 0 when b is 0; the minimum value divided by -1 wraps to itself.</summary>
    Div = 3,
    /// <summary>The remainder with the sign of the divisor (ONNX <c>Mod</c> with <c>fmod = 0</c>); 0 when b is 0 or -1.</summary>
    Mod = 4,
    /// <summary>The remainder with the sign of the dividend (ONNX <c>Mod</c> with <c>fmod = 1</c>); 0 when b is 0 or -1.</summary>
    Fmod = 5,
    /// <summary>The smaller of a and b.</summary>
    Min = 6,
    /// <summary>The larger of a and b.</summary>
    Max = 7,
}

/// <summary>Logical operations on bool tensors.</summary>
public enum GpuLogicOp
{
    /// <summary>a and b.</summary>
    And = 13,
    /// <summary>a or b.</summary>
    Or = 14,
    /// <summary>a xor b.</summary>
    Xor = 15,
}

/// <summary>Unary functions of int32 and int64 tensors.</summary>
public enum GpuIntegerUnaryOp
{
    /// <summary>-x, wrapping: the minimum value stays the minimum value.</summary>
    Neg = 16,
    /// <summary>|x|, wrapping: the minimum value stays the minimum value.</summary>
    Abs = 17,
}
