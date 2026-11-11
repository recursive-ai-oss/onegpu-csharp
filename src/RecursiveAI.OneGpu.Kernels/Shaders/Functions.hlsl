// Scalar functions shared by the element-wise kernels. Operation numbers match the enums in the host code.

#include "Common.hlsl"

// Complementary error function for x >= 0, Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7). It stays accurate relative to its
// own size in the tail, which the GELU functions rely on for large negative inputs.
float ErfcPositive(float a)
{
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    return poly * t * exp(-a * a);
}

float Erfc(float x)
{
    return x >= 0.0 ? ErfcPositive(x) : 2.0 - ErfcPositive(-x);
}

float Erf(float x)
{
    return x >= 0.0 ? 1.0 - ErfcPositive(x) : ErfcPositive(-x) - 1.0;
}

float Sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float Tanh(float x)
{
    // 1 - 2 / (e^2x + 1) is accurate for |x| above about 0.1 and saturates cleanly; a short series covers the middle.
    float a = abs(x);
    if (a < 0.1)
    {
        float x2 = x * x;
        return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0))));
    }

    float y = 1.0 - 2.0 / (exp(2.0 * min(a, 20.0)) + 1.0);
    return IsNaN(x) ? x : (x < 0.0 ? -y : y);
}

// The activations a matrix product can apply to its result; Unary below calls these, so a fused epilogue and the element-wise kernel run the same code.
float ReluF(float x) { return IsNaN(x) ? x : ((x < 0.0) ? 0.0 : x); }
float GeluErfF(float x) { return 0.5 * x * Erfc(-x * 0.70710678118654752); }
float GeluTanhF(float x) { return x / (1.0 + exp(-1.5957691216057308 * (x + 0.044715 * x * x * x))); }  // 0.5 (1 + tanh u) = 1 / (1 + e^-2u)

float Unary(uint op, float x, float p0, float p1)
{
    switch (op)
    {
        case 0u: return x;                                                            // Identity
        case 1u: return asfloat(asuint(x) ^ 0x80000000u);                             // Neg
        case 2u: return asfloat(asuint(x) & 0x7FFFFFFFu);                             // Abs
        case 3u: return sqrt(x);                                                      // Sqrt
        case 4u: return 1.0 / x;                                                      // Reciprocal
        case 5u: return exp(x);                                                       // Exp
        case 6u: return log(x);                                                       // Log
        case 7u: return Tanh(x);                                                      // Tanh
        case 8u: return Erf(x);                                                       // Erf
        case 9u: return Sigmoid(x);                                                   // Sigmoid
        case 10u: return ReluF(x);                                                    // Relu (NaN passes through)
        case 11u: return GeluErfF(x);                                                 // Gelu, exact form
        case 12u: return GeluTanhF(x);                                                // Gelu, tanh form
        case 13u: return x * Sigmoid(x);                                              // Silu / Swish
        case 14u: return IsNaN(x) ? x : max(0.0, min(1.0, p0 * x + p1));              // HardSigmoid(alpha = p0, beta = p1)
        case 15u: return IsNaN(x) ? x : x * max(0.0, min(1.0, x * (1.0 / 6.0) + 0.5)); // HardSwish
        case 16u: return IsNaN(x) ? x : min(max(x, p0), p1);                          // Clip(min = p0, max = p1)
        case 17u: return IsNaN(x) ? x : ((x < 0.0) ? p0 * x : x);                                     // LeakyRelu(alpha = p0)
        default: return x;
    }
}

float PowF(float a, float b)
{
    if (b == 0.0)
    {
        return 1.0;
    }

    if (a < 0.0)
    {
        // pow() of a negative base is undefined in HLSL; ONNX defines it for integral exponents.
        float r = pow(-a, b);
        bool isInt = (b == floor(b));
        if (!isInt)
        {
            return QuietNaN();
        }

        bool odd = (b != 2.0 * floor(b * 0.5));
        return odd ? -r : r;
    }

    return pow(a, b);
}

float Binary(uint op, float a, float b)
{
    switch (op)
    {
        case 0u: return a + b;
        case 1u: return a - b;
        case 2u: return a * b;
        case 3u: return a / b;
        case 4u: return PowF(a, b);
        case 5u: return IsNaN(a) ? a : (IsNaN(b) ? b : (b < a ? b : a));   // Min
        case 6u: return IsNaN(a) ? a : (IsNaN(b) ? b : (b > a ? b : a));   // Max
        default: return a;
    }
}

bool Compare(uint op, float a, float b)
{
    switch (op)
    {
        case 0u: return a == b;
        case 1u: return a < b;
        case 2u: return a > b;
        case 3u: return a <= b;
        case 4u: return a >= b;
        default: return false;
    }
}
