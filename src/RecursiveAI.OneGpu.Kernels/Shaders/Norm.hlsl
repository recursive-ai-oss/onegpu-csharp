// Layer normalisation and RMS normalisation over the last axis (ONNX LayerNormalization and SimplifiedLayerNormalization):
//   layer: y = (x - mean) / sqrt(variance + epsilon) * scale + bias
//   rms:   y = x / sqrt(mean(x^2) + epsilon) * scale
// Statistics are in single precision, also for half inputs. Each row is one workgroup; the sums use the block reduction of Block.hlsl, so
// the build without WAVE has a fixed summation order. F16 builds need even Len and even offsets (the host checks): they write whole words.

#include "Block.hlsl"

#if F16
typedef uint Elem;
#else
typedef float Elem;
#endif

struct NormParams
{
    uint Rows;
    uint Len;
    uint InOff;
    uint OutOff;
    uint ScaleOff;
    uint BiasOff;
    uint Flags;     // bit 0: scale, bit 1: bias, bit 2: RMS
    float Eps;
    uint GridX;
};

[[vk::push_constant]] ConstantBuffer<NormParams> NP;

[[vk::binding(0, 0)]] StructuredBuffer<Elem> NIn;
[[vk::binding(1, 0)]] StructuredBuffer<Elem> NScale;
[[vk::binding(2, 0)]] StructuredBuffer<Elem> NBias;
[[vk::binding(3, 0)]] RWStructuredBuffer<Elem> NOut;

float LoadN(uint i)
{
#if F16
    return LoadH(NIn, i);
#else
    return NIn[i];
#endif
}

float LoadScale(uint i)
{
#if F16
    return LoadH(NScale, NP.ScaleOff + i);
#else
    return NScale[NP.ScaleOff + i];
#endif
}

float LoadBias(uint i)
{
#if F16
    return LoadH(NBias, NP.BiasOff + i);
#else
    return NBias[NP.BiasOff + i];
#endif
}

float Normalise(uint col, float x, float mean, float inv)
{
    float y = (x - mean) * inv;
    if ((NP.Flags & 1u) != 0u)
    {
        y *= LoadScale(col);
    }

    if ((NP.Flags & 2u) != 0u)
    {
        y += LoadBias(col);
    }

    return y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void NormBlock(uint2 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    // A surplus workgroup of the rounded-up grid leaves before any barrier (see ReduceBlock).
    uint row = gid.y * NP.GridX + gid.x;
    if (row >= NP.Rows)
    {
        return;
    }

    bool rms = (NP.Flags & 4u) != 0u;
    uint rowIn = NP.InOff + row * NP.Len;
    float n = (float)NP.Len;

    float mean = 0.0;
    if (!rms)
    {
        float s = 0.0;
        for (uint i = tid; i < NP.Len; i += WG)
        {
            s += LoadN(rowIn + i);
        }

        mean = BlockReduce(0u, s, tid) / n;
    }

    float q = 0.0;
    for (uint j = tid; j < NP.Len; j += WG)
    {
        float dlt = LoadN(rowIn + j) - mean;
        q += dlt * dlt;
    }

    float variance = BlockReduce(0u, q, tid) / n;
    float inv = 1.0 / sqrt(variance + NP.Eps);

    uint rowOut = NP.OutOff + row * NP.Len;
#if F16
    for (uint w = tid; w < (NP.Len >> 1); w += WG)
    {
        uint c = 2u * w;
        uint lo = F2H(Normalise(c, LoadN(rowIn + c), mean, inv));
        uint hi = F2H(Normalise(c + 1u, LoadN(rowIn + c + 1u), mean, inv));
        NOut[(rowOut + c) >> 1] = lo | (hi << 16);
    }
#else
    for (uint c = tid; c < NP.Len; c += WG)
    {
        NOut[rowOut + c] = Normalise(c, NIn[rowIn + c], mean, inv);
    }
#endif
}
