// 8-bit integer matrix product with 32-bit accumulation on the packed dot product: C[m,n] = sum over k of (A[m,k] - zeroA) * (B[k,n] - zeroB),
// as in ONNX MatMulInteger, bit for bit the same integers as the portable kernel (Int8GemmKernel.hlsl).
//
// dot4add_i8packed multiplies four signed bytes pairwise, sums the products and adds a 32-bit accumulator, in one instruction. It only
// takes signed bytes, so every operand is first brought to the signed domain: an unsigned byte u is staged as s = u xor 0x80 = u - 128,
// which turns the zero point z into z' = z - 128 and leaves the sum untouched, because
//     sum (s - z')(t - w')  =  sum s*t  -  w' * sum s  -  z' * sum t  +  K * z' * w'
// holds exactly in the integers, and so also modulo 2^32, where the hardware wraps (Jacob et al., 2018, section 2.3 rearranges the same
// sum). The two byte sums come from the same instruction with an all-ones operand while the tiles are staged; the correction is applied
// once per output element at the end. Padding past K stages as zero bytes, which adds nothing to either sum or to the dot products.
//
// Tiles: BM x BN per workgroup of 256 threads, k in steps of 32 bytes (eight words), each thread owning a (BM/16) x (BN/16) register tile.
// The SMALL variant uses 32x32 tiles so that the 128-row products of small models still fill the device.

#include "Common.hlsl"

struct Int8GemmParams
{
    uint M;
    uint N;
    uint K;
    uint OffA;
    uint OffB;
    uint OffC;
    uint StrideA;
    uint StrideB;
    uint StrideC;
    uint Lda;
    uint Ldb;
    uint Flags;     // bit 0 A is int8, bit 1 B is int8, bit 2 B stored N x K, bit 3 zero point of A present, bit 4 one per row, bit 5 of B present, bit 6 one per column,
                    // bit 7 every row of A (offset, stride, Lda) starts on a word boundary, bit 8 the same for B,
                    // bit 9 the tiles are numbered row-major over a GridX-wide grid instead of x = column tile, y = row tile
    uint OffZA;
    uint OffZB;
    uint Epi;       // see Int8GemmEpilogue.hlsli
    uint OffBias;
    uint OffScale;
    uint OffFBias;
    uint GridX;     // workgroups along x when Flags bit 9 folds the tile grid into one dimension
};

[[vk::push_constant]] ConstantBuffer<Int8GemmParams> P;

[[vk::binding(0, 0)]] StructuredBuffer<uint> A;
[[vk::binding(1, 0)]] StructuredBuffer<uint> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> C;
[[vk::binding(3, 0)]] StructuredBuffer<uint> ZA;
[[vk::binding(4, 0)]] StructuredBuffer<uint> ZB;

#include "Int8GemmEpilogue.hlsli"

#if SMALL
static const uint BM = 32;
static const uint BN = 32;
#else
static const uint BM = 64;
static const uint BN = 64;
#endif
static const uint KW = 8;                        // words (four bytes each) of k per step
static const uint TILE_THREADS = 256;
static const uint A_WORDS = (BM * KW) / TILE_THREADS;
static const uint B_WORDS = (BN * KW) / TILE_THREADS;

groupshared uint As[KW][BM];
groupshared uint Bs[KW][BN];
groupshared int SumA[BM * KW];                   // partial byte sums per staged word, added up at the end
groupshared int SumB[BN * KW];

int Byte(StructuredBuffer<uint> buf, uint i, bool isSigned)
{
    uint b = (buf[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
    return isSigned ? ((int)(b << 24)) >> 24 : (int)b;
}

// Four bytes of a row starting at byte index i, k bytes at most valid. Bytes past the end are zero in the signed domain.
uint Word(StructuredBuffer<uint> buf, uint i, uint valid, bool aligned, uint flip)
{
    if (aligned && valid >= 4u)
    {
        return buf[i >> 2] ^ flip;
    }

    uint w = 0;
    for (uint e = 0; e < 4u; e++)
    {
        if (e < valid)
        {
            uint b = ((buf[(i + e) >> 2] >> (((i + e) & 3u) << 3)) & 0xFFu) ^ (flip & 0xFFu);
            w |= b << (e * 8u);
        }
    }

    return w;
}

// A row of C (or a column): its global index, the sum of its staged bytes and its zero point in the signed domain.
struct Axis
{
    uint idx;
    int sum;
    int z;
};

Axis RowAxis(uint local, uint m0, bool needSum, bool perRow, int scalarZ, bool isSigned)
{
    Axis a;
    a.idx = m0 + local;
    a.sum = 0;
    if (needSum)
    {
        [unroll] for (uint q = 0; q < KW; q++) { a.sum += SumA[local * KW + q]; }
    }

    a.z = scalarZ;
    if (perRow && a.idx < P.M) { a.z = Byte(ZA, P.OffZA + a.idx, isSigned) - (isSigned ? 0 : 128); }
    return a;
}

Axis ColAxis(uint local, uint n0, bool needSum, bool perCol, int scalarZ, bool isSigned)
{
    Axis a;
    a.idx = n0 + local;
    a.sum = 0;
    if (needSum)
    {
        [unroll] for (uint q = 0; q < KW; q++) { a.sum += SumB[local * KW + q]; }
    }

    a.z = scalarZ;
    if (perCol && a.idx < P.N) { a.z = Byte(ZB, P.OffZB + a.idx, isSigned) - (isSigned ? 0 : 128); }
    return a;
}

void Put(uint baseC, Axis r, Axis c, int dot)
{
    if (r.idx < P.M && c.idx < P.N)
    {
        C[baseC + r.idx * P.N + c.idx] = StoreValue(dot - c.z * r.sum - r.z * c.sum + (int)P.K * r.z * c.z, c.idx);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void Int8GemmDotTiled(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint m0 = group.y * BM;
    uint n0 = group.x * BN;
    if ((P.Flags & 512u) != 0u)
    {
        // A product with more tiles than one grid axis can hold: the host lays the tiles out over x and y, numbered row-major.
        // Which tile a workgroup owns depends only on its id, and surplus workgroups of the last row leave before touching anything.
        uint tilesN = (P.N + BN - 1u) / BN;
        uint flat = group.y * P.GridX + group.x;
        m0 = (flat / tilesN) * BM;
        n0 = (flat % tilesN) * BN;
        if (m0 >= P.M)
        {
            return;
        }
    }

    uint batch = group.z;
    uint baseA = P.OffA + batch * P.StrideA;
    uint baseB = P.OffB + batch * P.StrideB;
    uint baseC = P.OffC + batch * P.StrideC;
    bool aSigned = (P.Flags & 1u) != 0u;
    bool bSigned = (P.Flags & 2u) != 0u;
    bool transB = (P.Flags & 4u) != 0u;
    bool hasZA = (P.Flags & 8u) != 0u;
    bool rowZA = (P.Flags & 16u) != 0u;
    bool hasZB = (P.Flags & 32u) != 0u;
    bool colZB = (P.Flags & 64u) != 0u;
    bool alignedA = (P.Flags & 128u) != 0u;
    bool alignedB = (P.Flags & 256u) != 0u;
    uint flipA = aSigned ? 0u : 0x80808080u;
    uint flipB = bSigned ? 0u : 0x80808080u;

    // Zero points in the signed domain. A sum is only needed when the other operand's zero point can be non-zero.
    int zaScalar = (hasZA && !rowZA ? Byte(ZA, P.OffZA, aSigned) : 0) - (aSigned ? 0 : 128);
    int zbScalar = (hasZB && !colZB ? Byte(ZB, P.OffZB, bSigned) : 0) - (bSigned ? 0 : 128);
    bool needSumA = hasZB || !bSigned;
    bool needSumB = hasZA || !aSigned;

    uint tx = tid & 15u;
    uint ty = tid >> 4;

    // Per-thread staging roles stay the same for every k step, so each thread accumulates the byte sums of its own words.
    int partA[A_WORDS];
    int partB[B_WORDS];
    [unroll] for (uint pa = 0; pa < A_WORDS; pa++) { partA[pa] = 0; }
    [unroll] for (uint pb = 0; pb < B_WORDS; pb++) { partB[pb] = 0; }

#if SMALL
    int c00 = 0; int c01 = 0; int c10 = 0; int c11 = 0;
#else
    int c00 = 0; int c01 = 0; int c02 = 0; int c03 = 0; int c10 = 0; int c11 = 0; int c12 = 0; int c13 = 0; int c20 = 0; int c21 = 0; int c22 = 0; int c23 = 0; int c30 = 0; int c31 = 0; int c32 = 0; int c33 = 0;
#endif

    for (uint k0 = 0; k0 < P.K; k0 += KW * 4u)
    {
        [unroll] for (uint e = 0; e < A_WORDS; e++)
        {
            uint idx = tid + e * TILE_THREADS;
            uint r = idx / KW;
            uint w = idx % KW;
            uint gm = m0 + r;
            uint gk = k0 + w * 4u;
            uint v = 0;
            if (gm < P.M && gk < P.K)
            {
                v = Word(A, baseA + gm * P.Lda + gk, P.K - gk, alignedA, flipA);
            }

            As[w][r] = v;
            partA[e] = dot4add_i8packed(v, 0x01010101u, partA[e]);
        }

        [unroll] for (uint f = 0; f < B_WORDS; f++)
        {
            uint idx = tid + f * TILE_THREADS;
            uint c, w;
            if (transB) { c = idx / KW; w = idx % KW; }
            else { c = idx % BN; w = idx / BN; }
            uint gn = n0 + c;
            uint gk = k0 + w * 4u;
            uint v = 0;
            if (gn < P.N && gk < P.K)
            {
                uint valid = min(P.K - gk, 4u);
                if (transB)
                {
                    v = Word(B, baseB + gn * P.Ldb + gk, valid, alignedB, flipB);
                }
                else
                {
                    // Four rows of B, one byte each, packed along k.
                    for (uint e = 0; e < 4u; e++)
                    {
                        if (e < valid)
                        {
                            uint i = baseB + (gk + e) * P.Ldb + gn;
                            uint b = ((B[i >> 2] >> ((i & 3u) << 3)) & 0xFFu) ^ (flipB & 0xFFu);
                            v |= b << (e * 8u);
                        }
                    }
                }
            }

            Bs[w][c] = v;
            partB[f] = dot4add_i8packed(v, 0x01010101u, partB[f]);
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kw = 0; kw < KW; kw++)
        {
#if SMALL
            uint a0 = As[kw][ty + 0];
            uint a1 = As[kw][ty + 16];
            uint b0 = Bs[kw][tx + 0];
            uint b1 = Bs[kw][tx + 16];
            c00 = dot4add_i8packed(a0, b0, c00); c01 = dot4add_i8packed(a0, b1, c01);
            c10 = dot4add_i8packed(a1, b0, c10); c11 = dot4add_i8packed(a1, b1, c11);
#else
            uint a0 = As[kw][ty + 0];
            uint a1 = As[kw][ty + 16];
            uint a2 = As[kw][ty + 32];
            uint a3 = As[kw][ty + 48];
            uint b0 = Bs[kw][tx + 0];
            uint b1 = Bs[kw][tx + 16];
            uint b2 = Bs[kw][tx + 32];
            uint b3 = Bs[kw][tx + 48];
            c00 = dot4add_i8packed(a0, b0, c00); c01 = dot4add_i8packed(a0, b1, c01); c02 = dot4add_i8packed(a0, b2, c02); c03 = dot4add_i8packed(a0, b3, c03);
            c10 = dot4add_i8packed(a1, b0, c10); c11 = dot4add_i8packed(a1, b1, c11); c12 = dot4add_i8packed(a1, b2, c12); c13 = dot4add_i8packed(a1, b3, c13);
            c20 = dot4add_i8packed(a2, b0, c20); c21 = dot4add_i8packed(a2, b1, c21); c22 = dot4add_i8packed(a2, b2, c22); c23 = dot4add_i8packed(a2, b3, c23);
            c30 = dot4add_i8packed(a3, b0, c30); c31 = dot4add_i8packed(a3, b1, c31); c32 = dot4add_i8packed(a3, b2, c32); c33 = dot4add_i8packed(a3, b3, c33);
#endif
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Publish the byte sums: one partial per staged word, in the slot of its row (or column) and word.
    [unroll] for (uint ea = 0; ea < A_WORDS; ea++)
    {
        uint idx = tid + ea * TILE_THREADS;
        SumA[idx] = partA[ea];      // idx = row * KW + word
    }

    [unroll] for (uint eb = 0; eb < B_WORDS; eb++)
    {
        uint idx = tid + eb * TILE_THREADS;
        uint c, w;
        if (transB) { c = idx / KW; w = idx % KW; }
        else { c = idx % BN; w = idx / BN; }
        SumB[c * KW + w] = partB[eb];
    }

    GroupMemoryBarrierWithGroupSync();

    // C = sum s*t - w'[n] * sum_k s[m] - z'[m] * sum_k t[n] + K * z'[m] * w'[n], all modulo 2^32.
#if SMALL
    Axis r0 = RowAxis(ty + 0, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis r1 = RowAxis(ty + 16, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis k0a = ColAxis(tx + 0, n0, needSumB, colZB, zbScalar, bSigned);
    Axis k1a = ColAxis(tx + 16, n0, needSumB, colZB, zbScalar, bSigned);
    Put(baseC, r0, k0a, c00); Put(baseC, r0, k1a, c01);
    Put(baseC, r1, k0a, c10); Put(baseC, r1, k1a, c11);
#else
    Axis r0 = RowAxis(ty + 0, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis r1 = RowAxis(ty + 16, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis r2 = RowAxis(ty + 32, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis r3 = RowAxis(ty + 48, m0, needSumA, rowZA, zaScalar, aSigned);
    Axis k0a = ColAxis(tx + 0, n0, needSumB, colZB, zbScalar, bSigned);
    Axis k1a = ColAxis(tx + 16, n0, needSumB, colZB, zbScalar, bSigned);
    Axis k2a = ColAxis(tx + 32, n0, needSumB, colZB, zbScalar, bSigned);
    Axis k3a = ColAxis(tx + 48, n0, needSumB, colZB, zbScalar, bSigned);
    Put(baseC, r0, k0a, c00); Put(baseC, r0, k1a, c01); Put(baseC, r0, k2a, c02); Put(baseC, r0, k3a, c03);
    Put(baseC, r1, k0a, c10); Put(baseC, r1, k1a, c11); Put(baseC, r1, k2a, c12); Put(baseC, r1, k3a, c13);
    Put(baseC, r2, k0a, c20); Put(baseC, r2, k1a, c21); Put(baseC, r2, k2a, c22); Put(baseC, r2, k3a, c23);
    Put(baseC, r3, k0a, c30); Put(baseC, r3, k1a, c31); Put(baseC, r3, k2a, c32); Put(baseC, r3, k3a, c33);
#endif
}
