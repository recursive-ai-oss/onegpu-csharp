// Cooperative-matrix GEMM with eight subgroups per workgroup, F32 inputs (Metal only). See GemmCoop.hlsli.
#define GEMM_WIDE
#define GEMM_F32
#include "GemmCoop.hlsli"
