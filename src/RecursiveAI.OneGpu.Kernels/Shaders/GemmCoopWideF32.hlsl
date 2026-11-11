// Cooperative-matrix GEMM with eight subgroups per workgroup, F32 inputs (Metal only). See GemmCoop.hlsli.
#define GEMM_WIDE
#define GEMM_F32
#define GEMM_DIRECT
#include "GemmCoop.hlsli"
