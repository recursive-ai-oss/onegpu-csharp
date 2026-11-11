// Cooperative-matrix GEMM with eight subgroups per workgroup, F16 inputs. See GemmCoop.hlsli.
#define GEMM_WIDE
#define GEMM_F16
#include "GemmCoop.hlsli"
