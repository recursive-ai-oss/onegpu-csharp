// Cooperative-matrix GEMM with eight subgroups per workgroup, BF16 inputs. See GemmCoop.hlsli.
#define GEMM_WIDE
#define GEMM_BF16
#include "GemmCoop.hlsli"
