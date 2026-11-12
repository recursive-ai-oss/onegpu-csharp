// The WebGPU build of Attention.hlsl, with loops instead of unrolled statements (see the note there).
#define ATTN_LOOPS 1
#define VEC4 1
#include "Attention.hlsl"
