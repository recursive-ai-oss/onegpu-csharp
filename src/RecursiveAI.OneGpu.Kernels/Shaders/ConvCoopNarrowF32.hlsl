// The narrow tile of ConvCoopF32.hlsl: 64 x 32 per workgroup, for images too small to fill the GPU with 64 x 64 tiles.
#define NARROW
#include "ConvCoopF32.hlsl"
