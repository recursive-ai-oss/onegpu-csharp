// The narrow tile (64 x 32) of ConvCoopF32.hlsl, for images too small to fill the GPU with 64 x 64 tiles.
#define CONV_F32
#define NARROW
#include "ConvCoop.hlsli"
