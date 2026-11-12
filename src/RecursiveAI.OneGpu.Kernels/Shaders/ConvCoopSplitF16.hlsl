// Float16 convolution on simdgroup matrices that leaves float32 partial sums in a plane for the epilogue pass (a split k axis, or an activation the kernel does not fuse).
#define CONV_F16
#define SPLIT
#include "ConvCoop.hlsli"
