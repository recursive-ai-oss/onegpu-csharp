// Defect report: the Metal output drops a `continue` guarded by a condition inside nested [unroll] loops.
//
// Expected: elements with row >= Count.x or col >= Count.y are not written (the output buffer keeps its 7s outside the 3 x 5 block).
// Observed with Metal output (compiled MSL, run on an Apple M2): the condition is computed but the branch is missing, so every element
// is written. The SPIR-V output of the same file behaves as expected. Rewriting the loop body as `if (row < Count.x && col < Count.y) { ... }`
// works around it.

cbuffer P : register(b0) { uint2 Count; };
RWStructuredBuffer<uint> Out : register(u1);

[shader("compute")]
[numthreads(1, 1, 1)]
void Main()
{
    [unroll] for (uint row = 0; row < 4; row++)
    {
        if (row >= Count.x) { continue; }
        [unroll] for (uint col = 0; col < 4; col++)
        {
            if (col >= Count.y) { continue; }
            Out[row * 4 + col] = 1;
        }
    }
}
