// Three-level float32 accumulation for the 4 x 4 register tiles of the portable GEMM and convolution kernels.
//
// One chain over k rounds after every term, and the partial sum it rounds grows with k, so its error grows roughly in proportion
// to k (Higham, Accuracy and Stability of Numerical Algorithms, chapter 4). Summing in blocks and adding the block sums keeps
// every rounding on a smaller partial sum: a staged k step (32 terms) forms a block sum, four block sums (128 terms) a group sum that
// goes into the running total. The order depends only on the shape and the tile size: no atomics, no subgroup operations, nothing
// that varies with scheduling, so one device and driver give the same bits every run.
//
// The tile is sixteen named scalars (no array) because the Metal output keeps indexed arrays in thread-local memory.

#define TILE_DECLARE(p) float p##00 = 0.0; float p##01 = 0.0; float p##02 = 0.0; float p##03 = 0.0; float p##10 = 0.0; float p##11 = 0.0; float p##12 = 0.0; float p##13 = 0.0; float p##20 = 0.0; float p##21 = 0.0; float p##22 = 0.0; float p##23 = 0.0; float p##30 = 0.0; float p##31 = 0.0; float p##32 = 0.0; float p##33 = 0.0;

#define TILE_ZERO(p) p##00 = 0.0; p##01 = 0.0; p##02 = 0.0; p##03 = 0.0; p##10 = 0.0; p##11 = 0.0; p##12 = 0.0; p##13 = 0.0; p##20 = 0.0; p##21 = 0.0; p##22 = 0.0; p##23 = 0.0; p##30 = 0.0; p##31 = 0.0; p##32 = 0.0; p##33 = 0.0;

// d += s, element by element.
#define TILE_ADD(d, s) d##00 += s##00; d##01 += s##01; d##02 += s##02; d##03 += s##03; d##10 += s##10; d##11 += s##11; d##12 += s##12; d##13 += s##13; d##20 += s##20; d##21 += s##21; d##22 += s##22; d##23 += s##23; d##30 += s##30; d##31 += s##31; d##32 += s##32; d##33 += s##33;

// One k step of the tile: t (zeroed by the caller) takes the 16 products of the staged step as a chain.
#define TILE_STEP(t, a0, a1, a2, a3, b0, b1, b2, b3) \
    t##00 = mad(a0, b0, t##00); t##01 = mad(a0, b1, t##01); t##02 = mad(a0, b2, t##02); t##03 = mad(a0, b3, t##03); \
    t##10 = mad(a1, b0, t##10); t##11 = mad(a1, b1, t##11); t##12 = mad(a1, b2, t##12); t##13 = mad(a1, b3, t##13); \
    t##20 = mad(a2, b0, t##20); t##21 = mad(a2, b1, t##21); t##22 = mad(a2, b2, t##22); t##23 = mad(a2, b3, t##23); \
    t##30 = mad(a3, b0, t##30); t##31 = mad(a3, b1, t##31); t##32 = mad(a3, b2, t##32); t##33 = mad(a3, b3, t##33);

// Block sums (two staged k steps, 32 terms each) that collect in the middle level before it joins the total: 128 terms per group.
static const uint TILE_STEPS_PER_GROUP = 4;

// Cooperative-matrix products take a run of staged k steps into a second accumulator ("part"), which then joins the total. Each fold
// costs about as much as two steps of products on Apple hardware, so runs are long: 8 steps (128 terms) per part.
// The run length is fixed, so the order depends on the shape alone.
#define COOP_GROUP_STEPS 8
#define COOP_FOLD(acc, part) \
    acc[0] += part[0]; acc[1] += part[1]; acc[2] += part[2]; acc[3] += part[3]; \
    acc[4] += part[4]; acc[5] += part[5]; acc[6] += part[6]; acc[7] += part[7]; \
    part[0] = 0.0; part[1] = 0.0; part[2] = 0.0; part[3] = 0.0; part[4] = 0.0; part[5] = 0.0; part[6] = 0.0; part[7] = 0.0;
#if COOP_FOUR
#define COOP_FOLD_ALL COOP_FOLD(acc00, part00) COOP_FOLD(acc10, part10) COOP_FOLD(acc01, part01) COOP_FOLD(acc11, part11)
#else
#define COOP_FOLD_ALL COOP_FOLD(acc00, part00) COOP_FOLD(acc10, part10)
#endif
#define COOP_STEP_DONE if (++foldSteps == COOP_GROUP_STEPS) { COOP_FOLD_ALL foldSteps = 0; }
