// Three-level float32 accumulation for the 4 x 4 register tiles of the portable GEMM and convolution kernels.
//
// One chain over k rounds after every term, and the partial sum it rounds grows with k, so its error grows roughly in proportion
// to k (Higham, Accuracy and Stability of Numerical Algorithms, chapter 4). Summing in blocks and adding the block sums keeps
// every rounding on a smaller partial sum: a 16-term block (one staged k step) goes into a step sum, eight block sums (128 terms)
// go into the running total. The order depends only on the shape and the tile size: no atomics, no subgroup operations, nothing
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

// Staged k steps whose sums collect in the middle level before they join the total.
static const uint TILE_STEPS_PER_GROUP = 8;

// Cooperative-matrix products take a short run of staged k steps into a second accumulator ("part"), which joins a third ("mid"),
// which joins the total: three levels as in the register tile above, with 2 steps (32 terms) in a part and 4 parts (128 terms) in a
// mid. The chain rounds on smaller partial sums (see above). The run lengths are fixed, so the order depends on the shape alone.
#define COOP_GROUP_STEPS 2
#define COOP_MID_GROUPS 4
#define COOP_FOLD(acc, part) for (uint fi_ = 0; fi_ < part.Length(); fi_++) { acc[fi_] += part[fi_]; part[fi_] = 0.0; }
#if COOP_FOUR
#define COOP_FOLD_PART COOP_FOLD(mid00, part00) COOP_FOLD(mid10, part10) COOP_FOLD(mid01, part01) COOP_FOLD(mid11, part11)
#define COOP_FOLD_MID COOP_FOLD(acc00, mid00) COOP_FOLD(acc10, mid10) COOP_FOLD(acc01, mid01) COOP_FOLD(acc11, mid11)
#else
#define COOP_FOLD_PART COOP_FOLD(mid00, part00) COOP_FOLD(mid10, part10)
#define COOP_FOLD_MID COOP_FOLD(acc00, mid00) COOP_FOLD(acc10, mid10)
#endif
#define COOP_FOLD_ALL COOP_FOLD_PART COOP_FOLD_MID
#define COOP_STEP_DONE \
    if (++foldSteps == COOP_GROUP_STEPS) \
    { \
        COOP_FOLD_PART \
        foldSteps = 0; \
        if (++foldGroups == COOP_MID_GROUPS) { COOP_FOLD_MID foldGroups = 0; } \
    }
