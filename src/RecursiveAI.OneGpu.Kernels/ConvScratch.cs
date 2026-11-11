namespace RecursiveAI.OneGpu.Kernels;

/// <summary>The scratch a convolution needs, in elements of the scratch buffers' own types. See <see cref="ConvKernels.Scratch(in ConvProblem, bool)"/>.</summary>
/// <param name="ColumnElements">Elements of the column scratch (halves for a float16 convolution, floats otherwise) for the whole batch; 0 when none is needed.</param>
/// <param name="AccumulatorElements">Float32 elements of the accumulator scratch for the whole batch (float16 convolutions only); 0 when none is needed.</param>
/// <param name="ColumnElementsPerImage">The least column scratch that works: one image's worth. A smaller buffer than <paramref name="ColumnElements"/> makes the batch run in chunks.</param>
/// <param name="AccumulatorElementsPerImage">The least accumulator scratch that works: one image's worth.</param>
public readonly record struct ConvScratch(long ColumnElements, long AccumulatorElements, long ColumnElementsPerImage, long AccumulatorElementsPerImage);
