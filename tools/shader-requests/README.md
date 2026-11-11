# Requests for the shader compiler

Each file is the smallest program that needs one feature, with the output the GEMM and int8 kernels would use. They are kept as the record
of what was asked and of the output that was expected.

**Status.** Answered by OneShader 1.5.0: cooperative matrices (`CooperativeMatrix<T, R, C, MatrixUse::...>`), `bfloat16_t`, the
`dot4add_{i,u}8packed` intrinsics on every target, and a uniform `SV_GroupID` in WGSL. The float16 and bfloat16 GEMM kernels are now HLSL
(`src/RecursiveAI.OneGpu.Kernels/Shaders/GemmCoop.hlsli`) and the hand-written SPIR-V and Metal sources are gone. The syntax differs from the
files below in one place: the shader build needs `Enable16BitTypes` for `float16_t`. The packed dot product is available but no kernel
uses it yet.

## `cooperative_matrix.hlsl`, `bfloat16.hlsl`

Expected SPIR-V (Vulkan 1.2 with `VK_KHR_cooperative_matrix`):

```
OpCapability CooperativeMatrixKHR            ; also: OpCapability VulkanMemoryModel (the extension requires it) and Float16
OpExtension "SPV_KHR_cooperative_matrix"
OpMemoryModel Logical Vulkan
%mat_a = OpTypeCooperativeMatrixKHR %half %uint_3 %uint_16 %uint_16 %uint_0     ; Scope Subgroup = 3, Use MatrixAKHR = 0
%mat_b = OpTypeCooperativeMatrixKHR %half %uint_3 %uint_16 %uint_16 %uint_1     ; MatrixBKHR = 1
%mat_c = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_16 %uint_16 %uint_2    ; MatrixAccumulatorKHR = 2
OpCooperativeMatrixLoadKHR %mat_a %ptr %layout_row_major %stride NonPrivatePointer     ; pointer to an element in Workgroup or StorageBuffer
OpCooperativeMatrixMulAddKHR %mat_c %a %b %acc
OpCooperativeMatrixStoreKHR %ptr %acc %layout_row_major %stride NonPrivatePointer
```

Loads and stores from workgroup memory use `NonPrivatePointer`; `OpControlBarrier` between staging and loading uses acquire-release
semantics with `WorkgroupMemory | MakeAvailable | MakeVisible`. For bfloat16 the element type is `OpTypeFloat 16 BFloat16KHR` with
`OpCapability BFloat16TypeKHR`, `OpCapability BFloat16CooperativeMatrixKHR` and `OpExtension "SPV_KHR_bfloat16"`; float to bfloat16 is `OpFConvert`.
Device features: `cooperativeMatrix`, `vulkanMemoryModel`, and for bfloat16 `shaderBFloat16Type` and `shaderBFloat16CooperativeMatrix`.
A complete kernel of this kind is `Shaders/GemmCoop.hlsli`.

Expected MSL (Metal 3.1 for bfloat): a 16 x 16 matrix is a 2 x 2 arrangement of `simdgroup_matrix<T, 8, 8>`; `Load` becomes four
`simdgroup_load(m[i][j], ptr + (8*i)*stride + 8*j, stride)` calls (layout RowMajor; ColumnMajor passes `ulong2(0,0)` and `transpose = true`),
`MulAdd` four `simdgroup_multiply_accumulate` per k-step pair (`d[i][j] += a[i][k] * b[k][j]` over k = 0..1), `Store` four `simdgroup_store`.
Element types: `half`, `float`, `bfloat`, with float accumulators for 16-bit inputs. `Zero()` is `make_filled_simdgroup_matrix<T, 8, 8>(0)`.
The compiler's output for `Shaders/GemmCoop.hlsli` follows this scheme.

WGSL: unsupported. The compiler should report a diagnostic that names the feature and the target, so a kernel with a portable fallback
is built for the other targets only.

## `dot4_packed.hlsl`

SPIR-V: `OpSDot` / `OpUDot` (`OpCapability DotProduct`, `DotProductInput4x8BitPacked`, `OpExtension "SPV_KHR_integer_dot_product"`) with
`PackedVectorFormat4x8Bit`, then `OpIAdd` of the accumulator; or the accumulating form `OpSDotAccSat` is not wanted (the HLSL intrinsic wraps).
Device feature: `shaderIntegerDotProduct`.
MSL: no instruction; emit the four-lane expansion: `as_type<char4>(a)`, `as_type<char4>(b)` widened to `int4`, `dot()` then add (for unsigned, `uchar4`/`uint4`).
WGSL: `dot4I8Packed(a, b)` and `dot4U8Packed(a, b)` (language feature `packed_4x8_integer_dot_product`) plus the addition.
