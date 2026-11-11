#!/usr/bin/env python3
"""Writes the SPIR-V assembly of the cooperative-matrix GEMM kernels (float16 and bfloat16 inputs).

Why this exists: the shader compiler used for the other kernels has no cooperative-matrix types yet, so these two kernels are
written as SPIR-V directly, following the SPV_KHR_cooperative_matrix and SPV_KHR_bfloat16 specifications. The assembly is produced
by this script (unrolled loads and fixed tile geometry are easier to write as code than as text), assembled with spirv-as and checked
with spirv-val by tools/kernels/build-raw.sh, and the binary is committed beside the C# that loads it. Remove the script and the
binaries when the shader compiler can express the kernel.

Kernel shape: C = alpha * op(A) * op(B) + beta * C, one 64 x 64 tile of C per workgroup of 128 invocations (four 32-wide subgroups,
arranged 2 x 2, each owning a 32 x 32 sub-tile held as four 16 x 16 float accumulators). The k loop runs in steps of 16: the A and B
tiles are staged through workgroup memory in the matrix element type (this is where bounds, transposes and odd leading dimensions are
handled), then each subgroup multiplies its fragments with OpCooperativeMatrixMulAddKHR. The accumulators leave through workgroup memory so
alpha, beta and the matrix edges are handled by ordinary code.
"""
import sys

kind = sys.argv[1]  # "f16" or "bf16"
assert kind in ("f16", "bf16")

lines = []
consts = {}
counter = [0]


def t(prefix="t"):
    counter[0] += 1
    return f"%{prefix}{counter[0]}"


def U(n):
    name = f"%u{n}"
    consts[name] = f"{name} = OpConstant %uint {n}"
    return name


def op(name, text):
    lines.append(f"{name} = {text}")
    return name


def uadd(a, b): return op(t(), f"OpIAdd %uint {a} {b}")
def umul(a, b): return op(t(), f"OpIMul %uint {a} {b}")
def shr(a, n): return op(t(), f"OpShiftRightLogical %uint {a} {U(n)}")
def shl(a, n): return op(t(), f"OpShiftLeftLogical %uint {a} {U(n)}")
def band(a, n): return op(t(), f"OpBitwiseAnd %uint {a} {U(n)}")
def sel(c, a, b): return op(t(), f"OpSelect %uint {c} {a} {b}")
def ult(a, b): return op(t(), f"OpULessThan %bool {a} {b}")
def land(a, b): return op(t(), f"OpLogicalAnd %bool {a} {b}")


elem = "%half" if kind == "f16" else "%bf16"

# ---- loading one element of A or B from a buffer of 32-bit words holding two 16-bit values each ----
def load_element(buf, idx):
    wi = shr(idx, 1)
    p = op(t(), f"OpAccessChain %ptr_sb_uint {buf} {U(0)} {wi}")
    w = op(t(), f"OpLoad %uint {p}")
    odd = band(idx, 1)
    if kind == "f16":
        v2 = op(t(), f"OpExtInst %v2float %glsl UnpackHalf2x16 {w}")
        f = op(t(), f"OpVectorExtractDynamic %float {v2} {odd}")
    else:
        isodd = op(t(), f"OpINotEqual %bool {odd} {U(0)}")
        hi = op(t(), f"OpBitwiseAnd %uint {w} %u_ffff0000")
        lo = shl(w, 16)
        bits = sel(isodd, hi, lo)
        f = op(t(), f"OpBitcast %float {bits}")
    return op(t(), f"OpFConvert {elem} {f}")


def stage_element(buf, base, ld, rows_in, cols_in, trans, row0, col0, stored_trans_flag, shared, shared_stride, a_side, e, tid, k0, limit_rows, limit_k):
    """Loads element number e of this invocation's share of a tile and stores it into workgroup memory."""
    idx = uadd(tid, U(128 * e)) if e else tid
    if a_side:
        # tile is 64 (rows) x 16 (k). Normal: A stored M x K, consecutive threads walk k. Transposed: stored K x M, they walk m.
        rn, kn = shr(idx, 4), band(idx, 15)
        rt, kt = band(idx, 63), shr(idx, 6)
    else:
        # tile is 16 (k) x 64 (cols). Normal: B stored K x N, consecutive threads walk n. Transposed: stored N x K, they walk k.
        rn, kn = band(idx, 63), shr(idx, 6)     # here "r" is the column
        rt, kt = shr(idx, 4), band(idx, 15)
    r = sel(trans, rt, rn)
    k = sel(trans, kt, kn)
    g_r = uadd(row0, r)
    g_k = uadd(k0, k)
    inb = land(ult(g_r, limit_rows), ult(g_k, limit_k))
    addr_n = uadd(umul(g_r, ld), g_k) if a_side else uadd(umul(g_k, ld), g_r)
    addr_t = uadd(umul(g_k, ld), g_r) if a_side else uadd(umul(g_r, ld), g_k)
    # a_side: normal address = row * ld + k, transposed = k * ld + row.  b_side: normal = k * ld + col, transposed = col * ld + k.
    addr = uadd(base, sel(trans, addr_t, addr_n))
    safe = sel(inb, addr, U(0))
    h = load_element(buf, safe)
    v = op(t(), f"OpSelect {elem} {inb} {h} %zero_elem")
    if a_side:
        s_idx = uadd(umul(r, U(shared_stride)), k)
    else:
        s_idx = uadd(umul(k, U(shared_stride)), r)
    sp = op(t(), f"OpAccessChain %ptr_wg_elem {shared} {s_idx}")
    lines.append(f"OpStore {sp} {v} NonPrivatePointer")


header = []
header.append("OpCapability Shader")
header.append("OpCapability Float16")
header.append("OpCapability GroupNonUniform")
header.append("OpCapability CooperativeMatrixKHR")
header.append("OpCapability VulkanMemoryModel")
if kind == "bf16":
    header.append("OpCapability BFloat16TypeKHR")
    header.append("OpCapability BFloat16CooperativeMatrixKHR")
header.append('OpExtension "SPV_KHR_cooperative_matrix"')
if kind == "bf16":
    header.append('OpExtension "SPV_KHR_bfloat16"')
header.append('%glsl = OpExtInstImport "GLSL.std.450"')
header.append("OpMemoryModel Logical Vulkan")
header.append('OpEntryPoint GLCompute %main "main" %wg_id %local_index %sg_id %sg_lane %buf_a %buf_b %buf_c %push %tile_a %tile_b %tile_c')
header.append("OpExecutionMode %main LocalSize 128 1 1")
header.append('OpName %main "main"')
header += [
    "OpDecorate %wg_id BuiltIn WorkgroupId",
    "OpDecorate %local_index BuiltIn LocalInvocationIndex",
    "OpDecorate %sg_id BuiltIn SubgroupId",
    "OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId",
    "OpDecorate %rta_uint ArrayStride 4",
    "OpDecorate %rta_float ArrayStride 4",
    "OpDecorate %sb_uint Block",
    "OpDecorate %sb_float Block",
    "OpMemberDecorate %sb_uint 0 Offset 0",
    "OpMemberDecorate %sb_uint 0 NonWritable",
    "OpMemberDecorate %sb_float 0 Offset 0",
    "OpDecorate %buf_a DescriptorSet 0", "OpDecorate %buf_a Binding 0",
    "OpDecorate %buf_b DescriptorSet 0", "OpDecorate %buf_b Binding 1",
    "OpDecorate %buf_c DescriptorSet 0", "OpDecorate %buf_c Binding 2",
    "OpDecorate %push_t Block",
]
names = ["M", "N", "K", "Lda", "Ldb", "Ldc", "OffsetA", "OffsetB", "OffsetC", "StrideA", "StrideB", "StrideC", "Flags", "Alpha", "Beta"]
for i in range(len(names)):
    header.append(f"OpMemberDecorate %push_t {i} Offset {4 * i}")

types = [
    "%void = OpTypeVoid",
    "%fn_void = OpTypeFunction %void",
    "%bool = OpTypeBool",
    "%uint = OpTypeInt 32 0",
    "%float = OpTypeFloat 32",
    "%v2float = OpTypeVector %float 2",
    "%v3uint = OpTypeVector %uint 3",
    "%half = OpTypeFloat 16" if kind == "f16" else "%bf16 = OpTypeFloat 16 BFloat16KHR",
    f"%zero_elem = OpConstantNull {elem}",
    "%u_ffff0000 = OpConstant %uint 4294901760",
    "%f_zero = OpConstant %float 0",
    "%rta_uint = OpTypeRuntimeArray %uint",
    "%rta_float = OpTypeRuntimeArray %float",
    "%sb_uint = OpTypeStruct %rta_uint",
    "%sb_float = OpTypeStruct %rta_float",
    "%ptr_sb_sb_uint = OpTypePointer StorageBuffer %sb_uint",
    "%ptr_sb_sb_float = OpTypePointer StorageBuffer %sb_float",
    "%ptr_sb_uint = OpTypePointer StorageBuffer %uint",
    "%ptr_sb_float = OpTypePointer StorageBuffer %float",
    "%buf_a = OpVariable %ptr_sb_sb_uint StorageBuffer",
    "%buf_b = OpVariable %ptr_sb_sb_uint StorageBuffer",
    "%buf_c = OpVariable %ptr_sb_sb_float StorageBuffer",
    "%push_t = OpTypeStruct " + " ".join(["%uint"] * 13 + ["%float"] * 2),
    "%ptr_pc_push = OpTypePointer PushConstant %push_t",
    "%ptr_pc_uint = OpTypePointer PushConstant %uint",
    "%ptr_pc_float = OpTypePointer PushConstant %float",
    "%push = OpVariable %ptr_pc_push PushConstant",
    "%ptr_in_v3uint = OpTypePointer Input %v3uint",
    "%ptr_in_uint = OpTypePointer Input %uint",
    "%wg_id = OpVariable %ptr_in_v3uint Input",
    "%local_index = OpVariable %ptr_in_uint Input",
    "%sg_id = OpVariable %ptr_in_uint Input",
    "%sg_lane = OpVariable %ptr_in_uint Input",
    "%ptr_wg_elem = OpTypePointer Workgroup " + elem,
    "%ptr_wg_float = OpTypePointer Workgroup %float",
    "%arr_a = OpTypeArray " + elem + " %u_a_size",
    "%arr_b = OpTypeArray " + elem + " %u_b_size",
    "%arr_c = OpTypeArray %float %u_c_size",
    "%ptr_wg_arr_a = OpTypePointer Workgroup %arr_a",
    "%ptr_wg_arr_b = OpTypePointer Workgroup %arr_b",
    "%ptr_wg_arr_c = OpTypePointer Workgroup %arr_c",
    "%tile_a = OpVariable %ptr_wg_arr_a Workgroup",
    "%tile_b = OpVariable %ptr_wg_arr_b Workgroup",
    "%tile_c = OpVariable %ptr_wg_arr_c Workgroup",
    f"%mat_a = OpTypeCooperativeMatrixKHR {elem} %u3 %u16 %u16 %u0",
    f"%mat_b = OpTypeCooperativeMatrixKHR {elem} %u3 %u16 %u16 %u1",
    "%mat_c = OpTypeCooperativeMatrixKHR %float %u3 %u16 %u16 %u2",
    "%ptr_fn_mat_c = OpTypePointer Function %mat_c",
    "%zero_mat_c = OpConstantNull %mat_c",
]

# Workgroup memory: A tile 64 rows x (16 k + 8 pad) and B tile 16 k x (64 cols + 8 pad). The padding keeps rows 16-byte aligned for
# the fragment loads and spreads them across banks. C staging: 4 subgroups x 256 floats.
A_STRIDE, B_STRIDE = 24, 72
BARRIER = 0x6108  # acquire-release on workgroup memory, making writes available and visible
for n in (0, 1, 2, 3, 16, 128, BARRIER, A_STRIDE * 64, B_STRIDE * 16, 1024):
    U(n)
consts["%u_a_size"] = f"%u_a_size = OpConstant %uint {A_STRIDE * 64}"
consts["%u_b_size"] = f"%u_b_size = OpConstant %uint {B_STRIDE * 16}"
consts["%u_c_size"] = "%u_c_size = OpConstant %uint 1024"

body = []

def L(s):
    lines.append(s)

# ---- main ----
L("%main = OpFunction %void None %fn_void")
L("%entry = OpLabel")
for ai in range(2):
    for aj in range(2):
        L(f"%acc{ai}{aj} = OpVariable %ptr_fn_mat_c Function")
for ai in range(2):
    for aj in range(2):
        L(f"OpStore %acc{ai}{aj} %zero_mat_c")

pc = {}
for i, nme in enumerate(names):
    ty = "float" if nme in ("Alpha", "Beta") else "uint"
    p = op(f"%p_{nme}", f"OpAccessChain %ptr_pc_{ty} %push {U(i)}")
    pc[nme] = op(f"%pc_{nme}", f"OpLoad %{ty} {p}")

tid = op("%tid", "OpLoad %uint %local_index")
sg = op("%sg", "OpLoad %uint %sg_id")
lane = op("%lane", "OpLoad %uint %sg_lane")
wg = op("%wg", "OpLoad %v3uint %wg_id")
wgx = op("%wgx", "OpCompositeExtract %uint %wg 0")
wgy = op("%wgy", "OpCompositeExtract %uint %wg 1")
wgz = op("%wgz", "OpCompositeExtract %uint %wg 2")
n0 = umul(wgx, U(64))
m0 = umul(wgy, U(64))
base_a = uadd(pc["OffsetA"], umul(wgz, pc["StrideA"]))
base_b = uadd(pc["OffsetB"], umul(wgz, pc["StrideB"]))
base_c = uadd(pc["OffsetC"], umul(wgz, pc["StrideC"]))
trans_a = op(t(), f"OpINotEqual %bool {band(pc['Flags'], 1)} {U(0)}")
trans_b = op(t(), f"OpINotEqual %bool {band(pc['Flags'], 2)} {U(0)}")
sg_row = shr(sg, 1)
sg_col = band(sg, 1)
L("OpBranch %loop_head")

L("%loop_head = OpLabel")
L(f"%k0 = OpPhi %uint {U(0)} %entry %k0_next %loop_cont")
cond = ult("%k0", pc["K"])
L("OpLoopMerge %loop_merge %loop_cont None")
L(f"OpBranchConditional {cond} %loop_body %loop_merge")
L("%loop_body = OpLabel")
for e in range(8):
    stage_element("%buf_a", base_a, pc["Lda"], None, None, trans_a, m0, None, None, "%tile_a", A_STRIDE, True, e, tid, "%k0", pc["M"], pc["K"])
for e in range(8):
    stage_element("%buf_b", base_b, pc["Ldb"], None, None, trans_b, n0, None, None, "%tile_b", B_STRIDE, False, e, tid, "%k0", pc["N"], pc["K"])
L(f"OpControlBarrier {U(2)} {U(2)} {U(BARRIER)}")
frag_a, frag_b = [], []
for i in range(2):
    row = uadd(umul(sg_row, U(32)), U(16 * i)) if i else umul(sg_row, U(32))
    off = umul(row, U(A_STRIDE))
    p = op(t(), f"OpAccessChain %ptr_wg_elem %tile_a {off}")
    frag_a.append(op(t(), f"OpCooperativeMatrixLoadKHR %mat_a {p} {U(0)} {U(A_STRIDE)} NonPrivatePointer"))
for j in range(2):
    col = uadd(umul(sg_col, U(32)), U(16 * j)) if j else umul(sg_col, U(32))
    p = op(t(), f"OpAccessChain %ptr_wg_elem %tile_b {col}")
    frag_b.append(op(t(), f"OpCooperativeMatrixLoadKHR %mat_b {p} {U(0)} {U(B_STRIDE)} NonPrivatePointer"))
for i in range(2):
    for j in range(2):
        acc = op(t(), f"OpLoad %mat_c %acc{i}{j}")
        r = op(t(), f"OpCooperativeMatrixMulAddKHR %mat_c {frag_a[i]} {frag_b[j]} {acc}")
        L(f"OpStore %acc{i}{j} {r}")
L(f"OpControlBarrier {U(2)} {U(2)} {U(BARRIER)}")
L("OpBranch %loop_cont")
L("%loop_cont = OpLabel")
op("%k0_next", f"OpIAdd %uint %k0 {U(16)}")
L("OpBranch %loop_head")
L("%loop_merge = OpLabel")

cs_base = umul(sg, U(256))
beta_nz = op(t(), f"OpFOrdNotEqual %bool {pc['Beta']} %f_zero")
for i in range(2):
    for j in range(2):
        accv = op(t(), f"OpLoad %mat_c %acc{i}{j}")
        p = op(t(), f"OpAccessChain %ptr_wg_float %tile_c {cs_base}")
        L(f"OpCooperativeMatrixStoreKHR {p} {accv} {U(0)} {U(16)} NonPrivatePointer")
        L(f"OpControlBarrier {U(2)} {U(2)} {U(BARRIER)}")
        for e in range(8):
            ei = uadd(lane, U(32 * e)) if e else lane
            r = shr(ei, 4)
            c = band(ei, 15)
            cp = op(t(), f"OpAccessChain %ptr_wg_float %tile_c {uadd(cs_base, ei)}")
            v = op(t(), f"OpLoad %float {cp} NonPrivatePointer")
            gm = uadd(uadd(m0, umul(sg_row, U(32))), uadd(U(16 * i), r))
            gn = uadd(uadd(n0, umul(sg_col, U(32))), uadd(U(16 * j), c))
            inb = land(ult(gm, pc["M"]), ult(gn, pc["N"]))
            then, merge, then2, merge2 = t("L"), t("L"), t("L"), t("L")
            L(f"OpSelectionMerge {merge} None")
            L(f"OpBranchConditional {inb} {then} {merge}")
            L(f"{then} = OpLabel")
            at = uadd(base_c, uadd(umul(gm, pc["Ldc"]), gn))
            gp = op(t(), f"OpAccessChain %ptr_sb_float %buf_c {U(0)} {at}")
            res = op(t(), f"OpFMul %float {pc['Alpha']} {v}")
            L(f"OpSelectionMerge {merge2} None")
            L(f"OpBranchConditional {beta_nz} {then2} {merge2}")
            L(f"{then2} = OpLabel")
            old = op(t(), f"OpLoad %float {gp}")
            scaled = op(t(), f"OpFMul %float {pc['Beta']} {old}")
            sum_ = op(t(), f"OpFAdd %float {res} {scaled}")
            L(f"OpStore {gp} {sum_}")
            L(f"OpBranch {merge2}")
            L(f"{merge2} = OpLabel")
            # the beta == 0 path stores alpha * acc; the beta != 0 path above already stored its sum.
            then3, merge3 = t("L"), t("L")
            nb = op(t(), f"OpLogicalNot %bool {beta_nz}")
            L(f"OpSelectionMerge {merge3} None")
            L(f"OpBranchConditional {nb} {then3} {merge3}")
            L(f"{then3} = OpLabel")
            L(f"OpStore {gp} {res}")
            L(f"OpBranch {merge3}")
            L(f"{merge3} = OpLabel")
            L(f"OpBranch {merge}")
            L(f"{merge} = OpLabel")
        L(f"OpControlBarrier {U(2)} {U(2)} {U(BARRIER)}")
L("OpReturn")
L("OpFunctionEnd")

# Constants are emitted after types they depend on: all integer constants first, the types after.
out = header[:]
const_lines = sorted(consts.values(), key=lambda s: (0 if "%u_" not in s.split()[0] else 1, s))
fixed = [s for s in types if s.startswith(("%void", "%fn_void", "%bool", "%uint", "%float", "%v2float", "%v3uint", "%half", "%bf16"))]
rest = [s for s in types if s not in fixed]
out += fixed
out += const_lines
out += rest
out += lines
print("\n".join(out))
