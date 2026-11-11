#!/usr/bin/env bash
# Regenerates the hand-written SPIR-V kernels from gemm_coop.py. Needs spirv-as and spirv-val (SPIRV-Tools) on PATH, or SPIRV_TOOLS=<dir>.
# They are build-time tools of this script only; the produced .spv files are committed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../../src/RecursiveAI.OneGpu.Kernels/Raw"
tools="${SPIRV_TOOLS:-}"
as="${tools:+$tools/}spirv-as"; val="${tools:+$tools/}spirv-val"
mkdir -p "$out"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
for kind in f16 bf16; do
  python3 "$here/gemm_coop.py" "$kind" > "$tmp/gemm_coop_$kind.spvasm"
  "$as" --target-env spv1.5 "$tmp/gemm_coop_$kind.spvasm" -o "$out/gemm_coop_$kind.spv"
  "$val" --target-env vulkan1.2 "$out/gemm_coop_$kind.spv"
  echo "$kind: $(stat -c %s "$out/gemm_coop_$kind.spv") bytes"
done
