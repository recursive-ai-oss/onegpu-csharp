#!/usr/bin/env bash
# Regenerates src/RecursiveAI.OneGpu.Vulkan/Interop/VulkanApi.g.cs from the pinned Vulkan registry and the api list.
# The registry is downloaded to $ONEGPU_ARTIFACTS/vk (hash-checked); the output goes to the path given as $1 (default: the tracked file).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${ONEGPU_ARTIFACTS:-$ROOT/artifacts}"
out="${1:-$ROOT/src/RecursiveAI.OneGpu.Vulkan/Interop/VulkanApi.g.cs}"
"$ROOT/tools/fetch-vkxml.sh" "$ART/vk"
dotnet run -c Release --project "$ROOT/tools/RecursiveAI.OneGpu.VkBindGen" -- "$ART/vk/vk.xml" "$ROOT/tools/RecursiveAI.OneGpu.VkBindGen/vulkan-api.txt" "$out"
