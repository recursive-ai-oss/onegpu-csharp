# Sourced by the other scripts: where build output goes and which validation settings apply.
# ONEGPU_ARTIFACTS   build output (default: artifacts/ in the repository)
# VK_ICD_FILENAMES   pick a Vulkan driver (for example lavapipe's lvp_icd.json)
# VK_ADD_LAYER_PATH  folder holding the Khronos validation layer when the system does not install it
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ONEGPU_ARTIFACTS="${ONEGPU_ARTIFACTS:-$ROOT/artifacts}"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 MSBUILDDISABLENODEREUSE=1
export ONEGPU_REQUIRE_DEVICE=1
# Metal: API and shader validation make any misuse fatal.
export MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert MTL_SHADER_VALIDATION=1
# Vulkan: the test fixture turns the layer on and fails a test that reports anything; this makes it the Khronos one when several exist.
export VK_INSTANCE_LAYERS="${VK_INSTANCE_LAYERS:-VK_LAYER_KHRONOS_validation}"
cd "$ROOT"
