#!/usr/bin/env bash
# Puts the Khronos validation layer that the hosted CI installs (Ubuntu 24.04 "noble" package vulkan-validationlayers) in a folder you can
# hand to the Vulkan loader, without installing anything system-wide.
#
#   tools/ci/fetch-validation-layer.sh [folder]        folder defaults to $HOME/.cache/onegpu/vklayers-ci
#   export VK_ADD_LAYER_PATH=<folder>                    then run tools/ci/test-vulkan.sh as usual
#
# The version below is the one pin: the CI workflow reads it from here to install the same package. The layer is a test oracle
# only and is never part of a package. Newer layers find things this one does not, so also run once against the newest release.
set -euo pipefail

LAYER_PACKAGE_VERSION="1.3.275.0-1"
POOL="${ONEGPU_LAYER_MIRROR:-http://archive.ubuntu.com/ubuntu/pool/universe/v/vulkan-validationlayers}"
DEB="vulkan-validationlayers_${LAYER_PACKAGE_VERSION}_amd64.deb"

dest="${1:-$HOME/.cache/onegpu/vklayers-ci}"
if [ -f "$dest/VkLayer_khronos_validation.json" ] && [ -f "$dest/.version" ] && [ "$(cat "$dest/.version")" = "$LAYER_PACKAGE_VERSION" ]; then
  echo "$dest"
  exit 0
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/vklayer.XXXXXX")"
trap 'rm -rf "$work"' EXIT
curl -fsSL -o "$work/$DEB" "$POOL/$DEB"
dpkg -x "$work/$DEB" "$work/x"
mkdir -p "$dest"
cp "$work/x/usr/lib/x86_64-linux-gnu/libVkLayer_khronos_validation.so" "$dest/"
# The package names the library without a directory, which the loader looks up on the system library path; "./" makes it relative to the
# manifest, so the folder works from anywhere.
sed 's|"library_path": "libVkLayer|"library_path": "./libVkLayer|' \
  "$work/x/usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json" > "$dest/VkLayer_khronos_validation.json"
echo "$LAYER_PACKAGE_VERSION" > "$dest/.version"
echo "$dest"
