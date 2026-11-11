#!/usr/bin/env bash
# Publishes the browser test runner and runs it in headless Chrome on WebGPU. CHROME_GPU_FLAGS selects the adapter; the default is
# SwiftShader, which needs no GPU. Any failed test or uncaptured WebGPU error fails the run.
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="$ONEGPU_ARTIFACTS/web"
rm -rf "$out"
dotnet publish tests/RecursiveAI.OneGpu.WebRunner -c Release -o "$out"
node tools/web/run-page.mjs --root "$out/wwwroot" --path "/index.html?filter=${1:-}" --timeout 600 --out "$out/result.json" --log 1 2> "$out/console.log" || { cat "$out/console.log"; exit 1; }
# The page log holds every console message; an error or an uncaptured WebGPU error fails the run.
if grep -E '^(log error|error|exception):|uncaptured' "$out/console.log"; then
  echo "the page reported errors" >&2
  exit 1
fi
node tools/web/check-result.mjs "$out/result.json"
