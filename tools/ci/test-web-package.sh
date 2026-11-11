#!/usr/bin/env bash
# Proves a browser app can consume the packed WebGPU backend with no manual step: packs the libraries into a scratch feed, publishes the
# minimal consumer app (tests/RecursiveAI.OneGpu.WebConsumer) against that feed, checks that onegpu.js is in the published output and runs a
# compute kernel in headless Chrome. CHROME_GPU_FLAGS selects the adapter; the default is SwiftShader.
set -euo pipefail
. "$(dirname "$0")/env.sh"
feed="$ONEGPU_ARTIFACTS/consumer-feed"
out="$ONEGPU_ARTIFACTS/consumer"
# A fresh version each run, so a package cache never serves an older pack of the same version.
version="0.0.0-consumer.$(date +%s)"
rm -rf "$feed" "$out"
dotnet pack src/RecursiveAI.OneGpu.WebGpu -c Release -p:PackageVersion=$version -o "$feed"
dotnet pack src/RecursiveAI.OneGpu -c Release -p:PackageVersion=$version -o "$feed"
dotnet publish tests/RecursiveAI.OneGpu.WebConsumer -c Release -o "$out" -p:OneGpuPackageVersion=$version -p:RestoreAdditionalProjectSources="$feed"
script=$(find "$out/wwwroot" -name 'onegpu*.js' | head -1)
[ -n "$script" ] || { echo "onegpu.js is missing from the published output" >&2; exit 1; }
echo "published: ${script#$out/}"
node tools/web/run-page.mjs --root "$out/wwwroot" --path "/index.html" --timeout 300 --out "$out/result.json" --log 1 2> "$out/console.log" || { cat "$out/console.log"; exit 1; }
if grep -E '^(log error|error|exception):|uncaptured' "$out/console.log"; then
  echo "the page reported errors" >&2
  exit 1
fi
node -e '
const r = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
console.log(JSON.stringify(r));
if (r.fatal || r.wrong !== 0 || r.messages !== 0 || !(r.count > 0)) process.exit(1);
' "$out/result.json"
