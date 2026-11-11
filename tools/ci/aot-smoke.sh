#!/usr/bin/env bash
# Publishes the smoke app with NativeAOT and trimming (the build fails on any trim or AOT warning), then runs it on this machine's GPU.
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="$ONEGPU_ARTIFACTS/aot"
rm -rf "$out"
dotnet publish tests/RecursiveAI.OneGpu.AotSmoke -c Release -o "$out"
ONEGPU_VALIDATION=1 "$out/RecursiveAI.OneGpu.AotSmoke"
