#!/usr/bin/env bash
# Runs the GPU tests on Metal (macOS, Apple silicon); env.sh makes Metal API and shader validation fatal.
set -euo pipefail
. "$(dirname "$0")/env.sh"
dotnet build -c Release tests/RecursiveAI.OneGpu.Vulkan.Tests
dotnet test --no-build -c Release --project tests/RecursiveAI.OneGpu.Vulkan.Tests
dotnet build -c Release tests/RecursiveAI.OneGpu.Kernels.Tests
dotnet test --no-build -c Release --project tests/RecursiveAI.OneGpu.Kernels.Tests
