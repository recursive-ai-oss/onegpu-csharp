#!/usr/bin/env bash
# Runs the GPU tests on Vulkan with the validation layer required and clean. Select the device with VK_ICD_FILENAMES.
set -euo pipefail
. "$(dirname "$0")/env.sh"
dotnet restore OneGpu.slnx
dotnet build --no-restore -c Release tests/RecursiveAI.OneGpu.Vulkan.Tests
dotnet test --no-build -c Release --project tests/RecursiveAI.OneGpu.Vulkan.Tests
dotnet test --no-build -c Release --project tests/RecursiveAI.OneGpu.Tests
