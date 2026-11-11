#!/usr/bin/env bash
# Restores and builds the solution with warnings as errors, then runs the repository checks.
set -euo pipefail
. "$(dirname "$0")/env.sh"
dotnet restore OneGpu.slnx
dotnet build --no-restore OneGpu.slnx -c Release

# Interop lives only in the backend packages; the core and the kernels have none.
if git grep -nE 'DllImport|LibraryImport|JSImport|delegate\* unmanaged' -- 'src/RecursiveAI.OneGpu/**.cs' 'src/RecursiveAI.OneGpu.Kernels/**.cs'; then
  echo "interop found outside the backend packages" >&2
  exit 1
fi
# Shipping projects reference only the one* family or build-time tools.
if git grep -n 'PackageReference' -- 'src/**.csproj' 'src/**.props' | grep -v 'PrivateAssets="all"' | grep -v 'RecursiveAI.One'; then
  echo "a shipping project references a package outside the family" >&2
  exit 1
fi
if git ls-files | grep -E '(^|/)(bin|obj|artifacts)/'; then
  echo "build output is tracked" >&2
  exit 1
fi
