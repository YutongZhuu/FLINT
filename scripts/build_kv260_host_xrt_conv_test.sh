#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cxx=${CXX:-g++}
src=third_party/onnx-mlir/src/Accelerators/MyAccel/Test/HostXrtConvTest.cpp
out=${HOST_OUT:-build/kv260-hls/host_xrt_conv_test}

if ! command -v "$cxx" >/dev/null 2>&1; then
  echo "error: C++ compiler '$cxx' not found" >&2
  exit 2
fi

mkdir -p "$(dirname "$out")"

include_flags=()
lib_flags=(-lxrt_coreutil -pthread)

if [ -n "${XILINX_XRT:-}" ]; then
  include_flags+=("-I$XILINX_XRT/include")
  lib_flags=("-L$XILINX_XRT/lib" "-Wl,-rpath,$XILINX_XRT/lib" "${lib_flags[@]}")
fi

"$cxx" -std=c++17 -O2 "$src" "${include_flags[@]}" "${lib_flags[@]}" -o "$out"

printf 'built %s\n' "$out"
