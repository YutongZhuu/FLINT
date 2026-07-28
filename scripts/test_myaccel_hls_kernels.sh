#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cxx=${CXX:-clang++}
kernel_dir=third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime
test_src=third_party/onnx-mlir/src/Accelerators/MyAccel/Test/HlsConvKernelsTest.cpp
out_dir=build/hls-tests
test_exe=$out_dir/hls-conv-kernels-test

mkdir -p "$out_dir"

"$cxx" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -Werror \
  -Wno-unknown-pragmas \
  -Wno-unused-label \
  -I "$kernel_dir" \
  "$kernel_dir/Conv1x1Kernel.cpp" \
  "$kernel_dir/Conv3x3Kernel.cpp" \
  "$test_src" \
  -o "$test_exe"

"$test_exe"
