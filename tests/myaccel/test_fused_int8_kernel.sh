#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

cxx=${CXX:-clang++}
out_dir=build/hls-tests
test_exe=$out_dir/hls-fused-int8-kernel-test

mkdir -p "$out_dir"

"$cxx" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -Werror \
  -Wno-unknown-pragmas \
  -Wno-unused-label \
  -I hw/hls \
  hw/hls/Conv3x3ActDual1x1Int8Kernel.cpp \
  tests/myaccel/HlsFusedInt8KernelTest.cpp \
  -o "$test_exe"

"$test_exe"
