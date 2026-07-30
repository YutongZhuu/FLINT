#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

cxx=${CXX:-clang++}
out_dir=build/hls-tests
test_exe=$out_dir/conv6x6-stem-int8-kernel-test

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
  hw/hls/Conv6x6StemInt8Kernel.cpp \
  tests/myaccel/conv6x6_stem_int8_kernel_test.cpp \
  -o "$test_exe"

"$test_exe"
