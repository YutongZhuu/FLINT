#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

kernel_inc=third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime
conv1x1_src=$kernel_inc/Conv1x1Kernel.cpp
conv3x3_src=$kernel_inc/Conv3x3Kernel.cpp
target=${TARGET:-${VITIS_TARGET:-hw}}
out_dir=${VITIS_OUT_DIR:-${KV260_HLS_OUTPUT_DIR:-build/kv260-hls}}
platform=${PLATFORM:-${KV260_PLATFORM:-}}
kernel_clock_hz=${KV260_KERNEL_CLOCK_HZ:-100000000}

if [ -z "$platform" ]; then
  cat >&2 <<'EOF'
error: PLATFORM or KV260_PLATFORM is not set.

Set it to the KV260 Vitis platform name or .xpfm path, for example:

  export PLATFORM=/path/to/xilinx_kv260_*.xpfm

To discover installed platforms on a Vitis machine, try:

  platforminfo -l | grep -i kv260
EOF
  exit 2
fi

if ! command -v v++ >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: v++ was not found on PATH.

Run this on a Linux machine with AMD Vitis installed and sourced, for example:

  source /tools/Xilinx/Vitis/2023.2/settings64.sh
EOF
  exit 2
fi

mkdir -p "$out_dir"

conv1x1_xo="$out_dir/conv1x1_kernel.$target.xo"
conv3x3_xo="$out_dir/conv3x3_kernel.$target.xo"
xclbin="$out_dir/conv2d_kernel.$target.xclbin"

v++ -c \
  -t "$target" \
  --platform "$platform" \
  -k conv1x1_kernel \
  -I "$kernel_inc" \
  "$conv1x1_src" \
  -o "$conv1x1_xo"

v++ -c \
  -t "$target" \
  --platform "$platform" \
  -k conv3x3_kernel \
  -I "$kernel_inc" \
  "$conv3x3_src" \
  -o "$conv3x3_xo"

v++ -l \
  -t "$target" \
  --platform "$platform" \
  --clock.defaultFreqHz "$kernel_clock_hz" \
  "$conv1x1_xo" \
  "$conv3x3_xo" \
  -o "$xclbin"

printf 'built %s\n' "$xclbin"
