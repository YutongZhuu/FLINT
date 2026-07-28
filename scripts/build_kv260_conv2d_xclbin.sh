#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

kernel_name=conv2d_kernel
kernel_src=third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv2DKernel.cpp
kernel_inc=third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime
target=${TARGET:-hw}
out_dir=${VITIS_OUT_DIR:-build/kv260-hls}
platform=${PLATFORM:-${KV260_PLATFORM:-}}

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

xo="$out_dir/$kernel_name.$target.xo"
xclbin="$out_dir/$kernel_name.$target.xclbin"

v++ -c \
  -t "$target" \
  --platform "$platform" \
  -k "$kernel_name" \
  -I "$kernel_inc" \
  "$kernel_src" \
  -o "$xo"

v++ -l \
  -t "$target" \
  --platform "$platform" \
  "$xo" \
  -o "$xclbin"

printf 'built %s\n' "$xclbin"
