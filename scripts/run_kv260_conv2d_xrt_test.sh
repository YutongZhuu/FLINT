#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host=${HOST_EXE:-build/kv260-hls/host_xrt_conv_test}
xclbin=build/kv260-hls/conv2d_kernel.hw.xclbin
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  xclbin=$1
  shift
fi

if [ ! -x "$host" ]; then
  echo "error: host test is missing or not executable: $host" >&2
  echo "build it first with: scripts/build_kv260_host_xrt_conv_test.sh" >&2
  exit 2
fi

if [ ! -f "$xclbin" ]; then
  echo "error: xclbin not found: $xclbin" >&2
  exit 2
fi

"$host" "$xclbin" "$@"
