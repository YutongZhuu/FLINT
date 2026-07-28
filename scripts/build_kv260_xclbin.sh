#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${PLATFORM:-}" ]; then
  cat >&2 <<'EOF'
error: PLATFORM is not set.

First source the Vitis environment and point PLATFORM at the KV260 .xpfm:

  source /opt/Xilinx/Vitis/2022.1/settings64.sh
  export PLATFORM=/path/to/kv260_platform.xpfm
EOF
  exit 2
fi

if ! command -v v++ >/dev/null 2>&1; then
  echo "error: v++ is not on PATH; source the Vitis settings64.sh first" >&2
  exit 2
fi

export TARGET=${TARGET:-${VITIS_TARGET:-hw}}
export VITIS_OUT_DIR=${VITIS_OUT_DIR:-${KV260_HLS_OUTPUT_DIR:-build/kv260-hls}}

exec scripts/build_kv260_conv2d_xclbin.sh
