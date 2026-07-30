#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(dirname "$script_dir")

# Keep this compatibility entry point, but use the maintained HLS sources and
# report/profile flow under hw/. Absolute defaults preserve the legacy output
# location even though the delegated script changes directory to hw/.
export VITIS_OUT_DIR=${VITIS_OUT_DIR:-$project_root/build/kv260-int8-only}
export VITIS_REPORT_DIR=${VITIS_REPORT_DIR:-$project_root/build/kv260-int8-only/report}

exec "$project_root/hw/scripts/compile.sh" "$@"
