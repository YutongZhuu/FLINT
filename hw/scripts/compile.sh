#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

hw_dir=$PWD
kernel_dir=hls
conv1x1_src=$kernel_dir/Conv1x1Int8Kernel.cpp
conv3x3_src=$kernel_dir/Conv3x3Int8Kernel.cpp
target=${TARGET:-${VITIS_TARGET:-hw}}
out_dir=${VITIS_OUT_DIR:-build}
report_dir=${VITIS_REPORT_DIR:-report}
platform=${PLATFORM:-${KV260_PLATFORM:-}}
kernel_clock_hz=${KV260_KERNEL_CLOCK_HZ:-100000000}
link_temp_dir=$out_dir/vitis-link

case "$report_dir" in
/*) ;;
*) report_dir=$hw_dir/$report_dir ;;
esac

post_route_report_tcl=$hw_dir/scripts/post_route_reports.tcl

if [ -z "$platform" ]; then
  cat >&2 <<'EOF'
error: PLATFORM or KV260_PLATFORM is not set.

Source Vitis and point PLATFORM at the KV260 .xpfm, for example:

  source /tools/Xilinx/Vitis/2022.1/settings64.sh
  export PLATFORM=/path/to/xilinx_kv260_*.xpfm

To list installed platforms, try:

  platforminfo -l | grep -i kv260
EOF
  exit 2
fi

if ! command -v v++ >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: v++ was not found on PATH.

Run this script on a Linux Vitis machine after sourcing settings64.sh.
EOF
  exit 2
fi

if [ "$target" = "hw" ] && ! command -v bootgen >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: bootgen was not found on PATH.

Source the Vivado/Vitis settings64.sh that contains bootgen.
EOF
  exit 2
fi

if [ "$target" = "hw" ] && ! command -v vivado >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: vivado was not found on PATH.

Source the Vivado/Vitis settings64.sh so post-route reports can be generated.
EOF
  exit 2
fi

for source_file in "$conv1x1_src" "$conv3x3_src"; do
  if [ ! -f "$source_file" ]; then
    echo "error: package is missing $source_file" >&2
    exit 2
  fi
done

if [ ! -f "$post_route_report_tcl" ]; then
  echo "error: package is missing $post_route_report_tcl" >&2
  exit 2
fi

mkdir -p "$out_dir" "$report_dir"

conv1x1_xo="$out_dir/conv1x1_i8_kernel.$target.xo"
conv3x3_xo="$out_dir/conv3x3_i8_kernel.$target.xo"
xclbin="$out_dir/conv_int8_only.$target.xclbin"

v++ --compile \
  --target "$target" \
  --platform "$platform" \
  --kernel conv1x1_i8_kernel \
  --include "$kernel_dir" \
  "$conv1x1_src" \
  --output "$conv1x1_xo"

v++ --compile \
  --target "$target" \
  --platform "$platform" \
  --kernel conv3x3_i8_kernel \
  --include "$kernel_dir" \
  "$conv3x3_src" \
  --output "$conv3x3_xo"

v++ --link \
  --target "$target" \
  --platform "$platform" \
  --clock.defaultFreqHz "$kernel_clock_hz" \
  --report_level 1 \
  --save-temps \
  --temp_dir "$link_temp_dir" \
  "$conv1x1_xo" \
  "$conv3x3_xo" \
  --output "$xclbin"

printf '\nBuilt %s\n' "$xclbin"
printf 'Linked kernels: conv1x1_i8_kernel, conv3x3_i8_kernel\n'

if [ "$target" = "hw" ]; then
  routed_checkpoint=$(
    find "$link_temp_dir" -type f -name '*_routed.dcp' -print -quit
  )
  if [ -z "$routed_checkpoint" ] || [ ! -f "$routed_checkpoint" ]; then
    echo "error: routed Vivado checkpoint was not found under $link_temp_dir" >&2
    exit 1
  fi

  vivado \
    -mode batch \
    -notrace \
    -nojournal \
    -nolog \
    -source "$post_route_report_tcl" \
    -tclargs "$routed_checkpoint" "$report_dir"

  for final_report in timing_summary.rpt utilization.rpt; do
    if [ ! -s "$report_dir/$final_report" ]; then
      echo "error: Vivado did not produce $report_dir/$final_report" >&2
      exit 1
    fi
  done

  system_bit=$link_temp_dir/link/int/system.bit
  if [ ! -f "$system_bit" ]; then
    system_bit=$(find "$link_temp_dir" -type f -name system.bit -print -quit)
  fi
  if [ -z "$system_bit" ] || [ ! -f "$system_bit" ]; then
    echo "error: Vitis linked the XCLBIN but system.bit was not found under $link_temp_dir" >&2
    exit 1
  fi

  bit_name=conv_int8_only.$target.bit
  bit_file=$out_dir/$bit_name
  bit_bin=$bit_file.bin
  bif_name=conv_int8_only.$target.bif

  cp "$system_bit" "$bit_file"
  (
    cd "$out_dir"
    printf 'all:{%s}\n' "$bit_name" >"$bif_name"
    bootgen -w -arch zynqmp -process_bitstream bin -image "$bif_name"
  )

  if [ ! -f "$bit_bin" ]; then
    echo "error: bootgen did not produce $bit_bin" >&2
    exit 1
  fi
  printf 'Built %s\n' "$bit_bin"
  printf 'Timing report: %s/timing_summary.rpt\n' "$report_dir"
  printf 'Utilization report: %s/utilization.rpt\n' "$report_dir"
fi
