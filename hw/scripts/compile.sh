#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

hw_dir=$PWD
kernel_dir=hls
conv1x1_src=$kernel_dir/Conv1x1Int8Kernel.cpp
conv3x3_src=$kernel_dir/Conv3x3Int8Kernel.cpp
conv6x6_src=$kernel_dir/Conv6x6StemInt8Kernel.cpp
target=${TARGET:-${VITIS_TARGET:-hw}}
out_dir=${VITIS_OUT_DIR:-build}
report_dir=${VITIS_REPORT_DIR:-report}
platform=${PLATFORM:-${KV260_PLATFORM:-}}
kernel_clock_hz=${KV260_KERNEL_CLOCK_HZ:-100000000}
enable_profile=${VITIS_PROFILE:-0}
include_6x6_stem=${VITIS_INCLUDE_6X6_STEM:-0}
compile_temp_root=$out_dir/vitis-compile
compile_log_root=$out_dir/vitis-logs
link_temp_dir=$out_dir/vitis-link

case "$report_dir" in
/*) ;;
*) report_dir=$hw_dir/$report_dir ;;
esac

post_route_report_tcl=$hw_dir/scripts/post_route_reports.tcl

case "$enable_profile" in
0|1) ;;
*)
  echo "error: VITIS_PROFILE must be 0 or 1, got: $enable_profile" >&2
  exit 2
  ;;
esac

case "$include_6x6_stem" in
0|1) ;;
*)
  echo "error: VITIS_INCLUDE_6X6_STEM must be 0 or 1, got: $include_6x6_stem" >&2
  exit 2
  ;;
esac

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

required_sources=("$conv1x1_src" "$conv3x3_src")
if [ "$include_6x6_stem" = "1" ]; then
  required_sources+=("$conv6x6_src")
fi
for source_file in "${required_sources[@]}"; do
  if [ ! -f "$source_file" ]; then
    echo "error: package is missing $source_file" >&2
    exit 2
  fi
done

if [ ! -f "$post_route_report_tcl" ]; then
  echo "error: package is missing $post_route_report_tcl" >&2
  exit 2
fi

conv1x1_temp_dir=$compile_temp_root/conv1x1_i8_kernel
conv3x3_temp_dir=$compile_temp_root/conv3x3_i8_kernel
conv6x6_temp_dir=$compile_temp_root/conv6x6_stem_i8_kernel
conv1x1_vitis_report_dir=$report_dir/vitis/conv1x1_i8_kernel
conv3x3_vitis_report_dir=$report_dir/vitis/conv3x3_i8_kernel
conv6x6_vitis_report_dir=$report_dir/vitis/conv6x6_stem_i8_kernel
conv1x1_log_dir=$compile_log_root/conv1x1_i8_kernel
conv3x3_log_dir=$compile_log_root/conv3x3_i8_kernel
conv6x6_log_dir=$compile_log_root/conv6x6_stem_i8_kernel
link_report_dir=$report_dir/vitis/link
link_log_dir=$compile_log_root/link

mkdir -p \
  "$out_dir" \
  "$report_dir" \
  "$conv1x1_temp_dir" \
  "$conv3x3_temp_dir" \
  "$conv6x6_temp_dir" \
  "$conv1x1_vitis_report_dir" \
  "$conv3x3_vitis_report_dir" \
  "$conv6x6_vitis_report_dir" \
  "$conv1x1_log_dir" \
  "$conv3x3_log_dir" \
  "$conv6x6_log_dir" \
  "$link_report_dir" \
  "$link_log_dir"

conv1x1_xo="$out_dir/conv1x1_i8_kernel.$target.xo"
conv3x3_xo="$out_dir/conv3x3_i8_kernel.$target.xo"
conv6x6_xo="$out_dir/conv6x6_stem_i8_kernel.$target.xo"
xclbin="$out_dir/conv_int8_only.$target.xclbin"

compile_profile_args=()
link_profile_args=()
link_export_args=()
if [ "$enable_profile" = "1" ]; then
  # Stall ports must be enabled while compiling each kernel. Data, stall, and
  # execution monitors are then inserted while linking the system image.
  compile_profile_args+=(--profile.stall all:all:all)
  link_profile_args+=(
    --profile.data all:all:all:all
    --profile.stall all:all:all
    --profile.exec all:all:all
  )
fi
if [ "$target" = "hw" ]; then
  # Keep the post-link XSA so the DTBO can be generated from the exact routed
  # design rather than reused from an unrelated firmware application.
  link_export_args+=(--advanced.param compiler.addOutputTypes=hw_export)
fi

v++ --compile \
  --target "$target" \
  --platform "$platform" \
  --kernel conv1x1_i8_kernel \
  --include "$kernel_dir" \
  --report_level 2 \
  --save-temps \
  --temp_dir "$conv1x1_temp_dir" \
  --report_dir "$conv1x1_vitis_report_dir" \
  --log_dir "$conv1x1_log_dir" \
  ${compile_profile_args[@]+"${compile_profile_args[@]}"} \
  "$conv1x1_src" \
  --output "$conv1x1_xo"

v++ --compile \
  --target "$target" \
  --platform "$platform" \
  --kernel conv3x3_i8_kernel \
  --include "$kernel_dir" \
  --report_level 2 \
  --save-temps \
  --temp_dir "$conv3x3_temp_dir" \
  --report_dir "$conv3x3_vitis_report_dir" \
  --log_dir "$conv3x3_log_dir" \
  ${compile_profile_args[@]+"${compile_profile_args[@]}"} \
  "$conv3x3_src" \
  --output "$conv3x3_xo"

if [ "$include_6x6_stem" = "1" ]; then
  v++ --compile \
    --target "$target" \
    --platform "$platform" \
    --kernel conv6x6_stem_i8_kernel \
    --include "$kernel_dir" \
    --report_level 2 \
    --save-temps \
    --temp_dir "$conv6x6_temp_dir" \
    --report_dir "$conv6x6_vitis_report_dir" \
    --log_dir "$conv6x6_log_dir" \
    ${compile_profile_args[@]+"${compile_profile_args[@]}"} \
    "$conv6x6_src" \
    --output "$conv6x6_xo"
fi

copy_csynth_report() {
  local kernel=$1
  local vitis_report_dir=$2
  local temp_dir=$3
  local source_report=
  local expected_report=

  expected_report=$vitis_report_dir/$kernel.$target/hls_reports/${kernel}_csynth.rpt
  if [ -s "$expected_report" ]; then
    source_report=$expected_report
  else
    source_report=$(find "$vitis_report_dir" "$temp_dir" \
      -type f -path "*/hls_reports/${kernel}_csynth.rpt" -print -quit)
  fi
  if [ -z "$source_report" ] || [ ! -s "$source_report" ]; then
    echo "error: Vitis did not produce a non-empty ${kernel}_csynth.rpt" >&2
    exit 1
  fi

  cp "$source_report" "$report_dir/${kernel}_csynth.rpt"
  if [ ! -s "$report_dir/${kernel}_csynth.rpt" ]; then
    echo "error: failed to collect $report_dir/${kernel}_csynth.rpt" >&2
    exit 1
  fi
}

copy_csynth_report \
  conv1x1_i8_kernel "$conv1x1_vitis_report_dir" "$conv1x1_temp_dir"
copy_csynth_report \
  conv3x3_i8_kernel "$conv3x3_vitis_report_dir" "$conv3x3_temp_dir"
if [ "$include_6x6_stem" = "1" ]; then
  copy_csynth_report \
    conv6x6_stem_i8_kernel "$conv6x6_vitis_report_dir" "$conv6x6_temp_dir"
fi

kernel_xos=("$conv1x1_xo" "$conv3x3_xo")
if [ "$include_6x6_stem" = "1" ]; then
  kernel_xos+=("$conv6x6_xo")
fi

v++ --link \
  --target "$target" \
  --platform "$platform" \
  --clock.defaultFreqHz "$kernel_clock_hz" \
  --report_level 1 \
  --save-temps \
  --temp_dir "$link_temp_dir" \
  --report_dir "$link_report_dir" \
  --log_dir "$link_log_dir" \
  ${link_profile_args[@]+"${link_profile_args[@]}"} \
  ${link_export_args[@]+"${link_export_args[@]}"} \
  "${kernel_xos[@]}" \
  --output "$xclbin"

printf '\nBuilt %s\n' "$xclbin"
printf 'Linked kernels: conv1x1_i8_kernel, conv3x3_i8_kernel'
if [ "$include_6x6_stem" = "1" ]; then
  printf ', conv6x6_stem_i8_kernel'
fi
printf '\n'
printf 'HLS report: %s/conv1x1_i8_kernel_csynth.rpt\n' "$report_dir"
printf 'HLS report: %s/conv3x3_i8_kernel_csynth.rpt\n' "$report_dir"
if [ "$include_6x6_stem" = "1" ]; then
  printf 'HLS report: %s/conv6x6_stem_i8_kernel_csynth.rpt\n' "$report_dir"
fi
if [ "$enable_profile" = "1" ]; then
  printf 'XRT profiling instrumentation: data, stall, and execution\n'
fi

if [ "$target" = "hw" ]; then
  xsa=${xclbin%.xclbin}.xsa
  if [ ! -s "$xsa" ]; then
    echo "error: Vitis did not produce the linked hardware export $xsa" >&2
    exit 1
  fi

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

  for final_report in \
    timing_summary.rpt \
    utilization.rpt \
    utilization_hierarchical.rpt; do
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
  printf 'Linked hardware export: %s\n' "$xsa"
  printf 'Timing report: %s/timing_summary.rpt\n' "$report_dir"
  printf 'Utilization report: %s/utilization.rpt\n' "$report_dir"
  printf 'Hierarchical utilization: %s/utilization_hierarchical.rpt\n' \
    "$report_dir"
fi
