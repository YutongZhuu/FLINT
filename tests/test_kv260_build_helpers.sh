#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test_root=$(mktemp -d "${TMPDIR:-/tmp}/kv260-build-helpers.XXXXXX")
trap 'rm -rf "$test_root"' EXIT

mock_bin=$test_root/bin
call_log=$test_root/tool-calls.log
mkdir -p "$mock_bin"
export MOCK_TOOL_CALL_LOG=$call_log

cat >"$mock_bin/v++" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf 'v++' >>"$MOCK_TOOL_CALL_LOG"
printf ' %q' "$@" >>"$MOCK_TOOL_CALL_LOG"
printf '\n' >>"$MOCK_TOOL_CALL_LOG"

mode=
kernel=
target=
output=
temp_dir=
report_dir=
while [ $# -gt 0 ]; do
  case "$1" in
  --compile|--link)
    mode=$1
    shift
    ;;
  --kernel|--target|--output|--temp_dir|--report_dir|--log_dir|--platform|--include|--report_level|--advanced.param)
    option=$1
    value=$2
    case "$option" in
    --kernel) kernel=$value ;;
    --target) target=$value ;;
    --output) output=$value ;;
    --temp_dir) temp_dir=$value ;;
    --report_dir) report_dir=$value ;;
    esac
    shift 2
    ;;
  *) shift ;;
  esac
done

test -n "$mode"
test -n "$target"
test -n "$output"
mkdir -p "$(dirname "$output")"
if [ "$mode" = "--compile" ]; then
  test -n "$kernel"
  test -n "$report_dir"
  report=$report_dir/$kernel.$target/hls_reports/${kernel}_csynth.rpt
  mkdir -p "$(dirname "$report")"
  printf 'mock HLS report for %s\n' "$kernel" >"$report"
  printf 'mock XO\n' >"$output"
else
  test -n "$temp_dir"
  mkdir -p "$temp_dir/link/int"
  printf 'mock XCLBIN\n' >"$output"
  printf 'mock XSA\n' >"${output%.xclbin}.xsa"
  printf 'mock bitstream\n' >"$temp_dir/link/int/system.bit"
  printf 'mock checkpoint\n' >"$temp_dir/mock_routed.dcp"
fi
EOF

cat >"$mock_bin/vivado" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

report_dir=
while [ $# -gt 0 ]; do
  if [ "$1" = "-tclargs" ]; then
    test -f "$2"
    report_dir=$3
    break
  fi
  shift
done
test -n "$report_dir"
mkdir -p "$report_dir"
printf 'mock timing\n' >"$report_dir/timing_summary.rpt"
printf 'mock utilization\n' >"$report_dir/utilization.rpt"
printf 'mock hierarchy\n' >"$report_dir/utilization_hierarchical.rpt"
printf 'mock power\n' >"$report_dir/power.rpt"
EOF

cat >"$mock_bin/bootgen" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

output=
image=
process_bitstream=
while [ $# -gt 0 ]; do
  case "$1" in
  -o)
    output=$2
    shift 2
    ;;
  -image)
    image=$2
    shift 2
    ;;
  -process_bitstream)
    process_bitstream=$2
    shift 2
    ;;
  *) shift ;;
  esac
done
test "$process_bitstream" = bin
if [ -z "$output" ]; then
  test -n "$image"
  bit_file=$(sed -n 's/^all:{\(.*\)}$/\1/p' "$image")
  test -n "$bit_file"
  output=$bit_file.bin
fi
mkdir -p "$(dirname "$output")"
printf 'mock bit.bin\n' >"$output"
EOF

cat >"$mock_bin/xclbinutil" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

raw_bit=
while [ $# -gt 0 ]; do
  case "$1" in
  --dump-section)
    raw_bit=${2#BITSTREAM:RAW:}
    shift 2
    ;;
  *) shift ;;
  esac
done
test -n "$raw_bit"
mkdir -p "$(dirname "$raw_bit")"
printf 'mock raw bit\n' >"$raw_bit"
EOF

cat >"$mock_bin/dtc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

output=
input=
input_format=
while [ $# -gt 0 ]; do
  case "$1" in
  -o)
    output=$2
    shift 2
    ;;
  -I)
    input_format=$2
    shift 2
    ;;
  -O)
    shift 2
    ;;
  *)
    input=$1
    shift
    ;;
  esac
done
test -n "$output"
test -f "$input"
if [ "$input_format" = dts ]; then
  grep -Fq '/dts-v1/;' "$input"
  grep -Fq '/plugin/;' "$input"
  grep -Fq '/ {' "$input"
  grep -Fq 'firmware-name' "$input"
fi
cp "$input" "$output"
EOF

cat >"$mock_bin/xsct" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

xsct_script=$(mktemp)
trap 'rm -f "$xsct_script"' EXIT
cat >"$xsct_script"
grep -Fq 'hsi open_hw_design $::env(KV260_DTG_XSA)' "$xsct_script"
grep -Fq 'hsi set_property CONFIG.dt_overlay true [hsi::get_os]' "$xsct_script"
grep -Fq 'hsi set_property CONFIG.dt_zocl true [hsi::get_os]' "$xsct_script"
grep -Fq 'hsi close_hw_design [hsi current_hw_design]' "$xsct_script"

test -n "${KV260_DTG_XSA:-}"
test -f "$KV260_DTG_XSA"
test -n "${KV260_DTG_REPO:-}"
test -d "$KV260_DTG_REPO"
test -n "${KV260_DTG_OUT:-}"
mkdir -p "$KV260_DTG_OUT"
cat >"$KV260_DTG_OUT/pl.dtsi" <<'DTS'
/dts-v1/;
/plugin/;
/ {
  fragment@0 {
    __overlay__ {
      firmware-name = "vpl_gen_fixed.bit.bin";
    };
  };
};
DTS
EOF

cat >"$mock_bin/time" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-v" ]; then
  shift
fi
exec "$@"
EOF

chmod +x "$mock_bin"/*
printf 'mock platform\n' >"$test_root/platform.xpfm"

PATH="$mock_bin:$PATH" \
PLATFORM="$test_root/platform.xpfm" \
TARGET=hw \
VITIS_PROFILE=1 \
VITIS_INCLUDE_6X6_STEM=1 \
VITIS_OUT_DIR="$test_root/hw-out" \
VITIS_REPORT_DIR="$test_root/hw-report" \
  ./hw/scripts/compile.sh >/dev/null

test -s "$test_root/hw-report/conv1x1_i8_kernel_csynth.rpt"
test -s "$test_root/hw-report/conv3x3_i8_kernel_csynth.rpt"
test -s "$test_root/hw-report/conv6x6_stem_i8_kernel_csynth.rpt"
test -s "$test_root/hw-report/timing_summary.rpt"
test -s "$test_root/hw-report/utilization_hierarchical.rpt"
test -s "$test_root/hw-report/power.rpt"
test -s "$test_root/hw-out/conv_int8_only.hw.xsa"
test -s "$test_root/hw-out/conv_int8_only.hw.bit.bin"
grep -Fq -- '--profile.stall all:all:counters' "$call_log"
grep -Fq -- '--profile.data conv1x1_i8_kernel:conv1x1_i8_kernel_1:m_axi_gmem0:counters' "$call_log"
grep -Fq -- '--profile.data conv3x3_i8_kernel:conv3x3_i8_kernel_1:m_axi_gmem0:counters' "$call_log"
if grep -Fq -- '--profile.exec' "$call_log"; then
  echo "error: redundant execution monitor was requested" >&2
  exit 1
fi
grep -Fq -- 'compiler.addOutputTypes=hw_export' "$call_log"

: >"$call_log"
PATH="$mock_bin:$PATH" \
PLATFORM="$test_root/platform.xpfm" \
TARGET=hw \
VITIS_PROFILE=trace \
VITIS_OUT_DIR="$test_root/trace-out" \
VITIS_REPORT_DIR="$test_root/trace-report" \
  ./hw/scripts/compile.sh >/dev/null
grep -Fq -- '--profile.stall all:all:all' "$call_log"
grep -Fq -- '--profile.data conv1x1_i8_kernel:conv1x1_i8_kernel_1:m_axi_gmem0:all' "$call_log"
grep -Fq -- '--profile.data conv3x3_i8_kernel:conv3x3_i8_kernel_1:m_axi_gmem0:all' "$call_log"
grep -Fq -- '--profile.trace_memory FIFO:8K' "$call_log"

PATH="$mock_bin:$PATH" \
PLATFORM="$test_root/platform.xpfm" \
TARGET=hw \
VITIS_OUT_DIR="$test_root/wrapper-out" \
VITIS_REPORT_DIR="$test_root/wrapper-report" \
  ./scripts/build_kv260_int8_only_xclbin.sh >/dev/null
test -s "$test_root/wrapper-report/conv1x1_i8_kernel_csynth.rpt"
test -s "$test_root/wrapper-report/conv3x3_i8_kernel_csynth.rpt"
test ! -e "$test_root/wrapper-report/conv6x6_stem_i8_kernel_csynth.rpt"

printf 'mock XCLBIN\n' >"$test_root/input.xclbin"
printf 'mock XSA\n' >"$test_root/input.xsa"
mkdir -p "$test_root/device-tree-xlnx"
PATH="$mock_bin:$PATH" \
KV260_APP_NAME=profile-app \
DEVICE_TREE_REPO="$test_root/device-tree-xlnx" \
  ./scripts/generate_kv260_dtbo.sh \
    "$test_root/input.xsa" \
    "$test_root/generated/profile-app.dtbo" >/dev/null
test -s "$test_root/generated/profile-app.dtbo"
test -s "$test_root/generated/profile-app.dts"
grep -Fq 'firmware-name = "profile-app.bit.bin"' \
  "$test_root/generated/profile-app.dtbo"

cp "$test_root/generated/profile-app.dtbo" "$test_root/matching.dtbo"
PATH="$mock_bin:$PATH" \
KV260_APP_NAME=profile-app \
  ./scripts/package_kv260_firmware.sh \
    "$test_root/input.xclbin" \
    "$test_root/matching.dtbo" \
    "$test_root/firmware/profile-app" >/dev/null
test -s "$test_root/firmware/profile-app/profile-app.bit.bin"
test -s "$test_root/firmware/profile-app/profile-app.dtbo"
test -s "$test_root/firmware/profile-app/profile-app.xclbin"
test -s "$test_root/firmware/profile-app/shell.json"

printf 'firmware-name = "wrong-app.bit.bin";\n' >"$test_root/mismatched.dtbo"
if PATH="$mock_bin:$PATH" \
  KV260_APP_NAME=profile-app \
  ./scripts/package_kv260_firmware.sh \
    "$test_root/input.xclbin" \
    "$test_root/mismatched.dtbo" \
    "$test_root/firmware/should-not-exist" \
    >"$test_root/mismatch.stdout" 2>"$test_root/mismatch.stderr"; then
  echo "error: mismatched DTBO firmware-name unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'expected: profile-app.bit.bin' "$test_root/mismatch.stderr"
grep -Fq 'actual:   wrong-app.bit.bin' "$test_root/mismatch.stderr"

profile_package=$test_root/profile-package
mkdir -p "$profile_package/runtime-libs"
cp scripts/run_kv260_yolo_int8_profile.sh "$profile_package/run-profile.sh"
cp scripts/xrt-profile.ini "$profile_package/"
printf 'mock xclbin\n' >"$profile_package/profile.xclbin"
printf 'mock input\n' >"$profile_package/input.bin"
cat >"$profile_package/yolo-int8-myaccel-driver-aarch64" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "mock inference profile"
printf 'mock output\n' >"$2"
printf 'mock run summary\n' >xrt.run_summary
case "${MOCK_XRT_ARTIFACT_STYLE:-xrt-2.13}" in
xrt-2.13)
  printf 'mock summary\n' >summary.csv
  printf 'mock native trace\n' >native_trace.csv
  if [ "${MOCK_XRT_NO_DEVICE_EVENTS:-0}" = "1" ]; then
    printf 'HEADER\nEVENTS\n\nDEPENDENCIES\n' >device_trace_0.csv
  else
    printf 'HEADER\nEVENTS\n1,mock device event\nDEPENDENCIES\n' \
      >device_trace_0.csv
  fi
  ;;
newer)
  printf 'mock profile summary\n' >profile_summary.csv
  printf 'mock timeline\n' >timeline_trace.csv
  ;;
*)
  echo "error: unknown mock XRT artifact style" >&2
  exit 2
  ;;
esac
exit "${MOCK_XRT_EXIT_STATUS:-0}"
EOF
chmod +x "$profile_package/yolo-int8-myaccel-driver-aarch64"

(
  cd "$test_root"
  MYACCEL_XCLBIN="$profile_package/profile.xclbin" \
  MYACCEL_XRT_PROFILE_DIR="$test_root/xrt-output" \
  MYACCEL_XRT_INI="$profile_package/xrt-profile.ini" \
  TIME_BIN="$mock_bin/time" \
    "$profile_package/run-profile.sh" \
      "$profile_package/input.bin" \
      relative-output.bin \
      relative-profile.log >/dev/null
)
test -s "$test_root/relative-output.bin"
test -s "$test_root/relative-profile.log"
test -s "$test_root/xrt-output/xrt.run_summary"
test -s "$test_root/xrt-output/summary.csv"
test -s "$test_root/xrt-output/native_trace.csv"
test -s "$test_root/xrt-output/device_trace_0.csv"

(
  cd "$test_root"
  MOCK_XRT_ARTIFACT_STYLE=newer \
  MYACCEL_XCLBIN="$profile_package/profile.xclbin" \
  MYACCEL_XRT_PROFILE_DIR="$test_root/xrt-output-newer" \
  MYACCEL_XRT_INI="$profile_package/xrt-profile.ini" \
  TIME_BIN="$mock_bin/time" \
    "$profile_package/run-profile.sh" \
      "$profile_package/input.bin" \
      newer-output.bin \
      newer-profile.log >/dev/null
)
test -s "$test_root/newer-output.bin"
test -s "$test_root/newer-profile.log"
test -s "$test_root/xrt-output-newer/xrt.run_summary"
test -s "$test_root/xrt-output-newer/profile_summary.csv"
test -s "$test_root/xrt-output-newer/timeline_trace.csv"

if (
  cd "$test_root"
  MOCK_XRT_EXIT_STATUS=17 \
  MYACCEL_XCLBIN="$profile_package/profile.xclbin" \
  MYACCEL_XRT_PROFILE_DIR="$test_root/xrt-output-failed-run" \
  MYACCEL_XRT_INI="$profile_package/xrt-profile.ini" \
  TIME_BIN="$mock_bin/time" \
    "$profile_package/run-profile.sh" \
      "$profile_package/input.bin" \
      failed-run-output.bin \
      failed-run-profile.log \
      >"$test_root/failed-run.stdout" \
      2>"$test_root/failed-run.stderr"
); then
  echo "error: failed profiled inference unexpectedly passed" >&2
  exit 1
fi
test -s "$test_root/xrt-output-failed-run/xrt.run_summary"
test -s "$test_root/xrt-output-failed-run/summary.csv"
test -s "$test_root/xrt-output-failed-run/native_trace.csv"
grep -Fq 'XRT profile artifacts:' "$test_root/failed-run.stdout"
grep -Fq 'exited with status 17' "$test_root/failed-run.stderr"

if (
  cd "$test_root"
  MOCK_XRT_NO_DEVICE_EVENTS=1 \
  MYACCEL_XCLBIN="$profile_package/profile.xclbin" \
  MYACCEL_XRT_PROFILE_DIR="$test_root/xrt-output-no-device-events" \
  MYACCEL_XRT_INI="$profile_package/xrt-profile.ini" \
  TIME_BIN="$mock_bin/time" \
    "$profile_package/run-profile.sh" \
      "$profile_package/input.bin" \
      no-events-output.bin \
      no-events-profile.log \
      >"$test_root/no-events.stdout" \
      2>"$test_root/no-events.stderr"
); then
  echo "error: empty XRT device trace unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'device trace contains no events' "$test_root/no-events.stderr"
grep -Fq 'VITIS_PROFILE=trace' "$test_root/no-events.stderr"

echo "PASS KV260 build, firmware, and XRT profile helpers"
