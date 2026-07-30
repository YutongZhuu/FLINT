#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
invocation_dir="${PWD}"

absolute_path() {
  case "$1" in
  /*) printf '%s\n' "$1" ;;
  *) printf '%s/%s\n' "${invocation_dir}" "$1" ;;
  esac
}

xclbin="${MYACCEL_XCLBIN:-/lib/firmware/xilinx/conv2d-int8/conv2d-int8.xclbin}"
input="${1:-${script_dir}/bus-input-aarch64.bin}"
output="${2:-${script_dir}/bus-output-aarch64.bin}"
log="${3:-${script_dir}/yolo-profile.log}"
driver="${script_dir}/yolo-int8-myaccel-driver-aarch64"
xrt_ini="${MYACCEL_XRT_INI:-${script_dir}/xrt-profile.ini}"
profile_dir="${MYACCEL_XRT_PROFILE_DIR:-${script_dir}/xrt-profile}"
time_bin="${TIME_BIN:-/usr/bin/time}"

xclbin=$(absolute_path "${xclbin}")
input=$(absolute_path "${input}")
output=$(absolute_path "${output}")
log=$(absolute_path "${log}")
xrt_ini=$(absolute_path "${xrt_ini}")
profile_dir=$(absolute_path "${profile_dir}")

for required_file in "${driver}" "${xclbin}" "${input}" "${xrt_ini}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "error: required file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -x "${time_bin}" ]]; then
  echo "error: GNU time is unavailable at ${time_bin}" >&2
  exit 1
fi

export MYACCEL_XCLBIN="${xclbin}"
export MYACCEL_PROFILE=1
export XRT_INI_PATH="${xrt_ini}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export LD_LIBRARY_PATH="${script_dir}/runtime-libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CPU=1 and MYACCEL_FORCE_CPU=1 force the convolution onto the host.
unset CPU
unset MYACCEL_FORCE_CPU
unset XCL_EMULATION_MODE

mkdir -p "${profile_dir}" "$(dirname "${output}")" "$(dirname "${log}")"

echo "XCLBIN: ${MYACCEL_XCLBIN}"
echo "Input:  ${input}"
echo "Output: ${output}"
echo "Log:    ${log}"
echo "XRT INI:${XRT_INI_PATH}"
echo "XRT out: ${profile_dir}"

set +e
(
  cd "${profile_dir}"
  "${time_bin}" -v \
    "${driver}" "${input}" "${output}" \
    2>&1 | tee "${log}"
)
profile_status=$?
set -e

run_summary="${profile_dir}/xrt.run_summary"
if [[ ! -s "${run_summary}" ]]; then
  echo "error: XRT did not produce ${run_summary}" >&2
  exit 1
fi

# XRT 2.13 writes summary.csv/native_trace.csv/device_trace_N.csv. Newer XRT
# releases commonly use profile_summary.csv/timeline_trace.csv instead.
summary_artifact=""
for candidate in summary.csv profile_summary.csv; do
  if [[ -s "${profile_dir}/${candidate}" ]]; then
    summary_artifact="${profile_dir}/${candidate}"
    break
  fi
done
if [[ -z "${summary_artifact}" ]]; then
  echo "error: XRT produced neither summary.csv nor profile_summary.csv" >&2
  exit 1
fi

trace_artifacts=()
trace_artifact_count=0
device_trace_count=0
device_trace_with_events=0
for candidate in native_trace.csv timeline_trace.csv; do
  if [[ -s "${profile_dir}/${candidate}" ]]; then
    trace_artifacts+=("${profile_dir}/${candidate}")
    trace_artifact_count=$((trace_artifact_count + 1))
  fi
done
while IFS= read -r candidate; do
  if [[ -s "${candidate}" ]]; then
    trace_artifacts+=("${candidate}")
    trace_artifact_count=$((trace_artifact_count + 1))
    device_trace_count=$((device_trace_count + 1))
    if awk '
      $0 == "EVENTS" { in_events = 1; next }
      $0 == "DEPENDENCIES" { in_events = 0 }
      in_events && NF { found = 1 }
      END { exit(found ? 0 : 1) }
    ' "${candidate}"; then
      device_trace_with_events=$((device_trace_with_events + 1))
    fi
  fi
done < <(find "${profile_dir}" -maxdepth 1 -type f \
  -name 'device_trace_*.csv' -print | sort)
if (( trace_artifact_count == 0 )); then
  echo "error: XRT did not produce a native, timeline, or device trace CSV" >&2
  exit 1
fi
if (( device_trace_count > 0 && device_trace_with_events == 0 )); then
  echo "error: XRT device trace contains no events; use xrt-counters.ini " \
    "for a counter image, or rebuild the short-test image with " \
    "VITIS_PROFILE=trace" >&2
  exit 1
fi

echo "XRT profile artifacts:"
ls -lh \
  "${run_summary}" \
  "${summary_artifact}" \
  "${trace_artifacts[@]}"

if (( profile_status != 0 )); then
  echo "error: profiled inference exited with status ${profile_status}" >&2
  exit "${profile_status}"
fi
