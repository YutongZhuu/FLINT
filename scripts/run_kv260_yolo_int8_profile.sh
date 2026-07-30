#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

xclbin="${MYACCEL_XCLBIN:-/lib/firmware/xilinx/conv2d-int8/conv2d-int8.xclbin}"
input="${1:-${script_dir}/bus-input-aarch64.bin}"
output="${2:-${script_dir}/bus-output-aarch64.bin}"
log="${3:-${script_dir}/yolo-profile.log}"
driver="${script_dir}/yolo-int8-myaccel-driver-aarch64"

for required_file in "${driver}" "${xclbin}" "${input}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "error: required file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -x /usr/bin/time ]]; then
  echo "error: /usr/bin/time is unavailable; install it with: sudo apt install time" >&2
  exit 1
fi

export MYACCEL_XCLBIN="${xclbin}"
export MYACCEL_PROFILE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export LD_LIBRARY_PATH="${script_dir}/runtime-libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CPU=1 and MYACCEL_FORCE_CPU=1 force the convolution onto the host.
unset CPU
unset MYACCEL_FORCE_CPU

echo "XCLBIN: ${MYACCEL_XCLBIN}"
echo "Input:  ${input}"
echo "Output: ${output}"
echo "Log:    ${log}"

/usr/bin/time -v \
  "${driver}" "${input}" "${output}" \
  2>&1 | tee "${log}"
