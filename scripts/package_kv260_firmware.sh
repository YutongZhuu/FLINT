#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

app_name=${KV260_APP_NAME:-conv2d-kv260}
xclbin_input=${1:-build/kv260-hls/conv2d_kernel.hw.xclbin}
dtbo_input=${2:-}
output_dir=${3:-build/kv260-firmware/$app_name}

if [[ ! "$app_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: invalid KV260_APP_NAME: $app_name" >&2
  exit 2
fi

if [ ! -f "$xclbin_input" ]; then
  echo "error: xclbin not found: $xclbin_input" >&2
  exit 2
fi

if [ -z "$dtbo_input" ] || [ ! -f "$dtbo_input" ]; then
  cat >&2 <<EOF
usage: $0 [xclbin] <platform.dtbo> [output-directory]

The DTBO must be generated from the XSA exported by the same Vitis link as the
xclbin. Its firmware-name property must name $app_name.bit.bin.
EOF
  exit 2
fi

for required_tool in xclbinutil bootgen dtc; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "error: $required_tool is not on PATH" >&2
    exit 2
  fi
done

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

raw_bit="$temporary_dir/$app_name.bit"
dtbo_dts="$temporary_dir/$app_name.dts"

# Vitis 2022.1 ships DTC 1.5.0, whose interrupt checker aborts when it sees
# unresolved external phandles in an overlay. The check is not needed to read
# firmware-name, so disable only that diagnostic during decompilation.
dtc -Wno-interrupts_property -I dtb -O dts \
  -o "$dtbo_dts" "$dtbo_input"
firmware_name=$(sed -n \
  's/.*firmware-name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
  "$dtbo_dts")
expected_firmware_name=$app_name.bit.bin
if [ -z "$firmware_name" ]; then
  echo "error: DTBO has no firmware-name property: $dtbo_input" >&2
  exit 2
fi
if [ "$firmware_name" != "$expected_firmware_name" ]; then
  cat >&2 <<EOF
error: DTBO firmware-name does not match this firmware application.
  expected: $expected_firmware_name
  actual:   $firmware_name
  DTBO:     $dtbo_input
Regenerate the DTBO from the linked XSA and set firmware-name before packaging.
EOF
  exit 2
fi

xclbinutil \
  --dump-section "BITSTREAM:RAW:$raw_bit" \
  --input "$xclbin_input"

mkdir -p "$output_dir"

(
  cd "$temporary_dir"
  printf 'all:{%s}\n' "$app_name.bit" >"$app_name.bif"
  bootgen \
    -w \
    -arch zynqmp \
    -process_bitstream bin \
    -image "$app_name.bif"
)

generated_bit_bin=$raw_bit.bin
if [ ! -s "$generated_bit_bin" ]; then
  echo "error: bootgen did not produce $generated_bit_bin" >&2
  exit 1
fi
cp "$generated_bit_bin" "$output_dir/$app_name.bit.bin"

cp "$dtbo_input" "$output_dir/$app_name.dtbo"
cp "$xclbin_input" "$output_dir/$app_name.xclbin"

cat >"$output_dir/shell.json" <<'EOF'
{
  "shell_type": "XRT_FLAT",
  "num_slots": "1"
}
EOF

echo "[done] Kria firmware bundle:"
find "$output_dir" -maxdepth 1 -type f -print | sort
