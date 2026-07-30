#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

app_name=${KV260_APP_NAME:-conv2d-kv260}
xsa_input=${1:-build/kv260-int8-only/conv_int8_only.hw.xsa}
output=${2:-build/kv260-firmware/$app_name/$app_name.dtbo}
dtg_repo=${DEVICE_TREE_REPO:-${DTG_REPO:-}}

if [[ ! "$app_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: invalid KV260_APP_NAME: $app_name" >&2
  exit 2
fi
if [ ! -f "$xsa_input" ]; then
  echo "error: linked XSA not found: $xsa_input" >&2
  exit 2
fi
if [ -z "$dtg_repo" ] || [ ! -d "$dtg_repo" ]; then
  cat >&2 <<'EOF'
error: DEVICE_TREE_REPO (or DTG_REPO) must name a device-tree-xlnx checkout
matching the Vitis release used to link the XSA.
EOF
  exit 2
fi
for required_tool in xsct dtc; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "error: $required_tool is not on PATH" >&2
    exit 2
  fi
done

absolute_file() {
  local input=$1
  local directory
  directory=$(cd "$(dirname "$input")" && pwd -P)
  printf '%s/%s\n' "$directory" "$(basename "$input")"
}

xsa_input=$(absolute_file "$xsa_input")
dtg_repo=$(cd "$dtg_repo" && pwd -P)
output_dir=$(dirname "$output")
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
output=$output_dir/$(basename "$output")

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

export KV260_DTG_XSA=$xsa_input
export KV260_DTG_REPO=$dtg_repo
export KV260_DTG_OUT=$temporary_dir/dtg

# XSCT otherwise tries to validate DISPLAY and launch Xvfb, even though HSI is
# used entirely headlessly here. Build servers commonly provide neither.
xsct -nodisp <<'EOF'
hsi open_hw_design $::env(KV260_DTG_XSA)
hsi set_repo_path $::env(KV260_DTG_REPO)
hsi create_sw_design device-tree -os device_tree -proc psu_cortexa53_0
hsi set_property CONFIG.dt_overlay true [hsi::get_os]
hsi set_property CONFIG.dt_zocl true [hsi::get_os]
hsi generate_target -dir $::env(KV260_DTG_OUT)
hsi close_hw_design [hsi current_hw_design]
exit
EOF

generated_dts=$(find "$KV260_DTG_OUT" -type f -name pl.dtsi -print -quit)
if [ -z "$generated_dts" ] || [ ! -s "$generated_dts" ]; then
  echo "error: XSCT did not generate a non-empty pl.dtsi" >&2
  exit 1
fi
if ! grep -q 'firmware-name' "$generated_dts"; then
  echo "error: generated pl.dtsi has no firmware-name property" >&2
  exit 1
fi

patched_dts=$temporary_dir/$app_name.dts
sed -E \
  "s|(firmware-name[[:space:]]*=[[:space:]]*)\"[^\"]*\"|\\1\"$app_name.bit.bin\"|" \
  "$generated_dts" >"$patched_dts"

dtc -@ -I dts -O dtb -o "$output" "$patched_dts"
if [ ! -s "$output" ]; then
  echo "error: dtc did not produce a non-empty DTBO: $output" >&2
  exit 1
fi

verification_dts=$temporary_dir/verify.dts
# DTC 1.5.0 bundled with Vitis 2022.1 aborts in its interrupt checker while
# decompiling a valid unresolved overlay (external phandles are 0xffffffff
# until the base tree applies the fixups). Disable only that diagnostic for
# this round-trip property check; the board's newer DTC still validates the
# completed bundle before activation.
dtc -Wno-interrupts_property -I dtb -O dts \
  -o "$verification_dts" "$output"
if ! grep -Fq "firmware-name = \"$app_name.bit.bin\"" "$verification_dts"; then
  echo "error: compiled DTBO firmware-name verification failed" >&2
  exit 1
fi

cp "$patched_dts" "$output_dir/$app_name.dts"

printf 'Generated %s\n' "$output"
printf 'Source XSA: %s\n' "$xsa_input"
printf 'Firmware name: %s.bit.bin\n' "$app_name"
