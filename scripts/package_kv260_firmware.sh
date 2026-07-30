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
xclbin_info="$temporary_dir/$app_name.xclbin.info"

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

# A matching firmware-name alone does not prove that the overlay came from the
# same linked design. Compare every xclbin compute-unit instance/control address
# with the DTBO's generated symbol path before extracting the bitstream.
xclbinutil --info --input "$xclbin_input" >"$xclbin_info"
accelerator_map="$temporary_dir/$app_name.accelerator-map"
awk '
  $1 == "Instance:" {
    instance = $2
    next
  }
  instance != "" && $1 == "Base" && $2 == "Address:" {
    print instance, $3
    instance = ""
  }
' "$xclbin_info" >"$accelerator_map"
if [ ! -s "$accelerator_map" ]; then
  echo "error: xclbin contains no compute-unit instance/address metadata" >&2
  exit 2
fi

while read -r instance address; do
  if [[ ! "$instance" =~ ^[A-Za-z0-9_]+$ ]] ||
      [[ ! "$address" =~ ^0[xX][0-9A-Fa-f]+$ ]]; then
    echo "error: malformed xclbin compute-unit metadata: $instance $address" \
      >&2
    exit 2
  fi
  normalized_address=$(printf '%s' "$address" | tr 'A-F' 'a-f')
  normalized_address=${normalized_address#0x}
  normalized_address=$(printf '%s' "$normalized_address" | \
    sed 's/^0*//')
  if [ -z "$normalized_address" ]; then
    normalized_address=0
  fi
  if ! grep -Eq \
    "^[[:space:]]*${instance}[[:space:]]*=[[:space:]]*\"[^\"]*@${normalized_address}\";" \
    "$dtbo_dts"; then
    cat >&2 <<EOF
error: DTBO does not contain the xclbin compute-unit address mapping.
  instance: $instance
  address:  $address
  xclbin:   $xclbin_input
  DTBO:     $dtbo_input
Regenerate the DTBO from the XSA exported by the same Vitis link.
EOF
    exit 2
  fi
  node_name=$(printf '%s' "$instance" | sed -E 's/_[0-9]+$//')
  if ! awk -v node_name="$node_name" -v expected="$normalized_address" '
    function normalize_hex(value) {
      value = tolower(value)
      sub(/^0x/, "", value)
      sub(/^0+/, "", value)
      return value == "" ? "0" : value
    }
    {
      line = $0
      if (!in_node &&
          line ~ "^[[:space:]]*" node_name "@" expected \
                  "[[:space:]]*\\{") {
        in_node = 1
        depth = 0
      }
      if (in_node) {
        opened = gsub(/\{/, "{", line)
        closed = gsub(/\}/, "}", line)
        depth += opened - closed

        reg_line = $0
        if (reg_line ~ /^[[:space:]]*reg[[:space:]]*=/) {
          sub(/^[^<]*</, "", reg_line)
          sub(/>.*/, "", reg_line)
          cell_count = split(reg_line, cells, /[[:space:]]+/)
          if (cell_count >= 4 &&
              normalize_hex(cells[2]) == expected)
            found = 1
        }
        if (depth <= 0)
          in_node = 0
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$dtbo_dts"; then
    cat >&2 <<EOF
error: DTBO compute-unit node has no matching control-address reg property.
  instance: $instance
  node:     $node_name@$normalized_address
  address:  $address
  xclbin:   $xclbin_input
  DTBO:     $dtbo_input
Regenerate the DTBO from the XSA exported by the same Vitis link.
EOF
    exit 2
  fi
done <"$accelerator_map"

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
