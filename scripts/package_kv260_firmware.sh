#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

app_name=${KV260_APP_NAME:-conv2d-kv260}
xclbin_input=${1:-build/kv260-hls/conv2d_kernel.hw.xclbin}
dtbo_input=${2:-}
output_dir=${3:-build/kv260-firmware/$app_name}

if [ ! -f "$xclbin_input" ]; then
  echo "error: xclbin not found: $xclbin_input" >&2
  exit 2
fi

if [ -z "$dtbo_input" ] || [ ! -f "$dtbo_input" ]; then
  cat >&2 <<EOF
usage: $0 [xclbin] <platform.dtbo> [output-directory]

The DTBO must come from the same KV260 Vitis platform used to link the xclbin.
For a kernel-only change, pass the DTBO from the existing firmware package.
EOF
  exit 2
fi

for required_tool in xclbinutil bootgen; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "error: $required_tool is not on PATH" >&2
    exit 2
  fi
done

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

raw_bit="$temporary_dir/$app_name.bit"
bootgen_bif="$temporary_dir/$app_name.bif"

xclbinutil \
  --dump-section "BITSTREAM:RAW:$raw_bit" \
  --input "$xclbin_input"

cat >"$bootgen_bif" <<EOF
all:
{
  [destination_device = pl] $raw_bit
}
EOF

mkdir -p "$output_dir"

bootgen \
  -image "$bootgen_bif" \
  -arch zynqmp \
  -o "$output_dir/$app_name.bit.bin" \
  -w

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
