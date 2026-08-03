#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

kernel=hw/hls/Conv3x3Int8Kernel.cpp
header=hw/hls/Conv3x3Int8Kernel.h

grep -Fq '#define MYACCEL_CONV3X3_INT8_OUTPUT_STRIPE_WIDTH 32' "$header"
grep -Fq '#define MYACCEL_CONV3X3_INT8_INPUT_AXI_BITS 128' "$header"
grep -Fq 'constexpr int kOutputTileWidth = 8;' "$kernel"
grep -Fq 'ow_base += kOutputStripeWidth' "$kernel"
grep -Fq 'alignment_byte_size = 16' "$kernel"
grep -Fq 'max_widen_bitwidth = 128' "$kernel"
grep -Fq 'BIND_STORAGE variable = input_stripe type = ram_2p impl = bram' \
  "$kernel"

if grep -Fq 'ARRAY_PARTITION variable = input_stripe' "$kernel"; then
  echo 'FAIL INT8 3x3 stripe buffer must remain BRAM-backed' >&2
  exit 1
fi

if grep -Fq '#pragma HLS DATAFLOW' "$kernel"; then
  echo 'FAIL INT8 3x3 stripe path must not reintroduce DATAFLOW scheduling' >&2
  exit 1
fi

echo 'PASS INT8 3x3 stripe contract width=32 axi_bits=128 compute_tile=8'
