#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

kernel=hw/hls/Conv3x3Int8Kernel.cpp
header=hw/hls/Conv3x3Int8Kernel.h

grep -Fq '#define MYACCEL_CONV3X3_INT8_OUTPUT_STRIPE_WIDTH 32' "$header"
grep -Fq '#define MYACCEL_CONV3X3_INT8_INPUT_AXI_BITS 128' "$header"
grep -Fq 'typedef ap_uint<MYACCEL_CONV3X3_INT8_INPUT_AXI_BITS>' "$header"
grep -Fq 'const myaccel_conv3x3_i8_input_axi_t *x' "$header"
grep -Fq 'constexpr int kOutputTileWidth = 8;' "$kernel"
grep -Fq 'ow_base += kOutputStripeWidth' "$kernel"
grep -Fq 'constexpr int kInputStripeBeats =' "$kernel"
grep -Fq 'LoadInputStripeBeatLoop:' "$kernel"
grep -Fq 'BIND_STORAGE variable = input_stripe type = ram_2p impl = bram' \
  "$kernel"

if grep -Fq 'alignment_byte_size' "$kernel"; then
  echo 'FAIL Vitis HLS 2022.1 rejects alignment_byte_size on m_axi pragmas' >&2
  exit 1
fi

if grep -Fq 'max_widen_bitwidth' "$kernel"; then
  echo 'FAIL the input width must come from its explicit 128-bit type' >&2
  exit 1
fi

if grep -Fq 'ARRAY_PARTITION variable = input_stripe' "$kernel"; then
  echo 'FAIL INT8 3x3 stripe buffer must remain BRAM-backed' >&2
  exit 1
fi

if grep -Fq '#pragma HLS DATAFLOW' "$kernel"; then
  echo 'FAIL INT8 3x3 stripe path must not reintroduce DATAFLOW scheduling' >&2
  exit 1
fi

echo 'PASS INT8 3x3 stripe contract width=32 axi_bits=128 beats=5 compute_tile=8'
