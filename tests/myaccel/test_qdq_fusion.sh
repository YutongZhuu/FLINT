#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

onnx_mlir=${ONNX_MLIR_BIN:-third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir}
model=${AARCH64_MODEL:-build/yolov5n-int8.onnx}
expected_calls=${MYACCEL_EXPECTED_FUSED_CONVS:-60}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/myaccel-qdq-fusion.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
output="$test_dir/model"

"$onnx_mlir" --maccel=MyAccel --disable-recompose --EmitMLIR -O0 \
  -o "$output" "$model"
ir="$output.onnx.mlir"

fused_calls=$(grep -c 'funcName = "my_conv_qdq_i8"' "$ir" || true)
int8_outputs=$(grep 'funcName = "my_conv_qdq_i8"' "$ir" |
  grep -c 'memref<[^,]*xi8>' || true)
old_calls=$(grep -c 'funcName = "my_conv_qdq_i8_f32"' "$ir" || true)
fp32_calls=$(grep -c 'funcName = "my_conv_f32"' "$ir" || true)
fused_calls=${fused_calls:-0}
int8_outputs=${int8_outputs:-0}
old_calls=${old_calls:-0}
fp32_calls=${fp32_calls:-0}

test "$fused_calls" -eq "$expected_calls"
test "$int8_outputs" -eq "$expected_calls"
test "$old_calls" -eq 0
test "$fp32_calls" -eq 0

printf 'PASS fused_qdq_conv_calls=%d int8_outputs=%d fp32_fallbacks=%d\n' \
  "$fused_calls" "$int8_outputs" "$fp32_calls"
