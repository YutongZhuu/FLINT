#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

onnx_mlir=${ONNX_MLIR_BIN:-third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir}
model=${MODEL:-build/yolov5n-fp32.onnx}
out_base=${LOCAL_OUT_BASE:-build/yolov5n-myaccel}
driver_out=${LOCAL_DRIVER_OUT:-build/yolo-myaccel-driver}

if [ ! -x "$onnx_mlir" ]; then
  cat >&2 <<EOF
error: missing executable ONNX_MLIR_BIN: $onnx_mlir

Set ONNX_MLIR_BIN to a native onnx-mlir binary built with MyAccel enabled.
EOF
  exit 2
fi

if [ ! -f "$model" ]; then
  echo "error: missing model: $model" >&2
  echo "hint: run make build/yolov5n-fp32.onnx" >&2
  exit 2
fi

mkdir -p build

if [[ "$(uname -s)" == Darwin ]]; then
  sdk=$(xcrun --show-sdk-path)
  export SDKROOT="$sdk"
  export LIBRARY_PATH="$sdk/usr/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  rpath='@loader_path'
else
  rpath='$ORIGIN'
fi

"$onnx_mlir" \
  --maccel=MyAccel \
  --preserveLLVMIR \
  -O0 \
  -o "$out_base" \
  "$model"

c++ -std=c++17 -O2 -Wall -Wextra \
  -Ithird_party/onnx-mlir/include \
  -Ithird_party/onnx-mlir \
  driver/yolo_driver.cpp "${out_base}.so" \
  -Wl,-rpath,"$rpath" \
  -o "$driver_out"

file "${out_base}.so" "$driver_out"
