#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ONNX_MLIR=${ONNX_MLIR:-third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir}
MODEL=${MODEL:-build/yolov5n-fp32.onnx}
INPUT=${INPUT:-build/bus-input.bin}
OUT_PREFIX=${OUT_PREFIX:-build/yolov5n-profiled}
RUNS=${RUNS:-5}

if [ ! -x "$ONNX_MLIR" ]; then
  echo "missing onnx-mlir executable: $ONNX_MLIR" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi
if [ ! -f "$INPUT" ]; then
  echo "missing input tensor: $INPUT" >&2
  echo "create one with: python3 scripts/preprocess_yolo.py samples/bus.jpg build/bus-input.bin" >&2
  exit 1
fi

sdk=$(xcrun --show-sdk-path)
export SDKROOT="$sdk"
export LIBRARY_PATH="$sdk/usr/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

echo "[compile] profiled model: ${OUT_PREFIX}.so"
"$ONNX_MLIR" -O3 --EmitLib \
  --instrument-ops='onnx.Conv,onnx.Add,onnx.Sigmoid,onnx.Mul,onnx.MaxPool' \
  --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime \
  -o "$OUT_PREFIX" "$MODEL"

echo "[compile] profiled driver: build/yolo-driver-profiled"
c++ -std=c++17 -O2 -Wall -Wextra \
  -Ithird_party/onnx-mlir/include \
  driver/yolo_driver.cpp "${OUT_PREFIX}.so" \
  -Wl,-rpath,@loader_path \
  -o build/yolo-driver-profiled

mkdir -p build/profile
for i in $(seq 1 "$RUNS"); do
  echo "[run] $i/$RUNS"
  /usr/bin/time -p \
    ./build/yolo-driver-profiled "$INPUT" "build/profile/output-${i}.bin" \
    > "build/profile/run-${i}.txt" 2> "build/profile/time-${i}.txt"
done

python3 scripts/summarize_onnxmlir_profile.py build/profile/run-*.txt
