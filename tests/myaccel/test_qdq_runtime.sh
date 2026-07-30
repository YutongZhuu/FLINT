#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

test_binary=build/test-myaccel-qdq-runtime
mkdir -p build

${CC:-clang} -std=c11 -O2 -D_GNU_SOURCE \
  -Ithird_party/onnx-mlir/include \
  -Ithird_party/onnx-mlir \
  -Ithird_party/onnx-mlir/src/Runtime \
  -Ithird_party/onnx-mlir/src/Accelerators/MyAccel/Runtime \
  -Ihw/hls \
  tests/myaccel/qdq_conv_runtime_test.c \
  third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c \
  third_party/onnx-mlir/src/Runtime/OMTensor.c \
  third_party/onnx-mlir/src/Runtime/OMInstrument.c \
  third_party/onnx-mlir/src/Runtime/OnnxDataType.c \
  third_party/onnx-mlir/src/Support/SmallFPConversion.c \
  -lm -o "$test_binary"

"$test_binary"
