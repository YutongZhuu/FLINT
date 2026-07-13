#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

image=${1:-samples/bus.jpg}
input_bin=${2:-build/bus-input-aarch64.bin}
output_bin=${3:-build/bus-myaccel-aarch64.bin}
docker_image=${YOLO_AARCH64_DOCKER_IMAGE:-onnx-yolo-aarch64-jammy-runner}

driver=build/yolo-myaccel-driver-aarch64
model=build/yolov5n-myaccel-aarch64.so

if [ ! -x "$driver" ]; then
  cat >&2 <<EOF
error: missing ARM64 driver: $driver

Build it first with:

  make cross-yolo-accelerator-aarch64

or run scripts/cross_compile_aarch64_llvm.sh with AARCH64_SYSROOT and ONNX_MLIR_BIN set.
EOF
  exit 2
fi

if [ ! -f "$model" ]; then
  echo "error: missing ARM64 model shared library: $model" >&2
  exit 2
fi

if [ ! -f "$image" ]; then
  echo "error: image not found: $image" >&2
  exit 2
fi

mkdir -p build

echo "[docker] building/checking runtime image: $docker_image"
docker build \
  --platform linux/arm64 \
  -f Dockerfile.yolo-aarch64-run \
  -t "$docker_image" \
  .

echo "[docker/linux/arm64] preprocessing, running, and postprocessing"
docker run --rm \
  --platform linux/arm64 \
  -v "$PWD:/work" \
  -w /work \
  "$docker_image" \
  bash -lc '
    set -euo pipefail
    python3 scripts/preprocess_yolo.py "$0" "$1"
    /work/"$2" /work/"$1" /work/"$3"
    python3 scripts/postprocess_yolo.py "$0" "$3"
  ' "$image" "$input_bin" "$driver" "$output_bin"
