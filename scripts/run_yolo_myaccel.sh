#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

platform=${PLATFORM:-linux/amd64}
image=${MYACCEL_IMAGE:-onnx-mlir-myaccel-dev}
dockerfile=${MYACCEL_DOCKERFILE:-Dockerfile.myaccel}
mode=${MYACCEL_MODE:-preload}

docker build --platform "$platform" -f "$dockerfile" -t "$image" .

docker run --rm --platform "$platform" --entrypoint /bin/bash \
  -e MYACCEL_MODE="$mode" \
  -v "$PWD:/work" -w /work/third_party/onnx-mlir "$image" -lc '
set -euo pipefail
mkdir -p /work/build/preload

gcc -std=c11 -O3 -fPIC -I/work/third_party/onnx-mlir/include \
  -c src/Accelerators/MyAccel/Runtime/MyConv.c \
  -o /work/build/preload/MyConv.o
rm -f /usr/local/lib/libMyAccelRuntime.a
ar rcs /usr/local/lib/libMyAccelRuntime.a /work/build/preload/MyConv.o

if [ "$MYACCEL_MODE" = preload ]; then
  g++ -std=c++17 -O2 -fPIC -shared -DONNX_ML=1 -DONNX_NAMESPACE=onnx \
    -I/workdir/llvm-project/llvm/include \
    -I/workdir/llvm-project/build/include \
    -I/workdir/llvm-project/mlir/include \
    -I/workdir/llvm-project/build/tools/mlir/include \
    -I/work/third_party/onnx-mlir -I/work/third_party/onnx-mlir/build-myaccel \
    -I/work/third_party/onnx-mlir/third_party/onnx \
    -I/work/third_party/onnx-mlir/build-myaccel/third_party/onnx \
    src/Accelerators/MyAccel/MyAccelAccelerator.cpp \
    src/Accelerators/MyAccel/PreloadInit.cpp /work/build/preload/MyConv.o \
    -L/usr/local/lib -lOMCompilerPasses -lOMONNXToKrnl -lOMKrnlToLLVM \
    -lOMCompilerOptions -lOMAccelerator -lOMONNXOps -lOMKrnlOps \
    -Wl,-rpath,/usr/local/lib -o /work/build/libMyAccelPreload.so

  LD_PRELOAD=/work/build/libMyAccelPreload.so /usr/local/bin/onnx-mlir \
    --maccel=NNPA --preserveLLVMIR -O0 \
    -o /work/build/yolov5n-myaccel /work/build/yolov5n-fp32.onnx
else
  /usr/local/bin/onnx-mlir \
    --maccel=MyAccel --preserveLLVMIR -O0 \
    -o /work/build/yolov5n-myaccel /work/build/yolov5n-fp32.onnx
fi

g++ -std=c++17 -O2 -I/usr/local/include /work/driver/yolo_driver.cpp \
  /work/build/yolov5n-myaccel.so -Wl,-rpath,/work/build \
  -o /work/build/yolo-myaccel-driver

/work/build/yolo-myaccel-driver
'
