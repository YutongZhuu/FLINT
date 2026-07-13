# YOLOv5n with ONNX-MLIR and a custom Conv accelerator

This project demonstrates one focused workflow:

1. Build ONNX-MLIR with a custom accelerator named `MyAccel`.
2. Compile YOLOv5n so supported `onnx.Conv` ops lower to external C calls.
3. Run the compiled model on the local machine.
4. Cross-compile the same model/driver for Ubuntu 22.04 / GCC 12 arm64 boards
   such as Kria.

The custom Conv runtime lives in the ONNX-MLIR submodule:

```text
third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c
```

The model weights are embedded into the compiled model shared library. The
driver is intentionally small: it creates ONNX-MLIR runtime tensors, calls
`run_main_graph`, and optionally dumps the raw output tensor.

## Repository shape

```text
driver/                         C/C++ test drivers
scripts/                        setup, compile, package, profiling helpers
samples/bus.jpg                 small open-source inference sample
third_party/onnx-mlir           submodule: fork with MyAccel
Dockerfile.yolo-aarch64-run     optional Ubuntu 22.04 arm64 runtime smoke test
```

There is no LLVM fork. LLVM is a pinned build dependency of ONNX-MLIR; the
project only forks ONNX-MLIR because that is where `MyAccel` lives.

## Requirements

Common:

- Python 3
- `curl`
- `make`
- LLVM/Clang with `lld`
- a native `onnx-mlir` binary built with `MyAccel`

Optional:

- Docker, only for running the Linux/aarch64 artifacts locally in an Ubuntu
  22.04 arm64 container.

Initialize the ONNX-MLIR submodule:

```sh
git submodule update --init --recursive
```

The Makefile defaults to:

```text
third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
```

Override it when needed:

```sh
make ONNX_MLIR_BIN=/path/to/onnx-mlir ...
```

## Prepare the YOLO model

Download YOLOv5n and convert FP16 initializers to FP32:

```sh
make build/yolov5n-fp32.onnx
```

The model input is:

```text
1 x 3 x 640 x 640 float32, NCHW, normalized to [0, 1]
```

The helper:

```text
scripts/preprocess_yolo.py
```

turns an image into that raw tensor format.

## Build locally with MyAccel

Compile YOLOv5n with MyAccel and build the local driver:

```sh
make local-myaccel
```

Artifacts:

```text
build/yolov5n-myaccel.so
build/yolo-myaccel-driver
build/yolov5n-myaccel.ll
```

Run on the sample tensor:

```sh
make run-local-myaccel
```

Verify that Conv was lowered to the custom call path:

```sh
make verify-local-myaccel
```

You should see references to `my_conv_f32` in the generated LLVM IR and runtime
messages like:

```text
MYACCEL: my_conv_f32 invoked
```

## Cross-compile for Ubuntu 22.04 arm64

Fetch a Jammy arm64 sysroot:

```sh
make sysroot-jammy
```

This creates:

```text
build/aarch64-linux-jammy-sysroot
```

Then build Linux/aarch64 artifacts:

```sh
make cross-yolo-accelerator-aarch64
```

Artifacts:

```text
build/yolo-myaccel-driver-aarch64
build/yolov5n-myaccel-aarch64.so
```

The sysroot intentionally uses Ubuntu 22.04 / GCC 12 packages. This avoids
requiring newer C++ runtime symbols such as `GLIBCXX_3.4.32` on boards whose
`libstdc++.so.6` only goes up to `GLIBCXX_3.4.30`.

Check the resulting dependency versions:

```sh
llvm-readelf --version-info build/yolo-myaccel-driver-aarch64 | grep GLIBCXX
```

## Package for the board

Create a tarball with the arm64 driver, compiled model, sample input, and
minimal helper scripts:

```sh
make package-aarch64
```

Result:

```text
build/kria-yolo-myaccel.tar.gz
```

Copy to the board:

```sh
scp build/kria-yolo-myaccel.tar.gz ubuntu@kria:~/dev/
```

Run on the board:

```sh
cd ~/dev
tar -xzf kria-yolo-myaccel.tar.gz
cd kria-yolo-myaccel

./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

If you use a separate runtime library directory:

```sh
LD_LIBRARY_PATH="$PWD/build:$HOME/dev/runtime-libs" \
./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

Debug dynamic loading with:

```sh
ldd ./build/yolo-myaccel-driver-aarch64
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4 | tail
```

## Optional: run the arm64 artifacts locally with Docker

Docker is not used to compile the project. It is only a convenience for testing
the Linux/aarch64 output on a Mac before copying it to the board.

```sh
make run-aarch64-docker
```

This uses:

```text
Dockerfile.yolo-aarch64-run
```

and runs:

1. image preprocessing
2. the Linux/aarch64 driver
3. YOLO postprocessing

## Profiling

Compile an instrumented local model and summarize selected ONNX op times:

```sh
RUNS=5 ./scripts/profile_yolo_ops.sh
```

The summary includes:

- `onnx.Conv`
- `onnx.Add`
- `onnx.Sigmoid + onnx.Mul`
- `onnx.MaxPool`
- total instrumented time
- total process wall/CPU time

## Useful targets

```sh
make build/yolov5n-fp32.onnx
make local-myaccel
make run-local-myaccel
make verify-local-myaccel
make sysroot-jammy
make cross-yolo-accelerator-aarch64
make package-aarch64
make run-aarch64-docker
make clean
```

## Clean

```sh
make clean
```

Remove local ONNX-MLIR build directories too:

```sh
make distclean
```
