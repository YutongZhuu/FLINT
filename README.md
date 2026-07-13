# YOLOv5n with ONNX-MLIR and a custom Conv accelerator

This repo demonstrates the workflow we care about:

1. Compile a YOLOv5n ONNX model with ONNX-MLIR.
2. Lower supported `onnx.Conv` operations to custom external C calls.
3. Run the result locally.
4. Cross-compile Linux/aarch64 artifacts for an Ubuntu 22.04 / GCC 12 board,
   such as a Kria.

The custom accelerator is named `MyAccel`. Its runtime Conv implementation is:

```text
third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c
```

The compiled model is emitted as a shared library. The YOLO weights are embedded
inside that model library; the small driver only prepares tensors and calls the
ONNX-MLIR entry point.

```text
build/yolo-myaccel-driver-aarch64     # executable driver
build/yolov5n-myaccel-aarch64.so      # compiled graph + embedded weights
```

## Requirements

Common tools:

- Python 3
- `curl`
- `make`
- LLVM/Clang with `lld`
- Docker, only for the optional container-based local runners

For cross-compiling to Ubuntu 22.04 arm64 from macOS, you also need:

- a macOS-native `onnx-mlir` binary with `MyAccel` enabled
- a Linux/aarch64 Ubuntu 22.04 sysroot

The repo automates model download, sysroot download, cross-compilation, and
packaging. It does not rebuild the macOS-native ONNX-MLIR compiler by default;
that build is expensive and environment-specific. Provide an existing
`onnx-mlir` with `MyAccel`, or use the local build if present.

If you cloned this repo fresh, initialize the pinned ONNX-MLIR submodule:

```sh
git submodule update --init --recursive
```

The Makefile defaults to the local compiler path we have been using:

```text
third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
```

Override it if your compiler lives elsewhere:

```sh
make ONNX_MLIR_BIN=/path/to/onnx-mlir ...
```

## Fetch and prepare the YOLO model

Download YOLOv5n and convert FP16 initializers to FP32:

```sh
make model
make build/yolov5n-fp32.onnx
```

The input shape is fixed:

```text
1 x 3 x 640 x 640 float32, NCHW, normalized to [0, 1]
```

## Build and run the normal CPU model

This path uses the official ONNX-MLIR container and is useful as a baseline:

```sh
make all
make run
```

Artifacts:

```text
build/yolov5n.so
build/yolo_driver
```

## Build and run with MyAccel locally

Run YOLOv5n with supported Conv ops lowered to custom C calls:

```sh
make run-yolo-accelerator
```

Expected evidence that the custom path is active:

```text
MYACCEL: my_conv_f32 invoked
```

The local MyAccel artifacts are:

```text
build/yolov5n-myaccel.so
build/yolo-myaccel-driver
```

There is also a small Conv-only test:

```sh
make verify-accelerator
```

That test checks the generated IR for an external Conv call and compares the C
runtime output against expected values.

## Cross-compile for Ubuntu 22.04 / GCC 12 arm64

This is the recommended board path.

First fetch a Jammy arm64 sysroot:

```sh
make sysroot-jammy
```

This creates:

```text
build/aarch64-linux-jammy-sysroot
```

It intentionally uses Ubuntu 22.04 packages. This matters because Kria images
often provide `libstdc++.so.6` up to:

```text
GLIBCXX_3.4.30
```

Building against Ubuntu 24.04 / newer GCC can accidentally require symbols such
as `GLIBCXX_3.4.32`, which will not load on the board.

Then cross-compile:

```sh
make cross-yolo-accelerator-aarch64
```

Equivalent explicit form:

```sh
AARCH64_SYSROOT=build/aarch64-linux-jammy-sysroot \
ONNX_MLIR_BIN=third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir \
./scripts/cross_compile_aarch64_llvm.sh
```

Artifacts:

```text
build/yolo-myaccel-driver-aarch64
build/yolov5n-myaccel-aarch64.so
```

Check the required C++ runtime versions:

```sh
llvm-readelf --version-info build/yolo-myaccel-driver-aarch64 | grep GLIBCXX
```

With the Jammy sysroot, the driver should not require newer symbols than the
board provides.

## Package files for the board

Create a tarball containing the driver, model library, sample input, and helper
scripts:

```sh
make package-aarch64
```

This creates:

```text
build/kria-yolo-myaccel.tar.gz
```

Copy it to the board:

```sh
scp build/kria-yolo-myaccel.tar.gz ubuntu@kria:~/dev/
```

On the board:

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

If loading fails, inspect the dynamic dependencies:

```sh
ldd ./build/yolo-myaccel-driver-aarch64
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4 | tail
```

## Run the arm64 build locally with Docker

On an Apple Silicon Mac, you can sanity-check the Linux/aarch64 build in an
Ubuntu 22.04 arm64 container:

```sh
make run-aarch64-docker
```

Or directly:

```sh
./scripts/run_yolo_aarch64_docker.sh
```

This runs:

1. image preprocessing
2. the Linux/aarch64 driver
3. YOLO postprocessing

The default sample is:

```text
samples/bus.jpg
```

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

The C++ driver calls `omInstrumentPrint()` when the instrumented runtime is
present, so ONNX-MLIR timing reports are flushed automatically.

## Useful scripts

```text
scripts/fetch_ubuntu_aarch64_sysroot.sh   # download Ubuntu 22.04 arm64 sysroot
scripts/cross_compile_aarch64_llvm.sh     # cross-compile model + runtime + driver
scripts/package_aarch64_artifacts.sh      # package board artifacts
scripts/run_yolo_aarch64_docker.sh        # run arm64 artifacts in Docker
scripts/preprocess_yolo.py                # image -> raw input tensor
scripts/postprocess_yolo.py               # raw output tensor -> detections
scripts/profile_yolo_ops.sh               # ONNX-MLIR op timing summary
```

## Clean

Remove generated outputs:

```sh
make clean
```

Also remove local ONNX-MLIR build directories:

```sh
make distclean
```
