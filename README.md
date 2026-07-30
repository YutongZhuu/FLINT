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
- AMD Vitis on an x86_64 Linux machine, only for building the KV260 FPGA
  `.xclbin`.
- XRT runtime/development files on the KV260, only for running the FPGA-backed
  MyAccel path.

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

## Build the QDQ-aware MyAccel compiler and an INT8 model

Quantization and compilation are separate steps in this project:

1. Ultralytics/ONNX Runtime creates a QDQ ONNX model. Calibration happens in
   this step and the resulting model contains fixed scales, zero points, INT8
   weights, and `QuantizeLinear`/`DequantizeLinear` operations.
2. The modified ONNX-MLIR compiler recognizes a QDQ-wrapped `onnx.Conv` and
   lowers it to `my_conv_qdq_i8` instead of first materializing FP32 input
   and weight tensors for `my_conv_f32`.

ONNX-MLIR does **not** quantize an FP32 model during compilation. Passing
`yolov5n-fp32.onnx` to the compiler still produces an FP32 model.

### 1. Build ONNX-MLIR with MyAccel enabled

The QDQ rewrite is implemented in:

```text
third_party/onnx-mlir/src/Accelerators/MyAccel/MyAccelAccelerator.cpp
```

The AArch64 INT8/INT32 convolution runtime is implemented in:

```text
third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c
```

After building the LLVM/MLIR dependency under
`build/llvm-project-onnxmlir/build`, configure and build the compiler from the
repository root:

```sh
cmake -S third_party/onnx-mlir \
  -B third_party/onnx-mlir/build-host-exact \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_MLIR_ACCELERATORS=MyAccel \
  -DLLVM_DIR="$PWD/build/llvm-project-onnxmlir/build/lib/cmake/llvm" \
  -DMLIR_DIR="$PWD/build/llvm-project-onnxmlir/build/lib/cmake/mlir"

cmake --build third_party/onnx-mlir/build-host-exact \
  --config Release \
  --target onnx-mlir \
  -j "$(sysctl -n hw.ncpu)"
```

The compiler is generated at:

```text
third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
```

Check that this is the compiler being used:

```sh
ONNX_MLIR_BIN=third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
"$ONNX_MLIR_BIN" --version
"$ONNX_MLIR_BIN" --help | grep maccel
```

When only the MyAccel sources change, rerunning the `cmake --build` command is
enough; it is not necessary to recreate the QDQ model.

### 2. Export a QDQ INT8 ONNX model

The standard Ultralytics v8.4.60 export used by this repository is in:

```text
scripts/export_int8_model.py
```

Create the dedicated environment once if it is not already present:

```sh
python3 -m venv build/ultralytics-export-venv
build/ultralytics-export-venv/bin/pip install \
  'ultralytics==8.4.60' \
  'setuptools<81' \
  onnx onnxruntime onnxslim \
  pandas IPython tqdm gitpython thop seaborn
```

Run it with the dedicated Ultralytics environment. `PYTHONPATH` supplies the
class definitions needed to load the legacy YOLOv5 v7 checkpoint:

```sh
PYTHONPATH="$PWD/build/yolov5-v7.0" \
build/ultralytics-export-venv/bin/python \
  scripts/export_int8_model.py
```

The script calls the standard API:

```python
model.export(
    format="onnx",
    int8=True,
    data="coco128.yaml",
    fraction=0.25,
)
```

The current output is:

```text
models/yolov5n-ultralytics-standard_int8.onnx
```

Its external input and output are FP32. The weights and supported internal
activations are quantized using QDQ nodes. Verify that the graph really is QDQ:

```sh
build/ultralytics-export-venv/bin/python - <<'PY'
import collections
import onnx

model = onnx.load("models/yolov5n-ultralytics-standard_int8.onnx")
ops = collections.Counter(node.op_type for node in model.graph.node)
print("QuantizeLinear:", ops["QuantizeLinear"])
print("DequantizeLinear:", ops["DequantizeLinear"])
print("Conv:", ops["Conv"])
PY
```

Use a representative calibration dataset for accuracy measurements. The
calibration images affect the saved scales and zero points, but do not add any
calibration work to inference.

### 3. Confirm that MyAccel rewrites the QDQ convolutions

Emit MLIR without linking a native executable:

```sh
ONNX_MLIR_BIN=third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir
MODEL=models/yolov5n-ultralytics-standard_int8.onnx

"$ONNX_MLIR_BIN" \
  --maccel=MyAccel \
  --EmitMLIR \
  -O3 \
  -o build/yolov5n-int8-myaccel-inspect \
  "$MODEL"

grep -c 'my_conv_qdq_i8' \
  build/yolov5n-int8-myaccel-inspect.onnx.mlir
grep -c 'my_conv_f32' \
  build/yolov5n-int8-myaccel-inspect.onnx.mlir
```

`my_conv_qdq_i8` means that the Conv uses INT8 input/weights and INT32
accumulation in the custom runtime. `my_conv_f32` identifies an FP32 fallback.
For the current YOLOv5n export, `tests/myaccel/test_qdq_fusion.sh` expects 60
INT8 calls and no FP32 convolution fallbacks.
The exact counts can change when ONNX-MLIR optimizations fuse or reshape the
graph, so inspect both counts rather than assuming every source Conv remains a
separate call.

### 4. Cross-compile the QDQ model for KV260

Create the Jammy AArch64 sysroot once if it does not already exist:

```sh
make sysroot-jammy
```

Then explicitly pass the QDQ model to the cross-compile script:

```sh
AARCH64_SYSROOT=build/aarch64-linux-jammy-sysroot \
ONNX_MLIR_BIN=third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir \
AARCH64_MODEL=models/yolov5n-ultralytics-standard_int8.onnx \
AARCH64_OUT_BASE=build/yolov5n-int8-myaccel-aarch64 \
AARCH64_DRIVER_OUT=build/yolo-int8-myaccel-driver-aarch64 \
ONNX_MLIR_OPT_LEVEL=3 \
./scripts/cross_compile_aarch64_llvm.sh
```

Do not omit `AARCH64_MODEL`: the script's compatibility default is currently
`build/yolov5n-fp32.onnx`.

Artifacts:

```text
build/yolov5n-int8-myaccel-aarch64.o
build/yolov5n-int8-myaccel-aarch64.so
build/yolo-int8-myaccel-driver-aarch64
```

The build targets Cortex-A53, enables ONNX-MLIR parallel lowering, compiles at
`-O3`, and links the AArch64 LLVM OpenMP runtime (`libomp.so.5`). The KV260 must
have that runtime installed.

### 5. Run the compiled QDQ model on KV260

Keep the following board layout because the driver records the model library
dependency as `build/yolov5n-int8-myaccel-aarch64.so`:

```text
/home/ubuntu/dev/int8-myaccel-test/
├── yolo-int8-myaccel-driver-aarch64
└── build/
    └── yolov5n-int8-myaccel-aarch64.so
```

Run inference from that directory:

```sh
cd /home/ubuntu/dev/int8-myaccel-test

MYACCEL_PROFILE=1 \
OMP_NUM_THREADS=4 \
OMP_PROC_BIND=true \
/usr/bin/time -f 'elapsed=%e s, CPU=%P' \
  ./yolo-int8-myaccel-driver-aarch64 \
  /home/ubuntu/dev/build/bus-input-aarch64.bin \
  build/output-int8.bin
```

`bus-input-aarch64.bin` is a `[1,3,640,640]` FP32 input tensor, not the model.
The compiled model and embedded weights are in
`build/yolov5n-int8-myaccel-aarch64.so`. With `MYACCEL_PROFILE=1`, rewritten
convolutions print `MYACCEL_INT8` lines.

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

By default, `MyAccelXrt.cpp` builds a stub that returns failure, so
`my_conv_f32` uses the CPU fallback implementation. To build the arm64 runtime
with the real XRT wrapper, enable XRT and point the cross-compile script at
target/aarch64 XRT headers and libraries:

```sh
export MYACCEL_USE_XRT=1
export AARCH64_XRT_INCLUDE_DIR=/path/to/aarch64/include
export AARCH64_XRT_LIB_DIR=/path/to/aarch64/lib
make cross-yolo-accelerator-aarch64
```

The include directory must contain `xrt/xrt_bo.h`. The library directory must
contain `libxrt_coreutil.so` or `libxrt_coreutil.a`.

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

## Build the KV260 Conv2D xclbin

The KV260 is the runtime target, not the Vitis build machine. Build the FPGA
binary on an x86_64 Linux machine with Vitis installed, then copy the resulting
`.xclbin` to the board.

The accelerator contains matching specialized FP32 and INT8 HLS compute
units. The optimized INT8-only image also has an optional 6x6 stem:

```text
Conv1x1Kernel.cpp -> conv1x1_kernel
Conv3x3Kernel.cpp -> conv3x3_kernel
Conv1x1Int8Kernel.cpp -> conv1x1_i8_kernel
Conv3x3Int8Kernel.cpp -> conv3x3_i8_kernel
Conv6x6StemInt8Kernel.cpp -> conv6x6_stem_i8_kernel (opt-in)
```

The packed INT8 kernels preserve the logical NCHW/OIHW byte layout while using
32-bit AXI words. They apply INT32 bias, fixed-point requantization, and INT8
clamping in hardware, so the runtime no longer returns a full INT32 feature map
to the Arm for post-processing. The default image contains 1x1 and 3x3 only.
The 6x6 stem is a separate fit experiment and is enabled at build time with
`VITIS_INCLUDE_6X6_STEM=1`, then at runtime with
`MYACCEL_ENABLE_6X6_STEM=1`. Without both, the stem safely remains on the host.
The INT8 runtime accepts output only when XRT reports
`ERT_CMD_STATE_COMPLETED`. Other command states return to the CPU fallback;
the default per-layer timeout is 30 seconds and can be changed with a positive
`MYACCEL_XRT_RUN_TIMEOUT_MS` value.

Run the ordinary C++ numerical test before sending the sources to a Vitis
machine:

```sh
scripts/test_myaccel_hls_kernels.sh
```

It requires a KV260 Vitis platform `.xpfm`. If the platform is not already
installed, one known 2022.1 flow is to build the Kria platform on the Vitis
machine:

```sh
source /opt/Xilinx/Vitis/2022.1/settings64.sh

mkdir -p ~/xilinx-platforms
cd ~/xilinx-platforms
git clone --branch xlnx_rel_v2022.1 --recursive https://github.com/Xilinx/kria-vitis-platforms.git
cd kria-vitis-platforms/kv260
make platform PFM=kv260_ispMipiRx_vcu_DP
```

If platform packaging fails with `Xvfb is not available`, either install
`xvfb` on the Vitis machine or connect with a valid X display, for example
SSH X forwarding from XQuartz on macOS. The useful output platform is the final
`.xpfm` under:

```text
platforms/xilinx_kv260_ispMipiRx_vcu_DP_202210_1/
```

not the intermediate copy under `platforms/xsct/...`.

Verify the platform:

```sh
export PLATFORM=$HOME/xilinx-platforms/kria-vitis-platforms/kv260/platforms/xilinx_kv260_ispMipiRx_vcu_DP_202210_1/kv260_ispMipiRx_vcu_DP.xpfm
platforminfo "$PLATFORM"
```

Then build this project's kernel `.xclbin` from the repo root:

```sh
cd ~/capstone-compiler
source /opt/Xilinx/Vitis/2022.1/settings64.sh
export PLATFORM=$HOME/xilinx-platforms/kria-vitis-platforms/kv260/platforms/xilinx_kv260_ispMipiRx_vcu_DP_202210_1/kv260_ispMipiRx_vcu_DP.xpfm

scripts/build_kv260_conv2d_xclbin.sh
```

For the packed INT8-only image and its routed reports, use:

```sh
scripts/build_kv260_int8_only_xclbin.sh
```

The default build has no debug monitors. `VITIS_PROFILE=1` (or `counters`)
adds aggregate activation-bandwidth and CU-stall counters. The Vitis 2022.1
KV260 platform used here has no trace-master decoration, so full DDR-backed
device trace insertion fails before synthesis. `VITIS_PROFILE=trace` instead
uses an 8 KiB on-chip FIFO and is intended only for the short standalone XRT
tests, not a complete YOLO inference.

Each hardware build collects `csynth.rpt` for every included kernel plus
post-route timing, flat and hierarchical utilization, and estimated power.
Generate the overlay from that build's XSA, then package the matched XRT_FLAT
firmware set:

```sh
export DEVICE_TREE_REPO=$HOME/tools/device-tree-xlnx-2022.1
export KV260_APP_NAME=conv2d-int8

scripts/generate_kv260_dtbo.sh \
  build/kv260-int8-only/conv_int8_only.hw.xsa \
  build/kv260-firmware/conv2d-int8/conv2d-int8.dtbo

scripts/package_kv260_firmware.sh \
  build/kv260-int8-only/conv_int8_only.hw.xclbin \
  build/kv260-firmware/conv2d-int8/conv2d-int8.dtbo \
  build/kv260-firmware/conv2d-int8
```

The generator enables overlay/ZOCL nodes and verifies that the compiled DTBO
names `conv2d-int8.bit.bin`. Keep the resulting `.bit.bin`, `.dtbo`, `.xclbin`,
and `shell.json` together; do not reuse an overlay from another XSA.

The generic `build_kv260_conv2d_xclbin.sh` script compiles four `.xo` files
and links them into one xclbin:

```text
build/kv260-hls/conv1x1_kernel.hw.xo
build/kv260-hls/conv3x3_kernel.hw.xo
build/kv260-hls/conv1x1_i8_kernel.hw.xo
build/kv260-hls/conv3x3_i8_kernel.hw.xo
build/kv260-hls/conv2d_kernel.hw.xclbin
```

Copy it to the KV260:

```sh
scp build/kv260-hls/conv2d_kernel.hw.xclbin ubuntu@kria:~/dev/
```

## Run with XRT on KV260

QDQ convolutions enter MyAccel through `my_conv_qdq_i8`. The XRT wrapper sends
supported layers to `conv1x1_i8_kernel` or `conv3x3_i8_kernel`; those kernels
return the final requantized INT8 tensor. A 6x6 layer uses the optional stem
kernel only when both the matching image and runtime gate are enabled. Other
unsupported INT8 layers use the same bit-exact host implementation.

FP32 convolutions continue to enter through `my_conv_f32`: supported 1x1 and
3x3 layers use `conv1x1_kernel` and `conv3x3_kernel`, while other FP32 layers
fall back to the host reference.

On the KV260, set the xclbin path before running the compiled model:

```sh
export MYACCEL_XCLBIN=$HOME/dev/conv2d_kernel.hw.xclbin

./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

Use `CPU=1` to keep convolution on the host. `MYACCEL_FORCE_CPU=1` remains
available as a compatibility alias:

```sh
CPU=1 ./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

For a small board-side correctness test with transfer and compute timings,
build and run the standalone XRT test on the KV260:

```sh
scripts/build_kv260_host_xrt_conv_test.sh
scripts/run_kv260_conv2d_xrt_test.sh \
  build/kv260-hls/conv2d_kernel.hw.xclbin
```

It reports BO allocation, host writes, H2D synchronization, `run.wait()`, D2H
synchronization, and host reads separately for all four kernels.

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

For a KV260 run, select the XRT configuration that matches the linked image:

```sh
# No device monitors: host XRT API timeline plus MYACCEL per-layer timings.
MYACCEL_XRT_INI=$PWD/scripts/xrt-host-profile.ini \
  scripts/run_kv260_yolo_int8_profile.sh

# VITIS_PROFILE=counters image: add aggregate device bandwidth/stall counters.
MYACCEL_XRT_INI=$PWD/scripts/xrt-counters.ini \
  scripts/run_kv260_yolo_int8_profile.sh

# VITIS_PROFILE=trace image: one short HostXrtConvTest invocation only.
XRT_INI_PATH=$PWD/scripts/xrt-profile.ini \
  ./host_xrt_int8_test ./conv_int8_only.hw.xclbin --int8-3x3-only
```

Set `MYACCEL_XRT_PROFILE_DIR` to a new, nonexistent directory for each full
profile. The runner rejects an existing path so stale summaries or traces
cannot be accepted as evidence from a later invocation.

Firmware packaging verifies more than the mutable `firmware-name`: every
xclbin compute-unit instance/control address must also appear in the DTBO
symbol map. A mismatch requires regenerating the overlay from the XSA exported
by the same Vitis link.

`xdputil` inspects and benchmarks Vitis AI DPU/xmodel deployments. These are
custom HLS/XRT kernels, so their scheduling evidence comes from HLS reports,
their routed evidence from Vivado reports, and their runtime evidence from XRT
plus the MyAccel timing log.

## Scripts

The scripts are intentionally thin wrappers behind Make targets:

```text
build_yolo_myaccel_local.sh      local MyAccel compile + driver link
fetch_ubuntu_aarch64_sysroot.sh  Ubuntu 22.04 arm64 sysroot download
cross_compile_aarch64_llvm.sh    macOS -> Linux/aarch64 cross compile
package_aarch64_artifacts.sh     board tarball assembly
run_yolo_aarch64_docker.sh       optional Ubuntu 22.04 arm64 runtime smoke test
profile_yolo_ops.sh              local ONNX op profiling
build_kv260_int8_only_xclbin.sh  packed INT8 hardware image and reports
generate_kv260_dtbo.sh           matching XSA-to-overlay generation
package_kv260_firmware.sh        checked XRT_FLAT firmware bundle
run_kv260_yolo_int8_profile.sh   board XRT/MyAccel profiling
```

Python helpers:

```text
fp16_to_fp32.py                  ONNX model conversion used by Makefile
preprocess_yolo.py               image -> raw YOLO input tensor
postprocess_yolo.py              raw YOLO output tensor -> detections
summarize_onnxmlir_profile.py    parser used by profile_yolo_ops.sh
```

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
