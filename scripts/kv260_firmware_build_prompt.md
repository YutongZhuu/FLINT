# Prompt: Build a Safe KV260 Conv2D Firmware Application

You are working in the `capstone-compiler` repository on a Linux machine with
AMD Vitis, Vivado, and PetaLinux 2022.1 available. Execute this task, do not
only propose a plan.

## Problem

The standalone AArch64 XRT host test opens XRT device 0, but the KV260 stops
responding inside `xrt::device::load_xclbin()` while loading
`conv2d_kernel.hw.xclbin`.

The board boots with this flat application active:

```text
k26-starter-kits  XRT_FLAT  Active_slot=0
```

The Conv2D xclbin targets a different full platform:

```text
xilinx_kv260_ispMipiRx_vcu_DP_kv260_ispMipiRx_vcu_DP_1_0
```

Only the `.xclbin` was deployed. A Kria flat application must be installed as
a matched firmware directory containing its bitstream, device-tree overlay,
XRT metadata, and `shell.json`, then activated through `xmutil`.

## Required Output

Build an application named `conv2d-kv260` with exactly these non-empty files:

```text
build/kv260-firmware/conv2d-kv260/
  conv2d-kv260.bit.bin
  conv2d-kv260.dtbo
  conv2d-kv260.xclbin
  shell.json
```

The `.bit.bin`, `.dtbo`, and `.xclbin` must come from the same Vitis/Vivado
2022.1 hardware design and use the same basename. Do not combine artifacts
from different builds.

## Repository Inputs

- Kernel sources:
  - `third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv1x1Kernel.cpp`
  - `third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv3x3Kernel.cpp`
  - `third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv1x1Int8Kernel.cpp`
  - `third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv3x3Int8Kernel.cpp`
- Kernel names: `conv1x1_kernel`, `conv3x3_kernel`, `conv1x1_i8_kernel`,
  and `conv3x3_i8_kernel`
- Existing helper: `scripts/build_kv260_conv2d_xclbin.sh`
- Platform: `xilinx_kv260_ispMipiRx_vcu_DP_202210_1`
- Target: `hw`
- Kernel link clock: `100000000` Hz

Resolve the checkout dynamically:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
```

## Execution

1. Source the Vitis, Vivado, and PetaLinux 2022.1 environments. Print the
   versions of `v++`, `vivado`, `bootgen`, `xsct`, `dtc`, and
   `petalinux-build`.

2. Resolve the exact `.xpfm` with `platforminfo`. Stop if it is not the 2022.1
   `kv260_ispMipiRx_vcu_DP` platform.

3. Build the hardware kernel at 100 MHz:

   ```bash
   mkdir -p build/kv260-hls

   v++ --compile \
     --target hw \
     --platform "$KV260_PLATFORM" \
     --kernel conv1x1_kernel \
     --include third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime \
     third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv1x1Kernel.cpp \
     --output build/kv260-hls/conv1x1_kernel.hw.xo

   v++ --compile \
     --target hw \
     --platform "$KV260_PLATFORM" \
     --kernel conv3x3_kernel \
     --include third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime \
     third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv3x3Kernel.cpp \
     --output build/kv260-hls/conv3x3_kernel.hw.xo

   v++ --compile \
     --target hw \
     --platform "$KV260_PLATFORM" \
     --kernel conv1x1_i8_kernel \
     --include third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime \
     third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv1x1Int8Kernel.cpp \
     --output build/kv260-hls/conv1x1_i8_kernel.hw.xo

   v++ --compile \
     --target hw \
     --platform "$KV260_PLATFORM" \
     --kernel conv3x3_i8_kernel \
     --include third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime \
     third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/Conv3x3Int8Kernel.cpp \
     --output build/kv260-hls/conv3x3_i8_kernel.hw.xo

   v++ --link \
     --target hw \
     --platform "$KV260_PLATFORM" \
     --clock.defaultFreqHz 100000000 \
     build/kv260-hls/conv1x1_kernel.hw.xo \
     build/kv260-hls/conv3x3_kernel.hw.xo \
     build/kv260-hls/conv1x1_i8_kernel.hw.xo \
     build/kv260-hls/conv3x3_i8_kernel.hw.xo \
     --output build/kv260-hls/conv2d_kernel.hw.xclbin
   ```

4. Locate the linked `system.bit` and matching exported XSA. Stop if the XSA
   and bitstream cannot be tied to the same link output.

5. Convert `system.bit` to `.bit.bin`:

   ```bash
   printf 'all:{system.bit}\n' > bootgen.bif
   bootgen -w -arch zynqmp -process_bitstream bin -image bootgen.bif
   ```

6. Generate the DTBO from the matching XSA. Prefer the PetaLinux 2022.1
   `fpgamanager_dtg` flow from inside a configured KV260 PetaLinux project:

   ```bash
   petalinux-create -t apps --template fpgamanager_dtg \
     -n conv2d-kv260 --enable \
     --srcuri "conv2d-kv260.xsa conv2d-kv260.xclbin shell.json"
   petalinux-build
   ```

   If `fpgamanager_dtg` cannot be used, use XSCT with the 2022.1
   `device-tree-xlnx` branch to generate an overlay-enabled `pl.dtsi`, enable
   ZOCL metadata, review its addresses, clocks, interrupts, and ZOCL nodes,
   then compile it:

   ```bash
   dtc -@ -O dtb -o conv2d-kv260.dtbo pl.dtsi
   ```

   Do not create a generic DTBO or copy one from an unrelated application.

7. Create `shell.json`:

   ```json
   {
     "shell_type": "XRT_FLAT",
     "num_slots": "1"
   }
   ```

8. Copy and rename the matched artifacts into
   `build/kv260-firmware/conv2d-kv260/` using the common basename
   `conv2d-kv260`.

9. Validate them:

   ```bash
   xclbinutil --info \
     --input build/kv260-firmware/conv2d-kv260/conv2d-kv260.xclbin

   dtc -I dtb -O dts \
     build/kv260-firmware/conv2d-kv260/conv2d-kv260.dtbo \
     -o build/kv260-firmware/conv2d-kv260/conv2d-kv260.roundtrip.dts

   sha256sum build/kv260-firmware/conv2d-kv260/conv2d-kv260.bit.bin \
     build/kv260-firmware/conv2d-kv260/conv2d-kv260.dtbo \
     build/kv260-firmware/conv2d-kv260/conv2d-kv260.xclbin \
     build/kv260-firmware/conv2d-kv260/shell.json
   ```

10. Stop after validation. Do not SSH to the board and do not invoke `xmutil`
    from the build machine.

## Deployment From the Mac

After the completed directory is transferred into the Mac checkout, install it
without activating it:

```bash
KV260_PASSWORD='<board-password>' \
./scripts/build_package_deploy_kv260_runtime.sh \
  --firmware-dir build/kv260-firmware/conv2d-kv260 \
  --firmware-app conv2d-kv260
```

This installs the firmware under:

```text
/lib/firmware/xilinx/conv2d-kv260
```

Activation remains explicit:

```bash
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp conv2d-kv260
sudo xmutil listapps
xbutil examine
```

Only after `conv2d-kv260` is active and the XRT device is Ready should the tiny
host probe be attempted.

## Acceptance Criteria

- All four required files exist and are non-empty.
- The three binary files use the exact same `conv2d-kv260` basename.
- `shell.json` declares `XRT_FLAT` and one slot.
- `xclbinutil --info` reports `conv1x1_kernel`, `conv3x3_kernel`,
  `conv1x1_i8_kernel`, and `conv3x3_i8_kernel`, the KV260
  `ispMipiRx_vcu_DP` platform, hardware content, and a 100 MHz kernel clock.
- The DTBO decompiles and contains the expected FPGA overlay and ZOCL nodes.
- No board-side test is run.
- Report every generated path and validation result. If an XSA, platform, BSP,
  or PetaLinux input is missing, stop and identify it instead of fabricating a
  substitute.
