# Build the two-kernel INT8 XCLBIN

This source package contains exactly two HLS kernel implementations:

- `conv1x1_i8_kernel`
- `conv3x3_i8_kernel`

It contains no FP32 convolution kernel and no general-purpose INT8
convolution kernel. The link step consumes only the two INT8 `.xo` files.

## Build on the remote Vitis machine

Extract the archive into a new directory:

```bash
tar -xzf kv260-int8-only-xclbin-src.tar.gz
cd kv260-int8-only-xclbin
```

Source the Vitis environment and select the KV260 platform:

```bash
source /tools/Xilinx/Vitis/2022.1/settings64.sh
platforminfo -l | grep -i kv260
export PLATFORM=/absolute/path/to/the/kv260/platform.xpfm
```

Build a hardware XCLBIN:

```bash
./scripts/build_kv260_int8_only_xclbin.sh
```

For the `hw` target, the results are:

```text
build/kv260-int8-only/conv_int8_only.hw.xclbin
build/kv260-int8-only/conv_int8_only.hw.bit.bin
```

The script keeps Vitis's implementation output under
`build/kv260-int8-only/vitis-link` and uses `bootgen` to convert its
`system.bit` into the `.bit.bin` required by `xmutil`.

The default target is `hw` and the default kernel clock is 100 MHz. An
emulation build produces an XCLBIN but no `.bit.bin`:

```bash
TARGET=hw_emu KV260_KERNEL_CLOCK_HZ=150000000 \
  ./scripts/build_kv260_int8_only_xclbin.sh
```

The script accepts either `PLATFORM` or `KV260_PLATFORM`, and either may be a
platform name recognized by Vitis or the absolute path to its `.xpfm`.

## Confirm the linked kernels

If `xclbinutil` is installed, inspect the result with:

```bash
xclbinutil --info \
  --input build/kv260-int8-only/conv_int8_only.hw.xclbin
```

The kernel list should contain only `conv1x1_i8_kernel` and
`conv3x3_i8_kernel`.
