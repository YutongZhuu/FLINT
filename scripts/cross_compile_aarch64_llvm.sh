#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

target=${AARCH64_TARGET:-aarch64-unknown-linux-gnu}
sysroot=${AARCH64_SYSROOT:-}
onnx_mlir=${ONNX_MLIR_BIN:-onnx-mlir}
clang=${CLANG:-clang}
clangxx=${CLANGXX:-clang++}
out_base=${AARCH64_OUT_BASE:-build/yolov5n-myaccel-aarch64}
driver_out=${AARCH64_DRIVER_OUT:-build/yolo-myaccel-driver-aarch64}
gcc_toolchain=${AARCH64_GCC_TOOLCHAIN:-$sysroot/usr}
target_lib_dir=${AARCH64_TARGET_LIB_DIR:-$sysroot/usr/lib/aarch64-linux-gnu}

if [ -z "$sysroot" ]; then
  cat >&2 <<'EOF'
error: AARCH64_SYSROOT is not set.

Set it to a Linux/aarch64 sysroot, for example:

  export AARCH64_SYSROOT=/path/to/aarch64-linux-sysroot

The sysroot must contain target headers, libc, libm, libstdc++, startup files,
and the Linux dynamic linker for aarch64.
EOF
  exit 2
fi

if [ ! -d "$sysroot" ]; then
  echo "error: AARCH64_SYSROOT does not exist: $sysroot" >&2
  exit 2
fi

if [ -n "${AARCH64_GCC_LIB_DIR:-}" ]; then
  gcc_lib_dir=$AARCH64_GCC_LIB_DIR
else
  gcc_lib_dir=$(find "$sysroot/usr/lib/gcc/aarch64-linux-gnu" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -z "$gcc_lib_dir" ]; then
    echo "error: could not find GCC runtime directory under $sysroot/usr/lib/gcc/aarch64-linux-gnu" >&2
    exit 2
  fi
fi

if ! command -v "$onnx_mlir" >/dev/null 2>&1; then
  cat >&2 <<EOF
error: cannot find ONNX_MLIR_BIN '$onnx_mlir'.

Build or install a macOS-native onnx-mlir with MyAccel enabled, then either put
it on PATH or set:

  export ONNX_MLIR_BIN=/path/to/onnx-mlir
EOF
  exit 2
fi

if ! printf 'int main(void) { return 0; }\n' | \
  "$clang" --target="$target" --sysroot="$sysroot" --gcc-toolchain="$gcc_toolchain" \
    -B"$gcc_lib_dir" -B"$target_lib_dir" -fuse-ld=lld \
    -x c -Wl,--no-undefined -o /tmp/onnx-mlir-aarch64-probe - >/dev/null 2>&1; then
  cat >&2 <<EOF
error: clang cannot link a trivial $target program with this sysroot.

Check that AARCH64_SYSROOT points at a complete Linux/aarch64 sysroot and that
lld is available. Current values:

  CLANG=$clang
  AARCH64_TARGET=$target
  AARCH64_SYSROOT=$sysroot
  AARCH64_GCC_TOOLCHAIN=$gcc_toolchain
  AARCH64_GCC_LIB_DIR=$gcc_lib_dir
EOF
  exit 2
fi
rm -f /tmp/onnx-mlir-aarch64-probe

mkdir -p build/aarch64

"$onnx_mlir" \
  --maccel=MyAccel \
  --mtriple="$target" \
  --march=aarch64 \
  --EmitObj \
  -O0 \
  -o "$out_base" \
  build/yolov5n-fp32.onnx

runtime_sources=(
  third_party/onnx-mlir/src/Runtime/OMTensor.c
  third_party/onnx-mlir/src/Runtime/OMTensorList.c
  third_party/onnx-mlir/src/Runtime/OnnxDataType.c
  third_party/onnx-mlir/src/Runtime/OMInstrument.c
  third_party/onnx-mlir/src/Runtime/OMExternalConstant.c
  third_party/onnx-mlir/src/Runtime/OMIndexLookup.c
  third_party/onnx-mlir/src/Runtime/OMRandomNormal.c
  third_party/onnx-mlir/src/Runtime/OMRandomUniform.c
  third_party/onnx-mlir/src/Runtime/OMResize.c
  third_party/onnx-mlir/src/Runtime/OMSort.c
  third_party/onnx-mlir/src/Runtime/OMTopK.c
  third_party/onnx-mlir/src/Runtime/OMUnique.c
  third_party/onnx-mlir/src/Support/SmallFPConversion.c
  third_party/onnx-mlir/src/Accelerators/MyAccel/Runtime/MyConv.c
)

runtime_objects=()
for src in "${runtime_sources[@]}"; do
  obj="build/aarch64/$(basename "${src%.*}").o"
  "$clang" --target="$target" --sysroot="$sysroot" \
    --gcc-toolchain="$gcc_toolchain" -B"$gcc_lib_dir" -B"$target_lib_dir" \
    -std=c11 -O3 -fPIC -D_GNU_SOURCE \
    -Ithird_party/onnx-mlir/include \
    -Ithird_party/onnx-mlir \
    -Ithird_party/onnx-mlir/src/Runtime \
    -c "$src" -o "$obj"
  runtime_objects+=("$obj")
done

"$clang" --target="$target" --sysroot="$sysroot" \
  --gcc-toolchain="$gcc_toolchain" -B"$gcc_lib_dir" -B"$target_lib_dir" \
  -shared -fPIC -fuse-ld=lld \
  "$out_base.o" "${runtime_objects[@]}" \
  -lm \
  -o "$out_base.so"

"$clangxx" --target="$target" --sysroot="$sysroot" \
  --gcc-toolchain="$gcc_toolchain" -B"$gcc_lib_dir" -B"$target_lib_dir" \
  -std=c++17 -O2 -fuse-ld=lld \
  -Ithird_party/onnx-mlir/include \
  -Ithird_party/onnx-mlir \
  driver/yolo_driver.cpp "$out_base.so" \
  -Wl,-rpath,'$ORIGIN' \
  -o "$driver_out"

file "$out_base.o" "$out_base.so" "$driver_out"
