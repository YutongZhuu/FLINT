#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

target=${AARCH64_TARGET:-aarch64-unknown-linux-gnu}
sysroot=${AARCH64_SYSROOT:-$PWD/build/aarch64-linux-jammy-sysroot}
xrt_root=${AARCH64_XRT_ROOT:-$PWD/kv260-xrt-2.13.0}
clangxx=${CLANGXX:-clang++}
lld=${LLD:-lld}
source_file=${HOST_XRT_TEST_SOURCE:-third_party/onnx-mlir/src/Accelerators/MyAccel/Test/HostXrtConvTest.cpp}
output=${HOST_XRT_TEST_OUTPUT:-build/kv260-host/host_xrt_int8_test}
output_dir=$(dirname "$output")
object=$output_dir/HostXrtConvTest.o
archive=${HOST_XRT_TEST_ARCHIVE:-$output_dir/kv260-int8-xrt-host-aarch64.tar.gz}

if [ ! -d "$sysroot" ]; then
  echo "error: AArch64 sysroot not found: $sysroot" >&2
  exit 2
fi
if [ ! -d "$xrt_root" ]; then
  echo "error: AArch64 XRT root not found: $xrt_root" >&2
  exit 2
fi
if [ ! -f "$source_file" ]; then
  echo "error: host XRT test source not found: $source_file" >&2
  exit 2
fi
if ! command -v "$clangxx" >/dev/null 2>&1; then
  echo "error: C++ compiler not found: $clangxx" >&2
  exit 2
fi
lld_command=$lld
if [ "$lld" = "lld" ]; then
  lld_command=ld.lld
fi
if ! command -v "$lld_command" >/dev/null 2>&1 && [ ! -x "$lld_command" ]; then
  echo "error: LLVM ELF linker not found: $lld_command" >&2
  exit 2
fi

if [ -f "$xrt_root/include/xrt/xrt/xrt_bo.h" ]; then
  xrt_include_dir=$xrt_root/include/xrt
elif [ -f "$xrt_root/include/xrt/xrt_bo.h" ]; then
  xrt_include_dir=$xrt_root/include
else
  echo "error: XRT C++ headers were not found under $xrt_root/include" >&2
  exit 2
fi

if [ ! -f "$xrt_root/lib/libxrt_coreutil.so" ]; then
  echo "error: AArch64 libxrt_coreutil.so not found under $xrt_root/lib" >&2
  exit 2
fi

gcc_lib_dir=$(
  find "$sysroot/usr/lib/gcc/aarch64-linux-gnu" \
    -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null |
    sort -V |
    tail -1
)
if [ -z "$gcc_lib_dir" ]; then
  echo "error: AArch64 GCC runtime was not found in the sysroot" >&2
  exit 2
fi

target_lib_dir=$sysroot/usr/lib/aarch64-linux-gnu
shared_libm=$sysroot/lib/aarch64-linux-gnu/libm.so.6
if [ ! -f "$shared_libm" ]; then
  echo "error: target shared libm was not found: $shared_libm" >&2
  exit 2
fi

mkdir -p "$output_dir"

"$clangxx" \
  --target="$target" \
  --sysroot="$sysroot" \
  --gcc-toolchain="$sysroot/usr" \
  -B"$gcc_lib_dir" \
  -B"$target_lib_dir" \
  -std=c++17 \
  -O2 \
  -I"$xrt_include_dir" \
  -c "$source_file" \
  -o "$object"

"$clangxx" \
  --target="$target" \
  --sysroot="$sysroot" \
  --gcc-toolchain="$sysroot/usr" \
  -B"$gcc_lib_dir" \
  -B"$target_lib_dir" \
  "-fuse-ld=$lld" \
  "$object" \
  -L"$xrt_root/lib" \
  -Wl,-rpath,'$ORIGIN/runtime-libs' \
  -Wl,-rpath-link,"$target_lib_dir" \
  -lxrt_coreutil \
  -pthread \
  -Wl,--no-as-needed \
  "$shared_libm" \
  -Wl,--as-needed \
  -o "$output"

if ! file "$output" | grep -q 'ARM aarch64'; then
  file "$output" >&2
  echo "error: output is not an AArch64 executable" >&2
  exit 1
fi

runtime_lib_dir=$output_dir/runtime-libs
mkdir -p "$runtime_lib_dir"
cp "$xrt_root"/lib/libxrt_coreutil.so* "$runtime_lib_dir/"

# libxrt_coreutil depends on Boost.Filesystem on XRT 2.13. Some minimal
# cross-sysroots omit it even though the KV260 image provides it. Package a
# sysroot/toolchain copy when available, but do not discard an otherwise valid
# AArch64 host-test binary solely because this optional bundle dependency is
# absent.
boost_filesystem=${AARCH64_BOOST_FILESYSTEM_LIB:-}
if [ -z "$boost_filesystem" ]; then
  boost_filesystem=$(find "$target_lib_dir" "$xrt_root/lib" \
    -maxdepth 1 \( -type f -o -type l \) \
    -name 'libboost_filesystem.so*' -print -quit 2>/dev/null || true)
fi
if [ -n "$boost_filesystem" ]; then
  cp -L "$boost_filesystem" "$runtime_lib_dir/"
else
  echo "warning: Boost.Filesystem is not in the cross-sysroot; " \
    "the target image must provide libboost_filesystem.so" >&2
fi
cp -L "$target_lib_dir/libuuid.so.1" "$runtime_lib_dir/"
tar -czf "$archive" \
  -C "$output_dir" \
  "$(basename "$output")" \
  runtime-libs

file "$output"
printf 'built %s\n' "$output"
printf 'packaged %s\n' "$archive"
printf 'run on Kria with:\n'
printf '  LD_LIBRARY_PATH="$PWD/runtime-libs" \\\n'
printf '    ./%s ./conv_int8_only.hw.xclbin --int8-only\n' \
  "$(basename "$output")"
