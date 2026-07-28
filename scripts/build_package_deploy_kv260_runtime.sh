#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
usage: scripts/build_package_deploy_kv260_runtime.sh [options]

Build, package, upload, and install the KV260 runtime update.

Options:
  --remote USER@HOST          SSH target. Default: KV260_REMOTE or ubuntu@kria
  --runtime-dir PATH          Runtime dir on board. Default: KV260_RUNTIME_DIR or /home/ubuntu/dev/runtime
  --tmp-dir PATH              Temporary upload/extract dir on board. Default: KV260_TMP_DIR or /home/ubuntu/dev
  --password PASSWORD         Board SSH password. Prefer KV260_PASSWORD instead of this option.
  --host-src PATH             C++ host-test source to cross-compile.
  --host-name NAME            Executable name installed into runtime dir. Default: host_xrt_conv_test
  --host-out PATH             Local output path for compiled host executable.
  --xclbin PATH               Optional local xclbin to package.
  --firmware-dir PATH         Optional Kria firmware bundle directory.
  --firmware-app NAME         Firmware application name. Defaults to firmware directory basename.
  --firmware-root PATH        Board firmware root. Default: /lib/firmware/xilinx
  --activate-firmware         After installation, unload the active app and load this firmware.
  --package-only              Build/package locally but do not upload to the board.
  --help                      Show this help.

Environment overrides:
  KV260_REMOTE, KV260_RUNTIME_DIR, KV260_TMP_DIR, KV260_PASSWORD
  KV260_FIRMWARE_DIR, KV260_FIRMWARE_APP, KV260_FIRMWARE_ROOT
  AARCH64_SYSROOT, AARCH64_XRT_ROOT, ONNX_MLIR_BIN, AARCH64_TARGET, CLANGXX, LLD
  KV260_BUNDLE_NAME, KV260_BUNDLE_DIR, KV260_ARCHIVE
EOF
}

remote=${KV260_REMOTE:-ubuntu@kria}
remote_runtime_dir=${KV260_RUNTIME_DIR:-/home/ubuntu/dev/runtime}
remote_tmp_dir=${KV260_TMP_DIR:-/home/ubuntu/dev}
kv260_password=${KV260_PASSWORD:-}

sysroot=${AARCH64_SYSROOT:-$PWD/build/aarch64-linux-jammy-sysroot}
xrt_root=${AARCH64_XRT_ROOT:-$PWD/toolchains/kv260-xrt-2.13.0}
xrt_include_dir=$xrt_root/include/xrt
onnx_mlir=${ONNX_MLIR_BIN:-third_party/onnx-mlir/build-host-exact-fixed/Release/bin/onnx-mlir}
target=${AARCH64_TARGET:-aarch64-unknown-linux-gnu}
clangxx=${CLANGXX:-clang++}
lld=${LLD:-}

bundle_name=${KV260_BUNDLE_NAME:-kv260-runtime-update}
bundle_dir=${KV260_BUNDLE_DIR:-build/$bundle_name}
archive=${KV260_ARCHIVE:-build/$bundle_name.tar.gz}
host_name=${KV260_HOST_NAME:-host_xrt_conv_test}
host_test=${HOST_OUT:-}
host_test_src=${KV260_HOST_SRC:-third_party/onnx-mlir/src/Accelerators/MyAccel/Test/HostXrtConvTest.cpp}
local_xclbin=${KV260_XCLBIN:-build/kv260-hls/conv2d_kernel.hw.xclbin}
firmware_dir=${KV260_FIRMWARE_DIR:-}
firmware_app=${KV260_FIRMWARE_APP:-}
remote_firmware_root=${KV260_FIRMWARE_ROOT:-/lib/firmware/xilinx}
activate_firmware=0
deploy=1

need_value() {
  if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
    echo "error: missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --remote)
      need_value "$@"
      remote=$2
      shift 2
      ;;
    --runtime-dir)
      need_value "$@"
      remote_runtime_dir=$2
      shift 2
      ;;
    --tmp-dir)
      need_value "$@"
      remote_tmp_dir=$2
      shift 2
      ;;
    --password)
      need_value "$@"
      kv260_password=$2
      shift 2
      ;;
    --host-src)
      need_value "$@"
      host_test_src=$2
      shift 2
      ;;
    --host-name)
      need_value "$@"
      host_name=$2
      shift 2
      ;;
    --host-out)
      need_value "$@"
      host_test=$2
      shift 2
      ;;
    --xclbin)
      need_value "$@"
      local_xclbin=$2
      shift 2
      ;;
    --firmware-dir)
      need_value "$@"
      firmware_dir=$2
      shift 2
      ;;
    --firmware-app)
      need_value "$@"
      firmware_app=$2
      shift 2
      ;;
    --firmware-root)
      need_value "$@"
      remote_firmware_root=$2
      shift 2
      ;;
    --activate-firmware)
      activate_firmware=1
      shift
      ;;
    --package-only)
      deploy=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$host_test" ]; then
  host_test="$bundle_dir/$host_name"
fi
if [[ "$host_name" == */* ]]; then
  echo "error: --host-name must be a file name, not a path: $host_name" >&2
  exit 2
fi

if [ -n "$firmware_dir" ] && [ -z "$firmware_app" ]; then
  firmware_app=$(basename "$firmware_dir")
fi
if [ -n "$firmware_app" ] && [[ ! "$firmware_app" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: invalid firmware application name: $firmware_app" >&2
  exit 2
fi
if [ "$activate_firmware" = "1" ] && [ -z "$firmware_dir" ]; then
  echo "error: --activate-firmware requires --firmware-dir" >&2
  exit 2
fi
if [ -n "$firmware_dir" ] && [[ "$remote_firmware_root" != /* ]]; then
  echo "error: --firmware-root must be an absolute board path" >&2
  exit 2
fi

require_file() {
  if [ ! -e "$1" ]; then
    echo "error: missing $1" >&2
    exit 2
  fi
}

require_nonempty_file() {
  require_file "$1"
  if [ ! -s "$1" ]; then
    echo "error: file is empty: $1" >&2
    exit 2
  fi
}

write_sha256_manifest() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$@"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@"
  else
    echo "error: shasum or sha256sum is required for firmware packaging" >&2
    return 2
  fi
}

require_exe_or_command() {
  if [ -x "$1" ] || command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  echo "error: cannot execute $1" >&2
  exit 2
}

require_file "$sysroot"
require_file "$xrt_root/lib/libxrt_coreutil.so"
require_file "$host_test_src"
require_exe_or_command "$onnx_mlir"
require_exe_or_command "$clangxx"
if [ -n "$kv260_password" ]; then
  require_exe_or_command sshpass
fi

require_file "$xrt_include_dir/xrt/xrt_bo.h"

if [ -n "$firmware_dir" ]; then
  require_nonempty_file "$firmware_dir/$firmware_app.bit.bin"
  require_nonempty_file "$firmware_dir/$firmware_app.dtbo"
  require_nonempty_file "$firmware_dir/$firmware_app.xclbin"
  require_nonempty_file "$firmware_dir/shell.json"
  if ! grep -Eq '"shell_type"[[:space:]]*:[[:space:]]*"XRT_FLAT"' \
    "$firmware_dir/shell.json"; then
    echo "error: $firmware_dir/shell.json must declare shell_type XRT_FLAT" >&2
    exit 2
  fi
  if ! grep -Eq '"num_slots"[[:space:]]*:[[:space:]]*"?1"?' \
    "$firmware_dir/shell.json"; then
    echo "error: $firmware_dir/shell.json must declare num_slots 1" >&2
    exit 2
  fi
fi

if [ -z "$lld" ]; then
  if command -v ld.lld >/dev/null 2>&1; then
    lld=$(command -v ld.lld)
  elif [ -x /opt/homebrew/bin/ld.lld ]; then
    lld=/opt/homebrew/bin/ld.lld
  elif [ -x /usr/local/bin/ld.lld ]; then
    lld=/usr/local/bin/ld.lld
  else
    lld=lld
  fi
fi
linker_flag="-fuse-ld=$lld"

gcc_lib_dir=${AARCH64_GCC_LIB_DIR:-}
if [ -z "$gcc_lib_dir" ]; then
  gcc_lib_dir=$(find "$sysroot/usr/lib/gcc/aarch64-linux-gnu" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "$gcc_lib_dir" ]; then
  echo "error: could not find GCC runtime directory under $sysroot/usr/lib/gcc/aarch64-linux-gnu" >&2
  exit 2
fi

echo "==> Building XRT-enabled AArch64 YOLO runtime"
AARCH64_SYSROOT="$sysroot" \
AARCH64_XRT_ROOT="$xrt_root" \
AARCH64_XRT_INCLUDE_DIR="$xrt_include_dir" \
ONNX_MLIR_BIN="$onnx_mlir" \
MYACCEL_USE_XRT=1 \
make cross-yolo-accelerator-aarch64

echo "==> Building AArch64 standalone XRT Conv host test"
mkdir -p "$bundle_dir/runtime-libs"
mkdir -p "$(dirname "$host_test")"
"$clangxx" \
  --target="$target" \
  --sysroot="$sysroot" \
  --gcc-toolchain="$sysroot/usr" \
  -B"$gcc_lib_dir" \
  -B"$sysroot/usr/lib/aarch64-linux-gnu" \
  -std=c++17 -O2 "$linker_flag" \
  -I"$xrt_include_dir" \
  "$host_test_src" \
  -L"$xrt_root/lib" \
  -Wl,-rpath,'$ORIGIN/runtime-libs' \
  -lxrt_coreutil -pthread \
  -o "$host_test"

if ! file "$host_test" | grep -q 'ARM aarch64'; then
  file "$host_test" >&2
  echo "error: host test is not an AArch64 binary" >&2
  exit 2
fi
if [ "$host_test" != "$bundle_dir/$host_name" ]; then
  install -m 0755 "$host_test" "$bundle_dir/$host_name"
fi

echo "==> Packaging runtime update"
cp build/yolo-myaccel-driver-aarch64 "$bundle_dir/"
cp build/yolov5n-myaccel-aarch64.so "$bundle_dir/"
if [ -f build/bus-input-aarch64.bin ]; then
  cp build/bus-input-aarch64.bin "$bundle_dir/"
fi
if [ -f "$local_xclbin" ]; then
  cp "$local_xclbin" "$bundle_dir/conv2d_kernel.hw.xclbin"
elif [ -n "$firmware_dir" ]; then
  cp "$firmware_dir/$firmware_app.xclbin" "$bundle_dir/conv2d_kernel.hw.xclbin"
else
  echo "note: no local xclbin at $local_xclbin; board's existing runtime xclbin will be kept"
fi
cp "$xrt_root"/lib/libxrt_coreutil.so* "$bundle_dir/runtime-libs/"

if [ -n "$firmware_dir" ]; then
  packaged_firmware_dir="$bundle_dir/firmware/$firmware_app"
  mkdir -p "$packaged_firmware_dir"
  install -m 0644 "$firmware_dir/$firmware_app.bit.bin" "$packaged_firmware_dir/"
  install -m 0644 "$firmware_dir/$firmware_app.dtbo" "$packaged_firmware_dir/"
  install -m 0644 "$firmware_dir/$firmware_app.xclbin" "$packaged_firmware_dir/"
  install -m 0644 "$firmware_dir/shell.json" "$packaged_firmware_dir/"
  (
    cd "$packaged_firmware_dir"
    write_sha256_manifest \
      "$firmware_app.bit.bin" \
      "$firmware_app.dtbo" \
      "$firmware_app.xclbin" \
      shell.json > SHA256SUMS
  )
fi

COPYFILE_DISABLE=1 tar -czf "$archive" -C "$(dirname "$bundle_dir")" "$(basename "$bundle_dir")"
echo "packaged: $archive"

if [ "$deploy" = "0" ]; then
  echo "package-only requested; not uploading to board"
  exit 0
fi

echo "==> Uploading to $remote"
run_scp() {
  if [ -n "$kv260_password" ]; then
    SSHPASS="$kv260_password" sshpass -e scp "$@"
  else
    scp "$@"
  fi
}

run_ssh() {
  if [ -n "$kv260_password" ]; then
    SSHPASS="$kv260_password" sshpass -e ssh "$remote" "$@"
  else
    ssh "$remote" "$@"
  fi
}

run_root_ssh() {
  if run_ssh sudo -n true >/dev/null 2>&1; then
    run_ssh sudo bash -s -- "$@"
  elif [ -n "$kv260_password" ]; then
    {
      printf '%s\n' "$kv260_password"
      cat
    } | run_ssh sudo -S bash -s -- "$@"
  else
    echo "error: firmware installation needs passwordless sudo or KV260_PASSWORD" >&2
    return 1
  fi
}

run_scp "$archive" "$remote:$remote_tmp_dir/"

remote_archive="$remote_tmp_dir/$(basename "$archive")"
remote_bundle="$remote_tmp_dir/$bundle_name"

echo "==> Replacing files in $remote:$remote_runtime_dir"
run_ssh bash -s -- "$remote_archive" "$remote_bundle" "$remote_runtime_dir" "$host_name" <<'EOF'
set -euo pipefail

archive=$1
bundle=$2
runtime_dir=$3
host_name=$4

expand_path() {
  case "$1" in
    "~")
      printf '%s\n' "$HOME"
      ;;
    "~/"*)
      printf '%s/%s\n' "$HOME" "${1#"~/"}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

archive=$(expand_path "$archive")
bundle=$(expand_path "$bundle")
runtime_dir=$(expand_path "$runtime_dir")

mkdir -p "$runtime_dir"
rm -rf "$bundle"
tar -xzf "$archive" -C "$(dirname "$bundle")"

install -m 0755 "$bundle/$host_name" "$runtime_dir/$host_name"
install -m 0755 "$bundle/yolo-myaccel-driver-aarch64" "$runtime_dir/yolo-myaccel-driver-aarch64"
install -m 0644 "$bundle/yolov5n-myaccel-aarch64.so" "$runtime_dir/yolov5n-myaccel-aarch64.so"

if [ -f "$bundle/bus-input-aarch64.bin" ]; then
  install -m 0644 "$bundle/bus-input-aarch64.bin" "$runtime_dir/bus-input-aarch64.bin"
fi
if [ -f "$bundle/conv2d_kernel.hw.xclbin" ]; then
  install -m 0644 "$bundle/conv2d_kernel.hw.xclbin" "$runtime_dir/conv2d_kernel.hw.xclbin"
fi

mkdir -p "$runtime_dir/runtime-libs"
cp -f "$bundle"/runtime-libs/libxrt_coreutil.so* "$runtime_dir/runtime-libs/"

echo "updated runtime dir: $runtime_dir"
ls -lh "$runtime_dir/$host_name" \
  "$runtime_dir/yolo-myaccel-driver-aarch64" \
  "$runtime_dir/yolov5n-myaccel-aarch64.so"
EOF

if [ -n "$firmware_dir" ]; then
  remote_firmware_bundle="$remote_bundle/firmware/$firmware_app"
  echo "==> Installing Kria firmware $firmware_app under $remote_firmware_root"
  run_root_ssh \
    "$remote_firmware_bundle" \
    "$remote_firmware_root" \
    "$firmware_app" \
    "$activate_firmware" <<'EOF'
set -euo pipefail

source_dir=$1
firmware_root=$2
firmware_app=$3
activate_firmware=$4

case "$firmware_app" in
  ''|*[!A-Za-z0-9._-]*)
    echo "error: invalid firmware application name: $firmware_app" >&2
    exit 2
    ;;
esac
case "$firmware_root" in
  /*) ;;
  *)
    echo "error: firmware root must be absolute: $firmware_root" >&2
    exit 2
    ;;
esac
if [ "$firmware_root" = "/" ]; then
  echo "error: refusing to use / as firmware root" >&2
  exit 2
fi

for file in \
  "$firmware_app.bit.bin" \
  "$firmware_app.dtbo" \
  "$firmware_app.xclbin" \
  shell.json \
  SHA256SUMS; do
  if [ ! -f "$source_dir/$file" ]; then
    echo "error: missing packaged firmware file: $source_dir/$file" >&2
    exit 2
  fi
done

(
  cd "$source_dir"
  sha256sum -c SHA256SUMS
)

mkdir -p "$firmware_root"
stage_dir=$(mktemp -d "$firmware_root/.${firmware_app}.stage.XXXXXX")
trap 'rm -rf "$stage_dir"' EXIT

install -m 0644 "$source_dir/$firmware_app.bit.bin" "$stage_dir/"
install -m 0644 "$source_dir/$firmware_app.dtbo" "$stage_dir/"
install -m 0644 "$source_dir/$firmware_app.xclbin" "$stage_dir/"
install -m 0644 "$source_dir/shell.json" "$stage_dir/"
install -m 0644 "$source_dir/SHA256SUMS" "$stage_dir/"

target_dir="$firmware_root/$firmware_app"
previous_dir="$firmware_root/.${firmware_app}.previous"
if [ -e "$target_dir" ]; then
  rm -rf "$previous_dir"
  mv "$target_dir" "$previous_dir"
fi
mv "$stage_dir" "$target_dir"
trap - EXIT

echo "installed firmware application: $target_dir"
xmutil listapps
if ! xmutil listapps | grep -Fq "$firmware_app"; then
  echo "error: xmutil did not register firmware application $firmware_app" >&2
  exit 1
fi

if [ "$activate_firmware" = "1" ]; then
  echo "activating firmware application: $firmware_app"
  xmutil unloadapp
  xmutil loadapp "$firmware_app"
  xmutil listapps
fi
EOF
fi

echo
echo "Next on the board:"
if [ -n "$firmware_dir" ] && [ "$activate_firmware" = "0" ]; then
  echo "  sudo xmutil listapps"
  echo "  sudo xmutil unloadapp"
  echo "  sudo xmutil loadapp $firmware_app"
  echo "  sudo xmutil listapps"
fi
echo "  cd $remote_runtime_dir"
echo "  xbutil examine"
echo "  ./$host_name ./conv2d_kernel.hw.xclbin --probe-only"
