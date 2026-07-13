#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

onnx_mlir_dir=${ONNX_MLIR_DIR:-third_party/onnx-mlir}
patch_file=${MYACCEL_PATCH:-patches/onnx-mlir-myaccel.patch}

if [ ! -d "$onnx_mlir_dir/.git" ]; then
  echo "error: $onnx_mlir_dir is not a git checkout" >&2
  echo "hint: run git submodule update --init --recursive" >&2
  exit 2
fi

if [ ! -f "$patch_file" ]; then
  echo "error: missing patch file: $patch_file" >&2
  exit 2
fi

if ! git -C "$onnx_mlir_dir" diff --quiet || \
   ! git -C "$onnx_mlir_dir" diff --cached --quiet || \
   [ -n "$(git -C "$onnx_mlir_dir" ls-files --others --exclude-standard)" ]; then
  echo "error: $onnx_mlir_dir has local changes; refusing to apply patch" >&2
  git -C "$onnx_mlir_dir" status --short
  exit 2
fi

git -C "$onnx_mlir_dir" apply --check "../../$patch_file"
git -C "$onnx_mlir_dir" apply "../../$patch_file"
echo "applied $patch_file to $onnx_mlir_dir"
