#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

package_dir=${AARCH64_PACKAGE_DIR:-build/kria-yolo-myaccel}
archive=${AARCH64_PACKAGE_ARCHIVE:-build/kria-yolo-myaccel.tar.gz}

driver=build/yolo-myaccel-driver-aarch64
model=build/yolov5n-myaccel-aarch64.so
sample_input=build/bus-input-aarch64.bin

for path in "$driver" "$model"; do
  if [ ! -e "$path" ]; then
    echo "error: missing $path; run make cross-yolo-accelerator-aarch64 first" >&2
    exit 2
  fi
done

mkdir -p "$package_dir/build" "$package_dir/scripts" "$package_dir/samples"
cp "$driver" "$package_dir/build/"
cp "$model" "$package_dir/build/"
if [ -f "$sample_input" ]; then
  cp "$sample_input" "$package_dir/build/"
fi
cp scripts/preprocess_yolo.py "$package_dir/scripts/"
cp scripts/postprocess_yolo.py "$package_dir/scripts/"
if [ -f samples/bus.jpg ]; then
  cp samples/bus.jpg "$package_dir/samples/"
fi

cat > "$package_dir/README-on-board.md" <<'EOF'
# Running YOLOv5n + MyAccel on the Ubuntu 22.04 arm64 board

The driver is linked with `rpath=$ORIGIN`, so keep these two files together:

```sh
build/yolo-myaccel-driver-aarch64
build/yolov5n-myaccel-aarch64.so
```

Run the preprocessed bus sample:

```sh
./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

If you use a private runtime library directory, put the model directory first:

```sh
LD_LIBRARY_PATH="$PWD/build:$HOME/dev/runtime-libs" \
./build/yolo-myaccel-driver-aarch64 \
  build/bus-input-aarch64.bin \
  build/bus-myaccel-aarch64.bin
```

If library loading fails, inspect it with:

```sh
ldd ./build/yolo-myaccel-driver-aarch64
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4 | tail
```
EOF

tar -czf "$archive" -C "$(dirname "$package_dir")" "$(basename "$package_dir")"
echo "packaged: $archive"
