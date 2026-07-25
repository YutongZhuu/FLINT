#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

suite=${UBUNTU_SUITE:-jammy}
mirror=${UBUNTU_PORTS_MIRROR:-https://ports.ubuntu.com/ubuntu-ports}
deb_dir=${AARCH64_SYSROOT_DEB_DIR:-build/aarch64-sysroot-${suite}-debs}
sysroot=${AARCH64_SYSROOT:-build/aarch64-linux-${suite}-sysroot}

packages=(
  gcc-12-base
  libc6
  libc6-dev
  libgcc-s1
  libgcc-12-dev
  libstdc++6
  libstdc++-12-dev
  linux-libc-dev
  libuuid1
  uuid-dev
)

mkdir -p "$deb_dir" "$sysroot"

python3 - "$suite" "$mirror" "$deb_dir" "${packages[@]}" <<'PY'
import gzip
import os
import sys
import urllib.request

suite, mirror, deb_dir, *wanted = sys.argv[1:]
pockets = [f"{suite}-updates", f"{suite}-security", suite]
components = ["main", "universe"]

def parse_packages(data):
    current = {}
    for line in data.splitlines():
        if not line:
            if current:
                yield current
                current = {}
            continue
        if line.startswith(" ") and current:
            key = next(reversed(current))
            current[key] += "\n" + line
            continue
        key, value = line.split(":", 1)
        current[key] = value.strip()
    if current:
        yield current

found = {}
for pocket in pockets:
    for component in components:
        url = f"{mirror}/dists/{pocket}/{component}/binary-arm64/Packages.gz"
        print(f"[index] {url}", file=sys.stderr)
        try:
            with urllib.request.urlopen(url) as r:
                data = gzip.decompress(r.read()).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"warning: could not read {url}: {e}", file=sys.stderr)
            continue
        for stanza in parse_packages(data):
            name = stanza.get("Package")
            if name in wanted and name not in found:
                found[name] = (pocket, stanza)

missing = [p for p in wanted if p not in found]
if missing:
    raise SystemExit(f"missing packages in Ubuntu ports index: {', '.join(missing)}")

for name in wanted:
    pocket, stanza = found[name]
    filename = stanza["Filename"]
    version = stanza.get("Version", "?")
    url = f"{mirror}/{filename}"
    out = os.path.join(deb_dir, os.path.basename(filename))
    print(f"[download] {name} {version} from {pocket}")
    if os.path.exists(out) and os.path.getsize(out) == int(stanza.get("Size", "0")):
        print(f"  cached {out}")
        continue
    urllib.request.urlretrieve(url, out)
PY

for deb in "$deb_dir"/*.deb; do
  echo "[unpack] $(basename "$deb")"
  tmp=$(mktemp -d)
  (
    cd "$tmp"
    ar -x "$OLDPWD/$deb"
    data=$(ls data.tar.* | head -n 1)
    tar -xf "$data" -C "$OLDPWD/$sysroot"
  )
  rm -rf "$tmp"
done

echo
echo "sysroot ready: $sysroot"
echo
echo "Check libstdc++ symbol ceiling:"
echo "  strings \"$sysroot/usr/lib/aarch64-linux-gnu/libstdc++.so.6\" | grep GLIBCXX_3.4 | tail"
echo
echo "Build with:"
echo "  AARCH64_SYSROOT=\"$sysroot\" \\"
echo "  ONNX_MLIR_BIN=\"third_party/onnx-mlir/build-host-exact/Release/bin/onnx-mlir\" \\"
echo "  ./scripts/cross_compile_aarch64_llvm.sh"
