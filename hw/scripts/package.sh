#!/usr/bin/env bash
set -euo pipefail

hw_dir=$(cd "$(dirname "$0")/.." && pwd)
project_root=$(dirname "$hw_dir")
archive=${1:-$hw_dir/build/kv260-int8-hw-src.tar.gz}

case "$archive" in
/*) ;;
*) archive=$PWD/$archive ;;
esac

mkdir -p "$(dirname "$archive")"

(
  cd "$project_root"
  tar \
    --exclude='hw/build' \
    --exclude='hw/report' \
    -czf "$archive" \
    hw
)

archive_name=$(basename "$archive")
printf 'Created %s\n' "$archive"
printf '\nCopy the archive to the remote machine, then unpack it with:\n\n'
printf '  tar -xzf %q && cd hw\n' "$archive_name"
