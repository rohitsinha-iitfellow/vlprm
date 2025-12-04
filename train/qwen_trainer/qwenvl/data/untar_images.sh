#!/usr/bin/env bash

# Extract all .tar archives in a given folder.
# Accepts a directory or any file path inside that directory.

set -o nounset
IFS=$'\n\t'

usage() {
  echo "Usage: $(basename "$0") <directory-or-file-path>" >&2
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

INPUT_PATH=$1

if [[ -d "$INPUT_PATH" ]]; then
  TARGET_DIR="$INPUT_PATH"
elif [[ -f "$INPUT_PATH" ]]; then
  TARGET_DIR="$(dirname "$INPUT_PATH")"
else
  echo "Error: '$INPUT_PATH' is not a valid file or directory." >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "Error: 'tar' is not available in PATH." >&2
  exit 1
fi

# Enable case-insensitive and null globs
shopt -s nullglob nocaseglob

found_any=false
failed_any=false

for tar_file in "$TARGET_DIR"/*.tar; do
  found_any=true
  echo "Extracting: $tar_file"
  if ! tar -xf "$tar_file" -C "$TARGET_DIR"; then
    echo "Failed to extract: $tar_file" >&2
    failed_any=true
  fi
done

# Restore default globbing behavior
shopt -u nocaseglob nullglob

if [[ "$found_any" == false ]]; then
  echo "No .tar files found in: $TARGET_DIR"
  exit 0
fi

if [[ "$failed_any" == true ]]; then
  echo "One or more archives failed to extract." >&2
  exit 1
fi

echo "All .tar archives extracted into: $TARGET_DIR"

