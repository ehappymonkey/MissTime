#!/usr/bin/env bash
set -euo pipefail

RELEASE_URL="https://github.com/ehappymonkey/MissTime/releases/download/datasets-v1/misstime-datasets-v1.tar.gz"
ARCHIVE="${TMPDIR:-/tmp}/misstime-datasets-v1.tar.gz"

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "$RELEASE_URL" -o "$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$ARCHIVE" "$RELEASE_URL"
else
    echo "Error: curl or wget is required." >&2
    exit 1
fi

tar -xzf "$ARCHIVE" -C .
echo "Datasets extracted to ./dataset"
