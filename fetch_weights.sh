#!/usr/bin/env bash
# Download and unpack the trained model weights into ./models/.
#
# The archive (≈900 MB, pengwin-task2-v75-weights.tar.gz) is not in git.
# Download link and its SHA256 are in README.md -> "Model weights".
set -euo pipefail
cd "$(dirname "$0")"

URL="${WEIGHTS_URL:-}"
ARCHIVE="pengwin-task2-v75-weights.tar.gz"

if [ -f "$ARCHIVE" ]; then
    echo "[fetch] using local $ARCHIVE"
elif [ -n "$URL" ]; then
    echo "[fetch] $URL"
    curl -L --fail -o "$ARCHIVE" "$URL"
else
    echo "No $ARCHIVE and no WEIGHTS_URL set." >&2
    echo "Download the archive from the link in README.md into this directory, then re-run." >&2
    echo "Or:  WEIGHTS_URL=<direct-link> ./fetch_weights.sh" >&2
    exit 1
fi

echo "[fetch] sha256: $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
echo "        (must match the value in README.md)"

mkdir -p models
tar -xzf "$ARCHIVE" -C models --strip-components=1 2>/dev/null || tar -xzf "$ARCHIVE" -C .

echo "[fetch] verifying against models.md5 ..."
md5sum -c models.md5
echo "[done] weights in ./models/"
