#!/usr/bin/env bash
set -euo pipefail

URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
OUT="tiny-imagenet-200.zip"

wget -c --show-progress -O "$OUT" "$URL"

unzip -o "$OUT"

test -d tiny-imagenet-200/train && echo "✓ 解压完成：tiny-imagenet-200/" || echo "× 解压目录不存在"
