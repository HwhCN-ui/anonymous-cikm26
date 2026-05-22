set -euo pipefail


OUT_DIR="${1:-./data/cifar-100}"
URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
FILE_NAME="cifar-100-python.tar.gz"
FILE_PATH="${OUT_DIR}/${FILE_NAME}"

mkdir -p "${OUT_DIR}"

echo "==> Downloading CIFAR-100 to: ${FILE_PATH}"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 5 --retry-delay 2 -o "${FILE_PATH}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${FILE_PATH}" "${URL}"
else
  echo "Error: need curl or wget to download." >&2
  exit 1
fi

echo "==> Download complete."

if command -v sha256sum >/dev/null 2>&1; then
  echo "==> SHA256:"
  sha256sum "${FILE_PATH}"
fi

echo "==> Extracting..."
tar -xzf "${FILE_PATH}" -C "${OUT_DIR}"

echo "==> Done."
echo "==> Extracted folder: ${OUT_DIR}/cifar-100-python"
