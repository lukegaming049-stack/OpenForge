#!/usr/bin/env bash
# Build a single-file binary for the current platform (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -e . pyinstaller

if [[ "${1:-}" != "--no-gguf" ]]; then
  python3 -m pip install llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cpu || \
    echo "llama-cpp-python unavailable; building without GGUF support."
fi

rm -rf build dist
pyinstaller openforge.spec --noconfirm --clean
./dist/openforge --version
