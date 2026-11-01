# Build dist\openforge.exe on Windows.  Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\build-exe.ps1
param([switch]$NoGguf)

$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
pip install -e . pyinstaller

if (-not $NoGguf) {
    # Prebuilt CPU wheels; skip with -NoGguf if you do not want GGUF support.
    try {
        pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    } catch {
        Write-Warning "llama-cpp-python unavailable; building without GGUF support."
    }
}

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
pyinstaller openforge.spec --noconfirm --clean

Write-Host ""
Write-Host "Built: dist\openforge.exe" -ForegroundColor Green
& .\dist\openforge.exe --version
