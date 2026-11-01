# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: one self-contained executable per platform.

    pyinstaller openforge.spec --noconfirm --clean   ->  dist/openforge[.exe]

Textual ships CSS/TCSS assets and Rich ships themes, so both are collected in
full.  llama-cpp-python is bundled when it happens to be installed (that is what
gives the frozen binary local GGUF support); when it is absent the build still
succeeds and the binary simply offers the API providers.
"""

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve()

datas, binaries, hiddenimports = [], [], []

# Packages whose data files matter at runtime.
for pkg in ("textual", "rich", "httpx", "httpcore", "h11", "anyio", "sniffio", "certifi", "idna"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Optional: local GGUF inference. collect_all() returns empty lists rather than
# raising when a package is missing, so check for it explicitly.
if importlib.util.find_spec("llama_cpp") is not None:
    try:
        gguf_datas, gguf_binaries, gguf_hidden = collect_all("llama_cpp")
        datas += gguf_datas
        binaries += gguf_binaries
        hiddenimports += gguf_hidden
        print("[openforge.spec] bundling llama-cpp-python (GGUF support enabled)")
    except Exception as exc:
        print(f"[openforge.spec] llama-cpp-python present but not collectable: {exc}")
else:
    print("[openforge.spec] llama-cpp-python not installed - building without GGUF support")

hiddenimports += collect_submodules("openforge")

a = Analysis(
    [str(ROOT / "packaging" / "openforge_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="openforge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
