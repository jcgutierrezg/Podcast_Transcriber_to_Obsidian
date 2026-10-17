"""Windows CUDA DLL discovery.

The `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels install their DLLs into
site-packages\\nvidia\\<component>\\bin, which is on nobody's search path.
PyTorch registers these directories itself on import; CTranslate2 (underneath
faster-whisper) does not, so it fails with:

    Library cublas64_12.dll is not found or cannot be loaded

Since Python 3.8, Windows ignores PATH when resolving dependencies of extension
modules - you must call os.add_dll_directory(). We do both, since PATH still
matters for DLLs loaded indirectly by an already-loaded DLL.

This must run BEFORE faster_whisper / ctranslate2 is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_prepared = False


def nvidia_dll_dirs() -> list[Path]:
    """Locate the bin/lib folders shipped inside the nvidia-* wheels."""
    try:
        import nvidia  # namespace package created by the nvidia-*-cu12 wheels
        roots = [Path(p) for p in nvidia.__path__]
    except Exception:
        return []

    found: list[Path] = []
    for root in roots:
        for component in sorted(root.iterdir()) if root.is_dir() else []:
            for sub in ("bin", "lib"):  # bin on Windows, lib on Linux
                d = component / sub
                if d.is_dir():
                    found.append(d)
    return found


def prepare() -> list[Path]:
    """Register the NVIDIA DLL directories. Idempotent; safe to call anywhere."""
    global _prepared
    if _prepared:
        return []
    _prepared = True

    dirs = nvidia_dll_dirs()
    if not dirs:
        if sys.platform != "darwin":
            print("  note: NVIDIA CUDA libraries not installed in this environment.")
            print("        GPU transcription will fail. Fix with: uv sync")
        return []

    for d in dirs:
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    return dirs


def report() -> dict:
    """Diagnostics for the --doctor command."""
    dirs = nvidia_dll_dirs()
    wanted = ("cublas64_", "cublasLt64_", "cudnn_ops", "cudnn64_", "libcublas", "libcudnn")
    hits: dict[str, str] = {}
    for d in dirs:
        for f in d.iterdir():
            for w in wanted:
                if f.name.startswith(w) and w not in hits:
                    hits[w] = str(f)
    return {"dirs": [str(d) for d in dirs], "libs": hits}
