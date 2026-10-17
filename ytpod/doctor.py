"""Environment diagnostics - 'ytpod --doctor'.

Checks the four things that actually break on a fresh Windows machine, in the
order they'd bite you.
"""

from __future__ import annotations

import sys

import requests

from .config import Config
from . import cuda


def _ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


def _bad(msg: str, fix: str = "") -> None:
    print(f"  [FAIL] {msg}")
    if fix:
        print(f"         fix: {fix}")


def doctor(cfg: Config) -> int:
    problems = 0
    print("\nytpod environment check")
    print(f"  python {sys.version.split()[0]} on {sys.platform}\n")

    # 1. NVIDIA support libraries
    print("NVIDIA libraries")
    rep = cuda.report()
    if not rep["dirs"]:
        _bad("no nvidia-* wheels found", "uv sync")
        problems += 1
    else:
        for d in rep["dirs"]:
            print(f"         {d}")
        for name in ("cublas64_", "cudnn_ops", "libcublas", "libcudnn"):
            if name in rep["libs"]:
                _ok(f"{name}* present")
        if not any(k.startswith(("cublas64_", "libcublas")) for k in rep["libs"]):
            _bad("cuBLAS not found", "uv sync")
            problems += 1
        if not any(k.startswith(("cudnn", "libcudnn")) for k in rep["libs"]):
            _bad("cuDNN not found", 'uv add "nvidia-cudnn-cu12>=9,<10"')
            problems += 1

    # 2. CTranslate2 can actually open the GPU
    print("\nCTranslate2 / GPU")
    cuda.prepare()
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        print(f"  [ ok ] ctranslate2 {ctranslate2.__version__}")
        if count:
            _ok(f"{count} CUDA device(s) visible")
            types = ctranslate2.get_supported_compute_types("cuda")
            print(f"         compute types: {', '.join(sorted(types))}")
            if cfg.compute_type not in types:
                _bad(f"configured compute_type '{cfg.compute_type}' unsupported",
                     f"set compute_type to one of: {', '.join(sorted(types))}")
                problems += 1
        else:
            _bad("no CUDA devices visible", "check your NVIDIA driver, or set device='cpu'")
            problems += 1

        # An actual load is the only proof. Everything above can pass while the
        # real model still falls back to CPU at runtime.
        if count:
            try:
                import ctranslate2.models  # noqa: F401
                from faster_whisper import WhisperModel
                from .transcribe import ensure_model
                from huggingface_hub import snapshot_download  # noqa: F401
                print("  [ .. ] loading a model on the GPU to confirm...", end="", flush=True)
                path = ensure_model("tiny")
                m = WhisperModel(path, device="cuda", compute_type=cfg.compute_type)
                del m
                print("\r  [ ok ] real model loaded on the GPU            ")
            except Exception as exc:
                print()
                _bad(f"GPU load failed at runtime: {str(exc)[:160]}",
                     "this is what makes transcription silently use CPU")
                problems += 1
    except Exception as exc:
        _bad(f"ctranslate2 unusable: {exc}", "uv sync")
        problems += 1

    # 2b. yt-dlp
    print("\nyt-dlp")
    try:
        import yt_dlp
        version = yt_dlp.version.__version__
        _ok(f"yt-dlp {version}")
        print("         YouTube changes often; if downloads fail, update first:")
        print("         uv lock --upgrade-package yt-dlp, then uv sync")
    except Exception as exc:
        _bad(f"yt-dlp unusable: {exc}", "uv sync")
        problems += 1

    # 3. Ollama
    print("\nOllama")
    try:
        r = requests.get(f"{cfg.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        _ok(f"reachable at {cfg.ollama_url}")
        if any(n == cfg.llm_model or n.startswith(cfg.llm_model + ":") for n in names):
            _ok(f"model '{cfg.llm_model}' available")
        else:
            _bad(f"model '{cfg.llm_model}' not pulled", f"ollama pull {cfg.llm_model}")
            problems += 1
        print(f"         installed: {', '.join(names) or '(none)'}")
    except Exception:
        _bad(f"not reachable at {cfg.ollama_url}", "start the Ollama app, or run 'ollama serve'")
        problems += 1

    # 4. Whisper weights
    print("\nWhisper weights")
    try:
        from huggingface_hub import snapshot_download
        from .transcribe import _HF_PATTERNS
        try:
            from faster_whisper.utils import _MODELS
            repo = _MODELS.get(cfg.whisper_model, f"Systran/faster-whisper-{cfg.whisper_model}")
        except Exception:
            repo = f"Systran/faster-whisper-{cfg.whisper_model}"
        path = snapshot_download(repo, allow_patterns=_HF_PATTERNS, local_files_only=True)
        _ok(f"{repo} cached")
        print(f"         {path}")
    except Exception:
        print(f"  [note] '{cfg.whisper_model}' not cached yet "
              f"(~3 GB will download on first run)")

    print(f"\n{'All checks passed.' if not problems else f'{problems} problem(s) found.'}\n")
    return 1 if problems else 0
