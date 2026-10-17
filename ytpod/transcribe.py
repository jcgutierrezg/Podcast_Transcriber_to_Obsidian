"""Stage 2 - speech to text with segment-level timestamps.

Settings here are tuned for long-form conversational audio, where Whisper's
failure mode is looping ("...thanks for watching, thanks for watching...")
during music, silence, or crosstalk. Three defences:

  vad_filter=True                 -> never feed it silence in the first place
  condition_on_previous_text=False-> a loop can't feed itself
  temperature fallback ladder     -> retry a segment if it looks degenerate

Language handling is a two-pass affair. The formatting primer must be written in
the language being spoken, but we don't know that until Whisper looks at the
audio - so we do a cheap detection pass first, then transcribe properly.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Config
from . import lang as lang_mod

# faster-whisper's own downloader passes a disabled tqdm class to huggingface_hub,
# so a 3 GB first-run download looks like a hung process. We fetch the weights
# ourselves first, with the progress bar left switched on.
_HF_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def ensure_model(name: str) -> str:
    """Return a local path to the model weights, downloading them visibly if needed."""
    if Path(name).is_dir():
        return name

    try:
        from faster_whisper.utils import _MODELS
        repo = _MODELS.get(name, f"Systran/faster-whisper-{name}")
    except Exception:
        repo = f"Systran/faster-whisper-{name}"

    from huggingface_hub import snapshot_download

    print(f"  model: {repo}")
    return snapshot_download(repo, allow_patterns=_HF_PATTERNS)


def _detect_language(model, audio_file: str) -> tuple[str, float]:
    """Cheap detection pass.

    transcribe() returns its info object immediately - language detection happens
    during setup, before any decoding - so we can read the result and throw the
    unconsumed segment iterator away without paying for a transcription.
    """
    _iter, info = model.transcribe(audio_file, language=None, beam_size=1)
    del _iter
    return info.language, float(info.language_probability)


# Fields the child process needs. Config holds Paths, which do not survive a
# spawn cleanly, so we pass primitives and rebuild on the far side.
_CHILD_FIELDS = ("whisper_model", "device", "compute_type", "strict_gpu",
                 "language", "beam_size")


def _child_entry(payload: dict) -> None:
    """Runs in the spawned process. Writes transcript.json and exits."""
    from .config import Config
    cfg = Config()
    for k, v in payload["cfg"].items():
        setattr(cfg, k, v)
    cfg.vocab_file = Path(payload["vocab_file"])
    _transcribe_impl(payload["audio"], Path(payload["work"]), cfg)


def transcribe(audio_file: str, work: Path, cfg: Config) -> dict:
    """Transcribe, preferably in a child process.

    CTranslate2 frees the model on delete, but the CUDA context stays alive for
    the life of the process - so this process would still be sitting on VRAM
    while Ollama tries to load. Ollama sees too little free memory and quietly
    puts the LLM on the CPU. Letting a child process die returns everything.
    """
    if cfg.transcribe_in_subprocess and cfg.device == "cuda":
        import multiprocessing as mp
        payload = {
            "audio": audio_file,
            "work": str(work),
            "vocab_file": str(cfg.vocab_file),
            "cfg": {k: getattr(cfg, k) for k in _CHILD_FIELDS},
        }
        try:
            ctx = mp.get_context("spawn")
            proc = ctx.Process(target=_child_entry, args=(payload,))
            proc.start()
            proc.join()
            if proc.exitcode == 0:
                data = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
                print("  VRAM released (transcription ran in a child process)")
                return data
            print(f"  child process exited with {proc.exitcode}; retrying in-process")
        except Exception as exc:
            print(f"  could not spawn child ({exc}); running in-process")

    return _transcribe_impl(audio_file, work, cfg)


def _transcribe_impl(audio_file: str, work: Path, cfg: Config) -> dict:
    from . import cuda
    cuda.prepare()  # must happen before ctranslate2 loads

    from faster_whisper import WhisperModel  # imported late so the CLI starts fast

    model_path = ensure_model(cfg.whisper_model)

    device, compute_type = cfg.device, cfg.compute_type
    print(f"  loading model onto {device} ({compute_type})...", end="", flush=True)
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
    except Exception as exc:
        if device != "cuda":
            raise
        print(" FAILED")
        hint = ""
        if ".dll" in str(exc).lower() or "cannot be loaded" in str(exc):
            # Almost always the NVIDIA wheels being pruned from the venv by a sync.
            hint = ("\n  The CUDA libraries are likely missing from the environment"
                    " (pruned by a sync). Fix with: uv sync")
        if cfg.strict_gpu:
            raise RuntimeError(
                f"GPU init failed and strict_gpu is set: {exc}{hint}\n"
                f"  Diagnose with: uv run ytpod --doctor"
            ) from exc
        print()
        print("  " + "!" * 66)
        print("  !! GPU UNAVAILABLE - FALLING BACK TO CPU (5-8x SLOWER)")
        print(f"  !! reason: {str(exc)[:200]}")
        if hint:
            print("  !! likely cause: CUDA libraries pruned from the venv - fix with: uv sync")
        print("  !! diagnose with: uv run ytpod --doctor")
        print("  !! to fail instead of falling back: --strict-gpu")
        print("  " + "!" * 66)
        print()
        device, compute_type = "cpu", "int8"
        print(f"  loading model onto {device} ({compute_type})...", end="", flush=True)
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print(" ready")
    print(f"  DEVICE IN USE: {device} ({compute_type})")

    if cfg.language:
        code, prob = cfg.language, 1.0
        print(f"  language: {code} (forced)")
    else:
        print("  detecting language...", end="", flush=True)
        code, prob = _detect_language(model, audio_file)
        print(f" {code} ({prob:.0%})")

    profile = lang_mod.profile(code)
    if profile.code not in lang_mod.PROFILES:
        print(f"  note: no tuned profile for '{code}'; using neutral defaults")

    # Whisper imitates the formatting of whatever context it is given. With
    # condition_on_previous_text disabled it has none, and drifts into uncased,
    # unpunctuated output. A short well-formatted primer in the SAME language
    # restores the habit without creating a path a hallucination loop could ride.
    from .clean import load_vocab
    vocab = load_vocab(cfg, profile)
    initial_prompt = profile.initial_prompt or None
    if initial_prompt and vocab:
        initial_prompt += " " + ", ".join(vocab[:40]) + "."

    segments_iter, info = model.transcribe(
        audio_file,
        language=code,
        beam_size=cfg.beam_size,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
    )

    total = int(round(getattr(info, "duration", 0) or 0))
    out: list[dict] = []
    t_start = time.time()

    # The iterator is lazy - the actual work happens as we consume it, which is
    # exactly what lets us drive a progress bar off audio position.
    with tqdm(
        total=total or None,
        unit="s",
        desc="  transcribing",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        leave=True,
    ) as bar:
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
            if total:
                bar.update(min(int(seg.end), total) - bar.n)
            bar.set_postfix_str(f"{len(out)} segments", refresh=False)
        if total and bar.n < total:
            bar.update(total - bar.n)

    # Speed is the honest test of which device ran: large-v3 on a GPU manages
    # several times realtime, CPU int8 struggles to reach 1x.
    elapsed = time.time() - t_start
    if total and elapsed > 0:
        ratio = total / elapsed
        print(f"  {total / 60:.1f} min of audio in {elapsed / 60:.1f} min "
              f"= {ratio:.1f}x realtime on {device}")
        if device == "cuda" and ratio < 1.5:
            print("  note: that is slow for a GPU - check 'ytpod --doctor'")

    # Free the VRAM before the LLM stage needs it. On <8 GB this is not optional.
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    result = {"language": code, "language_probability": prob,
              "device": device, "compute_type": compute_type, "segments": out}
    (work / "transcript.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
