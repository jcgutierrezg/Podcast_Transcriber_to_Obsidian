"""Stage 3 - find topic boundaries.

The LLM never sees the full transcript. We hand it a *compressed outline*:
one line per ~45 s block. A 1 hr episode becomes ~80 lines instead of ~13k
tokens of prose - it fits comfortably in a small context window, and the model
is doing structural reasoning, not reading comprehension.

Think of it as looking at a book's page headers to find the chapter breaks
rather than reading every page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

from .config import Config, hms

SYSTEM = (
    "You are a podcast editor. You are given a time-indexed outline of an episode "
    "in {language}. "
    "Identify where the conversation genuinely shifts to a new subject. "
    "Ignore small digressions, jokes, and tangents that return to the same subject. "
    "A typical hour has 5 to 12 real topics. "
    "Write every title and summary in {language}, the same language as the outline. "
    "Do not translate them into English. "
    "Reply with JSON only, no prose, no markdown fences."
)

USER_TEMPLATE = """Episode: {title}
This excerpt covers {win_start} to {win_end} of the episode.

OUTLINE:
{outline}

Return JSON in exactly this shape:
{{"topics": [{{"start": "HH:MM:SS", "title": "short noun phrase, max 8 words",
"summary": "one sentence, max 25 words", "keywords": ["kw1", "kw2", "kw3"]}}]}}

Rules:
- "start" must be copied verbatim from a timestamp that appears in the OUTLINE.
- The first topic must start at the first timestamp in the OUTLINE.
- Titles describe the subject discussed, not "Introduction" or "Part 2".
- Write "title", "summary" and "keywords" in {language}.
- Keywords are single concepts, 1-3 words, singular, lowercase.
{vocabulary}"""


def build_outline(segments: list[dict], block_seconds: int) -> list[dict]:
    """Collapse ASR segments into fixed-duration blocks of text."""
    blocks: list[dict] = []
    start: float | None = None
    buf: list[str] = []
    for seg in segments:
        if start is None:
            start = seg["start"]
        buf.append(seg["text"])
        if seg["end"] - start >= block_seconds:
            blocks.append({"start": start, "text": " ".join(buf)})
            start, buf = None, []
    if buf and start is not None:
        blocks.append({"start": start, "text": " ".join(buf)})
    return blocks


def _render_outline(blocks: list[dict], max_chars: int = 240) -> str:
    lines = []
    for b in blocks:
        text = b["text"]
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{hms(b['start'])}] {text}")
    return "\n".join(lines)


def _windows(blocks: list[dict], cfg: Config):
    """Slide a window over the blocks so no topic is orphaned at a seam."""
    if not blocks:
        return
    span = cfg.window_minutes * 60
    step = max(60, span - cfg.overlap_minutes * 60)
    end_time = blocks[-1]["start"]
    t = blocks[0]["start"]
    while t <= end_time:
        chunk = [b for b in blocks if t <= b["start"] < t + span]
        if chunk:
            yield chunk
        t += step


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _extract_json(text: str) -> dict:
    text = _strip_thinking(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"no JSON found in LLM reply: {text[:200]}")
    return json.loads(match.group(0))


def _vocabulary_block(known: list[str]) -> str:
    """Show the model the vocabulary the library already uses, so it reuses it."""
    if not known:
        return ""
    return ("- Prefer these existing keywords wherever they fit, spelled exactly as "
            "given: " + ", ".join(known[:60]) + "\n")


def _ask_llm(prompt: str, cfg: Config, language: str) -> dict:
    payload = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(language=language)},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "think": False,  # ignored by non-reasoning models
        "options": {"temperature": 0.1, "num_ctx": cfg.llm_num_ctx},
    }
    r = requests.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return _extract_json(r.json()["message"]["content"])


def _parse_hms(value: str) -> float | None:
    parts = str(value).strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0)
    return nums[-3] * 3600 + nums[-2] * 60 + nums[-1]


def check_ollama(cfg: Config) -> None:
    """Fail fast with a useful message rather than mid-transcript."""
    try:
        r = requests.get(f"{cfg.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        raise RuntimeError(
            f"Ollama not reachable at {cfg.ollama_url}. Start the Ollama app, "
            f"or run 'ollama serve' in another terminal."
        )
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n == cfg.llm_model or n.startswith(cfg.llm_model + ":") for n in names):
        raise RuntimeError(
            f"Model '{cfg.llm_model}' not found in Ollama. Run: ollama pull {cfg.llm_model}\n"
            f"  Available: {', '.join(names) or '(none)'}"
        )


def report_placement(cfg: Config) -> None:
    """Say whether Ollama put the model on the GPU or the CPU.

    /api/ps reports size (total) and size_vram (the part actually resident on
    the GPU). If size_vram is zero the whole model is on the CPU, which is the
    difference between two minutes and two hours.
    """
    try:
        # a trivial request forces the model to load so /api/ps has something to report
        requests.post(f"{cfg.ollama_url}/api/chat", json={
            "model": cfg.llm_model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False, "think": False, "keep_alive": "10m",
            "options": {"num_ctx": cfg.llm_num_ctx, "num_predict": 1},
        }, timeout=300).raise_for_status()
        ps = requests.get(f"{cfg.ollama_url}/api/ps", timeout=10).json()
    except Exception:
        return

    for m in ps.get("models", []):
        if not m.get("name", "").startswith(cfg.llm_model.split(":")[0]):
            continue
        total = m.get("size", 0)
        vram = m.get("size_vram", 0)
        if not total:
            return
        pct = 100 * vram / total
        if vram == 0:
            print("  " + "!" * 66)
            print(f"  !! OLLAMA IS RUNNING '{cfg.llm_model}' ON THE CPU - this will be slow")
            print("  !! try: a smaller model (ollama pull qwen3:4b), or lower")
            print("  !!      llm_num_ctx in config.py, or close other GPU programs")
            print("  " + "!" * 66)
        elif pct < 95:
            print(f"  LLM: {pct:.0f}% on GPU, rest on CPU "
                  f"({vram / 1e9:.1f}/{total / 1e9:.1f} GB) - expect slowdown")
        else:
            print(f"  LLM: on GPU ({vram / 1e9:.1f} GB, ctx {cfg.llm_num_ctx})")
        return


def segment(segments: list[dict], meta: dict, work: Path, cfg: Config,
            language: str = "en", use_yt_chapters: bool = False,
            known_keywords: list[str] | None = None) -> list[dict]:
    from .lang import profile as get_profile
    lang_name = get_profile(language).name_en
    duration = meta.get("duration") or (segments[-1]["end"] if segments else 0.0)

    if use_yt_chapters and meta.get("yt_chapters"):
        print(f"  using {len(meta['yt_chapters'])} uploader chapters")
        topics = [
            {"start": c["start"], "title": c["title"], "summary": "", "keywords": []}
            for c in meta["yt_chapters"]
        ]
    else:
        check_ollama(cfg)
        report_placement(cfg)
        blocks = build_outline(segments, cfg.block_seconds)
        raw: list[dict] = []
        window_list = list(_windows(blocks, cfg))
        bar = tqdm(
            window_list,
            desc="  segmenting",
            unit="win",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} windows "
                       "[{elapsed}<{remaining}] {postfix}",
        )
        for chunk in bar:
            prompt = USER_TEMPLATE.format(
                title=meta.get("title", ""),
                win_start=hms(chunk[0]["start"]),
                win_end=hms(chunk[-1]["start"]),
                outline=_render_outline(chunk),
                language=lang_name,
                vocabulary=_vocabulary_block(known_keywords or []),
            )
            try:
                data = _ask_llm(prompt, cfg, lang_name)
            except Exception as exc:
                bar.write(f"  window at {hms(chunk[0]['start'])} failed ({exc}); skipping")
                continue
            for t in data.get("topics", []):
                start = _parse_hms(t.get("start", ""))
                if start is None:
                    continue
                raw.append({
                    "start": start,
                    "title": str(t.get("title", "")).strip() or "Untitled",
                    "summary": str(t.get("summary", "")).strip(),
                    "keywords": [str(k) for k in (t.get("keywords") or [])][:5],
                })
            bar.set_postfix_str(f"{len(raw)} candidates", refresh=False)
        bar.close()
        if window_list:
            per = bar.format_dict.get("elapsed", 0) / len(window_list)
            print(f"  {per:.0f}s per window ({len(window_list)} windows)")
        topics = _dedupe(raw, cfg.min_topic_seconds)

    topics = _snap_and_close(topics, segments, duration)
    (work / "topics.json").write_text(
        json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return topics


def _dedupe(raw: list[dict], min_gap: int) -> list[dict]:
    """Overlapping windows produce duplicate boundaries; keep the first of each cluster."""
    kept: list[dict] = []
    for t in sorted(raw, key=lambda x: x["start"]):
        if kept and t["start"] - kept[-1]["start"] < min_gap:
            # Prefer whichever entry has a real summary attached.
            if not kept[-1]["summary"] and t["summary"]:
                kept[-1]["summary"] = t["summary"]
                kept[-1]["keywords"] = kept[-1]["keywords"] or t["keywords"]
            continue
        kept.append(t)
    return kept


def _snap_and_close(topics: list[dict], segments: list[dict], duration: float) -> list[dict]:
    """Move each boundary to the nearest real segment start, then fill in end times."""
    if not topics:
        return [{"start": 0.0, "end": duration, "title": "Full episode",
                 "summary": "", "keywords": []}]

    starts = [s["start"] for s in segments]
    for t in topics:
        if starts:
            t["start"] = min(starts, key=lambda s: abs(s - t["start"]))
    topics.sort(key=lambda t: t["start"])
    topics[0]["start"] = segments[0]["start"] if segments else 0.0

    for i, t in enumerate(topics):
        t["end"] = topics[i + 1]["start"] if i + 1 < len(topics) else duration
    return topics
