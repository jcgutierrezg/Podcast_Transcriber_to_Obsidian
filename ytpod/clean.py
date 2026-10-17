"""Stage 2b - clean up ASR output without changing the words.

Three defects get fixed here, all of them artefacts of how Whisper decodes long
audio rather than anything the speaker actually did:

  seam stutters   "this didn't start" + "art recently"  -> decode-window overlap
  lost casing     long uncased runs                     -> no prior context to imitate
  no punctuation  ditto

The guiding rule: never change which words are present. The punctuation pass
uses an LLM, so it is verified word-for-word against the original and silently
discarded if a single word differs. Formatting is negotiable; content is not.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import requests
from tqdm import tqdm

from .config import Config
from .lang import LangProfile

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def _read_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def load_vocab(cfg: Config, profile: LangProfile) -> list[str]:
    """Shared vocab.txt plus the language-specific vocab.<code>.txt, in that order."""
    terms = _read_terms(cfg.vocab_file)
    terms += _read_terms(cfg.vocab_file.parent / profile.vocab_name)
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def apply_vocab(text: str, vocab: list[str]) -> str:
    """Force known terms to their canonical casing, however Whisper wrote them."""
    for term in vocab:
        text = re.sub(rf"\b{re.escape(term)}\b", term, text, flags=re.IGNORECASE)
    return text


# --------------------------------------------------------------------------
# seam stutters
# --------------------------------------------------------------------------

def _strip_seam_repeat(prev: str, nxt: str, gap: float = 0.0, max_words: int = 4) -> str:
    """Drop a duplicated word or phrase at the join between two segments.

    `gap` is the silence between the two segments. This is the discriminator:
    a decode-window artefact has no gap because the two segments describe the
    same instant of audio, whereas a speaker genuinely repeating a word leaves
    an audible pause. Without it, "the heart" + "art is subjective" gets eaten.
    """
    prev_words = prev.split()
    next_words = nxt.split()
    if not prev_words or not next_words:
        return nxt

    def norm(w: str) -> str:
        return re.sub(r"[^\w']", "", w).lower()

    # Case A: exact repeat of 1-4 words across the boundary.
    if gap <= 0.8:
        for n in range(min(max_words, len(prev_words), len(next_words)), 0, -1):
            if [norm(w) for w in prev_words[-n:]] == [norm(w) for w in next_words[:n]]:
                return " ".join(next_words[n:])

    # Case B: a word split across the boundary, e.g. "...start" + "art recently".
    # Much riskier, so it needs a tighter timing window.
    if gap <= 0.35:
        tail, head = norm(prev_words[-1]), norm(next_words[0])
        if len(head) >= 3 and len(tail) >= len(head) + 2 and tail.endswith(head):
            return " ".join(next_words[1:])

    return nxt


def fix_seams(segments: list[dict]) -> tuple[list[dict], int]:
    out: list[dict] = []
    fixed = 0
    for seg in segments:
        seg = dict(seg)
        if out:
            gap = max(0.0, seg["start"] - out[-1]["end"])
            new_text = _strip_seam_repeat(out[-1]["text"], seg["text"], gap)
            if new_text != seg["text"]:
                fixed += 1
                seg["text"] = new_text
            if not seg["text"].strip():
                continue
        out.append(seg)
    return out, fixed


# --------------------------------------------------------------------------
# casing
# --------------------------------------------------------------------------

def fix_titles(text: str, titles: tuple[str, ...]) -> str:
    """'dr peter levine' -> 'Dr. Peter Levine'. Sentence-casing alone misses these.

    Driven by the language profile: Spanish wants Sra./Srta./Lic., English wants
    Mr./Mrs./Jr.
    """
    if not titles:
        return text
    pattern = re.compile(rf"\b({'|'.join(titles)})\b\.?", re.IGNORECASE)
    return pattern.sub(lambda m: m.group(1).capitalize() + ".", text)


def capitalise_sentences(text: str) -> str:
    chars = list(text)
    cap_next = True
    for i, ch in enumerate(chars):
        if cap_next and ch.isalpha():
            chars[i] = ch.upper()
            cap_next = False
        elif ch in ".!?":
            cap_next = True
    return "".join(chars)


def fix_pronoun_i(text: str) -> str:
    """Standalone 'i' and its contractions. \\b treats the apostrophe as a boundary."""
    return re.sub(r"\bi\b", "I", text)


def looks_uncased(text: str) -> bool:
    """A segment Whisper gave up formatting: no capitals and no sentence punctuation."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 25:
        return False
    no_caps = not any(c.isupper() for c in letters)
    no_stops = not any(c in ".!?\u00bf\u00a1" for c in text)
    return no_caps or no_stops


# --------------------------------------------------------------------------
# punctuation restoration (LLM, verified)
# --------------------------------------------------------------------------

_REPUNCT_SYSTEM = (
    "You restore punctuation and capitalisation in {language} transcript text. "
    "You must not add, remove, reorder, substitute, or translate any word. "
    "Change only punctuation marks, accents, and letter case. "
    "Follow the punctuation conventions of {language}. "
    "Reply with the corrected text and nothing else."
)


def _word_signature(text: str) -> list[str]:
    """Words stripped of case, accents and punctuation - what must not change.

    Accents are folded because restoring them is exactly what we asked the model
    to do in Spanish: "cancion" -> "canci\u00f3n" is a correction, not a
    substitution. Word count and letter sequence still have to match exactly.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [w for w in re.sub(r"[^\w\s']", " ", folded).split() if w]


def _split_back(cleaned: str, counts: list[int]) -> list[str] | None:
    """Redistribute a cleaned batch across the segments it came from.

    Walks tokens, counting only those that carry an actual word, so a model that
    detaches punctuation ("\u00bf C\u00f3mo") still lands on the right boundaries.
    """
    tokens = cleaned.split()
    out, i = [], 0
    for n in counts:
        taken, words = [], 0
        while i < len(tokens) and words < n:
            tok = tokens[i]
            if _word_signature(tok):
                words += 1
            taken.append(tok)
            i += 1
        # absorb trailing bare punctuation into this segment
        while i < len(tokens) and not _word_signature(tokens[i]):
            taken.append(tokens[i])
            i += 1
        if words != n:
            return None
        out.append(" ".join(taken))
    return out if i == len(tokens) else None


def _ask_repunct(text: str, cfg: Config, profile: LangProfile) -> str | None:
    content = text
    if "qwen3" in cfg.llm_model.lower():
        content = "/no_think\n" + content  # older Ollama ignores the think field
    payload = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": _REPUNCT_SYSTEM.format(language=profile.name_en)},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",          # stop Ollama unloading between batches
        "options": {
            "temperature": 0.0,
            "num_ctx": cfg.repunct_ctx,
            "num_predict": cfg.repunct_ctx,
        },
    }
    try:
        r = requests.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        result = r.json()["message"]["content"]
    except Exception:
        return None

    result = re.sub(r"<think>.*?</think>", "", result, flags=re.S).strip()
    return result.strip('"').strip()


def _contiguous_batches(indices: list[int], segments: list[dict],
                        max_words: int) -> list[list[int]]:
    """Group neighbouring flagged segments so one request covers many of them."""
    batches: list[list[int]] = []
    current: list[int] = []
    words = 0
    for idx in indices:
        n = len(segments[idx]["text"].split())
        if current and (idx != current[-1] + 1 or words + n > max_words):
            batches.append(current)
            current, words = [], 0
        current.append(idx)
        words += n
    if current:
        batches.append(current)
    return batches


def repunctuate(segments: list[dict], cfg: Config,
                profile: LangProfile) -> tuple[list[dict], int, int]:
    targets = [i for i, s in enumerate(segments) if looks_uncased(s["text"])]
    if not targets:
        return segments, 0, 0

    batches = _contiguous_batches(targets, segments, cfg.repunct_batch_words)
    out = [dict(s) for s in segments]
    accepted = 0

    for batch in tqdm(batches, desc="  repunctuating", unit="batch",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                 "[{elapsed}<{remaining}]"):
        original = [out[i]["text"] for i in batch]
        joined = " ".join(original)
        fixed = _ask_repunct(joined, cfg, profile)

        # The safeguard: identical words, or the whole batch is discarded.
        if not fixed or _word_signature(fixed) != _word_signature(joined):
            continue
        pieces = _split_back(fixed, [len(_word_signature(o)) for o in original])
        if pieces is None:
            continue
        for i, piece in zip(batch, pieces):
            out[i]["text"] = piece
        accepted += len(batch)

    return out, accepted, len(targets)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def clean(segments: list[dict], work: Path, cfg: Config, language: str) -> list[dict]:
    from .lang import profile as get_profile
    profile = get_profile(language)

    vocab = load_vocab(cfg, profile)
    if vocab:
        print(f"  vocab: {len(vocab)} terms")

    if cfg.fix_seam_stutters:
        segments, fixed = fix_seams(segments)
        print(f"  seam stutters removed: {fixed}")

    if cfg.restore_punctuation:
        segments, ok, tried = repunctuate(segments, cfg, profile)
        if tried:
            print(f"  punctuation restored: {ok}/{tried} spans "
                  f"({tried - ok} rejected as word-altering)")

    out = []
    for seg in segments:
        text = seg["text"]
        text = capitalise_sentences(text)
        if profile.pronoun_i:
            text = fix_pronoun_i(text)
        text = fix_titles(text, profile.titles)
        if vocab:
            text = apply_vocab(text, vocab)
        out.append({**seg, "text": text})

    (work / "transcript.clean.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
