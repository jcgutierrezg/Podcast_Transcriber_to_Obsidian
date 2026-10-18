"""Stage 5 (optional) - condense each chapter into study notes.

The condensed text becomes the chapter's body; the full transcript stays right
below it in a collapsed callout, for when you want the exact words:

    ## Chapter title
    `00:03:05` – `00:06:11`

    Condensed notes...

    > [!quote]- Transcripción completa
    > The original paragraphs...

Runs on the local Ollama model, like segmentation. Two ways in:

  - during a run (--condense), from the fresh transcript
  - on notes already written (--condense-notes), from the note itself - so
    hand edits to a transcript are what gets condensed. Running it again
    re-condenses from the collapsed original instead of stacking summaries.
    The previous version is kept as <vault>/.ytpod/backups/<name>.<time>.md.bak,
    outside anything Obsidian indexes, so backups never show up as notes or
    backlinks.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from pathlib import Path

import requests

from .config import Config
from .lang import PROFILES, profile as get_profile

PROMPT_VERSION = 4   # bump to invalidate cached results after a prompt change

SYSTEM = ("You turn transcripts of talks and classes into condensed study notes. "
          "You write only in {language}, the transcript's own language, and you "
          "never add anything that is not in the text.")

PROMPT = """Condense this part of a {language} transcript into clear study notes.

Keep every distinct idea, argument, instruction, example and story, and every
name, number, date and reference (for example Bible verses, books, tools or
websites).

Remove filler words, greetings, repetition, false starts, small talk, and
asides that carry no content (technical hiccups, "can you see my screen?").

Rules:
- Write in {language}. Never translate.
- Paraphrase concisely in your own words, merging sentences that say the same
  thing, and say each point once.
- Do not quote the speaker. The only quotations allowed are texts read aloud
  (a Bible verse, a book passage) and a key saying the speaker attributes to
  someone by name.
- Leave out small talk and logistics entirely (screen sharing, "can you see
  this?", "are you excited?", questions about schedules) - do not list them.
- Use the speaker's own key terms. Add no facts, opinions or commentary.
- State the content directly; do not write "the speaker says".
- Short paragraphs. Use "- " bullets only for lists of items or steps.
- No headings, no title, no introduction, no closing remarks.
- Aim for about a third of the original length.

This is from the section "{chapter}" of "{title}".

TRANSCRIPT:
{text}

CONDENSED NOTES:"""

CALLOUT = "> [!quote]- {label}"
# Recognise our callout in any language's label, e.g. after switching profiles
_CALLOUT_RE = re.compile(r"^> \[!quote\]-\s*(.+?)\s*$")
_LABELS = {p.label("full_transcript") for p in PROFILES.values()} | {"Full transcript"}
_STAMP_RE = re.compile(r"^\[?`\d{2}:\d{2}:\d{2}`")
_MARK_RE = re.compile(r"^\*\[?`?\[?\d{2}:\d{2}:\d{2}")   # inline [hh:mm:ss] markers


# --------------------------------------------------------------------------
# the LLM step
# --------------------------------------------------------------------------

def _words(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, limit: int) -> list[str]:
    """A paragraph longer than one request, cut at sentence ends."""
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
    out, buf = [], []
    for s in sentences:
        if buf and _words(" ".join(buf + [s])) > limit:
            out.append(" ".join(buf))
            buf = []
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return out


def _chunks(paragraphs: list[str], limit: int) -> list[str]:
    """Group paragraphs into requests of at most ~limit words, never splitting
    a paragraph unless it is too long on its own."""
    pieces = [p for para in paragraphs for p in
              (_split_long(para, limit) if _words(para) > limit else [para])]
    out, buf, n = [], [], 0
    for p in pieces:
        w = _words(p)
        if buf and n + w > limit:
            out.append("\n\n".join(buf))
            buf, n = [], 0
        buf.append(p)
        n += w
    if buf:
        out.append("\n\n".join(buf))
    return out


def _clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines = text.splitlines()
    # small models sometimes echo the cue or open with a heading
    while lines and (lines[0].strip().upper().startswith("CONDENSED NOTES")
                     or lines[0].lstrip().startswith("#") or not lines[0].strip()):
        lines.pop(0)
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def condense_model(cfg: Config) -> str:
    return cfg.condense_model or cfg.llm_model


def _ask(prompt: str, cfg: Config, language_name: str) -> str:
    """One condense request. Thinking is ON for models that have it: with it
    off, qwen3 writes its reasoning into the answer or echoes the transcript.
    Ollama returns the reasoning in a separate field, which we drop; empty
    content means the reasoning used up the whole budget. Instruct models
    without thinking reject the flag, so the request is retried without it."""
    payload = {
        "model": condense_model(cfg),
        "messages": [
            {"role": "system", "content": SYSTEM.format(language=language_name)},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": True,
        "options": {"temperature": 0.2, "num_ctx": cfg.condense_num_ctx,
                    "num_predict": cfg.condense_num_predict},
    }
    r = requests.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=1800)
    if r.status_code == 400 and "think" in r.text.lower():
        payload.pop("think")
        r = requests.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=1800)
    r.raise_for_status()
    return _clean_output(r.json()["message"]["content"])


# Reasoning a model narrated into its answer instead of the thinking field
_LEAKED_REASONING = re.compile(
    r"\b(okay, let'?s|let me|the user (wants|says)|i need to|first, i'?ll)\b", re.I)


def _usable(out: str, chunk: str, language: str) -> bool:
    """Worth putting in the note: something came back, it is shorter than the
    transcript (dense passages may only shrink a little - still better), and
    it isn't the model thinking aloud in English over a non-English text."""
    if not out or _words(out) >= _words(chunk):
        return False
    return language == "en" or not _LEAKED_REASONING.search(out)


class Condenser:
    """Condenses chapters, caching each request under work/_condense/ so
    re-running a note (or a whole vault) only pays for what changed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache = Path(cfg.work_dir) / "_condense"
        self.words_in = 0
        self.words_out = 0
        self.requests = 0
        self.cached = 0
        self.failed = 0

    def _cache_path(self, *parts: str) -> Path:
        key = "\x1f".join([str(PROMPT_VERSION), condense_model(self.cfg), *parts])
        return self.cache / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".txt")

    def chapter(self, paragraphs: list[str], language: str, title: str,
                chapter: str) -> str | None:
        """Condensed text for one chapter, or None if it is too short to bother."""
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        total = sum(_words(p) for p in paragraphs)
        if total < self.cfg.condense_min_words:
            return None
        name = get_profile(language).name_en
        parts = []
        for chunk in _chunks(paragraphs, self.cfg.condense_chunk_words):
            cache = self._cache_path(language, title, chapter, chunk)
            if cache.exists():
                out = cache.read_text(encoding="utf-8")
                self.cached += 1
            else:
                prompt = PROMPT.format(language=name, chapter=chapter, title=title, text=chunk)
                out = ""
                for _attempt in range(2):          # one retry
                    out = _ask(prompt, self.cfg, name)
                    self.requests += 1
                    if _usable(out, chunk, language):
                        break
                if _usable(out, chunk, language):
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(out, encoding="utf-8")
                else:
                    # never lose or bloat content: keep the transcript for this
                    # part, and don't cache, so the next run tries again
                    out = chunk
                    self.failed += 1
                    print(f"\n  note: couldn't condense part of '{chapter}' "
                          f"(empty or not shorter); kept the original text there")
            parts.append(out)
        text = "\n\n".join(parts)
        self.words_in += total
        self.words_out += _words(text)
        return text

    def summary(self) -> str:
        pct = 100 * self.words_out / self.words_in if self.words_in else 0
        failed = f", {self.failed} part(s) left uncondensed" if self.failed else ""
        return (f"condensed {self.words_in} -> {self.words_out} words ({pct:.0f}%), "
                f"{self.requests} LLM requests, {self.cached} cached{failed}")


# --------------------------------------------------------------------------
# note layout, shared by render and the rewriter
# --------------------------------------------------------------------------

def chapter_body(original: list[str], condensed: str | None, label: str) -> list[str]:
    """Lines for a chapter body: condensed notes first, then the original in a
    collapsed callout. Without condensed text, the original as before."""
    original = _trim(original)
    if not condensed:
        return original + [""]
    lines = condensed.strip().splitlines() + ["", CALLOUT.format(label=label)]
    lines += [f"> {ln}" if ln.strip() else ">" for ln in original]
    return lines + [""]


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def paragraphs_of(lines: list[str]) -> list[str]:
    """Body lines -> paragraphs for the LLM, dropping inline timestamp markers."""
    paras, buf = [], []
    for ln in lines + [""]:
        if ln.strip():
            buf.append(ln.strip())
        elif buf:
            para = " ".join(buf)
            if not _MARK_RE.match(para):
                paras.append(para)
            buf = []
    return paras


# --------------------------------------------------------------------------
# postprocessing notes already written
# --------------------------------------------------------------------------

def _is_header_line(line: str) -> bool:
    """The lines render puts between a chapter heading and its text."""
    s = line.strip()
    return (not s or bool(_STAMP_RE.match(s))
            or (s.startswith("> ") and not s.startswith("> [!"))   # topic summary
            or (s.startswith("*") and "[[" in s))                  # keyword links


def _split_chapter(body: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(header, original transcript, anything after our callout).
    For an already-condensed chapter the original comes out of the callout."""
    i = 0
    while i < len(body) and _is_header_line(body[i]):
        i += 1
    header, rest = body[:i], body[i:]
    for j, ln in enumerate(rest):
        m = _CALLOUT_RE.match(ln)
        if m and m.group(1) in _LABELS:
            k = j + 1
            original = []
            while k < len(rest) and rest[k].startswith(">"):
                original.append(rest[k][2:] if rest[k].startswith("> ") else "")
                k += 1
            return header, original, rest[k:]
    return header, rest, []


def _frontmatter_value(fm: str, *keys: str) -> str:
    for key in keys:
        m = re.search(rf'^{re.escape(key)}:\s*"?(.*?)"?\s*$', fm, re.M)
        if m:
            return m.group(1)
    return ""


def _vault_root(note: Path) -> Path:
    """Nearest parent that looks like a vault (.obsidian or .ytpod)."""
    for d in note.resolve().parents:
        if (d / ".obsidian").is_dir() or (d / ".ytpod").is_dir():
            return d
    return note.resolve().parent


def condense_note(path: Path, condenser: Condenser) -> bool:
    """Condense every chapter of an existing note in place. A backup goes to
    <vault>/.ytpod/backups/ first. Returns True if the note changed."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    fm_end = 0
    if lines and lines[0].strip() == "---":
        fm_end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), 0)
    fm = "\n".join(lines[1:fm_end])

    code = _frontmatter_value(fm, "idioma", "language") or "en"
    profile = get_profile(code)
    title = (_frontmatter_value(fm, "título", "title")
             or next((ln[2:] for ln in lines if ln.startswith("# ")), path.stem))
    label = profile.label("full_transcript")

    out = lines[:fm_end + 1]
    i = fm_end + 1
    chapters = 0
    while i < len(lines):
        if not lines[i].startswith("## "):
            out.append(lines[i])
            i += 1
            continue
        heading = lines[i]
        j = i + 1
        while j < len(lines) and not lines[j].startswith("## "):
            j += 1
        body = lines[i + 1:j]
        first = next((ln.strip() for ln in body if ln.strip()), "")
        if not _STAMP_RE.match(first):        # Contents, or a section of your own
            out += [heading, *body]
        else:
            header, original, after = _split_chapter(body)
            chapters += 1
            print(f"\r  condensing: chapter {chapters}  {heading[3:][:50]:<50}", end="", flush=True)
            condensed = condenser.chapter(paragraphs_of(original), profile.code, title, heading[3:])
            out += [heading, *_trim(header), "", *chapter_body(original, condensed, label),
                    *after]
        i = j
    print()

    new = "\n".join(out)
    new = re.sub(r"\n{3,}", "\n\n", new).rstrip() + "\n"
    if new == text:
        return False
    # Backups must never take part in the vault: a dot-folder Obsidian doesn't
    # index, and a .bak extension so nothing parses them as notes or links
    # from them, even with hidden files shown. Rename to .md to restore.
    backups = _vault_root(path) / ".ytpod" / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backups / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.md.bak")
    path.write_text(new, encoding="utf-8")
    return True


def condense_notes(paths: list[Path], cfg: Config) -> int:
    """--condense-notes: files, or folders searched for episode notes."""
    from . import library
    from .segment import check_ollama, report_placement
    notes: list[Path] = []
    for p in paths:
        if p.is_dir():
            notes += library.episode_notes(p)
        elif p.suffix.lower() == ".md" and p.exists():
            notes.append(p)
        else:
            print(f"  skipped (not a note or folder): {p}")
    if not notes:
        print("no notes to condense")
        return 1
    # load the condensing model at its context, so the GPU check reports the real thing
    from dataclasses import replace
    ccfg = replace(cfg, llm_model=condense_model(cfg), llm_num_ctx=cfg.condense_num_ctx)
    check_ollama(ccfg)
    report_placement(ccfg)
    condenser = Condenser(cfg)
    changed = 0
    for n, note in enumerate(notes, 1):
        print(f"\n[{n}/{len(notes)}] {note.name}")
        t0 = time.time()
        if condense_note(note, condenser):
            changed += 1
            print(f"  updated ({time.time() - t0:.0f}s)")
        else:
            print("  unchanged")
    print(f"\n{changed} of {len(notes)} note(s) updated; {condenser.summary()}")
    if changed:
        print("previous versions kept in <vault>/.ytpod/backups/ as .md.bak "
              "(not indexed by Obsidian; rename to .md to restore)")
    return 0
