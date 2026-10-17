"""Finding episode notes wherever they live in the vault.

Notes can be sorted into folders (`URL | title | date | folder`) or moved by
hand, so the index, keyword pruning and duplicate checks can't assume every
episode sits in `<vault>/<lang>/`. An episode note is recognised by its
frontmatter instead: it has a video id key, and a `lang/<code>` tag.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

_ID_KEY = re.compile(r"^(?:id_video|video_id):\s*\"?([^\"\n]+?)\"?\s*$", re.M)
_HEAD = 4000  # frontmatter lives at the top; no need to read whole transcripts


def _frontmatter(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            text = fh.read(_HEAD)
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    return text[: end if end != -1 else len(text)]


def _markdown_files(root: Path):
    for p in root.rglob("*.md"):
        rel = p.relative_to(root).parts
        # .obsidian, .ytpod, .trash; and "_"-prefixed index notes
        if any(part.startswith(".") for part in rel[:-1]) or p.name.startswith("_"):
            continue
        yield p


def episode_notes(root: Path, language: str | None = None) -> list[Path]:
    """Every episode note under root, optionally only one language's."""
    tag = re.compile(rf"^\s*-\s*lang/{re.escape(language)}\s*$", re.M) if language else None
    out = []
    for p in _markdown_files(root):
        fm = _frontmatter(p)
        if _ID_KEY.search(fm) and (tag is None or tag.search(fm)):
            out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


_PUBLISHED = re.compile(r"^(?:publicado|published):\s*\"?(\d{4}-\d{2}-\d{2})", re.M)


def notes_for_video(root: Path, video_id: str) -> list[Path]:
    return [p for p in _markdown_files(root)
            if any(m == video_id for m in _ID_KEY.findall(_frontmatter(p)))]


def video_id_of(path: Path) -> str | None:
    m = _ID_KEY.search(_frontmatter(path))
    return m.group(1) if m else None


def published_of(path: Path) -> str:
    """The note's published date as YYYY-MM-DD, or "" if it has none."""
    m = _PUBLISHED.search(_frontmatter(path))
    return m.group(1) if m else ""


def notes_named(root: Path, filename: str) -> list[Path]:
    """Every note in the vault with this filename. Obsidian links by name alone,
    so two of them in different folders make [[name]] ambiguous."""
    want = filename.lower()
    return [p for p in _markdown_files(root) if p.name.lower() == want]


def safe_folder(folder: str) -> str:
    """Validate a folder given on an episode line: relative, inside the vault."""
    raw = folder.strip().replace("\\", "/")
    segments = [s.strip() for s in raw.split("/")]
    if re.match(r"^[A-Za-z]:", raw) or ".." in segments:
        raise ValueError(f"folder must be a path inside the vault, got {folder!r}")
    # Windows drops trailing dots and spaces from folder names; do it up front
    segments = [s.rstrip(". ") for s in segments]
    segments = [s for s in segments if s]
    if not segments:
        raise ValueError(f"folder must be a path inside the vault, got {folder!r}")
    if any(c in set('<>:"|?*') for s in segments for c in s):
        raise ValueError(f"folder contains characters Windows won't allow: {folder!r}")
    return str(PurePosixPath(*segments))
