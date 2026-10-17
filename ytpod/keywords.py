"""Cross-episode keyword index.

The problem with turning raw LLM keywords into links is drift: episode 1 says
"silencio", episode 7 says "el silencio", episode 12 says "silencios", and you
end up with three disconnected nodes describing one idea.

So the library keeps a canonical vocabulary in `_keywords.json` at the root of
each language folder. It does two jobs:

  1. new keywords are matched against it, and snap to an existing term if close
  2. it is fed back to the LLM during segmentation, so the model reuses your
     established vocabulary instead of inventing a synonym

Think of it as a parts bin: before machining a new bracket, check whether one
you already made will do.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import get_close_matches
from pathlib import Path

INDEX_NAME = "_keywords.json"


# Leading articles are noise for matching: "el silencio" and "silencio" are the
# same idea, and difflib won't merge them because the prefix shifts every char.
_ARTICLES = ("el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
             "the", "a", "an")


def fold(term: str) -> str:
    """Comparison key: lowercase, unaccented, punctuation-free, article-free."""
    s = unicodedata.normalize("NFKD", term.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    words = s.split()
    if len(words) > 1 and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def slug(term: str) -> str:
    """Filename- and tag-safe form."""
    return re.sub(r"[\s_]+", "-", fold(term)).strip("-")


def safe_hub_name(term: str) -> str:
    """Hub note filename for a term - must match what write_hub_notes creates."""
    return re.sub(r'[<>:"/\\|?*\[\]#^]', "-", term.strip())[:100]


def load_index(out_dir: Path) -> dict:
    path = out_dir / INDEX_NAME
    if not path.exists():
        return {"terms": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if "terms" in data else {"terms": {}}
    except Exception:
        return {"terms": {}}


def save_index(out_dir: Path, index: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / INDEX_NAME).write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def known_terms(index: dict) -> list[str]:
    """Display forms, most-used first - what gets shown to the LLM."""
    entries = index.get("terms", {})
    return [e["display"] for e in sorted(
        entries.values(), key=lambda e: -len(e.get("notes", [])))]


def canonicalise(term: str, index: dict, threshold: float = 0.88) -> str | None:
    """Snap a keyword to an existing canonical term, or register it as a new one."""
    key = fold(term)
    if len(key) < 3:
        return None
    terms = index.setdefault("terms", {})

    if key in terms:
        return terms[key]["display"]

    close = get_close_matches(key, list(terms.keys()), n=1, cutoff=threshold)
    if close:
        return terms[close[0]]["display"]

    # Store the article-free form as the display name too, so the vocabulary
    # shown to the LLM stays consistent.
    display = term.strip()
    first = display.split()[0].lower() if display.split() else ""
    if first in _ARTICLES and len(display.split()) > 1:
        display = display.split(" ", 1)[1]
    if display.islower():
        display = display[0].upper() + display[1:]
    terms[key] = {"display": display, "notes": []}
    return display


def register_note(index: dict, terms: list[str], note_stem: str) -> None:
    """Record which episode each canonical term appears in."""
    entries = index.setdefault("terms", {})
    for t in terms:
        entry = entries.get(fold(t))
        if entry is None:
            continue
        notes = entry.setdefault("notes", [])
        if note_stem not in notes:
            notes.append(note_stem)
            notes.sort()


def prune(index: dict, existing: set[str]) -> None:
    """Drop references to notes that no longer exist, and terms left orphaned.
    `existing` is every episode note's stem, wherever in the vault it lives."""
    entries = index.get("terms", {})
    for key in list(entries):
        entries[key]["notes"] = [n for n in entries[key].get("notes", []) if n in existing]
        if not entries[key]["notes"]:
            del entries[key]


def write_hub_notes(out_dir: Path, index: dict, folder: str, language: str,
                    min_notes: int = 1) -> int:
    """One note per canonical keyword, listing the episodes that mention it.

    These are plain wikilinks, so backlinks and graph view work with no plugins.
    A Dataview block is included as well for anyone who has it.
    """
    from .lang import profile as get_profile
    profile = get_profile(language)
    hub_dir = out_dir / folder
    hub_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    wanted: set[str] = set()
    for entry in index.get("terms", {}).values():
        notes = entry.get("notes", [])
        if len(notes) < min_notes:
            continue
        display = entry["display"]
        name = safe_hub_name(display)
        wanted.add(name)

        lines = ["---", "tags:", "  - hub", f"  - tema/{slug(display)}",
                 f"  - lang/{language}", "---", "", f"# {display}", "",
                 profile.label("mentioned", len(notes)).format(n=len(notes)), ""]
        for stem in notes:
            lines.append(f"- [[{stem}]]")
        lines += ["", "```dataview", "LIST",
                  f'FROM #tema/{slug(display)}', "```", ""]
        (hub_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        written += 1

    # remove hubs for terms that no longer qualify
    for stale in hub_dir.glob("*.md"):
        if stale.stem not in wanted:
            stale.unlink(missing_ok=True)
    return written
