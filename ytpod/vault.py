"""Per-vault settings.

Each vault covers its own subject, so it carries its own vocabulary. The files
live in a hidden `.ytpod` folder at the vault root: Obsidian ignores dot-folders,
and keeping them inside the vault means they sync with it to other machines.

    <vault>/
      .ytpod/
        vocab.txt        every language
        vocab.es.txt     Spanish episodes
        vocab.en.txt     English episodes
      es/  en/           notes, written as before
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = ".ytpod"
VOCAB_FILES = ("vocab.txt", "vocab.es.txt", "vocab.en.txt")

_FALLBACK_HEADER = (
    "# One name or term per line, written exactly how you want it to appear.\n"
    "# Lines starting with # are ignored.\n"
)


def vocab_dir(vault: Path) -> Path:
    return Path(vault) / VOCAB_DIR


def template_text(name: str) -> str:
    """Comment header for a new vocab file, taken from the project's own copy.
    Only the comments are kept: a vault's terms belong to that vault alone."""
    src = PROJECT_ROOT / name
    if src.exists():
        lines = src.read_text(encoding="utf-8").splitlines()
        header = [ln for ln in lines if ln.strip().startswith("#")]
        if header:
            return "\n".join(header) + "\n\n"
    return _FALLBACK_HEADER + "\n"


def ensure_vocab(vault: Path) -> Path:
    """Create the vault's vocab files if missing. Returns the shared vocab.txt,
    which is what Config.vocab_file expects; the language files sit beside it."""
    d = vocab_dir(vault).resolve()
    d.mkdir(parents=True, exist_ok=True)
    for name in VOCAB_FILES:
        f = d / name
        if not f.exists():
            f.write_text(template_text(name), encoding="utf-8")
    return d / VOCAB_FILES[0]
