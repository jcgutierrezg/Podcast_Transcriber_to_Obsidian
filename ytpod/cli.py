"""Command line entry point.

Every stage caches its output as JSON in work/<video_id>/. Re-running is cheap:
only missing stages execute. Use --redo to force a specific stage, which is how
you iterate on topic segmentation without re-transcribing an hour of audio.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import Config

STAGES = ("fetch", "transcribe", "clean", "segment", "render")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_transcript(path: Path) -> dict | None:
    """Accept both the current {language, segments} shape and the older bare list."""
    data = _load(path)
    if data is None:
        return None
    if isinstance(data, list):  # transcripts made before language support
        return {"language": "en", "segments": data}
    return data


_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


Entry = tuple[str, "str | None", "str | None", "str | None"]   # url, title, YYYYMMDD, folder


def parse_date(text: str) -> str:
    """'YYYY-MM-DD' -> 'YYYYMMDD'."""
    m = _DATE.match(text.strip())
    if not m:
        raise ValueError(f"date must be YYYY-MM-DD, got {text!r}")
    return "".join(m.groups())


def parse_entry(entry: str) -> Entry:
    """'URL | title | YYYY-MM-DD | folder' -> (url, title, YYYYMMDD, folder).
    Every field after the URL is optional and may be left empty ('URL | | | Juan').
    Title and date matter for player links, which carry neither; folder is
    relative to the vault root."""
    from .library import safe_folder
    url, *rest = [part.strip() for part in entry.split("|")]
    rest += [""] * (3 - len(rest))
    if len(rest) > 3:
        raise ValueError("too many '|' fields - expected URL | title | date | folder")
    title_s, date_s, folder_s = rest
    return (url, title_s or None, parse_date(date_s) if date_s else None,
            safe_folder(folder_s) if folder_s else None)


def process(entry: Entry, cfg: Config, redo: set[str],
            use_yt_chapters: bool) -> tuple[Path, Path, str]:
    """Run one episode. Returns (note path, its language's library folder, language)."""
    from . import clean as clean_mod
    from . import fetch as fetch_mod
    from . import keywords as kw_mod
    from . import library
    from . import render as render_mod
    from . import segment as segment_mod
    from . import transcribe as transcribe_mod

    t0 = time.time()
    url, title, date, folder = entry
    print(f"\n=== {fetch_mod.display_url(url)}")

    meta = fetch_mod.fetch(url, cfg.work_dir, cfg, title=title, date=date)
    work = cfg.work_dir / meta["video_id"]
    print(f"  {meta['title']}  ({meta['duration'] / 60:.0f} min)")

    raw = None if "transcribe" in redo else _load_transcript(work / "transcript.json")
    if raw is None:
        raw = transcribe_mod.transcribe(meta["audio_file"], work, cfg)
    else:
        print(f"  transcript cached ({len(raw['segments'])} segments, {raw['language']})")

    language = raw["language"]

    segments = None if redo & {"transcribe", "clean"} else _load(work / "transcript.clean.json")
    if segments is None:
        segments = clean_mod.clean(raw["segments"], work, cfg, language)
    else:
        print("  cleaned transcript cached")

    # Each language gets its own folder, so an Obsidian search never mixes them.
    ep_cfg = cfg
    if cfg.separate_by_language:
        ep_cfg = replace(cfg, out_dir=cfg.out_dir / language)

    # The canonical vocabulary is both an input to segmentation and an output
    # of rendering, so it is loaded before one and saved after the other.
    index = kw_mod.load_index(ep_cfg.out_dir)
    known = kw_mod.known_terms(index)

    topics = None if redo & {"transcribe", "clean", "segment"} else _load(work / "topics.json")
    if topics is None:
        topics = segment_mod.segment(segments, meta, work, cfg, language,
                                     use_yt_chapters, known)
    else:
        print(f"  topics cached ({len(topics)} topics)")

    # The keyword index and hubs stay with the language folder; only the note
    # itself moves into the requested folder (relative to the vault root).
    note_cfg = replace(ep_cfg, out_dir=cfg.out_dir / folder) if folder else ep_cfg
    path = render_mod.render(meta, segments, topics, note_cfg, language, index)
    all_kw = sorted({k for t in topics for k in t.get("canonical_keywords", [])})
    kw_mod.register_note(index, all_kw, path.stem)
    kw_mod.save_index(ep_cfg.out_dir, index)

    # A new title, date or folder means a new file. Never delete the old one -
    # it may be a hand-edited copy - but say where it is.
    for other in library.notes_for_video(cfg.out_dir, meta["video_id"]):
        if other.resolve() != path.resolve():
            print(f"  note: another note for this video exists: {other.relative_to(cfg.out_dir)}")

    new = sum(1 for k in all_kw if k not in known)
    print(f"  -> {path}")
    print(f"     {len(topics)} topics, {len(all_kw)} keywords ({new} new), "
          f"{language}, {time.time() - t0:.0f}s")
    return path, ep_cfg.out_dir, language


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ytpod", description=__doc__)
    p.add_argument("sources", nargs="*", metavar="source",
                   help="video URL(s), or a .txt file with one per line. YouTube or any "
                        "player link yt-dlp reads (e.g. a .m3u8). Optionally "
                        "\"URL | title | YYYY-MM-DD\" to name an episode.")
    p.add_argument("--doctor", action="store_true",
                   help="check the environment (CUDA, Ollama, model weights) and exit")
    p.add_argument("--vault", type=Path,
                   help="Obsidian vault to write into; also uses that vault's own "
                        "vocabulary from <vault>/.ytpod/")
    p.add_argument("--out", type=Path, help="output folder (default: ./output)")
    p.add_argument("--work", type=Path, help="scratch folder for audio + JSON (default: ./work)")
    p.add_argument("--model", help="whisper model, e.g. large-v3 / medium / distil-large-v3")
    p.add_argument("--llm", help="ollama model tag for segmentation")
    p.add_argument("--lang", help="force language code, e.g. es or en. Omit to autodetect.")
    p.add_argument("--no-split-lang", action="store_true",
                   help="write all languages into one folder instead of output/<code>/")
    p.add_argument("--redo", action="append", choices=STAGES, default=[],
                   help="force a stage to re-run (repeatable)")
    p.add_argument("--strict-gpu", action="store_true",
                   help="fail if the GPU is unavailable instead of falling back to CPU")
    p.add_argument("--cpu", action="store_true", help="force CPU transcription")
    p.add_argument("--list-formats", action="store_true",
                   help="show the formats YouTube offers this machine, then exit")
    p.add_argument("--format", metavar="SELECTOR",
                   help="override the yt-dlp format selector for this run")
    p.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="read cookies from a logged-in browser: firefox, chrome, edge, "
                        "brave, opera, vivaldi, safari. Optionally 'firefox:ProfileName'.")
    p.add_argument("--cookies", type=Path, metavar="FILE",
                   help="path to an exported cookies.txt (Netscape format)")
    p.add_argument("--referer", metavar="URL",
                   help="Referer header for player links whose CDN returns 403 without "
                        "it (Bunny is handled automatically)")
    p.add_argument("--repunctuate", action="store_true",
                   help="LLM pass to restore punctuation in uncased spans (slow)")
    p.add_argument("--no-repunctuate", action="store_true",
                   help="disable the punctuation pass even if enabled in config")
    p.add_argument("--yt-chapters", action="store_true",
                   help="use the uploader's own chapters instead of the LLM")
    p.add_argument("--keep-audio", action="store_true", help="don't delete audio when done")
    single = p.add_argument_group(
        "single episode", "Name one episode without the '|' syntax (one URL only).")
    single.add_argument("--title", help="note title (default: the video's own)")
    single.add_argument("--date", metavar="YYYY-MM-DD", help="published date for the note")
    single.add_argument("--folder", help="folder inside the vault, e.g. 'Module 2' or 'es/Juan'")
    args = p.parse_args(argv)

    cfg = Config()
    if args.vault and args.out:
        p.error("--vault already sets the output folder; use one or the other")
    if args.out:
        cfg.out_dir = args.out
    if args.work:
        cfg.work_dir = args.work
    if args.model:
        cfg.whisper_model = args.model
    if args.llm:
        cfg.llm_model = args.llm
    if args.lang:
        cfg.language = None if args.lang == "auto" else args.lang
    if args.no_split_lang:
        cfg.separate_by_language = False
    if args.strict_gpu:
        cfg.strict_gpu = True
    if args.cpu:
        cfg.device, cfg.compute_type = "cpu", "int8"
    if args.format:
        cfg.audio_formats = [args.format]
    if args.cookies_from_browser:
        cfg.cookies_from_browser = args.cookies_from_browser
    if args.cookies:
        cfg.cookie_file = args.cookies.resolve()
    if args.referer:
        cfg.referer = args.referer
    if args.repunctuate:
        cfg.restore_punctuation = True
    if args.no_repunctuate:
        cfg.restore_punctuation = False

    if args.doctor:
        from .doctor import doctor
        return doctor(cfg)

    if not args.sources:
        p.error("a URL or .txt file is required (or use --doctor)")

    if args.list_formats:
        from . import fetch as fetch_mod
        for source in args.sources:
            fetch_mod.list_formats(parse_entry(source)[0], cfg)
        return 0

    urls = []
    for source in args.sources:
        src = Path(source)
        if src.exists() and src.suffix.lower() == ".txt":
            urls += [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
        else:
            urls.append(source)

    # Catch a mistyped date or folder now, not an hour into a batch.
    from .fetch import display_url
    from .library import safe_folder
    entries: list[Entry] = []
    for line in urls:
        try:
            entries.append(parse_entry(line))
        except ValueError as exc:
            p.error(f"{exc}\n  in: {display_url(line.split('|', 1)[0].strip())[:100]}")

    if args.title or args.date or args.folder:
        if len(entries) != 1:
            p.error("--title / --date / --folder name a single episode; "
                    "for several, use 'URL | title | date | folder' lines")
        url, title, date, folder = entries[0]
        try:
            entries[0] = (url, args.title or title,
                          parse_date(args.date) if args.date else date,
                          safe_folder(args.folder) if args.folder else folder)
        except ValueError as exc:
            p.error(str(exc))

    if args.vault:
        from .vault import ensure_vocab
        cfg.out_dir = args.vault
        cfg.vocab_file = ensure_vocab(args.vault)

    # Resolve and create upfront, so a failure mid-run can't leave us writing
    # into a folder that was never made.
    cfg.work_dir = cfg.work_dir.resolve()
    cfg.out_dir = cfg.out_dir.resolve()
    cfg.vocab_file = cfg.vocab_file.resolve()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {cfg.out_dir}")
    print(f"work:   {cfg.work_dir}")
    print(f"vocab:  {cfg.vocab_file.parent}")

    failures = []
    written: dict[Path, str | None] = {}
    for entry in entries:
        try:
            _, lib_dir, language = process(entry, cfg, set(args.redo), args.yt_chapters)
            written[lib_dir] = language if cfg.separate_by_language else None
        except Exception as exc:
            # yt-dlp errors quote the URL; keep a player link's token out of the log
            raw = entry[0]
            shown = display_url(raw)
            err = str(exc).replace(raw, shown)
            print(f"  FAILED: {err}")
            failures.append((shown, err))

    if not args.keep_audio:
        for audio in cfg.work_dir.glob("*/audio.*"):
            if audio.suffix != ".json":
                audio.unlink(missing_ok=True)

    from . import render as render_mod
    from . import keywords as kw_mod
    from . import library
    from .lang import profile as get_profile
    # Nothing written: refresh the existing folders only. Writing a root index
    # here would leave an empty "_Podcast Index.md" beside the es/ en/ folders.
    if not written and not cfg.separate_by_language:
        written = {cfg.out_dir: None}
    for folder, code in sorted(written.items()):
        # Episodes may sit in any folder of the vault - sorted there by the
        # folder field, or moved by hand - so find them by frontmatter.
        notes = library.episode_notes(cfg.out_dir, code)
        index_path = render_mod.rebuild_index(folder, code, notes)
        print(f"\nIndex: {index_path}  ({len({n.stem for n in notes})} episodes)")
        if cfg.keyword_hubs:
            kw_index = kw_mod.load_index(folder)
            kw_mod.prune(kw_index, {n.stem for n in notes})
            kw_mod.save_index(folder, kw_index)
            n = kw_mod.write_hub_notes(
                folder, kw_index, get_profile(code).label("hub_folder"),
                code or "und", cfg.hub_min_notes)
            print(f"Hubs:  {n} keyword notes in {folder / get_profile(code).label('hub_folder')}")
    if failures:
        print(f"\n{len(failures)} failed:")
        for url, err in failures:
            print(f"  {url}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
