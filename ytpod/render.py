"""Stage 4 - write the Obsidian note.

Frontmatter is designed for Dataview: every note carries structured fields so
you can query the whole library later, e.g.

    TABLE topics, duration FROM #podcast WHERE contains(topics, "foo")
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Config, hms, safe_filename


def _yaml_str(value: str) -> str:
    """Quote a scalar so a colon or quote in a title can't break the frontmatter."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ts_url(base_url: str, seconds: float) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}t={int(seconds)}s"


def _stamp(seconds: float, link: str | None) -> str:
    """`hh:mm:ss`, linked to that moment when the source supports deep links."""
    return f"[`{hms(seconds)}`]({_ts_url(link, seconds)})" if link else f"`{hms(seconds)}`"


_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.rstrip()))


def _paragraphs(segments: list[dict], cfg: Config) -> list[str]:
    """Group segments into readable paragraphs.

    A pause is only a paragraph break if the speaker also finished a sentence -
    people pause mid-sentence constantly, and breaking there looks like a bug.
    The word limit is a safety valve for speakers who never punctuate.
    """
    paras: list[str] = []
    buf: list[str] = []
    words = 0
    prev_end: float | None = None

    for seg in segments:
        gap = seg["start"] - prev_end if prev_end is not None else 0.0
        wants_break = gap > cfg.paragraph_gap_seconds or words > cfg.paragraph_max_words
        can_break = buf and _ends_sentence(buf[-1])
        forced = words > cfg.paragraph_max_words * 2  # never let one run forever

        if buf and wants_break and (can_break or forced):
            paras.append(" ".join(buf))
            buf, words = [], 0

        buf.append(seg["text"])
        words += len(seg["text"].split())
        prev_end = seg["end"]

    if buf:
        paras.append(" ".join(buf))
    return paras


def _paragraphs_with_starts(segments: list[dict], cfg: Config) -> list[tuple[str, float]]:
    """Same grouping as _paragraphs, but keeps each paragraph's start time."""
    out: list[tuple[str, float]] = []
    buf: list[str] = []
    words = 0
    start: float | None = None
    prev_end: float | None = None

    for seg in segments:
        gap = seg["start"] - prev_end if prev_end is not None else 0.0
        wants_break = gap > cfg.paragraph_gap_seconds or words > cfg.paragraph_max_words
        can_break = buf and _ends_sentence(buf[-1])
        forced = words > cfg.paragraph_max_words * 2
        if buf and wants_break and (can_break or forced):
            out.append((" ".join(buf), start if start is not None else 0.0))
            buf, words, start = [], 0, None
        if start is None:
            start = seg["start"]
        buf.append(seg["text"])
        words += len(seg["text"].split())
        prev_end = seg["end"]

    if buf:
        out.append((" ".join(buf), start if start is not None else 0.0))
    return out


def render(meta: dict, segments: list[dict], topics: list[dict], cfg: Config,
           language: str = "en", index: dict | None = None) -> Path:
    from .lang import profile as get_profile
    from . import keywords as kw
    profile = get_profile(language)
    url = meta["url"]
    source = meta.get("source", "youtube")        # caches older than this key are YouTube
    watch = meta.get("link", url)                 # empty for a bare stream: nothing to open
    deep = url if source == "youtube" else None   # only YouTube understands ?t=
    date = meta.get("upload_date", "")
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else ""

    # Snap each raw keyword to the library's canonical vocabulary, so episodes
    # about the same idea link to the same node instead of near-duplicates.
    keywords: list[str] = []
    for t in topics:
        canon_for_topic = []
        for k in t.get("keywords", []):
            canon = (kw.canonicalise(k, index, cfg.keyword_match_threshold)
                     if index is not None else k.strip())
            if canon and canon not in keywords:
                keywords.append(canon)
            if canon and canon not in canon_for_topic:
                canon_for_topic.append(canon)
        t["canonical_keywords"] = canon_for_topic

    K = profile.key
    lines: list[str] = ["---"]
    lines.append(f"{K('title')}: {_yaml_str(meta['title'])}")
    lines.append(f"{K('channel')}: {_yaml_str(meta.get('channel', ''))}")
    lines.append(f"{K('video_id')}: {_yaml_str(meta['video_id'])}")
    lines.append(f"{K('url')}: {_yaml_str(url)}")
    if date_iso:
        lines.append(f"{K('published')}: {date_iso}")
    lines.append(f"{K('duration')}: {_yaml_str(hms(meta.get('duration', 0)))}")
    lines.append(f"{K('language')}: {_yaml_str(profile.code)}")
    lines.append("tags:")
    for tag in (*profile.tags, *cfg.extra_tags):
        lines.append(f"  - {tag}")
    lines.append(f"  - lang/{profile.code}")
    if cfg.keyword_tags:
        for k in keywords[:30]:
            lines.append(f"  - tema/{kw.slug(k)}")
    lines.append(f"{K('topics')}:")
    for t in topics:
        lines.append(f"  - {_yaml_str(t['title'])}")
    if keywords:
        lines.append(f"{K('keywords')}:")
        for k in keywords[:30]:
            lines.append(f"  - {_yaml_str(k)}")
    lines.append("---")
    lines.append("")

    lines.append(f"# {meta['title']}")
    lines.append("")
    byline = []
    if meta.get("channel"):
        byline.append(f"**{meta['channel']}**")
    if watch:
        label = profile.label("watch" if source == "youtube" else "watch_other")
        byline.append(f"[{label}]({watch})")
    if byline:
        lines.append(" · ".join(byline))
        lines.append("")

    if cfg.keyword_links and keywords:
        folder = profile.label("hub_folder")
        links = " · ".join(f"[[{folder}/{kw.safe_hub_name(k)}|{k}]]" for k in keywords[:20])
        lines.append(f"**{profile.label('related')}:** {links}")
        lines.append("")

    # Table of contents - the bit that makes browsing actually work.
    lines.append(f"## {profile.label('contents')}")
    lines.append("")
    for t in topics:
        stamp = hms(t["start"])
        anchor = safe_filename(t["title"])
        play = f" — [▶]({_ts_url(deep, t['start'])})" if deep else ""
        lines.append(f"- `{stamp}` [[#{anchor}|{t['title']}]]{play}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for t in topics:
        anchor = safe_filename(t["title"])
        lines.append(f"## {anchor}")
        lines.append("")
        lines.append(f"{_stamp(t['start'], deep)} – `{hms(t['end'])}`")
        lines.append("")
        if t.get("summary"):
            lines.append(f"> {t['summary']}")
            lines.append("")
        if cfg.keyword_links and t.get("canonical_keywords"):
            folder = profile.label("hub_folder")
            links = " · ".join(f"[[{folder}/{kw.safe_hub_name(k)}|{k}]]"
                               for k in t["canonical_keywords"])
            lines.append(f"*{links}*")
            lines.append("")

        chunk = [s for s in segments if t["start"] <= s["start"] < t["end"]]

        if cfg.inline_mark_seconds:
            # Coarse in-body markers every N seconds. Off by default.
            next_mark = t["start"] + cfg.inline_mark_seconds
            elapsed = t["start"]
            for para, para_start in _paragraphs_with_starts(chunk, cfg):
                if para_start >= next_mark:
                    lines.append(f"*{_stamp(para_start, deep)}*")
                    lines.append("")
                    while next_mark <= para_start:
                        next_mark += cfg.inline_mark_seconds
                lines.append(para)
                lines.append("")
                elapsed = para_start
        else:
            for para in _paragraphs(chunk, cfg):
                lines.append(para)
                lines.append("")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(f"{date_iso + ' - ' if date_iso else ''}{meta['title']}") + ".md"
    path = cfg.out_dir / name
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def rebuild_index(out_dir: Path, language: str | None = None,
                  notes: list[Path] | None = None) -> Path:
    """Regenerate the index note in out_dir. `notes` is the episode list, which
    may span the whole vault; by default, the notes directly in out_dir."""
    from .lang import profile as get_profile
    profile = get_profile(language)
    out_dir.mkdir(parents=True, exist_ok=True)
    if notes is None:
        # Skip any note starting with "_" - that is the index itself, whatever
        # it happens to be called in this language.
        notes = sorted(p for p in out_dir.glob("*.md") if not p.name.startswith("_"))
    # Two copies with one filename in different folders are one [[link]] to Obsidian
    stems = list(dict.fromkeys(n.stem for n in notes))

    heading = profile.label("index_title")
    if language:
        heading += f" — {profile.name_native}"

    lines = ["---", "tags:", "  - index"]
    if language:
        lines.append(f"  - lang/{language}")
    lines += ["---", "", f"# {heading}", "",
              profile.label("episodes", len(stems)).format(n=len(stems)), ""]
    for stem in stems:
        lines.append(f"- [[{stem}]]")
    path = out_dir / profile.index_file
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
