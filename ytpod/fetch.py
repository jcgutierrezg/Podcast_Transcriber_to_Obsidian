"""Stage 1 - pull audio + metadata from YouTube, or any other player yt-dlp reads.

We deliberately download the raw bestaudio stream and do NOT transcode it.
faster-whisper decodes m4a/opus/webm directly via PyAV, so ffmpeg is optional.

Besides YouTube, a direct stream link works too - e.g. a Bunny Stream
`.../playlist.m3u8` copied out of a web player. Those links carry no title or
date, and are usually signed with an expiring token, so:

  - the video id comes from the URL itself (the GUID in Bunny's path), which
    lets re-runs reuse the cache without touching the network after the token
    has expired
  - title and date come from the user: "URL | title | YYYY-MM-DD"
  - the token is stripped before the URL is stored or printed
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yt_dlp

_MEDIA_EXT = (".m3u8", ".mpd", ".mp4", ".m4a", ".mp3", ".webm", ".aac", ".ogg", ".wav")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# Names generic sites give a stream when they have nothing better to say.
_PLACEHOLDER_TITLES = {"playlist", "index", "master", "video", "manifest", "stream"}


def _audio_path(work: Path) -> Path | None:
    for p in sorted(work.glob("audio.*")):
        # .part / -FragN files are an interrupted download, not usable audio
        if p.suffix not in (".json", ".ytdl") and ".part" not in p.name:
            return p
    return None


def is_youtube(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com"))


def is_direct_stream(url: str) -> bool:
    """A media file or manifest, as opposed to a page yt-dlp has to scrape."""
    return urlsplit(url).path.lower().endswith(_MEDIA_EXT)


def clean_url(url: str) -> str:
    """The URL minus anything that grants access: query string, fragment, and
    path segments like Bunny's `bcdn_token=...`. Safe to log and to store."""
    parts = urlsplit(url)
    path = "/".join(seg for seg in parts.path.split("/") if "token=" not in seg)
    return urlunsplit((parts.scheme, parts.netloc, re.sub(r"/{2,}", "/", path), "", ""))


def stable_id(url: str) -> str:
    """An id for a direct stream that survives the token changing."""
    clean = clean_url(url)
    m = _UUID.search(urlsplit(clean).path)
    return m.group(0).lower() if m else hashlib.sha1(clean.encode()).hexdigest()[:12]


def display_url(url: str) -> str:
    return url if is_youtube(url) else clean_url(url)


def client_opts(cfg) -> dict:
    """Which YouTube extraction clients to try."""
    clients = getattr(cfg, "player_clients", None)
    if not clients:
        return {}
    return {"extractor_args": {"youtube": {"player_client": list(clients)}}}


def auth_opts(cfg) -> dict:
    """yt-dlp options for getting past YouTube's bot check.

    Applied to BOTH the metadata probe and the download - the check can fire on
    either, and half-authenticated is the same as not authenticated.
    """
    opts: dict = {}
    if cfg.cookie_file:
        opts["cookiefile"] = str(cfg.cookie_file)
    elif cfg.cookies_from_browser:
        # yt-dlp wants (browser, profile, keyring, container)
        parts = cfg.cookies_from_browser.split(":")
        browser = parts[0].strip().lower()
        profile = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        opts["cookiesfrombrowser"] = (browser, profile, None, None)
    return opts


def header_opts(url: str, cfg) -> dict:
    """Referer for CDNs that only serve requests coming from their own player."""
    if cfg is None:
        return {}
    referer = cfg.referer
    if not referer:
        host = urlsplit(url).hostname or ""
        referer = next((r for suffix, r in cfg.referers.items() if host.endswith(suffix)), None)
    return {"http_headers": {"Referer": referer}} if referer else {}


def _progress_hook(d: dict) -> None:
    """yt-dlp calls this regardless of quiet=True, so we get progress without its noise."""
    if d.get("status") == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        done = d.get("downloaded_bytes", 0)
        speed = (d.get("speed") or 0) / 1e6
        if total:
            print(
                f"\r  audio: {100 * done / total:5.1f}%  "
                f"({done / 1e6:6.1f}/{total / 1e6:.1f} MB, {speed:4.1f} MB/s)",
                end="", flush=True,
            )
        else:
            print(f"\r  audio: {done / 1e6:6.1f} MB ({speed:4.1f} MB/s)", end="", flush=True)
    elif d.get("status") == "finished":
        print("\r  audio: downloaded" + " " * 40)


class _Silent:
    """yt-dlp logger for the download pass. It prints 'ERROR: Requested format
    is not available' before raising, even with quiet=True - expected noise while
    walking the format fallbacks. Failures still surface as exceptions."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _apply_overrides(meta: dict, title: str | None, date: str | None) -> dict:
    if title:
        meta["title"] = title
    if date:
        meta["upload_date"] = date
    return meta


def _cached_stream_meta(url: str, work_root: Path) -> dict | None:
    """For a direct stream that has already been fetched, skip the network:
    its signed link has often expired by the time you re-run --redo segment."""
    work = work_root / stable_id(url)
    meta_file = work / "meta.json"
    if not meta_file.exists():
        return None
    if _audio_path(work) is None and not (work / "transcript.json").exists():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _refused_hint(url: str) -> str:
    return (
        "The server refused the request (HTTP 403).\n"
        "  Signed player links expire - copy a fresh one from the player and retry.\n"
        "  If it is fresh, the CDN may want a different Referer: --referer \"<page URL>\"\n"
        f"  ({clean_url(url)})"
    )


def fetch(url: str, work_root: Path, cfg=None, title: str | None = None,
          date: str | None = None) -> dict:
    """Download audio into work_root/<video_id>/ and return the metadata dict.
    `title` / `date` (YYYYMMDD) override whatever the source reports."""
    youtube = is_youtube(url)
    stream = not youtube and is_direct_stream(url)

    if stream:
        cached = _cached_stream_meta(url, work_root)
        if cached is not None:
            print("  metadata cached")
            meta = _apply_overrides(cached, title, date)
            (work_root / meta["video_id"] / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            return meta

    auth = auth_opts(cfg) if cfg is not None else {}
    clients = client_opts(cfg) if cfg is not None and youtube else {}
    headers = header_opts(url, cfg)
    if auth:
        which = auth.get("cookiefile") or auth["cookiesfrombrowser"][0]
        print(f"  cookies: {which}")
    if headers:
        print(f"  referer: {headers['http_headers']['Referer']}")

    # First pass: metadata only, so we know the video id before creating folders.
    print("  fetching metadata...", end="", flush=True)
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True,
                               **clients, **auth, **headers}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        if "not a bot" in str(exc) or "Sign in to confirm" in str(exc):
            raise RuntimeError(
                "YouTube demanded sign-in verification. Pass cookies from a browser "
                "where you are logged in, e.g.:\n"
                "  uv run ytpod \"<url>\" --cookies-from-browser firefox\n"
                "or export a cookies.txt and use --cookies cookies.txt"
            ) from exc
        if not youtube and "403" in str(exc):
            raise RuntimeError(_refused_hint(url)) from None
        raise
    print(" done")

    vid = stable_id(url) if stream else info["id"]
    work = work_root / vid
    work.mkdir(parents=True, exist_ok=True)

    reported = info.get("title") or ""
    if stream and (reported.lower() in _PLACEHOLDER_TITLES or reported == info.get("id")):
        reported = ""
    meta = {
        "video_id": vid,
        "title": reported or vid,
        "channel": info.get("uploader") or info.get("channel") or "",
        "upload_date": info.get("upload_date") or "",          # YYYYMMDD
        "duration": float(info.get("duration") or 0.0),
        # Never the signed link: tokens are credentials, and they expire anyway.
        "url": info.get("webpage_url", url) if youtube else clean_url(url),
        # What a "watch" link points at; empty for a bare stream.
        "link": "" if stream else info.get("webpage_url", url),
        "source": "youtube" if youtube else ("stream" if stream else "web"),
        "description": info.get("description", "") or "",
        # Uploader-authored chapters, if any. Free ground truth worth having.
        "yt_chapters": [
            {"start": float(c["start_time"]), "end": float(c["end_time"]), "title": c["title"]}
            for c in (info.get("chapters") or [])
        ],
    }
    _apply_overrides(meta, title, date)
    if stream and not title:
        print("  note: this stream has no title - add one as \"URL | title | YYYY-MM-DD\"")

    existing = _audio_path(work)
    if existing is None:
        if youtube:
            formats = list(getattr(cfg, "audio_formats", None) or ["bestaudio/best"])
        else:
            formats = list(getattr(cfg, "stream_audio_formats", None) or ["bestaudio/worst"])
        # YouTube's canonical page URL drops playlist parameters; anything else
        # must be fetched exactly as given, token included.
        source = meta["url"] if youtube else url
        last_error: Exception | None = None
        for i, fmt in enumerate(formats):
            opts = {
                "format": fmt,
                "outtmpl": str(work / "audio.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "logger": _Silent(),
                "progress_hooks": [_progress_hook],
                **clients,
                **auth,
                **headers,
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([source])
                if i:
                    print(f"  (used fallback format selector: {fmt})")
                break
            except Exception as exc:
                last_error = exc
                if not youtube and "403" in str(exc):
                    raise RuntimeError(_refused_hint(url)) from None
                if "Requested format is not available" not in str(exc):
                    raise
                print(f"\r  format '{fmt}' unavailable, trying next..." + " " * 20)
        existing = _audio_path(work)
        if existing is None:
            raise RuntimeError(
                f"No downloadable audio format found. Last error: {last_error}\n"
                f"  Diagnose with: uv run ytpod --list-formats \"{display_url(url)}\"\n"
                f"  Most often this means yt-dlp is out of date: "
                f"uv lock --upgrade-package yt-dlp, then uv sync"
            )
    else:
        print(f"  audio: cached ({existing.stat().st_size / 1e6:.1f} MB)")

    if existing is None:
        raise RuntimeError(f"audio download failed for {display_url(url)}")

    meta["audio_file"] = str(existing)
    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def list_formats(url: str, cfg) -> None:
    """Print what the source is actually offering this machine. Diagnostic only."""
    print(f"yt-dlp {yt_dlp.version.__version__}")
    opts = {"quiet": True, "skip_download": True, "listformats": True,
            **(client_opts(cfg) if is_youtube(url) else {}), **auth_opts(cfg),
            **header_opts(url, cfg)}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=False)
