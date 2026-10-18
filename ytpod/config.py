"""Central configuration. Tweak defaults here rather than hunting through modules."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- paths ---
    # Both default to folders inside the project directory. Point --out at your
    # Obsidian vault only once you're happy with the output.
    work_dir: Path = Path("work")          # audio + JSON intermediates (scratch space)
    out_dir: Path = Path("output")         # the generated .md files

    # --- YouTube auth (stage 1) ---
    # YouTube sometimes demands proof you are not a bot. Either works:
    #   cookies_from_browser = "firefox"        (read live from a logged-in browser)
    #   cookie_file = Path("cookies.txt")       (exported Netscape-format file)
    cookies_from_browser: str | None = None
    cookie_file: Path | None = None

    # Format selectors tried in order until one works. YouTube serves different
    # format sets to different clients, IPs and accounts, so a single selector
    # that works on one machine can fail on another.
    audio_formats: list[str] = field(default_factory=lambda: [
        "bestaudio[ext=m4a]/bestaudio",
        "bestaudio*",
        "worstaudio/worst",
        "best",
    ])
    # Extraction clients. The web client is increasingly served SABR-only
    # streams with no downloadable formats; tv and ios still expose real ones.
    player_clients: list[str] = field(default_factory=lambda: [
        "tv", "ios", "web_safari", "default",
    ])

    # --- other players (stage 1) ---
    # Direct stream links (.m3u8 etc.) usually carry audio muxed into each video
    # rendition, so the smallest rendition with audio is the cheapest download.
    stream_audio_formats: list[str] = field(default_factory=lambda: [
        "bestaudio", "worst[acodec!=none]", "worst",
    ])
    # CDNs that refuse requests without the embedding player as Referer (403).
    # Matched on the end of the host name. --referer overrides for one run.
    referers: dict[str, str] = field(default_factory=lambda: {
        "b-cdn.net": "https://iframe.mediadelivery.net/",          # Bunny Stream
        "mediadelivery.net": "https://iframe.mediadelivery.net/",
    })
    referer: str | None = None

    # --- ASR (stage 2) ---
    whisper_model: str = "large-v3"
    device: str = "cuda"
    strict_gpu: bool = False                # True = fail loudly instead of falling back to CPU
    transcribe_in_subprocess: bool = True   # release ALL VRAM before the LLM stage runs
    compute_type: str = "int8_float16"     # ~2.5 GB VRAM. Use "float16" if you have headroom.
    language: str | None = None            # None = autodetect per episode; or force "es" / "en"
    separate_by_language: bool = True      # write into output/<code>/ subfolders
    beam_size: int = 5

    # --- cleanup (stage 2b) ---
    vocab_file: Path = Path("vocab.txt")   # names/jargon, cased the way you want them
    fix_seam_stutters: bool = True         # remove words duplicated across decode windows
    restore_punctuation: bool = False      # LLM repair of uncased runs. Off: slow, and the
                                           # per-language initial_prompt mostly prevents the
                                           # problem it fixes. Enable with --repunctuate.
    repunct_batch_words: int = 400         # words per LLM request (was 1 segment = ~10 words)
    repunct_ctx: int = 4096

    # --- segmentation (stage 3) ---
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b"            # see README for alternatives
    llm_num_ctx: int = 4096    # a 14-min window needs ~1.7k tokens; the KV cache is
                               # allocated for the full value, and over-provisioning
                               # here is what pushes Ollama off the GPU on small cards
    block_seconds: int = 45                # granularity of the outline fed to the LLM
    window_minutes: int = 14               # how much of the outline the LLM sees at once
    overlap_minutes: int = 3               # overlap so topics spanning a window edge survive
    min_topic_seconds: int = 150           # merge anything shorter than this into its neighbour

    # --- condensing (optional stage 5) ---
    # Each chapter rewritten as condensed study notes, with the full transcript
    # kept below it in a collapsed callout. Enable with --condense.
    condense: bool = False
    condense_chunk_words: int = 800    # words per LLM request
    condense_min_words: int = 80       # shorter chapters are already short: left as-is
    # qwen3 must reason before it condenses well (with thinking off it narrates
    # its reasoning into the answer, or copies the text). Reasoning needs room:
    # ~1.2k tokens in, up to ~4k thinking, the notes after. 8k context for
    # qwen3:4b is ~3.9 GB, still all on a 6 GB card.
    condense_num_ctx: int = 8192
    condense_num_predict: int = 6000
    # Model for condensing; None = same as llm_model. The non-thinking instruct
    # build answers directly: on a 14-min class, 31 s and 39% length with no
    # failures, against 12 min and frequent failures for qwen3:4b, which has to
    # reason first. Install with: ollama pull qwen3:4b-instruct-2507-q4_K_M
    condense_model: str | None = "qwen3:4b-instruct-2507-q4_K_M"

    # --- rendering (stage 4) ---
    inline_mark_seconds: int = 0           # 0 = off. Set to e.g. 300 for a [mm:ss] every 5 min.
    paragraph_gap_seconds: float = 1.6     # silence longer than this starts a new paragraph
    paragraph_max_words: int = 110         # ...or this many words, whichever comes first
    extra_tags: list[str] = field(default_factory=list)  # added on top of the language defaults

    # --- cross-episode linking ---
    keyword_links: bool = True         # wikilinks to keyword hub notes (backlinks + graph)
    keyword_tags: bool = True          # also emit #tema/<keyword> nested tags
    keyword_hubs: bool = True          # generate one hub note per keyword
    hub_min_notes: int = 1             # only create a hub once N episodes mention it
    keyword_match_threshold: float = 0.88   # lower = merges more aggressively


def hms(seconds: float) -> str:
    """3725.4 -> '01:02:05'"""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def safe_filename(name: str, max_len: int = 120) -> str:
    """Windows is fussy about filenames; Obsidian is fussy about square brackets."""
    bad = '<>:"/\\|?*[]#^'
    cleaned = "".join("-" if c in bad else c for c in name)
    cleaned = " ".join(cleaned.split())          # collapse whitespace
    cleaned = cleaned.strip(" .")                # Windows hates trailing dots/spaces
    return cleaned[:max_len] or "untitled"
