# ytpod

YouTube podcast → topic-segmented Obsidian markdown. Runs entirely on your machine.

## Setup (Windows, uv)

From the folder containing `pyproject.toml`:

```powershell
uv sync
```

That single command creates the virtual environment, installs the dependencies,
installs this project in editable mode, and pulls in the cuBLAS/cuDNN DLLs that
CTranslate2 needs for GPU transcription. No activation step required.

The NVIDIA libraries are ordinary dependencies, so no later `uv sync` or
`uv add` can prune them. `uv sync --extra cuda` from older notes still works;
the extra is now an empty alias.

The Python version is pinned in `.python-version`, so uv won't quietly move the
environment to a new interpreter (which breaks the `.venv\Scripts` launchers
with "uv trampoline failed to spawn python child process").

**CPU only?** Open `ytpod/config.py` and set `device = "cpu"` and
`compute_type = "int8"`. Roughly 5–8× slower, still fine overnight.

### Ollama (topic segmentation)

Install Ollama from ollama.com, then:

```powershell
ollama pull qwen3:4b
```

On a sub-8 GB card `qwen3:4b` at 16k context is the safe choice. If you have
headroom, `qwen3:8b` gives noticeably better topic titles — pass `--llm qwen3:8b`.
Check what you actually have with `ollama list`.

## Desktop app

```powershell
uv run ytpod-gui
```

Or double-click `.venv\Scripts\ytpod-gui.exe`, which opens without a console
window. Right-click it and choose *Pin to taskbar*, or send a shortcut to the
desktop, for one-click launching.

- **Vault**: *Browse...* picks the folder. Recent vaults stay in the dropdown,
  and each one remembers its own language setting.
- **Single video** (the default): one URL, with its own fields:
  - **Title**: leave blank to use the video's own. Player links have none, so
    name those.
  - **Date**: `YYYY-MM-DD`, optional.
  - **Folder**: inside the vault. Pick an existing one from the dropdown, browse
    with *Choose...*, or type a new one, which is created if missing.
  - **Referer**: only for a player link that's refused. Bunny needs none.
- **Batch**: several URLs, one per line, using the `URL | title | date | folder`
  format (see "Sorting episodes into folders").
- **Options**, shared by both modes:
  - **Language**: auto, or force `es` / `en`.
  - **Re-run**: how much cached work to redo. *Topics only* re-segments,
    *Cleanup + topics* applies vocab edits, *Everything* transcribes again.
  - **Topics from**: the AI, or a YouTube video's own chapters.
  - **AI model**: any model Ollama has installed. The dropdown lists them.
- **Vocabulary** tab: edits the selected vault's own vocab files (see below).
- **Run** streams the log live. **Stop** ends the whole run, including
  transcription, so the GPU is freed. **Check setup** runs `--doctor`.

Bad input (a malformed date, a folder outside the vault, a title pasted into the
URL box) is caught before anything downloads. The app remembers your mode and
AI model between sessions.

The app uses the same pipeline as the command line. Single mode is
`ytpod <url> --vault <folder> --title ... --date ... --folder ...`, so a title
can contain `|` without confusing anything.

## Usage

`uv run` resolves the environment automatically, so these work from any fresh
terminal without activating anything:

By default everything lands in `output/` inside the project folder. Once you're
happy with the results, point `--out` at your Obsidian vault instead.

```powershell
# into a vault, using that vault's own vocabulary
uv run ytpod "https://..." --vault "D:\Vaults\Atlas"

# one episode, named and sorted
uv run ytpod "https://..." --vault "D:\Vaults\Curso" --title "Lección 1" --date 2026-10-11 --folder "Módulo 1"

# several URLs at once
uv run ytpod "https://...1" "https://...2" --vault "D:\Vaults\Atlas"

# single episode -> ./output
uv run ytpod "https://www.youtube.com/watch?v=..."

# batch: one URL per line in urls.txt, # for comments
uv run ytpod urls.txt

# straight into your vault
uv run ytpod "https://..." --out "D:\Vault\Podcasts"

# re-run only the topic segmentation (transcript is cached, costs nothing)
uv run ytpod "https://..." --redo clean      # re-clean, reusing the transcript
uv run ytpod "https://..." --redo segment --llm qwen3:8b

# use the uploader's own chapters instead of the LLM
uv run ytpod "https://..." --yt-chapters
```

Always quote the URL — YouTube URLs contain `&`, which PowerShell treats as a
command separator otherwise.

`uv run python -m ytpod.cli ...` does exactly the same thing if you prefer the
explicit form.

### Adding dependencies later

```powershell
uv add somepackage          # installs AND records it in pyproject.toml
```

Prefer this over `uv pip install`, which only mutates the environment and leaves
`pyproject.toml` out of date.

## Sorting episodes into folders

Each line takes up to four fields, separated by `|`. Only the URL is required,
and any field can be left blank:

```
URL | title | YYYY-MM-DD | folder
```

```
https://www.youtube.com/watch?v=... | | | Juan
https://vz-....b-cdn.net/.../playlist.m3u8 | Enseñanza: Juan 15 | 2026-10-11 | Juan
https://www.youtube.com/watch?v=... | | | Series/Hechos
```

The folder is relative to the **vault root**, and is created if missing. `Juan`
puts the note in `<vault>\Juan\`. For a folder under the language folder, write
`es/Juan`. Without a folder, notes go to `<vault>\es\` as before. Folders can't
leave the vault: `..` and drive letters are rejected before the run starts.

Only the note moves. The keyword vocabulary (`_keywords.json`), the hub notes
(`temas/`) and the index stay in the language folder, and they cover episodes in
every folder.

**Moving notes by hand is safe too.** The index and keyword hubs find episodes
anywhere in the vault by their frontmatter (`id_video` plus the `lang/es` tag),
not by their folder. A run never forgets a note you dragged into another folder.

If a re-run would write a note under a new name (a different title, date or
folder), the old one is left alone. ytpod never deletes notes, since one might
be your hand-edited copy, and the log names it so you can tidy up.

## How it works

| Stage | Output | Cached as |
|---|---|---|
| 1. fetch | audio + metadata via yt-dlp | `work/<id>/meta.json` |
| 2. transcribe | timestamped segments via faster-whisper | `work/<id>/transcript.json` |
| 2b. clean | seam repair, casing, punctuation | `work/<id>/transcript.clean.json` |
| 3. segment | topic boundaries via Ollama | `work/<id>/topics.json` |
| 4. render | Obsidian note | `output/<date> - <title>.md` |

Stages are skipped if their cache exists. `--redo segment` invalidates downstream
stages only, so you can iterate on prompts and models without re-transcribing.

The LLM never sees the raw transcript — it sees a compressed outline, one line per
45 s block. An hour becomes ~80 lines instead of ~13k tokens.

Whisper and the LLM cannot share a sub-8 GB card, so stage 2 explicitly unloads the
model before stage 3 starts.

## First run: what to expect

The very first run downloads the Whisper weights (~3 GB for `large-v3`) from
Hugging Face into your user cache. This happens once, ever. You'll see a
progress bar for it. Subsequent runs skip straight to loading.

A typical run looks like:

```
=== https://www.youtube.com/watch?v=...
  fetching metadata... done
  Some Podcast Episode Title  (58 min)
  audio: downloaded
  model: Systran/faster-whisper-large-v3
  loading model onto cuda (int8_float16)... ready
  language: en (99%)
  transcribing: 100%|##########| 3480/3480s [08:12<00:00] 1243 segments
  segmenting: 100%|##########| 5/5 windows [01:44<00:00] 9 candidates
  -> vault\2026-01-14 - Some Podcast Episode Title.md  [7 topics, 612s]
```

## Is it actually using the GPU?

**Windows Task Manager will lie to you here.** Its GPU graphs default to *3D*
and *Copy*; CUDA work appears under **Compute_0**. Click a graph title and
change it, or watch the *Dedicated GPU memory* figure instead.

Better, don't rely on it. Every run prints the device it loaded onto and the
speed it achieved:

```
  DEVICE IN USE: cuda (int8_float16)
  ...
  13.0 min of audio in 1.4 min = 9.3x realtime on cuda
```

Speed is the honest test. `large-v3` on a GPU manages several times realtime;
CPU `int8` rarely reaches 1x. If a 13-minute video takes 13 minutes, it is on
the CPU whatever any graph says.

If the GPU is unavailable, ytpod falls back to CPU and prints a loud banner
saying so. To fail instead of falling back:

```powershell
uv run ytpod "https://..." --strict-gpu
```

`uv run ytpod --doctor` now attempts a real model load on the GPU, which is the
only check that catches a fallback — the library checks can all pass while the
actual load still fails. Use `--cpu` to force CPU deliberately.

## Is the LLM on the GPU?

Same question as for Whisper, different process. Before segmenting, ytpod asks
Ollama where it actually loaded the model:

```
  LLM: on GPU (3.1 GB, ctx 4096)
```

If it went to the CPU you get a banner instead. Three things cause it:

**1. `llm_num_ctx` too high.** The KV cache is allocated for the *whole* context
whether you use it or not. A 14-minute window needs about 1.7k tokens; the
default is 4096, with headroom. Raising it to 16k wastes roughly 9x the memory
and is the most common reason Ollama falls back to CPU on a small card.

**2. Something else holding VRAM.** ytpod runs transcription in a child process
precisely so this cannot be itself — CTranslate2 frees the model on delete, but
the CUDA context lives as long as the process, so the parent would still be
sitting on VRAM while Ollama tried to load. Letting the child die returns all of
it. Disable with `transcribe_in_subprocess = False` if it ever causes trouble.

**3. The model is simply too big.** `qwen3:8b` at q4 needs ~5 GB before context.
On a 6 GB card with a browser open, that will not fit. `qwen3:4b` will.

Each run also reports seconds per window, so a slow one is obvious without
watching Task Manager.

## Expected runtime

On a 6 GB NVIDIA card, per 1 hour episode: roughly 6–12 min transcription,
1–3 min segmentation.

## YouTube sign-in checks

On some machines and networks YouTube responds with:

```
Sign in to confirm you're not a bot.
```

Pass cookies from a browser where you are logged into YouTube:

```powershell
uv run ytpod "https://..." --cookies-from-browser firefox
```

Supported: `firefox`, `chrome`, `edge`, `brave`, `opera`, `vivaldi`, `safari`.
Add a profile if you use more than one: `--cookies-from-browser "firefox:default"`.

To avoid typing it every run, set it once in `ytpod/config.py`:

```python
cookies_from_browser: str | None = "firefox"
```

**Firefox is the reliable choice on Windows.** Chrome and Edge encrypt their
cookie store (App-Bound Encryption, Chrome 127+), and yt-dlp often cannot read
it while the browser is running — sometimes not at all. If you must use Chrome,
close it completely first.

**If reading the browser fails**, export a `cookies.txt` with a Netscape-format
cookie extension while logged into YouTube, then:

```powershell
uv run ytpod "https://..." --cookies cookies.txt
```

Cookies are applied to both the metadata probe and the download — the check can
fire on either. Treat a `cookies.txt` as a password: it grants access to your
account. It is already in `.gitignore`.

## Other players (Bunny Stream, .m3u8 links)

Not just YouTube: any link yt-dlp can read works, including a direct stream
copied out of a web player, such as Bunny Stream's
`https://vz-....b-cdn.net/.../playlist.m3u8`.

**Getting the link.** Open the page with the player, press F12, go to the
*Network* tab, filter for `m3u8`, and play the video. Right-click the
`playlist.m3u8` request and choose *Copy URL*.

**Name it.** A raw stream carries no title or date, so add them on the same
line. Both are optional, and the date sets the note's filename and `publicado`:

```
https://vz-....b-cdn.net/.../playlist.m3u8 | Enseñanza: Juan 15 | 2026-10-11
```

This works the same in the app, on the command line, and in a `urls.txt`.

What ytpod does for you:

- **Referer.** Bunny refuses requests that don't come from its player (HTTP
  403). ytpod sends the expected Referer for `b-cdn.net` automatically. For
  another CDN that needs one, pass `--referer "<page URL>"` or add it to
  `referers` in `ytpod/config.py`.
- **Small download.** These streams mix audio into every video quality, so the
  smallest one with audio is downloaded. Only its audio is used.
- **Expiring links.** The link is signed with a token that expires within hours,
  so run it soon after copying. If you get *refused (HTTP 403)*, copy a fresh
  link. The video's id comes from the GUID in the path, so a fresh link for the
  same video still reuses everything already cached. `--redo segment` works
  after the link has expired, because it doesn't touch the network.
- **No tokens in your notes.** The token is stripped before the URL is printed
  or stored. The note's `url` field keeps the clean address for reference, and
  there's no *Watch* link or per-topic ▶ links, since a bare stream has no page
  to open.

## "Requested format is not available"

Usually means one of two things.

**1. yt-dlp is out of date.** This is the most common difference between two
machines. YouTube changes its streaming setup often, and the version pinned when
you first ran `uv sync` may already be stale:

```powershell
uv lock --upgrade-package yt-dlp
uv sync
```

Check what each machine has with `uv run ytpod --doctor`.

**2. YouTube served that machine a format set with no downloadable audio.** The
web client is increasingly given SABR-only streams. ytpod asks for the `tv` and
`ios` clients first, which still expose real formats, and falls back through
four format selectors before giving up.

To see what your machine is actually being offered:

```powershell
uv run ytpod --list-formats "https://www.youtube.com/watch?v=..."
```

Then force a specific one if you spot a workable audio format:

```powershell
uv run ytpod "https://..." --format 140
```

Both the client list and the fallback chain are in `ytpod/config.py` as
`player_clients` and `audio_formats`.

Note that cookies can *change* which formats you are offered — a logged-in
session sometimes gets a different set than an anonymous one. If the format
error appeared right after you added `--cookies-from-browser`, try it both ways.

## Languages

Language is detected per episode and everything downstream follows it: the
Whisper formatting primer, the casing rules, the punctuation repair prompt, and
the language topic titles are written in.

```powershell
uv run ytpod "https://..."                # autodetect (default)
uv run ytpod "https://..." --lang es      # force Spanish, skips detection
uv run ytpod "https://..." --lang auto    # explicit autodetect
```

Output is split by language:

```
output/
  es/  _Podcast Index.md  +  Spanish episodes
  en/  _Podcast Index.md  +  English episodes
```

Each note carries a `lang/es` tag, so Obsidian searches never mix the two. Pass
`--no-split-lang` for a single folder.

**A Spanish episode produces a fully Spanish note.** Not just the transcript and
topic titles, but the headings, link labels, tags and frontmatter keys:

```yaml
---
título: "Episodio 42: La ciencia y el café"
canal: "Café Científico"
publicado: 2026-03-01
duración: "01:12:40"
idioma: "es"
tags:
  - pódcast
  - transcripción
  - lang/es
temas:
  - "Introducción al programa"
---

# Episodio 42: La ciencia y el café

**Café Científico** · [Ver en YouTube](...)

## Contenido
```

The index note is `_Índice de pódcast.md` and counts episodes with correct
singular/plural agreement.

One field is deliberately **not** translated: `tags`. Obsidian indexes that key
by name, so renaming it to `etiquetas` would silently break tag search. Every
other key is localised in `ytpod/lang.py` under `keys`.

**The trade-off with localised keys:** a Dataview query has to name the language's
keys, so one query can't span both folders. If you would rather query everything
at once, empty the `keys` dict in the Spanish profile — the note body stays
Spanish, only the frontmatter field names revert to English.

Tuned profiles exist for Spanish and English (`ytpod/lang.py`). Other languages
still work — Whisper handles 99 — but fall back to neutral defaults with no
formatting primer. Adding one is a single entry in `PROFILES`.

**Forcing the language is worth it when you know it.** Detection reads only the
first 30 seconds, so an English cold-open on a Spanish podcast can mislead it.
For a batch that is all one language, `--lang es` is both faster and safer.

## Connecting episodes

Every note also carries per-keyword tags and wikilinks, so episodes about the
same idea find each other.

```
output/es/
  _Índice de pódcast.md
  _keywords.json                  <- the canonical vocabulary
  2026-03-01 - Episodio 42.md
  temas/
    Silencio.md                   <- hub: every episode mentioning it
    Trauma.md
```

Each episode note gets:

- nested tags — `tema/silencio`, `tema/trauma` — so `#tema/silencio` finds them all
- a **Temas** line of wikilinks under the title, and per-topic links under each
  section heading, so backlinks and graph view work with no plugins installed

### Keyword drift, and why the index exists

Left alone, an LLM will write "silencio" in one episode, "el silencio" in the
next and "silencios" in a third, and the graph fragments into near-duplicate
nodes describing one idea.

`_keywords.json` is a canonical vocabulary that grows with your library, and it
works both directions:

1. **Before segmentation** the established terms are shown to the LLM, which is
   asked to reuse them wherever they fit.
2. **After segmentation** anything new is fuzzy-matched against the vocabulary
   and snapped to an existing term if close enough. Leading articles are
   stripped and accents folded first, so "el silencio", "Silencios" and
   "silenció" all land on `Silencio`.

Deleting notes is safe: stale references are pruned and orphaned hubs removed on
the next run.

### Tuning the vocabulary

- `keyword_match_threshold` (default `0.88`) — lower merges more aggressively,
  higher keeps more distinct terms. If unrelated ideas are being merged, raise it.
- `hub_min_notes` — set to `2` to only create a hub once a term appears in more
  than one episode, which keeps the folder clean on a large library.
- `keyword_tags` / `keyword_links` / `keyword_hubs` — turn each mechanism off
  independently.

You can edit `_keywords.json` by hand to merge or rename terms; the display name
is what appears in notes, and the next run picks it up.

**Cross-language linking is deliberately absent** — Spanish and English keyword
vocabularies are separate, since they live in separate folders. If you want a
Spanish episode to link to an English one on the same subject, add the shared
term to both and link the two hub notes manually.

## Transcript cleanup

Whisper's raw output has three recurring defects, all fixed in stage 2b:

**Seam stutters** — a word emitted at the end of one decode window and again at
the start of the next ("this didn't start" / "art recently"). Repair is gated on
the silence between segments: an artefact has no gap because both segments cover
the same instant of audio, whereas a speaker genuinely repeating themselves
leaves an audible pause. That timing check is what stops "the heart" / "art is
subjective" being mangled.

**Lost casing and punctuation** — long uncased runs, caused by
`condition_on_previous_text=False` (which we need, to stop hallucination loops).
The fix is the per-language `initial_prompt`, which primes Whisper with
well-formatted text so it has a habit to imitate. This is prevention, and it
handles most cases on its own.

There is also an optional LLM repair pass for anything that still slips through.
**It is off by default** — it costs real time, and the primer usually makes it
unnecessary. Enable per run:

```powershell
uv run ytpod "https://..." --redo clean --repunctuate
```

The repair pass **cannot change your words**. Its output is compared against the
original word-for-word and discarded entirely if a single word differs. The
comparison ignores case, punctuation and accents — restoring `canción` from
`cancion` is the correction we asked for — but any added, dropped, reordered or
substituted word rejects the whole span. This also blocks the model from
quietly translating instead of repunctuating. The run reports how many spans
were rejected.

**Mid-sentence paragraph breaks** — a pause now only starts a new paragraph if
the speaker also finished a sentence.

### Vocabulary files

One name or term per line, cased and accented how you want it. Used twice: fed
to Whisper as a recognition hint, and applied afterwards to force consistent
casing.

| File | Applies to |
|---|---|
| `vocab.txt` | every language |
| `vocab.es.txt` | Spanish episodes |
| `vocab.en.txt` | English episodes |

The shared file plus the matching language file are combined. Keep the total
under ~40 entries — Whisper's prompt window is small.

**Each vault has its own set.** With `--vault`, or from the desktop app, the
files are read from a hidden `.ytpod` folder at the vault root. That way a vault
of sermons and a vault of tech podcasts never share terms:

```
<vault>/
  .ytpod/
    vocab.txt  vocab.es.txt  vocab.en.txt
  es/  en/
```

They're created empty on first use. Obsidian ignores dot-folders, and the files
sync with the vault to your other machines. The files in the project folder are
used only when you run without `--vault`.

Vocab is applied when transcribing and cleaning, and both are cached. After you
edit a vault's vocab, re-run an episode it should affect with `--redo clean`
(the casing fix) or `--redo transcribe` (the Whisper hint too).

Accents matter here: writing `Martín Fernández` in `vocab.es.txt` fixes it
everywhere Whisper wrote `Martin Fernandez`.

## Tuning

Everything lives in `ytpod/config.py`:

- `language` — set to `"es"` to skip detection entirely if your library is single-language.
- `separate_by_language` — set `False` to pool all languages in one folder.
- `restore_punctuation` — set `True` to enable the LLM repair pass by default.
- `repunct_batch_words` — words per LLM request. Lower if a small model loses track over long spans.
- `fix_seam_stutters` — set `False` to keep the transcript strictly as Whisper emitted it.
- `min_topic_seconds` — raise to 240 if you're getting too many topics, lower to 90 for finer granularity.
- `block_seconds` — outline granularity. Lower = more precise boundaries, more tokens.
- `inline_mark_seconds` — set to `300` for a timestamp link every 5 min inside the body text.
- `whisper_model` — `distil-large-v3` is ~2× faster with a small accuracy cost if throughput matters more than perfection.

## Troubleshooting

Run the built-in check first — it tests CUDA, Ollama, and the model cache in the
order they'd break, and works even when the environment is broken:

```powershell
uv run ytpod --doctor
```

| Symptom | Cause |
|---|---|
| `cublas64_12.dll is not found or cannot be loaded` | NVIDIA wheels missing from the environment — re-run `uv sync` |
| `Could not locate cudnn_ops64_9.dll` | Same cause as above |
| `The server refused the request (HTTP 403)` on a player link | The link's token expired — copy a fresh one (see "Other players") |
| `Connection refused` on port 11434 | Ollama isn't running — launch the app or `ollama serve` |
| `Sign in to confirm you're not a bot` | Add `--cookies-from-browser firefox` (see above) |
| `Requested format is not available` | Update yt-dlp first: `uv lock --upgrade-package yt-dlp`, then `uv sync` (see above) |
| Topics look wrong | `uv run ytpod <url> --redo segment --llm qwen3:8b` — reuses the cached transcript |
| Segmenting pegs the CPU | Ollama loaded on CPU — see "Is the LLM on the GPU?" above |
| Repunctuation ETA is enormous | It is off by default now. If you enabled it, raise `repunct_batch_words` or use a smaller LLM. |
| Long silent pause, high network | First-run model download. Wait it out; it only happens once. |

## Obsidian

Frontmatter is Dataview-ready. Drop this into a note to browse your library:

````
```dataview
TABLE channel, duration, topics
FROM #podcast
SORT published DESC
```

For Spanish notes, use the localised keys:

```dataview
TABLE canal, duración, temas
FROM #pódcast
SORT publicado DESC
```
````
