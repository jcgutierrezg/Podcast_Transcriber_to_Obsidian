"""Small desktop front end: pick a vault, give it a video (or a batch), run,
watch the log.

The pipeline runs as a child process (`python -m ytpod.cli --vault ...`), so the
window stays responsive and Stop can end the whole run, including the
transcription child it spawns. Launch with `uv run ytpod-gui`, or the
`.venv\\Scripts\\ytpod-gui.exe` that opens without a console window.
"""

from __future__ import annotations

import codecs
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import Config
from .vault import PROJECT_ROOT, VOCAB_FILES, template_text, vocab_dir

SETTINGS_FILE = PROJECT_ROOT / ".ytpod-gui.json"
LANGS = ("auto", "es", "en")
# How much of the cached work to throw away. Each step implies the ones after it.
RERUN = (
    ("Nothing - reuse what's cached", []),
    ("Topics only", ["--redo", "segment"]),
    ("Cleanup + topics (after vocab edits)", ["--redo", "clean"]),
    ("Everything - transcribe again", ["--redo", "transcribe"]),
)
TOPIC_SOURCES = (
    ("AI (Ollama)", []),
    ("Video's own chapters (YouTube)", ["--yt-chapters"]),
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SKIP_DIRS = {"temas", "topics"}  # keyword hub folders, not places for episodes
VOCAB_LABELS = {
    "vocab.txt": "vocab.txt  (every language)",
    "vocab.es.txt": "vocab.es.txt  (Spanish)",
    "vocab.en.txt": "vocab.en.txt  (English)",
}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _python_exe() -> str:
    """The console interpreter. Under the windowless launcher sys.executable is
    pythonw.exe, whose children would have no stdout to stream."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(data: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    except OSError:
        pass  # remembering the last vault is a convenience, never worth an error


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.out_q: queue.Queue = queue.Queue()
        self._cr = False           # a bare \r arrived: next text overwrites the line
        self._stopping = False
        self._llm_names: list[str] | None = None
        self.vocab_name = VOCAB_FILES[0]
        self.settings = _load_settings()
        self.settings.setdefault("vaults", {})

        root.title("ytpod")
        # sizes are for a 96-dpi screen; tk scaling is 1.333 there (pixels per point)
        k = float(root.tk.call("tk", "scaling")) / (96 / 72)
        root.geometry(f"{int(900 * k)}x{int(780 * k)}")
        root.minsize(int(720 * k), int(600 * k))
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(4, weight=1)

        # vault
        ttk.Label(frm, text="Vault").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.vault_var = tk.StringVar(value=self.settings.get("last_vault", ""))
        self.vault_box = ttk.Combobox(frm, textvariable=self.vault_var,
                                      values=self._recent_vaults())
        self.vault_box.grid(row=0, column=1, sticky="ew")
        self.vault_box.bind("<<ComboboxSelected>>", lambda e: self.on_vault_changed())
        self.vault_box.bind("<FocusOut>", lambda e: self.on_vault_changed())
        ttk.Button(frm, text="Browse...", command=self.browse).grid(row=0, column=2, padx=(8, 0))

        # mode
        modes = ttk.Frame(frm)
        modes.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.mode_var = tk.StringVar(value=self.settings.get("mode", "single"))
        ttk.Radiobutton(modes, text="Single video", value="single", variable=self.mode_var,
                        command=self.on_mode_changed).pack(side="left")
        ttk.Radiobutton(modes, text="Batch (several URLs)", value="batch",
                        variable=self.mode_var, command=self.on_mode_changed
                        ).pack(side="left", padx=(16, 0))

        # the two episode panels share one grid cell; only one is shown
        self.single_frame = self._build_single(frm)
        self.batch_frame = self._build_batch(frm)
        for f in (self.single_frame, self.batch_frame):
            f.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        self._build_options(frm).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        # tabs: log + vocabulary
        self.tabs = ttk.Notebook(frm)
        self.tabs.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

        log_tab = ttk.Frame(self.tabs)
        self.log = ScrolledText(log_tab, wrap="word", state="disabled",
                                font=("Consolas", 9), height=10)
        self.log.pack(fill="both", expand=True)
        self.tabs.add(log_tab, text="Log")

        vocab_tab = ttk.Frame(self.tabs, padding=(0, 6, 0, 0))
        vocab_tab.columnconfigure(1, weight=1)
        vocab_tab.rowconfigure(1, weight=1)
        ttk.Label(vocab_tab, text="File").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.vocab_choice = ttk.Combobox(vocab_tab, state="readonly",
                                         values=[VOCAB_LABELS[n] for n in VOCAB_FILES])
        self.vocab_choice.current(0)
        self.vocab_choice.grid(row=0, column=1, sticky="w")
        self.vocab_choice.bind("<<ComboboxSelected>>", lambda e: self.on_vocab_file_changed())
        self.vocab_status = ttk.Label(vocab_tab, foreground="gray")
        self.vocab_status.grid(row=0, column=2, sticky="e", padx=8)
        ttk.Button(vocab_tab, text="Save", command=self.save_vocab).grid(row=0, column=3)
        self.vocab_text = tk.Text(vocab_tab, wrap="none", undo=True, font=("Consolas", 10))
        self.vocab_text.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        self.vocab_text.bind("<<Modified>>", lambda e: self._update_vocab_status())
        self.tabs.add(vocab_tab, text="Vocabulary")

        # controls
        bar = ttk.Frame(frm)
        bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.status = ttk.Label(bar, foreground="gray")
        self.status.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.run_btn = ttk.Button(bar, text="Run", command=self.run)
        self.run_btn.pack(side="right", padx=(0, 6))
        self.doctor_btn = ttk.Button(bar, text="Check setup", command=self.doctor)
        self.doctor_btn.pack(side="right", padx=(0, 6))

        self.on_mode_changed()
        self.on_vault_changed()
        threading.Thread(target=self._fetch_llm_models, daemon=True).start()
        root.after(50, self._drain)

    # ---------------------------------------------------------------- layout

    def _build_single(self, parent) -> ttk.Frame:
        f = ttk.Frame(parent)
        f.columnconfigure(1, weight=1)
        hint = {"foreground": "gray"}

        ttk.Label(f, text="Video URL").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.url_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.url_var).grid(row=0, column=1, columnspan=3, sticky="ew")

        ttk.Label(f, text="Title").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.title_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.title_var).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(f, text="Date").grid(row=1, column=2, sticky="e", padx=(12, 6), pady=(6, 0))
        self.date_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.date_var, width=12).grid(row=1, column=3, sticky="w",
                                                                pady=(6, 0))
        ttk.Label(f, text="Blank title = the video's own (player links have none, so name "
                          "those). Date as YYYY-MM-DD, optional.", **hint
                  ).grid(row=2, column=1, columnspan=3, sticky="w")

        ttk.Label(f, text="Folder").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.folder_var = tk.StringVar()
        self.folder_box = ttk.Combobox(f, textvariable=self.folder_var)
        self.folder_box.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(f, text="Choose...", command=self.choose_folder
                   ).grid(row=3, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Label(f, text="Inside the vault, e.g. Module 2 or es/Juan - created if missing. "
                          "Blank = the language folder.", **hint
                  ).grid(row=4, column=1, columnspan=3, sticky="w")

        ttk.Label(f, text="Referer").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.referer_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.referer_var).grid(row=5, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(f, text="Only if a player link is refused (403). Bunny needs none.", **hint
                  ).grid(row=5, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(6, 0))
        return f

    def _build_batch(self, parent) -> ttk.Frame:
        f = ttk.Frame(parent)
        f.columnconfigure(0, weight=1)
        self.urls = tk.Text(f, height=6, wrap="none", undo=True)
        self.urls.grid(row=0, column=0, sticky="ew")
        ttk.Label(f, text="One URL per line, optionally  URL | title | YYYY-MM-DD | folder  "
                          "with any field blank (URL | | | Juan).  # skips a line.",
                  foreground="gray").grid(row=1, column=0, sticky="w")
        return f

    def _build_options(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="Options", padding=(8, 4))
        ttk.Label(f, text="Language").grid(row=0, column=0, sticky="w")
        self.lang_var = tk.StringVar(value="auto")
        ttk.Combobox(f, textvariable=self.lang_var, values=LANGS, width=6,
                     state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(f, text="Re-run").grid(row=0, column=2, sticky="w")
        self.rerun_box = ttk.Combobox(f, state="readonly", width=28,
                                      values=[label for label, _ in RERUN])
        self.rerun_box.current(0)
        self.rerun_box.grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(f, text="Topics from").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.topics_box = ttk.Combobox(f, state="readonly", width=30,
                                       values=[label for label, _ in TOPIC_SOURCES])
        self.topics_box.current(0)
        self.topics_box.grid(row=1, column=1, sticky="w", padx=(6, 18), pady=(6, 0))

        ttk.Label(f, text="AI model").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.llm_var = tk.StringVar(value=self.settings.get("llm") or Config().llm_model)
        self.llm_box = ttk.Combobox(f, textvariable=self.llm_var, width=26)
        self.llm_box.grid(row=1, column=3, sticky="w", padx=(6, 18), pady=(6, 0))
        return f

    def on_mode_changed(self) -> None:
        single = self.mode_var.get() == "single"
        (self.batch_frame if single else self.single_frame).grid_remove()
        (self.single_frame if single else self.batch_frame).grid()
        self.settings["mode"] = self.mode_var.get()

    def _fetch_llm_models(self) -> None:
        """Fill the model dropdown with what Ollama actually has installed."""
        try:
            with urllib.request.urlopen(f"{Config().ollama_url}/api/tags", timeout=2) as r:
                names = sorted(m["name"] for m in json.load(r).get("models", []))
        except Exception:
            return  # Ollama not running: the box stays editable with the default
        self._llm_names = names  # picked up by _drain - Tk isn't thread-safe

    # ---------------------------------------------------------------- vault

    def _recent_vaults(self) -> list[str]:
        return self.settings.get("recent", [])

    def vault(self) -> Path | None:
        v = self.vault_var.get().strip().strip('"')
        return Path(v) if v else None

    def browse(self) -> None:
        start = self.vault_var.get() or str(Path.home())
        chosen = filedialog.askdirectory(title="Choose the Obsidian vault", initialdir=start,
                                         mustexist=True)
        if chosen:
            self.vault_var.set(str(Path(chosen)))
            self.on_vault_changed()

    def on_vault_changed(self) -> None:
        vault = self.vault()
        key = str(vault) if vault else ""
        if key == getattr(self, "_loaded_vault", None):
            return
        if not self._confirm_vocab_discard():
            self.vault_var.set(self._loaded_vault or "")
            return
        self._loaded_vault = key
        lang = self.settings["vaults"].get(key, {}).get("lang", "auto")
        self.lang_var.set(lang if lang in LANGS else "auto")
        self.folder_box["values"] = self._vault_folders(vault) if vault else []
        self.load_vocab()

    def _vault_folders(self, vault: Path, depth: int = 3) -> list[str]:
        """Existing folders, for the Folder dropdown. Skips .obsidian, .ytpod, hubs."""
        out: list[str] = []

        def walk(d: Path, rel: str, level: int) -> None:
            try:
                children = sorted((c for c in d.iterdir() if c.is_dir()), key=lambda c: c.name.lower())
            except OSError:
                return
            for c in children:
                if c.name.startswith(".") or c.name in _SKIP_DIRS:
                    continue
                path = f"{rel}/{c.name}" if rel else c.name
                out.append(path)
                if level < depth:
                    walk(c, path, level + 1)

        if vault.is_dir():
            walk(vault, "", 1)
        return out

    def choose_folder(self) -> None:
        vault = self.vault()
        if vault is None or not vault.is_dir():
            messagebox.showwarning("ytpod", "Choose the vault first.")
            return
        start = vault / self.folder_var.get() if self.folder_var.get() else vault
        chosen = filedialog.askdirectory(title="Folder inside the vault for this note",
                                         initialdir=str(start if start.is_dir() else vault))
        if not chosen:
            return
        try:
            rel = Path(chosen).resolve().relative_to(vault.resolve())
        except ValueError:
            messagebox.showerror("ytpod", "Pick a folder inside the vault.")
            return
        # the vault itself isn't a sorting folder; blank already means the default
        self.folder_var.set("" if rel == Path(".") else rel.as_posix())

    def _remember_vault(self, vault: Path) -> None:
        key = str(vault)
        recent = [key] + [v for v in self._recent_vaults() if v != key]
        self.settings["recent"] = recent[:10]
        self.settings["last_vault"] = key
        self.settings["vaults"].setdefault(key, {})["lang"] = self.lang_var.get()
        self.settings["llm"] = self.llm_var.get().strip()
        self.vault_box["values"] = self.settings["recent"]
        _save_settings(self.settings)

    # ---------------------------------------------------------------- vocab

    def _vocab_path(self) -> Path | None:
        vault = self.vault()
        return vocab_dir(vault) / self.vocab_name if vault else None

    def load_vocab(self) -> None:
        path = self._vocab_path()
        self.vocab_text.delete("1.0", "end")
        if path is None:
            self._vocab_saved = None
        elif path.exists():
            self.vocab_text.insert("1.0", path.read_text(encoding="utf-8"))
            self._vocab_saved = True
        else:
            self.vocab_text.insert("1.0", template_text(self.vocab_name))
            self._vocab_saved = False
        self.vocab_text.edit_reset()
        self.vocab_text.edit_modified(False)
        self._update_vocab_status()

    def _update_vocab_status(self) -> None:
        if self._vocab_saved is None:
            text = "choose a vault first"
        elif self.vocab_text.edit_modified():
            text = "unsaved changes"
        elif not self._vocab_saved:
            text = "new - created on Save or on the first run"
        else:
            text = str(self._vocab_path().parent)
        self.vocab_status.config(text=text)

    def save_vocab(self) -> bool:
        path = self._vocab_path()
        if path is None:
            messagebox.showwarning("ytpod", "Choose a vault first.")
            return False
        if not path.parent.parent.is_dir():
            messagebox.showerror("ytpod", f"Vault folder not found:\n{path.parent.parent}")
            return False
        path.parent.mkdir(exist_ok=True)
        text = self.vocab_text.get("1.0", "end-1c")
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        self._vocab_saved = True
        self.vocab_text.edit_modified(False)
        self._update_vocab_status()
        return True

    def _confirm_vocab_discard(self) -> bool:
        """True to go ahead. Offers to save edits first; Cancel keeps them."""
        if not self.vocab_text.edit_modified():
            return True
        answer = messagebox.askyesnocancel("ytpod", f"Save changes to {self.vocab_name}?")
        if answer is None:
            return False
        return self.save_vocab() if answer else True

    def on_vocab_file_changed(self) -> None:
        new = VOCAB_FILES[self.vocab_choice.current()]
        if new == self.vocab_name:
            return
        if not self._confirm_vocab_discard():
            self.vocab_choice.current(VOCAB_FILES.index(self.vocab_name))
            return
        self.vocab_name = new
        self.load_vocab()

    # ---------------------------------------------------------------- running

    def run(self) -> None:
        vault = self.vault()
        if vault is None:
            messagebox.showwarning("ytpod", "Choose a vault first.")
            return
        if not vault.is_dir():
            messagebox.showerror("ytpod", f"Vault folder not found:\n{vault}")
            return
        if self.mode_var.get() == "single":
            episode = self._single_args()
            if episode is None:
                return
            label = f"Running 1 episode into {vault.name}"
        else:
            episode = [ln.strip() for ln in self.urls.get("1.0", "end").splitlines()
                       if ln.strip() and not ln.strip().startswith("#")]
            if not episode:
                messagebox.showwarning("ytpod", "Paste at least one video URL.")
                return
            label = f"Running {len(episode)} episode(s) into {vault.name}"
        if not self._confirm_vocab_discard():
            return
        self._remember_vault(vault)
        args = [*episode, "--vault", str(vault), "--lang", self.lang_var.get()]
        args += RERUN[self.rerun_box.current()][1]
        args += TOPIC_SOURCES[self.topics_box.current()][1]
        if self.llm_var.get().strip():
            args += ["--llm", self.llm_var.get().strip()]
        self._start(args, label)

    def _single_args(self) -> list[str] | None:
        """The single-video fields as CLI arguments, or None after telling the
        user what's wrong. Checked here so a typo never starts a download."""
        from .library import safe_folder
        url = self.url_var.get().strip()
        if not url.lower().startswith(("http://", "https://")):
            messagebox.showwarning("ytpod", "Paste the video's URL (starting with https://).")
            return None
        if "|" in url or " " in url:
            messagebox.showwarning("ytpod", "The URL box takes the URL only - title, date and "
                                            "folder have their own fields.")
            return None
        args = [url]
        title = self.title_var.get().strip()
        if title:
            args += ["--title", title]
        date = self.date_var.get().strip()
        if date:
            if not _DATE.match(date):
                messagebox.showwarning("ytpod", f"Date must be YYYY-MM-DD, e.g. 2026-10-11.\n"
                                                f"Got: {date}")
                return None
            args += ["--date", date]
        folder = self.folder_var.get().strip()
        if folder:
            try:
                args += ["--folder", safe_folder(folder)]
            except ValueError as exc:
                messagebox.showwarning("ytpod", str(exc))
                return None
        referer = self.referer_var.get().strip()
        if referer:
            args += ["--referer", referer]
        return args

    def doctor(self) -> None:
        self._start(["--doctor"], "Checking setup")

    def _start(self, args: list[str], label: str) -> None:
        if self.proc:
            return
        cmd = [_python_exe(), "-u", "-m", "ytpod.cli", *args]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        from .fetch import display_url
        self._stopping = False
        self._clear_log()
        # player links carry access tokens; echo them without
        shown = [display_url(a.split("|", 1)[0].strip()) + (" | ..." if "|" in a else "")
                 if a.startswith("http") else a for a in args]
        shown = [f'"{a}"' if " " in a else a for a in shown]
        self._append(f"> ytpod {' '.join(shown)}\n\n")
        try:
            self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
        except OSError as exc:
            self._append(f"could not start: {exc}\n")
            return
        self.tabs.select(0)
        self._set_running(True, label)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := proc.stdout.read1(4096):
            self.out_q.put(decoder.decode(chunk))
        self.out_q.put(decoder.decode(b"", final=True))
        self.out_q.put(("done", proc.wait()))

    def stop(self) -> None:
        if not self.proc:
            return
        self._stopping = True
        self._append("\n--- stopping ---\n")
        if sys.platform == "win32":
            # /T takes the transcription child down too, freeing the GPU.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True, creationflags=_NO_WINDOW)
        else:
            self.proc.terminate()

    def _drain(self) -> None:
        try:
            while True:
                item = self.out_q.get_nowait()
                if isinstance(item, tuple):
                    code = item[1]
                    if self._stopping:
                        self._append("\n--- stopped ---\n")
                        label = "stopped"
                    else:
                        self._append(f"\n--- finished (exit {code}) ---\n")
                        label = "done" if code == 0 else f"failed (exit {code}) - see log"
                    self.proc = None
                    self._set_running(False, label)
                    vault = self.vault()
                    if vault:  # a run may have created a new folder
                        self.folder_box["values"] = self._vault_folders(vault)
                else:
                    self._append(item)
        except queue.Empty:
            pass
        if self._llm_names is not None:
            self.llm_box.configure(values=self._llm_names)
            self._llm_names = None
        self.root.after(50, self._drain)

    def _set_running(self, running: bool, label: str) -> None:
        self.run_btn.config(state="disabled" if running else "normal")
        self.doctor_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")
        self.status.config(text=label)

    # ---------------------------------------------------------------- log

    def _clear_log(self) -> None:
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self._cr = False

    def _append(self, text: str) -> None:
        """Progress bars redraw with a bare \\r; honour that by overwriting the
        current line rather than printing every frame."""
        self.log.config(state="normal")
        for part in re.split(r"(\r\n|\r|\n)", text):
            if part in ("\n", "\r\n"):
                self.log.insert("end", "\n")
                self._cr = False
            elif part == "\r":
                self._cr = True
            elif part:
                if self._cr:
                    self.log.delete("end-1c linestart", "end-1c")
                    self._cr = False
                self.log.insert("end", part)
        self.log.see("end")
        self.log.config(state="disabled")

    def log_text(self) -> str:
        return self.log.get("1.0", "end-1c")

    # ---------------------------------------------------------------- close

    def on_close(self) -> None:
        if self.proc and not messagebox.askokcancel("ytpod", "A run is in progress. Stop it and quit?"):
            return
        if not self._confirm_vocab_discard():
            return
        self.stop()
        self.settings["llm"] = self.llm_var.get().strip()
        _save_settings(self.settings)  # mode and model choice
        self.root.destroy()


def dpi_aware() -> None:
    """Render at the screen's real resolution. Without this, Windows stretches
    the window on a scaled display (125%, 150%) and the text comes out blurry."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main() -> int:
    dpi_aware()
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
