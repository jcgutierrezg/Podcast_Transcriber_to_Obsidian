"""Language profiles.

Everything language-specific lives here rather than being scattered through the
pipeline. Adding a language means adding one entry below, not hunting for
hardcoded English through four modules.

The `initial_prompt` matters more than it looks. Whisper imitates the formatting
of whatever context it is given, so the primer must be in the target language -
priming Spanish audio with English text degrades accents and punctuation, and
can nudge the model toward translating rather than transcribing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LangProfile:
    code: str
    name_en: str                  # for prompting the LLM
    name_native: str              # for display
    initial_prompt: str
    titles: tuple[str, ...] = ()  # abbreviations to capitalise + dot
    pronoun_i: bool = False       # English "i" -> "I" repair
    vocab_suffix: str = ""        # vocab.<suffix>.txt
    ui: dict = field(default_factory=dict)      # headings and link labels
    tags: tuple[str, ...] = ()                  # frontmatter tags
    keys: dict = field(default_factory=dict)    # frontmatter key names
    index_file: str = "_Podcast Index.md"

    @property
    def vocab_name(self) -> str:
        return f"vocab.{self.vocab_suffix or self.code}.txt"

    def label(self, key: str, n: int | None = None) -> str:
        """Look up a UI string. Pass n for anything that inflects for number -
        Spanish needs 'episodio transcrito' vs 'episodios transcritos'."""
        if n == 1 and f"{key}_one" in self.ui:
            key = f"{key}_one"
        elif n == 1 and f"{key}_one" in _EN_UI and key not in self.ui:
            key = f"{key}_one"
        return self.ui.get(key, _EN_UI.get(key, _EN_UI[key.removesuffix("_one")]))

    def key(self, name: str) -> str:
        """Localised frontmatter key. 'tags' is never translated - Obsidian
        indexes that field by name, and renaming it breaks tag search."""
        if name == "tags":
            return "tags"
        return self.keys.get(name, name)


_EN_UI = {
    "contents": "Contents",
    "watch": "Watch on YouTube",
    "watch_other": "Watch video",
    "full_transcript": "Full transcript",
    "index_title": "Podcast Index",
    "episodes": "{n} episodes transcribed.",
    "episodes_one": "{n} episode transcribed.",
    "hub_folder": "topics",
    "related": "Topics",
    "mentioned": "Mentioned in {n} episodes:",
    "mentioned_one": "Mentioned in {n} episode:",
}


ENGLISH = LangProfile(
    code="en",
    name_en="English",
    name_native="English",
    initial_prompt=(
        "Hello, and welcome back to the show. What follows is a clearly "
        "punctuated transcript, with correct capitalisation and full sentences."
    ),
    titles=("dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st"),
    pronoun_i=True,
    ui=dict(_EN_UI),
    tags=("podcast", "transcript"),
)

SPANISH = LangProfile(
    code="es",
    name_en="Spanish",
    name_native="Español",
    initial_prompt=(
        "Hola, y bienvenidos de nuevo al programa. Lo que sigue es una "
        "transcripción claramente puntuada, con mayúsculas y acentos correctos, "
        "signos de interrogación y exclamación de apertura, y frases completas."
    ),
    titles=("dr", "dra", "sr", "sra", "srta", "prof", "lic", "ing"),
    pronoun_i=False,
    ui={
        "contents": "Contenido",
        "watch": "Ver en YouTube",
        "watch_other": "Ver vídeo",
        "full_transcript": "Transcripción completa",
        "index_title": "Índice de pódcast",
        "episodes": "{n} episodios transcritos.",
        "episodes_one": "{n} episodio transcrito.",
        "hub_folder": "temas",
        "related": "Temas",
        "mentioned": "Mencionado en {n} episodios:",
        "mentioned_one": "Mencionado en {n} episodio:",
    },
    tags=("pódcast", "transcripción"),
    keys={
        "title": "título",
        "channel": "canal",
        "video_id": "id_video",
        "published": "publicado",
        "duration": "duración",
        "language": "idioma",
        "topics": "temas",
        "keywords": "palabras_clave",
    },
    index_file="_Índice de pódcast.md",
)

PROFILES: dict[str, LangProfile] = {p.code: p for p in (ENGLISH, SPANISH)}


def profile(code: str | None) -> LangProfile:
    """Fall back to a neutral profile for languages we haven't tuned."""
    if code and code in PROFILES:
        return PROFILES[code]
    return LangProfile(
        code=code or "und",
        name_en=(code or "the source language"),
        name_native=(code or "unknown"),
        initial_prompt="",
        titles=(),
        pronoun_i=False,
        ui=dict(_EN_UI),
        tags=("podcast", "transcript"),
    )
