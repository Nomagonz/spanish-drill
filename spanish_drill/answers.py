"""The record of every answer: its audio, and how it was judged.

Every answer is kept, not just the misses, so a verdict can always be traced
back to the audio that produced it.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .audio import save_wav
from .config import ANSWER_LOG, ANSWERS_DIR, SAMPLE_RATE
from .text import normalize

MIN_CLIP_SECONDS = 0.1


def slug(word):
    """A word as a filename: no accents, no spaces, nothing a shell dislikes.

    Returns "" for anything that leaves no usable characters, so the caller
    can fall back rather than writing a file called "-.wav".
    """
    return normalize(word).replace(" ", "-") if word else ""


@lru_cache(maxsize=1)
def _index_by_cue_and_answer():
    """(English cue, answer) -> deck position.

    The cue alone is ambiguous and the answer alone can be too: `vivimos` is
    both "we live" and "we lived". Together they are not, because the deck
    tests refuse to hold two cards that sound the same and accept different
    words. This is what identifies a conjugated form in an old record.
    """
    from .deck import load_deck
    out = {}
    for index, card in enumerate(load_deck()):
        for answer in card.answers:
            out.setdefault((normalize(card.prompt), normalize(answer)), index)
    return out


@lru_cache(maxsize=1)
def _index_by_answer():
    """Accepted answer -> deck position, vocabulary only.

    The last resort for records written before card_id existed. A vocabulary
    answer is unique across the deck, so it identifies the card on its own.
    Conjugated forms are deliberately left out: `vivimos` belongs to two of
    them, and guessing between them is the misfiling this exists to stop.
    """
    from .deck import load_deck
    out = {}
    for index, card in enumerate(load_deck()):
        if not card.lemma:
            for answer in card.answers:
                out.setdefault(normalize(answer), index)
    return out


def resolve_index(record, deck=None):
    """Which card in today's deck an answer was about, or None.

    A stored position is only meaningful against the deck that produced it.
    This deck went from 253 cards to 1670, so every index written before that
    now points at a different word: re-checking an old answer would have
    repaired somebody else's card. Records written since carry the card's
    stable id, and older ones are matched back by the answer that was
    expected. Anything that cannot be resolved returns None and is left alone
    rather than guessed at.
    """
    from .deck import index_by_id, load_deck
    deck = deck or load_deck()
    card_id = record.get("card_id") if isinstance(record, dict) else None
    if card_id:
        return index_by_id(deck).get(card_id)
    expected = (record.get("expected") or [None])[0]
    if not expected:
        return None
    cue = record.get("prompt") or record.get("en") or ""
    found = _index_by_cue_and_answer().get(
        (normalize(cue), normalize(expected)))
    if found is not None:
        return found
    return _index_by_answer().get(normalize(expected))


@dataclass
class AnswerRecord:
    id: str
    card_index: int
    prompt: str
    expected: list
    heard: str                  # what the local model produced
    quality: int
    correct: bool
    silent: bool
    elapsed: float
    audio: str = None           # basename, or None when nothing was captured
    before: dict = None         # card state prior to grading
    api_text: str = None        # what the second opinion produced
    api_checked: bool = False
    api_echoed: bool = False
    overturned: bool = False
    verdict: str = None
    # The card's stable id. `card_index` is a position, and a position is
    # only meaningful against the deck that wrote it; editing the deck
    # silently repoints every one of them at a different word.
    card_id: str = ""
    # How the answer was given. Kept because a typed answer and a spoken one
    # are not the same evidence about the same word, and a log that does not
    # say which is which can never be split apart afterwards.
    mode: str = "voice"
    at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class AnswerLog:
    """Append-only JSONL beside the audio it describes."""

    def __init__(self, directory=None, path=None):
        # Resolved when constructed, not when this function was defined, so the
        # destination can actually be redirected. Binding the module constants
        # as default arguments meant tests wrote into the real answers/ folder.
        self.dir = Path(directory or ANSWERS_DIR)
        self.path = Path(path or ANSWER_LOG)

    def save_audio(self, card_index, audio, word=None):
        """Write the clip first: a live re-check needs a file to send.

        Named by the word rather than the deck position. An index is only
        meaningful against the deck that produced it, so reordering the deck
        leaves a folder of clips nobody can identify; the word stays readable
        for as long as the file does. The index is still on the record in the
        log, which is where the schedule linkage belongs.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + (
            slug(word) or f"{card_index:03d}")
        if audio is None or len(audio) <= SAMPLE_RATE * MIN_CLIP_SECONDS:
            return stamp, None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{stamp}.wav"
        try:
            save_wav(path, audio)
            return stamp, path
        except OSError:
            return stamp, None

    def append(self, record):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def all(self):
        if not self.path.exists():
            return []
        out = []
        for line in open(self.path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue            # never let one bad line hide the rest
        return out

    def rewrite(self, records):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def audio_path(self, record):
        name = record.get("audio") if isinstance(record, dict) else record.audio
        if not name:
            return None
        path = self.dir / name
        return path if path.exists() else None
