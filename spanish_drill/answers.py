"""The record of every answer: its audio, and how it was judged.

Every answer is kept, not just the misses, so a verdict can always be traced
back to the audio that produced it.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .audio import save_wav
from .config import ANSWER_LOG, ANSWERS_DIR, SAMPLE_RATE

MIN_CLIP_SECONDS = 0.1


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

    def save_audio(self, card_index, audio):
        """Write the clip first: a live re-check needs a file to send."""
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{card_index:03d}"
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
