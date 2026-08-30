"""Saved progress and settings.

One file, written atomically. Losing weeks of scheduling to a crash mid-write
is not an acceptable failure mode for something that runs every day.
"""
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import MAIN_MODEL, PROGRESS_PATH
from .deck import load_deck
from . import scheduler
from .scheduler import Card, is_mature, migrate


@dataclass
class Progress:
    path: Path = PROGRESS_PATH
    cards: dict = field(default_factory=dict)      # deck index -> Card

    # settings
    dialect: str = "es-MX"
    input_device: str = ""      # by NAME; indices shift as devices connect
    model: str = MAIN_MODEL
    new_per: int = 20
    window: float = 6.0
    hints: bool = True
    verify_live: bool = True

    # daily counters
    day: int = 0
    new_done: int = 0
    missed_today: int = 0

    # lifetime second-opinion tally
    kept: int = 0
    overturned: int = 0

    # test hook: force a specific queue instead of building one
    queue_override: list = None

    _SETTINGS = ("dialect", "input_device", "model", "new_per", "window",
                 "hints", "verify_live")
    _COUNTERS = ("day", "new_done", "missed_today", "kept", "overturned")

    # -- persistence ------------------------------------------------------
    @classmethod
    def load(cls, path=None, today=None):
        path = Path(path or PROGRESS_PATH)
        now = today if today is not None else scheduler.today()
        p = cls(path=path)
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            raw = {}
        for key in cls._SETTINGS + cls._COUNTERS:
            if key in raw:
                setattr(p, key, raw[key])
        p.cards = {int(k): migrate(v, today=now) for k, v in raw.get("cards", {}).items()}
        if p.day != now:                # a new day resets the daily caps
            p.day, p.new_done, p.missed_today = now, 0, 0
        return p

    def save(self):
        data = {"cards": {str(k): v.to_dict() for k, v in self.cards.items()}}
        for key in self._SETTINGS + self._COUNTERS:
            data[key] = getattr(self, key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)      # atomic: a crash cannot shred progress

    # -- queries ----------------------------------------------------------
    def card(self, index):
        return self.cards.get(index)

    def card_or_new(self, index, today=None):
        return self.cards.get(index) or Card.new(today)

    def due_indexes(self, today=None):
        now = today if today is not None else scheduler.today()
        return [i for i, c in self.cards.items() if c.due <= now]

    def unseen_indexes(self):
        return [i for i in range(len(load_deck())) if i not in self.cards]

    def new_remaining(self):
        return max(0, self.new_per - self.new_done)

    def learning_count(self):
        """Answered right and scheduled ahead. Moves the moment you get one."""
        return sum(1 for c in self.cards.values() if c.interval >= 1)

    def mature_count(self):
        """Anki's sense of stuck: three weeks or more between reviews."""
        return sum(1 for c in self.cards.values() if is_mature(c))

    def build_queue(self, today=None, rng=None):
        """Due reviews, shuffled, with the day's new words spread through."""
        if self.queue_override is not None:
            return list(self.queue_override)
        rng = rng or random
        due = self.due_indexes(today)
        rng.shuffle(due)
        fresh = self.unseen_indexes()[: self.new_remaining()]
        queue = list(due)
        if fresh:
            step = max(1, (len(queue) + len(fresh)) // len(fresh))
            at = 0
            for index in fresh:
                queue.insert(min(at, len(queue)), index)
                at += step + 1
        return queue
