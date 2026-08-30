"""Re-judging past answers, and repairing a card without losing later work."""
import json

import numpy as np
import pytest

from spanish_drill.answers import AnswerLog, AnswerRecord
from spanish_drill.audio import save_wav
from spanish_drill.deck import load_deck
from spanish_drill.progress import Progress
from spanish_drill.review import repair, review
from spanish_drill.scheduler import Card, schedule

DECK = load_deck()


def index_of(word):
    return next(i for i, c in enumerate(DECK) if c.answers[0] == word)


@pytest.fixture
def log(tmp_path):
    d = tmp_path / "answers"
    d.mkdir()
    return AnswerLog(directory=d, path=d / "answers.jsonl")


def add(log, card_index, heard, quality, before, with_audio=True):
    name = None
    if with_audio:
        name = f"clip-{card_index}.wav"
        save_wav(log.dir / name, np.zeros(16000, dtype=np.float32))
    return log.append(AnswerRecord(
        id=f"id-{card_index}", card_index=card_index,
        prompt=DECK[card_index].prompt, expected=list(DECK[card_index].answers),
        heard=heard, quality=quality, correct=quality >= 3, silent=False,
        elapsed=1.0, audio=name, before=before))


class TestRepair:
    def test_the_ease_penalty_is_reversed(self, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        i = index_of("llevar")
        card = Card.new(today=0)
        clean = card.ease
        schedule(card, 1, today=0)
        p.cards[i] = card
        repaired, _ = repair(p, {"card_index": i, "quality": 1,
                                 "before": Card.new(today=0).to_dict()}, 5)
        assert repaired.ease > clean
        assert repaired.lapses == 0

    def test_a_later_correct_answer_is_not_trampled(self, tmp_path):
        """A missed word is requeued and may be answered right in the same
        session. That repetition is real and must survive the repair."""
        p = Progress(path=tmp_path / "p.json")
        i = index_of("llevar")
        card = Card.new(today=0)
        schedule(card, 1, today=0)      # the bad miss
        schedule(card, 5, today=0)      # answered correctly later
        p.cards[i] = card
        interval_after_retry = card.interval
        repaired, restored = repair(
            p, {"card_index": i, "quality": 1,
                "before": Card.new(today=0).to_dict()}, 5)
        assert not restored
        assert repaired.interval == interval_after_retry


class TestReview:
    def test_a_misheard_miss_is_overturned(self, log, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        i = index_of("llevar")
        card = Card.new(today=0)
        schedule(card, 1, today=0)
        p.cards[i] = card
        add(log, i, "Y ahora", 1, Card.new(today=0).to_dict())
        out = review(log=log, progress=p, verifier=lambda path: ("llevar", False))
        assert len(out["overturned"]) == 1
        assert p.cards[i].lapses == 0

    def test_a_real_miss_is_confirmed(self, log, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        i = index_of("comer")
        add(log, i, "dormir", 1, Card.new(today=0).to_dict())
        out = review(log=log, progress=p, verifier=lambda path: ("dormir", False))
        assert len(out["confirmed"]) == 1 and not out["overturned"]

    def test_an_echo_is_not_treated_as_a_confirmed_miss(self, log, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        add(log, index_of("llegar"), "Y errar", 1, Card.new(today=0).to_dict())
        out = review(log=log, progress=p, verifier=lambda path: (None, True))
        assert len(out["unusable"]) == 1
        assert not out["confirmed"], "a prompt echo is not evidence of a miss"

    def test_a_missing_clip_is_skipped(self, log, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        add(log, index_of("ser"), "x", 1, Card.new(today=0).to_dict(),
            with_audio=False)
        out = review(log=log, progress=p, verifier=lambda path: ("ser", False))
        assert len(out["unusable"]) == 1

    def test_records_are_not_re_reviewed(self, log, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        add(log, index_of("comer"), "dormir", 1, Card.new(today=0).to_dict())
        review(log=log, progress=p, verifier=lambda path: ("dormir", False))
        calls = []
        review(log=log, progress=p,
               verifier=lambda path: calls.append(path) or ("x", False))
        assert calls == []

    def test_it_reads_logs_written_by_the_previous_version(self, log, tmp_path):
        """Old lines used cid/en instead of card_index/prompt."""
        i = index_of("llevar")
        save_wav(log.dir / "old.wav", np.zeros(16000, dtype=np.float32))
        legacy = {"id": "old", "cid": i, "en": DECK[i].prompt,
                  "expected": list(DECK[i].answers), "heard": "Y ahora",
                  "quality": 1, "live_ok": False, "silent": False,
                  "elapsed": 1.0, "audio": "old.wav",
                  "before": Card.new(today=0).to_dict(), "verdict": None}
        with open(log.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(legacy) + "\n")
        p = Progress(path=tmp_path / "p.json")
        out = review(log=log, progress=p, verifier=lambda path: ("llevar", False))
        assert len(out["overturned"]) == 1
