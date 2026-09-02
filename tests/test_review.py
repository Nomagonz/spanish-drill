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
                                 "before": Card.new(today=0).to_dict()},
                             5, index=i)
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
                "before": Card.new(today=0).to_dict()}, 5, index=i)
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


class TestAnAnswerIsTiedToACardNotAPosition:
    """A stored index belongs to whatever deck was loaded that day.

    This deck went from 253 cards to 1670. Every position written before that
    now names a different word, so re-checking an old answer repaired somebody
    else's schedule. Records carry the card's stable id instead, and older
    ones are matched back by the cue and the answer they expected.
    """

    def test_a_record_with_an_id_resolves_by_it(self):
        from spanish_drill.answers import resolve_index
        deck = load_deck()
        index = next(i for i, c in enumerate(deck) if c.id == "hablar:pret-el")
        record = {"card_id": "hablar:pret-el", "card_index": 4, "expected": ["x"]}
        assert resolve_index(record, deck) == index

    def test_a_stale_position_is_ignored_when_an_id_is_present(self):
        from spanish_drill.answers import resolve_index
        deck = load_deck()
        record = {"card_id": "ser", "card_index": 999, "expected": ["ser"]}
        assert deck[resolve_index(record, deck)].id == "ser"

    def test_an_old_vocabulary_record_resolves_by_its_answer(self):
        from spanish_drill.answers import resolve_index
        deck = load_deck()
        record = {"card_index": 0, "expected": ["comer"], "en": "to eat"}
        assert deck[resolve_index(record, deck)].answers[0] == "comer"

    def test_a_conjugation_needs_the_cue_as_well_as_the_answer(self):
        """`vivimos` is both "we live" and "we lived"; the cue separates them."""
        from spanish_drill.answers import resolve_index
        deck = load_deck()
        present = {"expected": ["vivimos"], "prompt": "we live"}
        past = {"expected": ["vivimos"], "prompt": "we lived yesterday"}
        assert deck[resolve_index(present, deck)].id == "vivir:pres-nos"
        assert deck[resolve_index(past, deck)].id == "vivir:pret-nos"

    def test_an_unidentifiable_record_resolves_to_nothing(self):
        from spanish_drill.answers import resolve_index
        assert resolve_index({"expected": ["notaword"], "prompt": "?"}) is None

    def test_review_leaves_a_card_it_cannot_identify_alone(self, tmp_path):
        """Better to skip it than to repair a word it was never about."""
        from spanish_drill.answers import AnswerLog
        from spanish_drill.progress import Progress
        log = AnswerLog(directory=tmp_path, path=tmp_path / "a.jsonl")
        log.rewrite([{"id": "x", "card_index": 3, "expected": ["notaword"],
                      "prompt": "?", "quality": 1, "elapsed": 1.0,
                      "heard": "", "audio": None, "verdict": None,
                      "before": {"ease": 2.5, "interval": 0, "reps": 0,
                                 "lapses": 0, "due": 0}}])
        progress = Progress(path=tmp_path / "p.json")
        buckets = review(log=log, progress=progress,
                         verifier=lambda p: ("comer", False))
        assert buckets["overturned"] == []
        assert progress.cards == {}, "it repaired a card it could not identify"

    def test_every_record_on_file_can_be_identified(self):
        """The real log, after the migration. A record that cannot be placed
        is one that --review would either skip or, before this, misfile."""
        from spanish_drill.answers import AnswerLog, resolve_index
        from spanish_drill.config import ANSWER_LOG
        if not ANSWER_LOG.exists():
            pytest.skip("no answers recorded on this machine")
        deck = load_deck()
        records = AnswerLog().all()
        lost = [r for r in records if resolve_index(r, deck) is None]
        assert not lost, f"{len(lost)} answers cannot be tied to a card"

    def test_repair_refuses_a_card_it_cannot_name(self, tmp_path):
        """Filing under None would invent a card and leave the real one broken."""
        p = Progress(path=tmp_path / "p.json")
        with pytest.raises(ValueError):
            repair(p, {"expected": ["notaword"], "prompt": "?", "quality": 1,
                       "before": Card.new(today=0).to_dict()}, 5)
        assert p.cards == {}
