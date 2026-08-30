"""Rapid placement: sorting the deck into known and not-yet-known."""
import numpy as np
import pytest

from spanish_drill.deck import load_deck
from spanish_drill.placement import PlacementSession, mark_known, mark_to_learn
from spanish_drill.progress import Progress
from spanish_drill.scheduler import MATURE_AT, is_mature, today

DECK = load_deck()


def index_of(word):
    return next(i for i, c in enumerate(DECK) if c.answers[0] == word)


class Answers:
    """Answers each card correctly or not, by deck index."""

    def __init__(self, correct_for):
        self.correct_for = correct_for
        self.asked = []
        self.last_audio = None

    def bind(self, session):
        self.session = session
        return self

    def listen(self, window, should_stop=None, accept=None):
        index = self.session.current
        self.asked.append(index)
        if index in self.correct_for:
            return DECK[index].answers[0]
        return "comer" if DECK[index].answers[0] != "comer" else "dormir"

    def calibrate(self):
        pass


@pytest.fixture
def progress(tmp_path):
    return Progress(path=tmp_path / "p.json", verify_live=False, new_per=5)


def run(progress, correct_for, limit=None, only=None):
    if only is not None:
        progress.queue_override = only
    listener = Answers(correct_for)
    session = PlacementSession(progress, listener, verifier=None, limit=limit)
    listener.bind(session)
    session.listener = listener
    session.run()
    return session


class TestMarking:
    def test_known_is_parked_three_weeks_out(self, progress):
        card = mark_known(progress, index_of("ser"), 2, today_=100)
        assert card.interval == MATURE_AT and card.due == 100 + MATURE_AT
        assert is_mature(card)

    def test_to_learn_is_due_now_with_no_progress(self, progress):
        card = mark_to_learn(progress, index_of("ser"), today_=100)
        assert card.interval == 0 and card.reps == 0 and card.due == 100
        assert not is_mature(card)


class TestPassing:
    def test_one_correct_answer_is_not_enough(self, progress):
        i = index_of("ser")
        session = run(progress, correct_for={i}, only=[i])
        assert session.listener.asked.count(i) == 2, \
            "it must confirm a correct answer a second time"
        assert i in session.known

    def test_two_correct_answers_mark_it_known(self, progress):
        i = index_of("ser")
        run(progress, correct_for={i}, only=[i])
        assert is_mature(progress.card(i))

    def test_one_wrong_answer_sends_it_to_the_learning_pile(self, progress):
        i = index_of("ser")
        session = run(progress, correct_for=set(), only=[i])
        assert i in session.to_learn
        assert progress.card(i).interval == 0
        assert session.listener.asked.count(i) == 1, \
            "a missed word is classified, not drilled here"


class TestSorting:
    def test_a_mixed_run_splits_the_deck(self, progress):
        known = {index_of(w) for w in ("ser", "estar", "tener")}
        rest = {index_of(w) for w in ("hacer", "poder", "decir")}
        session = run(progress, correct_for=known, only=sorted(known | rest))
        assert set(session.known) == known
        assert len(session.to_learn) == 3
        assert not (set(session.known) & set(session.to_learn))

    def test_every_card_ends_up_classified(self, progress):
        session = run(progress, correct_for={0, 2, 4}, only=[0, 1, 2, 3, 4, 5])
        s = session.summary()
        assert s["tested"] == 6
        assert len(progress.cards) == 6

    def test_it_ignores_the_daily_new_word_cap(self, progress):
        progress.new_per = 2
        session = run(progress, correct_for=set(), only=list(range(10)))
        assert session.summary()["tested"] == 10, \
            "the cap limits review load, and triage creates none"

    def test_already_classified_cards_are_skipped(self, progress):
        i = index_of("ser")
        mark_known(progress, i, 2)
        session = run(progress, correct_for=set(), limit=5)
        assert i not in session.listener.asked


class TestFeedback:
    """Fast, but never silent: answering into a void is worse than slow."""

    def test_a_correct_answer_is_confirmed(self, progress, monkeypatch):
        played = []
        monkeypatch.setattr("spanish_drill.cues.correct",
                            lambda: played.append("correct"))
        i = index_of("ser")
        run(progress, correct_for={i}, only=[i])
        assert played, "you must be told when you got it right"

    def test_a_miss_sounds_different_and_says_the_word(self, progress, monkeypatch):
        played, spoken = [], []
        monkeypatch.setattr("spanish_drill.cues.wrong",
                            lambda: played.append("wrong"))
        monkeypatch.setattr("spanish_drill.placement.say_spanish",
                            lambda *a, **k: spoken.append(a[0]))
        i = index_of("ser")
        run(progress, correct_for=set(), only=[i])
        assert played == ["wrong"]
        assert spoken == ["ser"], "a miss should tell you the answer"

    def test_it_never_reads_the_example_sentence(self, progress, monkeypatch):
        """That is the teaching step, and it is seconds per card."""
        spoken = []
        monkeypatch.setattr("spanish_drill.placement.say_spanish",
                            lambda *a, **k: spoken.append(a[0]))
        i = index_of("ser")
        run(progress, correct_for=set(), only=[i])
        assert all(DECK[i].example not in s for s in spoken)


class TestItTerminates:
    """A run must end, and end having classified everything it started."""

    @pytest.mark.parametrize("accuracy", [1.0, 0.5, 0.0])
    def test_a_full_category_finishes(self, tmp_path, accuracy):
        import random
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        rng = random.Random(1)
        verbs = [i for i, c in enumerate(DECK) if c.pos == "verb"]

        class Listener:
            last_audio = None
            def listen(self, window, should_stop=None, accept=None):
                card = DECK[session.current]
                return card.answers[0] if rng.random() < accuracy else "zzz"
            def calibrate(self):
                pass

        session = PlacementSession(progress, Listener(), verifier=None,
                                   rng=random.Random(2))
        session.run()
        assert not session.running
        classified = len(session.known) + len(session.to_learn)
        assert classified == len(verbs), (
            f"stopped after {classified} of {len(verbs)}")

    def test_a_word_is_asked_at_most_twice(self, tmp_path):
        """Right twice passes; wrong once classifies. Neither loops."""
        import random
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        asked = []

        class Listener:
            last_audio = None
            def listen(self, window, should_stop=None, accept=None):
                asked.append(session.current)
                return DECK[session.current].answers[0]
            def calibrate(self):
                pass

        session = PlacementSession(progress, Listener(), verifier=None,
                                   rng=random.Random(3))
        session.run()
        from collections import Counter
        assert max(Counter(asked).values()) == 2
