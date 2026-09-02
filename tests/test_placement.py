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

    def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
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
            def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
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
            def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
                asked.append(session.current)
                return DECK[session.current].answers[0]
            def calibrate(self):
                pass

        session = PlacementSession(progress, Listener(), verifier=None,
                                   rng=random.Random(3))
        session.run()
        from collections import Counter
        assert max(Counter(asked).values()) == 2


class TestProgressReporting:
    """The run has a known length, so it can show how far along it is."""

    def _run_with_progress(self, progress, only, correct_for=frozenset()):
        listener = Answers(set(correct_for))
        session = PlacementSession(progress, listener, verifier=None)
        listener.bind(session)
        session.listener = listener
        progress.queue_override = list(only)
        seen = []
        session.on_progress = lambda done, total: seen.append((done, total))
        session.run()
        return session, seen

    def test_the_total_is_announced_before_the_first_card(self, progress):
        _, seen = self._run_with_progress(progress, [0, 1, 2])
        assert seen[0] == (0, 3)

    def test_it_advances_once_per_classified_word(self, progress):
        _, seen = self._run_with_progress(progress, [0, 1, 2])
        assert [d for d, _ in seen] == [0, 1, 2, 3]

    def test_a_word_awaiting_its_second_pass_does_not_advance_it(self, progress):
        """Getting it right once is not a classification yet, so the bar holds."""
        i = index_of("ser")
        _, seen = self._run_with_progress(progress, [i], correct_for={i})
        steps = [d for d, _ in seen]
        assert steps[1] == 0, "advanced before the word was confirmed"
        assert steps[-1] == 1 and steps == sorted(steps)

    def test_it_ends_at_the_total(self, progress):
        session, seen = self._run_with_progress(progress, [0, 1, 2])
        assert seen[-1] == (session.total, session.total)

    def test_the_total_matches_what_was_classified(self, progress):
        session, _ = self._run_with_progress(progress, [0, 1, 2, 3])
        assert session.total == session.classified == 4


class TestScopeIsHonest:
    """Placement skips words it has already classified. That has to be visible.

    A bare "2 / 48" over a 149-verb category tells you nothing about the 104
    that were left out, and looks like a bug.
    """

    def test_already_classified_words_are_counted_as_skipped(self, tmp_path):
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        verbs = [i for i, c in enumerate(DECK) if c.pos == "verb"]
        for i in verbs[:10]:
            mark_known(progress, i, 2)
        session = PlacementSession(progress, Answers(set()), verifier=None)
        session.listener.bind(session)
        queue = session.next_queue()
        assert session.skipped == 10
        assert len(queue) == len(verbs) - 10

    def test_retest_includes_them_again(self, tmp_path):
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        verbs = [i for i, c in enumerate(DECK) if c.pos == "verb"]
        for i in verbs[:10]:
            mark_known(progress, i, 2)
        session = PlacementSession(progress, Answers(set()), verifier=None,
                                   retest=True)
        session.listener.bind(session)
        queue = session.next_queue()
        assert session.skipped == 0
        assert len(queue) == len(verbs)

    def test_the_skip_count_is_reported(self, tmp_path):
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        for i in [i for i, c in enumerate(DECK) if c.pos == "verb"][:5]:
            mark_known(progress, i, 2)
        session = PlacementSession(progress, Answers(set()), verifier=None)
        session.listener.bind(session)
        said = []
        session.on_status = said.append
        session.next_queue()
        assert any("already classified" in s for s in said)

    def test_nothing_is_reported_when_nothing_is_skipped(self, tmp_path):
        progress = Progress(path=tmp_path / "p.json", verify_live=False,
                            category="verb")
        session = PlacementSession(progress, Answers(set()), verifier=None)
        session.listener.bind(session)
        said = []
        session.on_status = said.append
        session.next_queue()
        assert not any("already classified" in s for s in said)


class TestItIsActuallyFast:
    """A sorting run must not pay for the main model.

    Measured on a two-second clip: the main model takes about six seconds, the
    scout about one. The main model only ever changes misses, and a miss is
    re-checked by the second opinion, which is quicker and more accurate.
    """

    def test_placement_asks_for_the_fast_path(self, progress):
        session = PlacementSession(progress, Answers(set()), verifier=None)
        assert session.fast_recognition is True

    def test_the_normal_drill_does_not(self, progress):
        from spanish_drill.session import DrillSession
        assert DrillSession(progress, Answers(set()), verifier=None).fast_recognition is False

    def test_the_flag_reaches_the_listener(self, progress):
        seen = []

        class Watcher(Answers):
            def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
                seen.append(fast)
                return super().listen(window, should_stop, accept, fast)

        listener = Watcher(set())
        session = PlacementSession(progress, listener, verifier=None)
        listener.bind(session)
        progress.queue_override = [0]
        session.run()
        assert seen and all(seen), "placement must ask for the scout model"


class TestPlacementCanBeTyped:
    """Sorting silently, for the same reason the drill can be.

    Placement was voice-only, so the one mode built for a place you cannot
    talk in could not be used to sort a new part of speech.
    """

    def test_it_grades_typed_answers(self, tmp_path):
        from spanish_drill.typed import TypedListener
        deck = load_deck()
        index = next(i for i, c in enumerate(deck) if c.pos == "noun")
        progress = Progress(path=tmp_path / "p.json", verify_live=False)
        progress.queue_override = [index]
        listener = TypedListener()
        session = PlacementSession(progress, listener, verifier=None,
                                   typed=True, passes_needed=1)

        def on_status(text):
            if text == "Type it":
                listener.submit(deck[session.current].answers[0])

        session.on_status = on_status
        session.on_result = lambda r: session.stop()
        session.run()
        assert session.known == [index], "a typed answer was not accepted"

    def test_it_makes_no_sound(self, tmp_path, monkeypatch):
        """The whole point of the mode."""
        from spanish_drill import cues, placement as placement_module
        from spanish_drill.typed import TypedListener
        noise = []
        monkeypatch.setattr(cues, "play", lambda *a, **k: noise.append("tone"))
        monkeypatch.setattr(placement_module, "say_spanish",
                            lambda *a, **k: noise.append("speech"))
        deck = load_deck()
        index = next(i for i, c in enumerate(deck) if c.pos == "noun")
        progress = Progress(path=tmp_path / "p.json", verify_live=False)
        progress.queue_override = [index]
        listener = TypedListener()
        session = PlacementSession(progress, listener, verifier=None,
                                   typed=True)

        def on_status(text):
            if text == "Type it":
                listener.submit("definitelywrong")

        session.on_status = on_status
        session.on_result = lambda r: session.stop()
        session.run()
        assert session.to_learn == [index]
        assert noise == [], f"typed placement made a sound: {noise}"

    def test_the_spoken_run_still_beeps(self, tmp_path, monkeypatch):
        """Silencing the typed path must not silence the one it came from."""
        from spanish_drill import cues
        deck = load_deck()
        index = next(i for i, c in enumerate(deck) if c.pos == "noun")
        heard = []
        monkeypatch.setattr(cues, "play", lambda *a, **k: heard.append("tone"))
        progress = Progress(path=tmp_path / "p.json", verify_live=False)
        progress.queue_override = [index]
        class Speaking:
            last_audio = None

            def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
                return deck[index].answers[0]

            def calibrate(self):
                pass

        session = PlacementSession(progress, Speaking(), verifier=None,
                                   passes_needed=1)
        session.on_result = lambda r: session.stop()
        session.run()
        assert heard, "the spoken placement run lost its feedback tone"

    def test_a_category_filter_limits_what_is_sorted(self, tmp_path):
        """Which is what makes a nouns-only run possible at all."""
        deck = load_deck()
        progress = Progress(path=tmp_path / "p.json", category="noun")
        session = PlacementSession(progress, None, verifier=None)
        queue = session.next_queue()
        assert queue, "nothing was offered"
        assert {deck[i].pos for i in queue} == {"noun"}
