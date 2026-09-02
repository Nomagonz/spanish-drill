"""The drill loop itself, with no Qt and no microphone.

The session takes a listener and a verifier as plain callables, so the whole
loop is exercised here without audio hardware or a network call.
"""
import numpy as np
import pytest

from spanish_drill.deck import load_deck
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card
from spanish_drill.session import DrillSession, Outcome

DECK = load_deck()


def index_of(word):
    for i, c in enumerate(DECK):
        if c.answers[0] == word:
            return i
    raise AssertionError(word)


class ScriptedListener:
    """Returns canned transcripts, one per call."""

    def __init__(self, *transcripts, audio=b"x"):
        self.script = list(transcripts)
        self.calls = 0
        self.last_audio = np.zeros(16000, dtype=np.float32) if audio else None

    def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
        i, self.calls = self.calls, self.calls + 1
        return self.script[i] if i < len(self.script) else None

    def calibrate(self):
        pass


@pytest.fixture
def progress(tmp_path):
    return Progress(path=tmp_path / "progress.json", new_per=5, verify_live=False)


def run(progress, listener, limit=1, verifier=None):
    session = DrillSession(progress, listener, verifier=verifier)
    results = []
    session.on_result = lambda r: (results.append(r),
                                   session.stop() if len(results) >= limit else None)
    session.run()
    return results


class TestGrading:
    def test_a_correct_answer_starts_the_learning_ladder(self, progress):
        """One right answer no longer finishes a new word for the day.

        It used to schedule straight out to tomorrow, which meant a word you
        had just been told counted as learned. It now has to come back at
        every rung of LEARNING_STEPS first, so it stays due today.
        """
        progress.queue_override = [index_of("ser")]
        r = run(progress, ScriptedListener("ser"))[0]
        assert r.outcome is Outcome.CORRECT
        card = progress.card(index_of("ser"))
        assert (card.reps, card.interval) == (1, 0)

    def test_a_wrong_answer_stays_due_and_lapses(self, progress):
        progress.queue_override = [index_of("ser")]
        r = run(progress, ScriptedListener("comer"))[0]
        assert r.outcome is Outcome.MISS
        card = progress.card(index_of("ser"))
        assert card.interval == 0 and card.lapses == 1

    def test_silence_is_a_miss_and_scores_lower_than_a_guess(self, progress):
        progress.queue_override = [index_of("ser"), index_of("estar")]
        silent = run(progress, ScriptedListener(None), limit=1)[0]
        assert silent.outcome is Outcome.MISS and silent.quality == 0


class TestSecondOpinion:
    def test_a_misheard_answer_is_overturned(self, progress):
        progress.verify_live = True
        progress.queue_override = [index_of("querer")]
        r = run(progress, ScriptedListener("Quieres."),
                verifier=lambda path: ("querer", False))[0]
        assert r.outcome is Outcome.OVERTURNED
        assert progress.card(index_of("querer")).lapses == 0

    def test_a_genuinely_wrong_answer_is_kept(self, progress):
        progress.verify_live = True
        progress.queue_override = [index_of("comer")]
        r = run(progress, ScriptedListener("dormir"),
                verifier=lambda path: ("dormir", False))[0]
        assert r.outcome is Outcome.MISS
        assert progress.kept == 1 and progress.overturned == 0

    def test_an_echo_yields_no_verdict_and_is_not_counted(self, progress):
        progress.verify_live = True
        progress.queue_override = [index_of("llegar")]
        r = run(progress, ScriptedListener("Y errar."),
                verifier=lambda path: (None, True))[0]
        assert r.outcome is Outcome.MISS
        assert progress.kept == 0 and progress.overturned == 0, \
            "an echoed prompt is not evidence the miss was right"

    def test_correct_answers_are_never_sent_for_verification(self, progress):
        progress.verify_live = True
        progress.queue_override = [index_of("ser")]
        calls = []
        run(progress, ScriptedListener("ser"),
            verifier=lambda path: calls.append(path) or ("ser", False))
        assert calls == []


class TestVoiceCommands:
    def test_skip_requeues_without_grading(self, progress):
        i = index_of("ser")
        progress.queue_override = [i, index_of("estar")]
        session = DrillSession(progress, ScriptedListener("skip", "estar"))
        results = []
        session.on_result = lambda r: (results.append(r), session.stop())
        session.run()
        assert progress.card(i) is None, "a skipped card must not be graded"

    def test_stop_ends_the_session(self, progress):
        progress.queue_override = [index_of("ser")]
        session = DrillSession(progress, ScriptedListener("para"))
        session.run()
        assert not session.running


class TestPersistence:
    def test_progress_survives_a_reload(self, progress, tmp_path):
        progress.queue_override = [index_of("ser")]
        run(progress, ScriptedListener("ser"))
        reloaded = Progress.load(tmp_path / "progress.json")
        assert reloaded.card(index_of("ser")).reps == 1

    def test_the_daily_counters_reset_on_a_new_day(self, tmp_path):
        """Every one of them. The dashboard reports these as today's work, so
        one left behind reads as a day's progress that never happened."""
        p = Progress(path=tmp_path / "p.json", day=0, missed_today=9,
                     new_done=9, reviews_done=9)
        p.save()
        fresh = Progress.load(tmp_path / "p.json", today=1)
        assert (fresh.missed_today, fresh.new_done, fresh.reviews_done) == (0, 0, 0)

    def test_the_daily_counters_survive_the_same_day(self, tmp_path):
        """Reopening the app mid-day must not wipe what has been done."""
        p = Progress(path=tmp_path / "p.json", day=7, missed_today=9,
                     new_done=4, reviews_done=11)
        p.save()
        fresh = Progress.load(tmp_path / "p.json", today=7)
        assert (fresh.new_done, fresh.reviews_done) == (4, 11)

    def test_the_lifetime_tallies_are_not_daily(self, tmp_path):
        """kept and overturned count second opinions across all time."""
        p = Progress(path=tmp_path / "p.json", day=0, kept=30, overturned=5)
        p.save()
        fresh = Progress.load(tmp_path / "p.json", today=1)
        assert (fresh.kept, fresh.overturned) == (30, 5)


class TestAnswersThatLookLikeCommands:
    """Four deck answers are also control words: para, parar, alto, siguiente.

    Saying the right answer used to end or derail the session.
    """

    def _card_index(self, word):
        return index_of(word)

    def test_para_answers_its_card_instead_of_stopping(self, progress):
        i = index_of("para")
        progress.queue_override = [i]
        results = run(progress, ScriptedListener("para"))
        assert results and results[0].outcome is Outcome.CORRECT

    def test_parar_answers_its_card_instead_of_stopping(self, progress):
        i = index_of("parar")
        progress.queue_override = [i]
        results = run(progress, ScriptedListener("parar"))
        assert results and results[0].outcome is Outcome.CORRECT

    def test_siguiente_answers_its_card_instead_of_skipping(self, progress):
        i = index_of("siguiente")
        progress.queue_override = [i]
        results = run(progress, ScriptedListener("siguiente"))
        assert results and results[0].outcome is Outcome.CORRECT

    def test_stop_still_stops_on_an_unrelated_card(self, progress):
        progress.queue_override = [index_of("ser")]
        session = DrillSession(progress, ScriptedListener("para"), verifier=None)
        session.run()
        assert not session.running

    def test_skip_still_skips_on_an_unrelated_card(self, progress):
        i = index_of("ser")
        progress.queue_override = [i, index_of("estar")]
        session = DrillSession(progress, ScriptedListener("siguiente", "estar"),
                               verifier=None)
        results = []
        session.on_result = lambda r: (results.append(r), session.stop())
        session.run()
        assert progress.card(i) is None, "a skipped card must not be graded"


class TestTheDayTurnsOverLocally:
    """A drill is a daily habit, so its day has to be the calendar day.

    time.time() // 86400 turns over at UTC midnight, which in the Americas is
    early evening: the new-word allowance came back at 7pm and cards scheduled
    for tomorrow fell due before dinner.
    """

    def test_the_day_follows_the_local_clock(self, monkeypatch):
        import time as time_module
        from spanish_drill import scheduler

        class Evening:
            """9pm local, five hours behind UTC, so UTC is already tomorrow."""
            tm_gmtoff = -5 * 3600

        stamp = 20695 * 86400 + 2 * 3600        # 02:00 UTC = 21:00 local
        monkeypatch.setattr(scheduler.time, "time", lambda: stamp)
        monkeypatch.setattr(scheduler.time, "localtime", lambda t=None: Evening())
        assert int(stamp // 86400) == 20695     # what UTC would have said
        assert scheduler.today() == 20694, (
            "the day rolled over at 7pm local instead of at midnight")

    def test_it_still_turns_over_at_local_midnight(self, monkeypatch):
        from spanish_drill import scheduler

        class Central:
            tm_gmtoff = -5 * 3600

        monkeypatch.setattr(scheduler.time, "localtime", lambda t=None: Central())
        before = 20695 * 86400 + 4 * 3600 + 3599    # 23:59:59 local
        after = 20695 * 86400 + 5 * 3600            # 00:00:00 local
        monkeypatch.setattr(scheduler.time, "time", lambda: before)
        first = scheduler.today()
        monkeypatch.setattr(scheduler.time, "time", lambda: after)
        assert scheduler.today() == first + 1

    def test_the_scale_is_unchanged_so_stored_due_dates_still_mean_something(
            self, monkeypatch):
        """Any fix that renumbered the days would repoint every schedule."""
        from spanish_drill import scheduler

        class UTC:
            tm_gmtoff = 0

        monkeypatch.setattr(scheduler.time, "localtime", lambda t=None: UTC())
        monkeypatch.setattr(scheduler.time, "time", lambda: 20695 * 86400 + 60)
        assert scheduler.today() == 20695


class TestTheDayCanTurnOverWhileTheAppIsOpen:
    """The reset used to live in load(), so an app left open past midnight kept
    yesterday's tallies and its spent allowance until it was restarted."""

    def make(self, tmp_path):
        return Progress(path=tmp_path / "p.json", day=20695, new_done=20,
                        reviews_done=9, missed_today=7, new_per=20)

    def test_it_resets_without_reloading(self, tmp_path):
        p = self.make(tmp_path)
        assert p.roll_over(20696) is True
        assert (p.new_done, p.reviews_done, p.missed_today) == (0, 0, 0)
        assert p.day == 20696

    def test_the_same_day_is_left_alone(self, tmp_path):
        p = self.make(tmp_path)
        assert p.roll_over(20695) is False
        assert (p.new_done, p.reviews_done, p.missed_today) == (20, 9, 7)

    def test_the_allowance_comes_back_with_the_new_day(self, tmp_path):
        p = self.make(tmp_path)
        assert p.new_remaining() == 0
        p.roll_over(20696)
        assert p.new_remaining() == 20

    def test_a_session_running_past_midnight_counts_against_the_new_day(
            self, tmp_path, monkeypatch):
        from spanish_drill import scheduler
        from spanish_drill.session import DrillSession
        p = self.make(tmp_path)
        session = DrillSession.__new__(DrillSession)
        session.progress, session.queue = p, []
        session.rng = __import__("random").Random(0)
        session.rungs, session.weights = {}, {}
        monkeypatch.setattr(scheduler, "today", lambda: 20696)
        monkeypatch.setattr("spanish_drill.progress.scheduler.today", lambda: 20696)
        session._apply(0, 5)
        assert p.day == 20696
        assert p.new_done == 1, "it kept counting against yesterday"


class TestTheLearningLadder:
    """A new word climbs LEARNING_STEPS before it counts as learned.

    One correct answer used to send a word away for a day. The gap between
    missing it and being asked again was about five cards, so you were
    repeating something you had been told moments earlier and it was banked
    as recall. Each rung puts more distance between you and the last sighting.
    """

    def ladder_session(self, progress, index):
        from spanish_drill.session import DrillSession
        session = DrillSession.__new__(DrillSession)
        session.progress, session.queue = progress, list(range(60))
        session.rng = __import__("random").Random(0)
        session.rungs, session.weights = {}, {}
        return session

    def test_the_gaps_are_the_configured_ladder(self, progress):
        from spanish_drill.config import LEARNING_STEPS
        session = self.ladder_session(progress, 99)
        gaps = []
        for _ in LEARNING_STEPS:
            session.queue = list(range(60))
            session._apply(99, 5)
            gaps.append(session.queue.index(99))
        assert gaps == list(LEARNING_STEPS)

    def test_it_takes_the_whole_ladder_to_finish_a_word(self, progress):
        from spanish_drill.config import LEARNING_STEPS
        from spanish_drill.scheduler import today
        session = self.ladder_session(progress, 99)
        for _ in LEARNING_STEPS:
            session._apply(99, 5)
            assert progress.card(99).interval == 0, "banked before the top"
        session._apply(99, 5)                   # clears the last rung
        card = progress.card(99)
        assert (card.reps, card.interval) == (1, 1)
        assert card.due == today() + 1

    def test_a_miss_steps_down_one_rung_not_to_the_bottom(self, progress):
        """One slip near the top must not throw away the whole climb."""
        session = self.ladder_session(progress, 99)
        for _ in range(3):
            session._apply(99, 5)
        assert session.rungs[99] == 3
        session._apply(99, 1)
        assert session.rungs[99] == 2, "a miss reset the ladder instead of stepping"

    def test_a_word_already_in_review_is_untouched(self, progress):
        """The ladder is for learning. A known word still schedules ahead."""
        from spanish_drill.scheduler import Card
        progress.cards[99] = Card(ease=2.5, interval=6, reps=2, lapses=0, due=0)
        session = self.ladder_session(progress, 99)
        session._apply(99, 5)
        assert progress.card(99).interval > 1
        assert 99 not in session.rungs

    def test_a_lapse_drops_a_known_word_into_the_ladder(self, progress):
        from spanish_drill.scheduler import Card
        progress.cards[99] = Card(ease=2.5, interval=30, reps=5, lapses=0, due=0)
        session = self.ladder_session(progress, 99)
        session._apply(99, 1)
        assert 99 in session.rungs
        assert progress.card(99).interval == 0

    def test_climbing_does_not_pile_up_lapses(self, progress):
        """Eight steps down would otherwise brand a hard word a leech."""
        session = self.ladder_session(progress, 99)
        session._apply(99, 1)                   # falls in: one real lapse
        for _ in range(6):
            session._apply(99, 1)               # stepping down, not relapsing
        assert progress.card(99).lapses == 1


class TestTheProgressBarMoves:
    """A bar that renders and never moves is worse than no bar.

    The first version counted cards put to bed. With the ladder nothing is
    finished until it has been answered five times, so over a hundred-card
    queue it read 0 for hundreds of answers. It looked correct in a unit test
    and was useless in the window.
    """

    def run_some(self, progress, answers):
        from spanish_drill.typed import TypedListener
        deck = load_deck()
        listener = TypedListener()
        session = DrillSession(progress, listener, verifier=None, typed=True)
        seen = []
        session.on_progress = lambda d, t: seen.append((d, t))
        count = {"n": 0}

        def on_status(text):
            if text == "Type it":
                listener.submit(deck[session.current].answers[0])

        def on_result(_):
            count["n"] += 1
            if count["n"] >= answers:
                session.stop()

        session.on_status, session.on_result = on_status, on_result
        session.run()
        return seen

    def test_it_advances_on_every_correct_answer(self, progress):
        deck = load_deck()
        progress.queue_override = [i for i, c in enumerate(deck)
                                   if c.id in ("ser", "estar", "tener", "ir")]
        seen = self.run_some(progress, 6)
        values = [d for d, _ in seen]
        assert values == sorted(values), f"the bar went backwards: {values}"
        assert max(values) >= 5, (
            f"six correct answers moved the bar to {max(values)}; it is "
            f"counting something that barely changes")

    def test_the_total_is_the_work_not_the_card_count(self, progress):
        """Four cards on the ladder is twenty passes, not four."""
        deck = load_deck()
        progress.queue_override = [i for i, c in enumerate(deck)
                                   if c.id in ("ser", "estar", "tener", "ir")]
        seen = self.run_some(progress, 2)
        assert seen and seen[0][1] == 4 * DrillSession.LADDER_PASSES

    def test_a_known_review_costs_one_pass_not_a_whole_ladder(self, progress):
        from spanish_drill.scheduler import Card
        deck = load_deck()
        index = next(i for i, c in enumerate(deck) if c.id == "ser")
        progress.cards[index] = Card(ease=2.5, interval=6, reps=2,
                                     lapses=0, due=0)
        progress.queue_override = [index]
        seen = self.run_some(progress, 1)
        assert seen[0][1] == 1, "a card already in review was priced as new"
