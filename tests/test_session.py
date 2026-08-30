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

    def listen(self, window, should_stop=None, accept=None):
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
    def test_a_correct_answer_is_scheduled_ahead(self, progress):
        progress.queue_override = [index_of("ser")]
        r = run(progress, ScriptedListener("ser"))[0]
        assert r.outcome is Outcome.CORRECT
        assert progress.card(index_of("ser")).interval == 1

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
        assert reloaded.card(index_of("ser")).interval == 1

    def test_the_daily_counters_reset_on_a_new_day(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", day=0, missed_today=9, new_done=9)
        p.save()
        assert Progress.load(tmp_path / "p.json", today=1).missed_today == 0
