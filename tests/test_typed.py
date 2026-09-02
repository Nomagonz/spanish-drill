"""Answering with a keyboard instead of a voice.

The mode exists for the times speaking is not an option, so the tests that
matter most are the ones proving nothing here makes a sound and nothing here
reaches for a microphone or the network.

The other half is accents. Speech cannot be graded on them and this can, which
is the only place the two modes are deliberately allowed to disagree about
whether the same string is a correct answer.
"""
import threading
import time

import pytest

from spanish_drill import session as session_module
from spanish_drill.deck import load_deck
from spanish_drill.grading import (check, quality, typed_quality,
                                   typing_budget)
from spanish_drill.progress import Progress
from spanish_drill.session import DrillSession, Outcome
from spanish_drill.text import normalize
from spanish_drill.typed import Gate, TypedListener

DECK = load_deck()


def card_with_id(card_id):
    for c in DECK:
        if c.id == card_id:
            return c
    raise AssertionError(card_id)


def index_of_id(card_id):
    for i, c in enumerate(DECK):
        if c.id == card_id:
            return i
    raise AssertionError(card_id)


class TestNormalizeKeepsAccentsOnRequest:
    def test_accents_still_go_by_default(self):
        assert normalize("habló") == "hablo"

    def test_they_survive_when_asked_for(self):
        assert normalize("habló", accents=True) == "habló"

    def test_the_enye_survives_either_way(self):
        assert normalize("año") == "año"
        assert normalize("año", accents=True) == "año"

    def test_punctuation_and_case_still_go(self):
        assert normalize("¿Habló?", accents=True) == "habló"

    def test_the_default_is_unchanged_for_every_deck_answer(self):
        """The spoken path must not move at all."""
        for c in DECK:
            for a in c.answers:
                assert normalize(a) == normalize(a, accents=False)


class TestAccentsAreNotGraded:
    """Typed answers are held to the same standard as spoken ones.

    A keyboard could be stricter, and briefly was: a missing accent scored a 3
    rather than a 5. Turned off on purpose. Reaching for the option key on
    every other word taxes the mode that exists for answering quickly and
    quietly, and `hablo` for `habló` is a spelling slip rather than a failure
    to know the word.
    """

    def test_a_missing_accent_is_a_clean_pass(self):
        card = card_with_id("hablar:pret-el")       # the answer is habló
        match = check("hablo", card)
        assert match and not match.close, "a missing accent cost marks"

    def test_the_accented_form_is_a_clean_pass_too(self):
        card = card_with_id("hablar:pret-el")
        match = check("habló", card)
        assert match and not match.close

    def test_a_wrong_word_is_still_wrong(self):
        assert check("comer", card_with_id("hablar:pret-el")) is None

    def test_a_different_form_of_the_same_verb_is_still_wrong(self):
        assert check("hablaste", card_with_id("hablar:pret-el")) is None

    def test_saying_it_inside_a_longer_answer_still_counts(self):
        card = card_with_id("hablar:pret-el")
        match = check("el hablo", card)
        assert match and not match.close

    def test_a_leading_article_is_still_free(self):
        noun = next(c for c in DECK if c.answers[0] == "casa")
        match = check("la casa", noun)
        assert match and not match.close

    def test_every_deck_answer_passes_unaccented(self):
        """Whatever the card holds, typing it without accents is accepted."""
        from spanish_drill.text import normalize
        for card in DECK:
            plain = normalize(card.answers[0])      # accents already stripped
            match = check(plain, card)
            assert match and not match.close, card.id


class TestTypingHasItsOwnClock:
    def test_a_longer_word_is_allowed_longer(self):
        assert typing_budget("encontrar") > typing_budget("ir")

    def test_a_quick_answer_scores_top_marks(self):
        assert typed_quality(True, False, False, 1.0, "ir") == 5

    def test_a_laboured_one_scores_below_it(self):
        assert typed_quality(True, False, False, 60.0, "ir") == 4

    def test_a_missing_accent_lands_on_the_near_miss_grade(self):
        assert typed_quality(True, True, False, 0.1, "habló") == 3

    def test_a_blank_answer_is_the_worst_grade(self):
        assert typed_quality(False, False, True, 0.1, "ir") == 0

    def test_a_wrong_answer_beats_no_answer(self):
        assert typed_quality(False, False, False, 1.0, "ir") == 1

    def test_a_long_word_typed_steadily_is_not_punished(self):
        """The spoken window would have called this laboured; typing is slower."""
        slow_but_fine = 3.0
        assert quality(True, False, False, slow_but_fine, 6.0) == 4
        assert typed_quality(True, False, False, slow_but_fine, "encontrar") == 5


class TestTypedListener:
    def test_it_returns_what_was_submitted(self):
        listener = TypedListener()
        listener.submit("hablo")
        assert listener.listen(should_stop=lambda: False) == "hablo"

    def test_surrounding_space_is_trimmed(self):
        listener = TypedListener()
        listener.submit("  hablo  ")
        assert listener.listen(should_stop=lambda: False) == "hablo"

    def test_an_empty_answer_reads_as_silence(self):
        listener = TypedListener()
        listener.submit("   ")
        assert listener.listen(should_stop=lambda: False) is None

    def test_it_gives_up_when_the_session_stops(self):
        listener = TypedListener()
        assert listener.listen(should_stop=lambda: True) is None

    def test_it_waits_for_an_answer_that_arrives_late(self):
        listener = TypedListener()
        threading.Timer(0.15, lambda: listener.submit("hablo")).start()
        assert listener.listen(should_stop=lambda: False) == "hablo"

    def test_it_notices_a_stop_while_it_is_waiting(self):
        """A stop must not hang until somebody types something."""
        listener = TypedListener()
        stopped = []
        threading.Timer(0.1, lambda: stopped.append(True)).start()
        started = time.time()
        assert listener.listen(should_stop=lambda: bool(stopped)) is None
        assert time.time() - started < 2.0

    def test_clear_drops_what_was_typed_too_early(self):
        listener = TypedListener()
        listener.submit("stale")
        listener.clear()
        listener.submit("fresh")
        assert listener.listen(should_stop=lambda: False) == "fresh"

    def test_it_records_no_audio(self):
        assert TypedListener().last_audio is None


class TestTheAnswerWindowAppliesToTyping:
    """The same clock the spoken drill runs on.

    A typed card that waited forever meant the two modes graded on different
    terms: silence is a miss when spoken and was a free pass when typed.
    """

    def test_running_out_of_time_reads_as_no_answer(self):
        listener = TypedListener()
        started = time.time()
        assert listener.listen(0.2, should_stop=lambda: False) is None
        assert 0.15 < time.time() - started < 2.0

    def test_an_answer_inside_the_window_still_lands(self):
        listener = TypedListener()
        threading.Timer(0.05, lambda: listener.submit("hablo")).start()
        assert listener.listen(2.0, should_stop=lambda: False) == "hablo"

    def test_an_answer_after_the_window_is_too_late(self):
        listener = TypedListener()
        threading.Timer(0.4, lambda: listener.submit("hablo")).start()
        assert listener.listen(0.15, should_stop=lambda: False) is None

    def test_no_window_still_waits_indefinitely(self):
        """What a bare call in a test wants, and the default."""
        listener = TypedListener()
        threading.Timer(0.3, lambda: listener.submit("hablo")).start()
        assert listener.listen(None, should_stop=lambda: False) == "hablo"

    def test_a_stop_still_beats_the_window(self):
        listener = TypedListener()
        started = time.time()
        assert listener.listen(30.0, should_stop=lambda: True) is None
        assert time.time() - started < 1.0

    def test_the_session_hands_over_the_configured_window(self, tmp_path):
        """The spinner in the settings is what times a typed card."""
        seen = []

        class Watching(TypedListener):
            def listen(self, window=None, **kw):
                seen.append(window)
                return "habló"

        progress = Progress(path=tmp_path / "p.json", window=9.0,
                            verify_live=False)
        progress.queue_override = [index_of_id("hablar:pret-el")]
        session = DrillSession(progress, Watching(), verifier=None, typed=True)
        session.on_result = lambda r: session.stop()
        session.run()
        assert seen == [9.0]


class TestTheGate:
    def test_it_lets_through_once_released(self):
        gate = Gate()
        threading.Timer(0.1, gate.release).start()
        started = time.time()
        gate(should_stop=lambda: False)
        assert 0.05 < time.time() - started < 3.0

    def test_it_gives_up_when_the_session_stops(self):
        """Shutting down must never wait on a keypress that is not coming."""
        gate = Gate()
        started = time.time()
        gate(should_stop=lambda: True)
        assert time.time() - started < 1.0

    def test_it_says_when_it_is_actually_waiting(self):
        """A key can only release a hold that exists."""
        gate = Gate()
        assert not gate.waiting
        seen = []
        threading.Timer(0.1, lambda: seen.append(gate.waiting)).start()
        threading.Timer(0.2, gate.release).start()
        gate(should_stop=lambda: False)
        assert seen == [True]
        assert not gate.waiting

    def test_it_can_be_used_again_for_the_next_miss(self):
        gate = Gate()
        for _ in range(3):
            threading.Timer(0.05, gate.release).start()
            gate(should_stop=lambda: False)
        assert not gate.waiting


@pytest.fixture
def progress(tmp_path):
    return Progress(path=tmp_path / "progress.json", new_per=5)


def run_typed(progress, *answers, limit=1, verifier=None):
    """Answer each card as it is asked, the way a person does.

    Typed only when the drill says "Type it", not queued up in advance: the
    session discards anything typed before a card is on screen, so pre-loading
    the answers would test a path the app never takes and hang waiting for one
    that never arrives.
    """
    listener = TypedListener()
    pending = list(answers)
    session = DrillSession(progress, listener, verifier=verifier, typed=True)
    results = []
    session.seen_statuses = []

    def on_status(text):
        session.seen_statuses.append(text)
        if text == "Type it" and pending:
            listener.submit(pending.pop(0))

    session.on_status = on_status
    session.on_result = lambda r: (results.append(r),
                                   session.stop() if len(results) >= limit else None)
    session.run()
    return results, session


class TestTheTypedDrill:
    def test_a_correct_answer_starts_the_learning_ladder(self, progress):
        """Typed or spoken, a new word still has to climb the ladder."""
        index = index_of_id("hablar:pret-el")
        progress.queue_override = [index]
        results, _ = run_typed(progress, "habló")
        assert results[0].outcome is Outcome.CORRECT
        card = progress.card(index)
        assert (card.reps, card.interval) == (1, 0)

    def test_a_missing_accent_costs_nothing(self, progress):
        """`hablo` for `habló` is a clean answer, scored like any other."""
        index = index_of_id("hablar:pret-el")
        progress.queue_override = [index]
        results, _ = run_typed(progress, "hablo")
        assert results[0].correct and not results[0].close
        assert results[0].quality == 5
        assert progress.card(index).ease > 2.5      # rewarded, not penalised

    def test_a_wrong_answer_lapses_the_card(self, progress):
        index = index_of_id("hablar:pret-el")
        progress.queue_override = [index]
        results, _ = run_typed(progress, "comer")
        assert results[0].outcome is Outcome.MISS
        assert progress.card(index).lapses == 1

    def test_it_never_speaks(self, progress, monkeypatch):
        """The one thing this mode must not do."""
        spoken = []
        monkeypatch.setattr(session_module, "say_english",
                            lambda *a, **k: spoken.append(a))
        monkeypatch.setattr(session_module, "say_spanish",
                            lambda *a, **k: spoken.append(a))
        progress.queue_override = [index_of_id("hablar:pret-el")]
        run_typed(progress, "comer")        # a miss speaks the most
        assert spoken == []

    def test_it_never_calls_the_second_opinion(self, progress):
        """There is no clip to send, so asking would be billing for nothing."""
        called = []
        progress.queue_override = [index_of_id("hablar:pret-el")]
        progress.verify_live = True
        run_typed(progress, "comer",
                  verifier=lambda p: called.append(p) or ("habló", False))
        assert called == []

    def test_the_answer_log_says_it_was_typed(self, progress):
        progress.queue_override = [index_of_id("hablar:pret-el")]
        _, session = run_typed(progress, "habló")
        assert session.log.all()[-1]["mode"] == "typed"

    def test_a_spoken_session_still_logs_as_voice(self, progress):
        from tests.test_session import ScriptedListener
        progress.queue_override = [index_of_id("hablar:pret-el")]
        progress.verify_live = False
        session = DrillSession(progress, ScriptedListener("habló"), verifier=None)
        session.on_result = lambda r: session.stop()
        session.run()
        assert session.log.all()[-1]["mode"] == "voice"

    def test_typing_stop_ends_the_session(self, progress):
        progress.queue_override = [index_of_id("hablar:pret-el"),
                                   index_of_id("hablar:pres-yo")]
        # "para" is not the answer to either card, so it reads as the command.
        results, session = run_typed(progress, "para")
        assert results == []
        assert not session.running

    def test_it_never_claims_to_be_listening(self, progress):
        """There is no microphone open, and the status bar said there was.

        The wait status was emitted from one place for both modes, so a typing
        session announced "Type it", overwrote it with "Listening" a moment
        later, and sat there naming hardware it had never touched.
        """
        progress.queue_override = [index_of_id("hablar:pret-el")]
        _, session = run_typed(progress, "habló")
        assert "Listening" not in session.seen_statuses
        assert "Speaking" not in session.seen_statuses
        assert "Type it" in session.seen_statuses

    def test_the_field_is_cleared_before_it_asks_for_an_answer(self, progress):
        """Clearing afterwards threw away the answer it had just invited.

        The helper types the instant the drill says "Type it", which is what a
        person does. Clearing after that announcement discarded it and the
        card waited forever for a second one.
        """
        progress.queue_override = [index_of_id("hablar:pret-el")]
        results, _ = run_typed(progress, "habló")
        assert results and results[0].correct

    def test_a_blank_entry_is_a_miss_not_a_crash(self, progress):
        index = index_of_id("hablar:pret-el")
        progress.queue_override = [index]
        results, _ = run_typed(progress, "")
        assert results[0].silent and results[0].quality == 0


class TestAMissWaitsToBeRead:
    """The answer used to flash past while you were still looking at the cue.

    Only in the typing mode. The spoken drill is hands-free on purpose, and
    making it stop for a keypress on every miss would take away the whole
    reason it exists.
    """

    def held_run(self, progress, *answers):
        """Run with a gate, releasing it from another thread as it catches."""
        gate = Gate()
        listener = TypedListener()
        pending = list(answers)
        session = DrillSession(progress, listener, verifier=None, typed=True,
                               hold_on_miss=gate)
        results, holds = [], []

        def on_status(text):
            if text == "Type it" and pending:
                listener.submit(pending.pop(0))
            if text == "Press Enter to continue":
                holds.append(text)
                threading.Timer(0.05, gate.release).start()

        session.on_status = on_status
        session.on_result = lambda r: (results.append(r), session.stop())
        session.run()
        return results, holds

    def test_a_miss_holds_until_it_is_released(self, progress):
        progress.queue_override = [index_of_id("hablar:pret-el")]
        results, holds = self.held_run(progress, "comer")
        assert results[0].outcome is Outcome.MISS
        assert holds == ["Press Enter to continue"]

    def test_a_correct_answer_does_not_hold(self, progress):
        """Nothing to read, so nothing to wait for."""
        progress.queue_override = [index_of_id("hablar:pret-el")]
        results, holds = self.held_run(progress, "habló")
        assert results[0].correct
        assert holds == []

    def test_an_unaccented_answer_does_not_hold(self, progress):
        """It is a pass, and a pass keeps moving."""
        progress.queue_override = [index_of_id("hablar:pret-el")]
        results, holds = self.held_run(progress, "hablo")
        assert results[0].correct and not results[0].close
        assert holds == []

    def test_a_session_with_no_gate_never_waits(self, progress):
        """This is what keeps the spoken drill hands-free."""
        progress.queue_override = [index_of_id("hablar:pret-el")]
        _, session = run_typed(progress, "comer")       # no hold_on_miss
        assert "Press Enter to continue" not in session.seen_statuses

    def test_stopping_releases_a_held_card(self, progress):
        """Otherwise Stop hangs until somebody presses a key."""
        gate = Gate()
        listener = TypedListener()
        pending = ["comer"]
        session = DrillSession(progress, listener, verifier=None, typed=True,
                               hold_on_miss=gate)
        progress.queue_override = [index_of_id("hablar:pret-el")]

        def on_status(text):
            if text == "Type it" and pending:
                listener.submit(pending.pop(0))
            if text == "Press Enter to continue":
                threading.Timer(0.05, session.stop).start()

        session.on_status = on_status
        started = time.time()
        session.run()
        assert time.time() - started < 5.0, "stop did not break the hold"


class TestPausing:
    """A pause has to stop the clock, not just the asking.

    The answer window is the thing that would otherwise keep running while
    you are away, and a card that times out is scored as silence — the worst
    grade there is. So the wait is held before the window starts.
    """

    def test_it_holds_until_resumed(self):
        import threading
        from spanish_drill.typed import TypedListener
        listener = TypedListener()
        listener.pause()
        got = []
        thread = threading.Thread(
            target=lambda: got.append(listener.listen(window=0.2)), daemon=True)
        thread.start()
        time.sleep(0.4)
        assert not got, "it answered while paused"
        listener.submit("tengo")
        listener.resume()
        thread.join(timeout=2)
        assert got == ["tengo"]

    def test_the_window_starts_only_after_resuming(self):
        """Paused for longer than the window, the card still gets its full
        window when you come back."""
        import threading
        from spanish_drill.typed import TypedListener
        listener = TypedListener()
        listener.pause()
        got = []
        thread = threading.Thread(
            target=lambda: got.append(listener.listen(window=1.0)), daemon=True)
        thread.start()
        time.sleep(1.4)                 # longer than the window
        listener.resume()
        listener.submit("tengo")
        thread.join(timeout=3)
        assert got == ["tengo"], "the window ran out while paused"

    def test_a_stop_still_gets_through_a_pause(self):
        import threading
        from spanish_drill.typed import TypedListener
        listener = TypedListener()
        listener.pause()
        stopping = []
        got = []
        thread = threading.Thread(
            target=lambda: got.append(
                listener.listen(window=None,
                                should_stop=lambda: bool(stopping))),
            daemon=True)
        thread.start()
        time.sleep(0.2)
        stopping.append(True)
        thread.join(timeout=2)
        assert got == [None]

    def test_it_reports_whether_it_is_paused(self):
        from spanish_drill.typed import TypedListener
        listener = TypedListener()
        assert listener.paused is False
        listener.pause()
        assert listener.paused is True
        listener.resume()
        assert listener.paused is False
