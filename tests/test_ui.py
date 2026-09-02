"""The window's button states and its responsiveness.

The rule this file exists to enforce: nothing slow ever runs on the UI thread.
Loading a model takes seconds and calibrating the microphone takes about one,
and doing either on the main thread froze the window and made the button look
broken.
"""
import time

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from spanish_drill import ui as ui_module
from spanish_drill.config import CONJUGATION_LOG
from spanish_drill.paradigm import ConjugationSession
from spanish_drill.progress import Progress
from spanish_drill.session import DrillSession
from spanish_drill.ui import SessionWorker, Window


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class SlowListener:
    """Stands in for real hardware: calibration costs about a second."""

    def __init__(self, floor=0.02, calibrate_seconds=0.9):
        self.floor = floor
        self.calibrate_seconds = calibrate_seconds
        self.calibrated = False
        self.closed = False
        self.last_audio = None

    def calibrate(self):
        time.sleep(self.calibrate_seconds)
        self.calibrated = True
        return self.floor

    def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
        return None

    def set_device(self, name):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(ui_module.Progress, "load",
                        classmethod(lambda cls, *a, **k: Progress(path=tmp_path / "p.json")))
    # never load a real model in a test
    monkeypatch.setattr(ui_module.Window, "_start_preload", lambda self: None)
    w = Window("small")
    w.listener = SlowListener()
    yield w
    # A QThread destroyed while still running is a qFatal abort, not an
    # exception, so every test must leave the worker properly shut down.
    if w.thread is not None:
        w.worker.stop()
        w.thread.quit()
        assert w.thread.wait(5000), "worker thread did not shut down"
    w.close()


class TestTheCueMarksWhoItIsAbout:
    """Rendered, not just formatted. The cue is a styled QLabel, and a
    stylesheet `color` can swallow an inline one, so a test that only reads
    back the markup would pass on a screen that is still entirely black.
    """

    def _red_pixels(self, window):
        from PyQt6.QtWidgets import QApplication
        window.resize(900, 700)
        window.show()
        QApplication.processEvents()
        image = window.prompt_label.grab().toImage()
        return sum(image.pixelColor(x, y).name() == ui_module.MISS.lower()
                   for x in range(image.width())
                   for y in range(image.height()))

    def _card(self, prompt):
        from spanish_drill.deck import load_deck
        return next(c for c in load_deck() if c.prompt == prompt)

    def test_the_subject_is_red_in_the_word_drill(self, window):
        window._on_prompt(self._card("I think"), "review 2")
        assert "I" in window.prompt_label.text()
        assert self._red_pixels(window) > 50, "the subject is not painted red"

    def test_plain_vocabulary_is_left_alone(self, window):
        window._on_prompt(self._card("to think"), "new")
        assert self._red_pixels(window) == 0

    def test_the_conjugations_drill_marks_nothing(self, window):
        """Every card there has a subject, so marking it says nothing."""
        class Worker:
            session = object.__new__(ConjugationSession)
        window.worker = Worker()
        window._on_prompt(self._card("I think"), "review 2")
        assert self._red_pixels(window) == 0


class TestButtonStates:
    def test_it_starts_disabled_until_models_load(self, app, tmp_path, monkeypatch):
        monkeypatch.setattr(ui_module.Progress, "load",
                            classmethod(lambda cls, *a, **k: Progress(path=tmp_path / "p.json")))
        started = []
        monkeypatch.setattr(ui_module.Window, "_start_preload",
                            lambda self: started.append(True))
        w = Window("small")
        assert started, "models should load without being asked"

    def test_pressing_go_does_not_block_the_ui_thread(self, window):
        """The whole point: calibration costs ~1s and must not happen here."""
        started = time.time()
        window.toggle()
        elapsed = time.time() - started
        assert elapsed < 0.2, (
            f"toggle() held the UI thread for {elapsed:.2f}s; the window is "
            f"frozen and unresponsive for that whole time")
        assert window.go.text() == "STOP"

    def test_a_click_while_loading_is_ignored_not_queued(self, window):
        window.listener = None
        window.toggle()
        assert window.thread is None
        assert "loading" in window.status_label.text().lower()

    def test_stopping_disables_the_button_until_it_really_stops(self, window):
        window.toggle()
        window._request_stop()
        assert window.go.text() == "STOPPING…"
        assert not window.go.isEnabled(), (
            "showing GO while the worker is still winding down means the "
            "next click silently does nothing")

    def test_toggle_refuses_to_run_while_disabled(self, window):
        window.go.setEnabled(False)
        window.toggle()
        assert window.thread is None, (
            "toggle must not depend on the widget alone to stop re-entry")


class TestTheConjugationMode:
    """It is a separate sitting, and the window has to treat it as one."""

    def test_it_drills_its_own_tracker_not_the_vocabulary_one(self, window):
        window.toggle_conjugations()
        session = window.worker.session
        assert isinstance(session, ConjugationSession)
        assert session.progress is not window.progress, (
            "a conjugation run that holds the main Progress would move "
            "vocabulary review dates, which is the one thing it must not do")

    def test_its_answers_go_to_their_own_log(self, window):
        window.toggle_conjugations()
        assert window.worker.session.log.path == CONJUGATION_LOG, (
            "sharing the log lets --review repair a conjugation answer "
            "into the vocabulary schedule")

    def test_it_borrows_the_window_settings(self, window):
        """One set of dials, not two.

        Set on the controls rather than on Progress, because starting a run
        applies the controls first: dialect decides whether vosotros is in
        the chain at all, so a second copy that could disagree would be a
        second way to be wrong.
        """
        window.dialect.setCurrentText("es-MX")
        window.window_seconds.setValue(7)
        window.toggle_conjugations()
        p = window.worker.session.progress
        assert (p.dialect, p.window) == ("es-MX", 7.0)

    def test_a_category_filter_does_not_follow_it_in(self, window):
        """Scope here is the verb batch. A noun filter would empty the queue."""
        window.progress.category = "noun"
        window.toggle_conjugations()
        assert window.worker.session.progress.category == "all"

    def test_it_takes_over_the_other_buttons_while_it_runs(self, window):
        window.toggle_conjugations()
        assert window.conjugations.text() == "STOP"
        assert not window.go.isEnabled()
        assert not window.placement.isEnabled()
        assert not window.sentences.isEnabled()

    def test_stopping_it_names_the_button_that_started_it(self, window):
        window.toggle_conjugations()
        window._request_stop()
        assert window.conjugations.text() == "STOPPING…"
        assert window.go.text() == Window.WORDS, (
            "the words button never started this run, so it must not become "
            "its stop control"
        )

    def test_the_panel_counts_the_schedule_being_moved(self, window):
        """Otherwise the counters describe a deck nobody is drilling."""
        window.toggle_conjugations()
        assert window._active_progress() is window.worker.session.progress

    def test_and_goes_back_to_the_vocabulary_one_when_idle(self, window):
        assert window._active_progress() is window.progress


class TestShutdown:
    def test_closing_the_window_releases_the_microphone(self, window):
        """The stream is held open all session, so something has to let go."""
        listener = window.listener
        window.close()
        assert listener.closed


class TestStoppingEarly:
    def test_stop_during_calibration_does_not_start_drilling(self, tmp_path):
        """Calibration runs before the loop. A stop arriving during it used to
        be forgotten, and the session drilled on regardless."""
        listener = SlowListener(calibrate_seconds=0.3)
        progress = Progress(path=tmp_path / "p.json")
        session = DrillSession(progress, listener, verifier=None)
        worker = SessionWorker(session)
        prompts = []
        worker.prompt.connect(lambda *a: prompts.append(a))

        import threading
        t = threading.Thread(target=worker.run)
        t.start()
        time.sleep(0.05)            # while it is still calibrating
        worker.stop()
        t.join(timeout=5)

        assert not t.is_alive()
        assert prompts == [], "stop was ignored and it drilled anyway"


class TestWorkerCalibration:
    def test_the_worker_calibrates_before_drilling(self, tmp_path):
        listener = SlowListener(calibrate_seconds=0)
        session = DrillSession(Progress(path=tmp_path / "p.json"), listener,
                               verifier=None)
        session.progress.queue_override = []
        SessionWorker(session).run()
        assert listener.calibrated

    def test_a_dead_microphone_is_reported_not_crashed(self, tmp_path):
        class Broken(SlowListener):
            def calibrate(self):
                raise OSError("no input device")

        session = DrillSession(Progress(path=tmp_path / "p.json"), Broken(),
                               verifier=None)
        worker = SessionWorker(session)
        seen = []
        worker.status.connect(seen.append)
        finished = []
        worker.finished.connect(lambda: finished.append(True))
        worker.run()
        assert finished, "a failed microphone must still end the session"
        assert any("Microphone failed" in s for s in seen)

    def test_a_silent_microphone_warns(self, tmp_path):
        session = DrillSession(Progress(path=tmp_path / "p.json"),
                               SlowListener(floor=0.0001, calibrate_seconds=0),
                               verifier=None)
        session.progress.queue_override = []
        worker = SessionWorker(session)
        seen = []
        worker.status.connect(seen.append)
        worker.run()
        assert any("near silence" in s for s in seen)


class TestWhichSettingsTakeEffectMidSession:
    """Turning a dial mid-drill used to do nothing at all.

    Every control wrote into progress only from _apply_settings, which runs on
    GO, so a change made during a session sat in the widget until the next one.
    The plumbing behind two of them was already live and only the write was
    missing: the drill re-reads the answer window before listening to each
    card, and rebuilds the queue whenever it drains.
    """

    def test_the_answer_window_reaches_the_running_drill(self, window):
        window.window_seconds.setValue(11)
        assert window.progress.window == 11.0, (
            "the drill reads progress.window on every card, and the spinner "
            "was not writing to it until the next session started")

    def test_it_keeps_reaching_it_after_a_session_has_begun(self, window):
        window._apply_settings()
        window.window_seconds.setValue(7)
        assert window.progress.window == 7.0

    def test_the_new_word_allowance_reaches_it_too(self, window):
        window._apply_settings()
        window.new_per.setValue(35)
        assert window.progress.new_per == 35

    def test_the_queue_refills_rather_than_ending_the_session(self):
        """This is what makes new/day and the category live at all: an empty
        queue is refilled in the loop, not treated as the end."""
        import inspect
        from spanish_drill.session import DrillSession
        body = inspect.getsource(DrillSession._loop)
        assert "self.queue = self.next_queue()" in body
        assert body.index("while") < body.index("self.queue = self.next_queue()"), (
            "the queue is built once before the loop, so nothing can be live")

    def test_a_bigger_allowance_introduces_more_on_the_next_refill(self, tmp_path):
        from spanish_drill.progress import Progress
        small = Progress(path=tmp_path / "a.json", new_per=1)
        large = Progress(path=tmp_path / "b.json", new_per=40)
        assert len(large.build_queue()) > len(small.build_queue())


class TestTheDashboardShowsTodaysWork:
    """The panel used to open with what was left, never with what was done.

    A session could go well and the top of the window would show a smaller
    "new left" and nothing else. Words learned and repetitions made are the
    two things a day's work consists of, so both are counted and shown.
    """

    def test_it_shows_words_learned_and_how_many_remain(self, window):
        window.progress.new_done, window.progress.new_per = 8, 20
        window._refresh_counters()
        assert window.counters["learned"].text() == "8"
        assert "12/20 LEFT" in window.counter_labels["learned"].text()

    def test_lowering_the_allowance_below_the_day_reads_as_itself(self, window):
        """Learn forty with new/day at forty, then set the dial to ten: none
        are left, and the panel has to say so next to the ten rather than
        showing a bare zero that looks like a broken count."""
        window.progress.new_done, window.progress.new_per = 40, 10
        window._refresh_counters()
        assert window.counters["learned"].text() == "40"
        assert "0/10 LEFT" in window.counter_labels["learned"].text()

    def test_it_shows_repetitions_made_today(self, window):
        window.progress.reviews_done = 23
        window._refresh_counters()
        assert window.counters["reviews"].text() == "23"

    def test_the_two_are_counted_separately(self, tmp_path):
        """Meeting a word for the first time is not a repetition of it."""
        from spanish_drill.progress import Progress
        from spanish_drill.scheduler import Card
        from spanish_drill.session import DrillSession
        progress = Progress(path=tmp_path / "p.json")
        progress.cards[1] = Card(ease=2.5, interval=6, reps=2, lapses=0, due=0)
        session = DrillSession.__new__(DrillSession)
        session.progress, session.queue, session.rng = progress, [], __import__("random").Random(0)
        session.rungs, session.weights = {}, {}
        session._apply(0, 5)        # never seen before
        session._apply(1, 5)        # already known
        assert (progress.new_done, progress.reviews_done) == (1, 1)

    def test_the_ladder_names_the_step_each_card_is_on(self, window):
        from spanish_drill.scheduler import Card
        window.progress.cards = {
            0: Card(ease=2.5, interval=1, reps=1, lapses=0, due=0),
            1: Card(ease=2.5, interval=6, reps=2, lapses=0, due=0),
            2: Card(ease=2.5, interval=30, reps=5, lapses=0, due=0),
        }
        text = window._ladder_text()
        assert "1d 1" in text and "6d 1" in text and "21d+ 1" in text

    def test_the_ladder_reads_soonest_first(self, window):
        from spanish_drill.scheduler import Card
        window.progress.cards = {
            0: Card(ease=2.5, interval=40, reps=6, lapses=0, due=0),
            1: Card(ease=2.5, interval=1, reps=1, lapses=0, due=0),
            2: Card(ease=2.5, interval=0, reps=0, lapses=3, due=0),
        }
        text = window._ladder_text()
        assert text.index("RELEARNING") < text.index("1d") < text.index("21d+")

    def test_a_card_in_progress_names_which_repetition_it_is_on(self):
        """"Review · 6d" said how long the gap was but not how far along the
        word is. Both matter: the sixth day of a second repetition and of a
        fifth are different situations."""
        from spanish_drill.scheduler import Card, describe_state
        assert describe_state(Card(ease=2.5, interval=6, reps=2, lapses=0,
                                   due=0)) == "Review 2 · day 6 · ease 2.50"
        assert "day 21+" in describe_state(
            Card(ease=2.5, interval=30, reps=5, lapses=0, due=0))


class TestInQueueMatchesWhatTheDrillWillAsk:
    """The counter used to call due_indexes directly, which does not apply the
    category filter and does not know about new words.

    With "verbs only" chosen it counted every due card in the deck, and it
    never included the new words the session was about to introduce. It
    promised a number the drill had no intention of asking.
    """

    def make(self, tmp_path, **kwargs):
        from spanish_drill.progress import Progress
        from spanish_drill.scheduler import Card
        from spanish_drill.deck import load_deck
        progress = Progress(path=tmp_path / "p.json", **kwargs)
        deck = load_deck()
        verbs = [i for i, c in enumerate(deck) if c.pos == "verb"][:6]
        nouns = [i for i, c in enumerate(deck) if c.pos == "noun"][:9]
        for i in verbs + nouns:                     # all due today
            progress.cards[i] = Card(ease=2.5, interval=1, reps=1, lapses=0, due=0)
        return progress

    def test_it_counts_what_build_queue_will_return(self, tmp_path):
        progress = self.make(tmp_path, new_per=4)
        due, fresh = progress.queue_parts()
        assert len(due) + len(fresh) == len(progress.build_queue())

    def test_it_respects_the_category_filter(self, tmp_path):
        progress = self.make(tmp_path, new_per=0, category="verb")
        due, _ = progress.queue_parts()
        assert len(due) == 6, "nouns leaked into a verbs-only queue"

    def test_it_includes_the_new_words_that_are_about_to_be_asked(self, tmp_path):
        progress = self.make(tmp_path, new_per=4)
        due, fresh = progress.queue_parts()
        assert len(fresh) == 4
        assert len(due) + len(fresh) > len(due)

    def test_an_exhausted_allowance_adds_nothing(self, tmp_path):
        progress = self.make(tmp_path, new_per=10, new_done=40)
        _, fresh = progress.queue_parts()
        assert fresh == []

    def test_the_panel_says_how_many_of_the_queue_are_new(self, window, tmp_path):
        window.progress = self.make(tmp_path, new_per=4)
        window._refresh_counters()
        assert window.counters["due"].text() == str(15 + 4)
        assert "4 NEW" in window.counter_labels["due"].text()


class TestThePanelNoticesANewDay:
    def test_reading_the_counters_rolls_the_day(self, window):
        from spanish_drill import scheduler
        window.progress.day = scheduler.today() - 1
        window.progress.new_done = 20
        window.progress.new_per = 20
        window._refresh_counters()
        assert window.counters["learned"].text() == "0"
        assert "20/20 LEFT" in window.counter_labels["learned"].text()

    def test_it_checks_on_its_own_while_idle(self, window):
        """Nothing else touches the panel overnight, so it needs a heartbeat."""
        assert window._day_timer.isActive()
        assert window._day_timer.interval() <= 60_000


class TestTheWindowAndTheBrowserAreOneDrill:
    """Not one schedule. One session.

    The window used to build a session of its own, so it and the phone agreed
    about what you knew and still sat on different cards, each waiting for an
    answer the other could not give. The window now asks the hub for the
    session and only runs it, which is what makes every screen a view of the
    same card rather than the start of another drill.
    """

    @pytest.fixture
    def shared(self, app, tmp_path):
        import json as _json
        import re as _re
        import time as _time
        import urllib.request as _req
        from spanish_drill.progress import Progress
        from spanish_drill.scheduler import Card, today
        from spanish_drill.serve import Hub, start_server

        hub = Hub(Progress.load(path=tmp_path / "p.json"))
        # Enough due to have something to ask without touching the real file.
        for i in range(12):
            hub.progress.cards[i] = Card(interval=1, reps=1, due=0)
        hub.progress.day = today()

        window = Window(hub=hub)
        window.go.setEnabled(True)          # the model preload normally does this
        server, token = start_server(hub, port=0, host="127.0.0.1", token="tok")
        base = "http://127.0.0.1:%d" % server.server_address[1]

        def pump(seconds=1.0):
            end = _time.time() + seconds
            while _time.time() < end:
                app.processEvents()
                _time.sleep(0.01)

        def post(path, body=None):
            r = _req.Request(f"{base}{path}?t=tok", method="POST",
                             data=_json.dumps(body or {}).encode(),
                             headers={"Content-Type": "application/json"})
            return _req.urlopen(r, timeout=30).read()

        def browser_card(since):
            with _req.urlopen(f"{base}/poll?t=tok&since={since}", timeout=30) as a:
                d = _json.loads(a.read())
            shown = [_json.loads(e) for e in d["events"]
                     if _json.loads(e)["event"] == "prompt"]
            return (shown[-1]["text"] if shown else None), d["next"]

        def on_screen():
            return _re.sub(r"<[^>]+>", "", window.prompt_label.text())

        try:
            yield dict(hub=hub, window=window, pump=pump, post=post,
                       browser_card=browser_card, on_screen=on_screen)
        finally:
            hub.stop()
            pump(0.5)
            server.shutdown()
            window.close()

    def start(self, s):
        s["window"].typing.setChecked(True)
        s["pump"](0.2)
        s["window"].toggle()
        s["pump"](3.0)

    def test_both_screens_are_shown_the_same_card(self, shared):
        self.start(shared)
        theirs, _ = shared["browser_card"](0)
        assert theirs, "the browser was never told about a card"
        assert shared["on_screen"] == theirs or shared["on_screen"]() == theirs

    def test_answering_in_the_browser_moves_the_window(self, shared):
        self.start(shared)
        hub = shared["hub"]
        was = shared["on_screen"]()
        card = hub.deck[hub.session.current]
        shared["post"]("/answer", {"text": card.answers[0]})
        shared["pump"](4.0)
        assert shared["on_screen"]() != was, (
            "the window ignored an answer given on another screen")

    def test_answering_in_the_window_moves_the_browser(self, shared):
        self.start(shared)
        hub, window = shared["hub"], shared["window"]
        _, since = shared["browser_card"](0)
        was = shared["on_screen"]()
        card = hub.deck[hub.session.current]
        window.answer_box.setText(card.answers[0])
        window._submit_typed()
        shared["pump"](4.0)
        theirs, _ = shared["browser_card"](since)
        assert shared["on_screen"]() != was
        assert theirs == shared["on_screen"](), (
            "the browser is still showing the card the window has left")

    def test_there_is_only_ever_one_session(self, shared):
        self.start(shared)
        assert shared["window"].worker.session is shared["hub"].session
