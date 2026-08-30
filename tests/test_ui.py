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
        self.last_audio = None

    def calibrate(self):
        time.sleep(self.calibrate_seconds)
        self.calibrated = True
        return self.floor

    def listen(self, window, should_stop=None, accept=None):
        return None

    def set_device(self, name):
        pass


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
