"""Capture behaviour, with a fake sound device.

The bug worth guarding here: checking mid-window blocks the capture loop for a
second or more while a model runs. The audio callback keeps filling its queue
throughout, and if nothing pulls those blocks back out, the saved clip is
missing exactly the part recorded during the check. Windows set to 5s were
saving as little as 1s, and that truncated clip was what got sent for a second
opinion.
"""
import threading
import time

import numpy as np
import pytest

from spanish_drill import audio as audio_module
from spanish_drill.audio import Recorder, read_wav, resolve_device, save_wav
from spanish_drill.config import SAMPLE_RATE

BLOCK = int(SAMPLE_RATE * 0.05)


class FakeStream:
    """Feeds blocks to the callback on its own thread, like PortAudio does.

    Mirrors the start/stop/close lifecycle the Recorder now uses: the stream is
    opened once per session rather than once per card.
    """

    def __init__(self, callback, level=0.0, **_):
        self.callback = callback
        self.level = level
        self._stop = threading.Event()
        self._thread = None
        self.starts = 0
        self.stops = 0

    def _pump(self):
        while not self._stop.is_set():
            block = np.full((BLOCK, 1), self.level, dtype=np.float32)
            self.callback(block, BLOCK, None, None)
            time.sleep(0.05)

    def start(self):
        self.starts += 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self):
        self.stops += 1
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def close(self):
        self.stop()


def _install(monkeypatch, level):
    """One stream instance, so the test can count how often it is opened.

    The recorder deliberately leaves streams open now, so the fixture has to
    stop them itself or the pump threads pile up and abort the interpreter.
    """
    made = []

    def factory(**kw):
        stream = FakeStream(kw["callback"], level=level)
        made.append(stream)
        return stream

    monkeypatch.setattr(audio_module.sd, "InputStream", factory)
    yield made
    for stream in made:
        stream.stop()


@pytest.fixture
def loud(monkeypatch):
    """A device that is always emitting well above the noise floor."""
    yield from _install(monkeypatch, 0.5)


@pytest.fixture
def quiet(monkeypatch):
    yield from _install(monkeypatch, 0.0)


class TestWindow:
    def test_the_whole_window_is_captured(self, loud):
        r = Recorder()
        r.floor = 0.01
        _, window, heard, _ = r.record(1.0)
        assert heard
        assert len(window) / SAMPLE_RATE >= 0.85

    def test_silence_still_keeps_the_audio(self, quiet):
        """A miss blamed on silence is only ever provable from the recording."""
        r = Recorder()
        r.floor = 0.01
        speech, window, heard, _ = r.record(1.0)
        assert speech is None and not heard
        assert window is not None and len(window) / SAMPLE_RATE >= 0.85

    def test_a_check_that_outlasts_the_window_does_not_lose_audio(self, loud):
        """The regression, in the shape that actually caused it.

        The check has to still be running when the window expires. The loop
        then exits immediately, and everything the callback queued while the
        model was busy is stranded unless it is drained afterwards. This is why
        a 5s window was saving barely a second of audio.
        """
        r = Recorder()
        r.floor = 0.01
        r._pause_threshold = 0.0        # check at the first opportunity
        r._check_interval = 0.0

        def slow_check(partial):
            time.sleep(1.2)             # outlasts the window below
            return False

        _, window, _, _ = r.record(0.6, on_pause=slow_check)
        captured = len(window) / SAMPLE_RATE
        assert captured >= 1.0, (
            f"audio recorded during the check was dropped: kept {captured:.2f}s, "
            f"but the microphone ran for at least 1.2s")


    def test_a_later_check_sees_what_was_recorded_during_an_earlier_one(self, loud):
        """A second attempt spoken while the first check was running must be
        part of what the next check looks at, or repeating yourself is wasted.
        """
        r = Recorder()
        r.floor = 0.01
        r._pause_threshold = 0.0
        r._check_interval = 0.0
        sizes = []

        def check(partial):
            sizes.append(len(partial) / SAMPLE_RATE)
            time.sleep(0.3)
            return False

        r.record(1.5, on_pause=check)
        assert len(sizes) >= 2, "expected more than one check in the window"
        assert sizes[1] > sizes[0] + 0.2, (
            f"the second check saw {sizes[1]:.2f}s, barely more than the first "
            f"{sizes[0]:.2f}s: audio from during the first check was dropped")


class TestEarlyExit:
    def test_accepting_ends_the_window_immediately(self, loud):
        r = Recorder()
        r.floor = 0.01
        r._pause_threshold = 0.0
        r._check_interval = 0.0
        started = time.time()
        speech, _, _, early = r.record(5.0, on_pause=lambda partial: True)
        assert early and speech is not None
        assert time.time() - started < 2.0, "should not wait out the window"

    def test_refusing_runs_the_full_window(self, loud):
        r = Recorder()
        r.floor = 0.01
        started = time.time()
        _, _, _, early = r.record(0.8, on_pause=lambda partial: False)
        assert not early
        assert time.time() - started >= 0.7

    def test_should_stop_ends_it(self, loud):
        r = Recorder()
        r.floor = 0.01
        _, _, _, early = r.record(5.0, should_stop=lambda: True)
        assert not early


class TestStreamLifetime:
    """The microphone is opened once per session, never once per card.

    Opening and closing a CoreAudio stream per card deadlocks: the drill hung
    twice inside AudioOutputUnitStop waiting on CoreAudio's HAL mutex, with
    every thread parked at zero percent CPU.
    """

    def test_many_recordings_open_one_stream(self, loud):
        r = Recorder()
        r.floor = 0.01
        for _ in range(5):
            r.record(0.2)
        assert len(loud) == 1, f"opened {len(loud)} streams for 5 cards"
        assert loud[0].starts == 1
        assert loud[0].stops == 0, "the stream must stay open between cards"

    def test_calibrating_reuses_the_same_stream(self, loud):
        r = Recorder()
        r.calibrate(seconds=0.2)
        r.record(0.2)
        assert len(loud) == 1

    def test_close_releases_it(self, loud):
        r = Recorder()
        r.record(0.2)
        r.close()
        assert loud[0].stops >= 1

    def test_closing_twice_is_harmless(self, loud):
        r = Recorder()
        r.record(0.2)
        r.close()
        r.close()

    def test_a_failure_to_close_does_not_propagate(self, loud):
        r = Recorder()
        r.record(0.2)
        stream = loud[0]
        real_stop = stream.stop
        stream.stop = lambda: (_ for _ in ()).throw(OSError("HAL busy"))
        try:
            r.close()   # must not raise
        finally:
            stream.stop = real_stop

    def test_changing_device_replaces_the_stream(self, loud, monkeypatch):
        monkeypatch.setattr(audio_module, "input_devices",
                            lambda: [(0, "Built-in"), (3, "Headset")])
        r = Recorder()
        r.record(0.2)
        r.set_device("Headset")
        r.record(0.2)
        assert len(loud) == 2, "a new input needs a new stream"

    def test_stale_audio_is_dropped_before_a_new_card(self, loud):
        """Whatever was captured while the prompt was spoken is not an answer."""
        r = Recorder()
        r.floor = 0.01
        r.open()
        time.sleep(0.4)                 # audio piles up while "speaking"
        _, window, _, _ = r.record(0.3)
        assert len(window) / SAMPLE_RATE < 0.55


class TestDeviceResolution:
    def test_unknown_name_falls_back_to_the_default(self):
        assert resolve_device("no such microphone") is None

    def test_empty_name_is_the_default(self):
        assert resolve_device("") is None

    def test_a_known_name_resolves(self, monkeypatch):
        monkeypatch.setattr(audio_module, "input_devices",
                            lambda: [(0, "Built-in"), (3, "Headset")])
        assert resolve_device("Headset") == 3


class TestWavRoundTrip:
    def test_audio_survives_a_save_and_load(self, tmp_path):
        original = np.sin(np.linspace(0, 40, SAMPLE_RATE)).astype(np.float32) * 0.5
        path = tmp_path / "clip.wav"
        save_wav(path, original)
        assert np.allclose(read_wav(path), original, atol=1e-4)

    def test_clipping_is_handled(self, tmp_path):
        path = tmp_path / "loud.wav"
        save_wav(path, np.full(1000, 3.0, dtype=np.float32))
        assert read_wav(path).max() <= 1.0
