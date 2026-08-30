"""The right/wrong tones."""
import numpy as np
import pytest

from spanish_drill import cues
from spanish_drill.config import SAMPLE_RATE


@pytest.fixture
def captured(monkeypatch):
    played = []
    monkeypatch.setattr(cues, "play", lambda name, notes: played.append(notes))
    return played


class TestCues:
    def test_right_and_wrong_are_distinguishable(self, captured):
        cues.correct()
        cues.wrong()
        right, bad = captured
        assert {f for f, _ in right} != {f for f, _ in bad}

    def test_correct_rises(self, captured):
        cues.correct()
        pitches = [f for f, _ in captured[0]]
        assert pitches == sorted(pitches) and len(set(pitches)) > 1

    def test_wrong_is_lower_than_correct(self, captured):
        cues.correct()
        cues.wrong()
        assert max(f for f, _ in captured[1]) < min(f for f, _ in captured[0])

    def test_both_are_brief(self, captured):
        """Long enough to ring, short enough not to run into the next card."""
        cues.correct()
        cues.wrong()
        for notes in captured:
            assert 0.2 < sum(s for _, s in notes) < 0.7


class TestWaveform:
    def test_it_is_the_requested_length(self):
        wave = cues._wave(440, 0.1, 0.2)
        assert len(wave) == pytest.approx(SAMPLE_RATE * 0.1, abs=2)

    def test_it_stays_in_range(self):
        assert np.abs(cues._wave(440, 0.1, 0.9)).max() <= 1.0

    def test_it_fades_in_and_out(self):
        """A hard edge on a sine wave is an audible click."""
        wave = cues._wave(440, 0.1, 0.5)
        assert abs(wave[0]) < 0.01 and abs(wave[-1]) < 0.01

    def test_a_very_short_tone_does_not_break(self):
        assert len(cues._wave(440, 0.001, 0.5)) > 0


class TestRobustness:
    def test_a_missing_player_does_not_stop_the_drill(self, monkeypatch):
        monkeypatch.setattr(cues.subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("no afplay")))
        cues.correct()      # must not raise

    def test_it_does_not_wait_for_playback(self, monkeypatch):
        """Waiting costs ~1s of process startup for a third of a second of
        sound, which is most of the time budget for a card."""
        waited = []
        class FakeProc:
            def poll(self): return 0
            def wait(self, *a): waited.append(True)
        monkeypatch.setattr(cues.subprocess, "Popen", lambda *a, **k: FakeProc())
        cues.correct()
        assert not waited

    def test_it_never_uses_the_in_process_audio_library(self):
        """sd.play() from the worker thread deadlocked inside CoreAudio's
        AudioOutputUnitStop, because the same process constantly opens and
        closes input streams for the microphone. Playing out of process shares
        no audio state with our recording, so it cannot deadlock against it.
        """
        import ast
        tree = ast.parse(open(cues.__file__, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "sounddevice" not in imported, (
            "cues must not touch the audio library the microphone uses")

    def test_finished_players_are_reaped(self, monkeypatch):
        class Done:
            def poll(self): return 0
        monkeypatch.setattr(cues.subprocess, "Popen", lambda *a, **k: Done())
        for _ in range(30):
            cues.correct()
        assert len(cues._running) <= 1
