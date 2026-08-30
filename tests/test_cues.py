"""The right/wrong tones."""
import numpy as np
import pytest

from spanish_drill import cues
from spanish_drill.config import SAMPLE_RATE


@pytest.fixture
def captured(monkeypatch):
    played = []
    monkeypatch.setattr(cues, "play",
                        lambda *notes, **kw: played.append(notes))
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
        cues.correct()
        cues.wrong()
        for notes in captured:
            assert sum(s for _, s in notes) < 0.35, "feedback must not slow the run"


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
    def test_a_missing_output_device_does_not_stop_the_drill(self, monkeypatch):
        import sounddevice as sd
        monkeypatch.setattr(sd, "play",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("no device")))
        cues.correct()      # must not raise
