"""Short tones for right and wrong.

The normal drill tells you how you did by speaking the answer and an example
sentence, which is how you learn a word but costs several seconds. During a
placement run you are only being sorted, so the feedback has to be instant --
but it still has to exist, or you are answering into a void.

Playback goes through `afplay` rather than through sounddevice. The first
version called sd.play() from the drill's worker thread, which deadlocked:
CoreAudio's HAL takes a mutex in AudioOutputUnitStop, and the same process is
constantly opening and closing input streams for the microphone. It hung
permanently mid-session, inside AudioOutputUnitStop, waiting on that lock.
A separate process shares no audio state with our recording, so it cannot.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import save_wav
from .config import SAMPLE_RATE

_FADE_SECONDS = 0.005       # a hard edge on a sine wave is an audible click
_CACHE = Path(tempfile.gettempdir()) / "spanish-drill-cues"

# A bright two-note bell going up, against a dull two-note buzz going down.
# The pitch direction carries the meaning, so it still reads at low volume.
CORRECT = ((988, 0.09), (1319, 0.30))   # B5 -> E6, ringing
WRONG = ((196, 0.16), (147, 0.30))      # G3 -> D3, flat and low
VOLUME = 0.5


def _wave(frequency, seconds, volume):
    """A note with a plucked decay and a quiet octave above it.

    The decay is what makes it read as a ding rather than a flat beep.
    """
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * frequency * t)
    wave += 0.35 * np.sin(2 * np.pi * frequency * 2 * t)
    wave *= np.exp(-t * (3.0 if seconds > 0.2 else 1.2))
    wave *= volume / 1.35
    fade = min(int(SAMPLE_RATE * _FADE_SECONDS), n // 2)
    if fade:
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
    return wave


def _render(name, notes, volume=VOLUME):
    """Write the tone once and reuse the file."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{name}.wav"
    if not path.exists():
        save_wav(path, np.concatenate([_wave(f, s, volume) for f, s in notes]))
    return path


_running = []           # keep handles so finished players can be reaped


def play(name, notes):
    """Start a cue and return immediately.

    Waiting for afplay costs about a second of process startup for a third of a
    second of sound, which is most of the time budget for a card. The tone
    plays while the drill moves on.
    """
    try:
        path = _render(name, notes)
        _running[:] = [p for p in _running if p.poll() is None]     # reap
        _running.append(subprocess.Popen(
            ["afplay", str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    except Exception:
        pass            # no afplay or no output device: carry on silently


def correct():
    play("correct", CORRECT)


def wrong():
    play("wrong", WRONG)
