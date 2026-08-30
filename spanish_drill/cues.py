"""Short tones for right and wrong.

The normal drill tells you how you did by speaking the answer and an example
sentence, which is how you learn a word but costs several seconds. During a
placement run you are only being sorted, so the feedback has to be instant --
but it still has to exist, or you are answering into a void.
"""
import numpy as np

from .config import SAMPLE_RATE

_FADE_SECONDS = 0.005       # a hard edge on a sine wave is an audible click


def _wave(frequency, seconds, volume):
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * frequency * t) * volume
    fade = min(int(SAMPLE_RATE * _FADE_SECONDS), n // 2)
    if fade:
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
    return wave


def play(*notes, volume=0.18):
    """Play (frequency, seconds) pairs back to back, blocking until done."""
    import sounddevice as sd
    try:
        audio = np.concatenate([_wave(f, s, volume) for f, s in notes])
        sd.play(audio.astype(np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass            # a missing output device must never stop a drill


def correct():
    """Two rising notes. About a sixth of a second."""
    play((660, 0.07), (990, 0.09))


def wrong():
    """One low note, clearly different from the rising pair."""
    play((280, 0.18))
