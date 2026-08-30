"""Speaking.

Phrases are recorded once and replayed from disk; see voice.py for why. This
module is the vocabulary the drill speaks in, and stays free of the mechanics.

Interruptible on purpose: a Stop that waits out the rest of a sentence reads as
a frozen button.
"""
from . import voice
from .config import ENGLISH_RATE, VOICES


def say(text, voice_name=None, rate=ENGLISH_RATE):
    """Speak, blocking until finished or interrupted."""
    voice.speak(text, voice_name, rate or ENGLISH_RATE)


def stop_speaking():
    """Cut off whatever is being spoken, so Stop feels immediate."""
    voice.stop()


def spanish_voice(dialect):
    return VOICES.get(dialect, VOICES["es-MX"])


def say_english(text):
    say(text, rate=ENGLISH_RATE)


def say_spanish(text, dialect):
    say(text, voice_name=spanish_voice(dialect))
