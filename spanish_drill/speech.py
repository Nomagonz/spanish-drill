"""Speaking, via the macOS `say` voice.

Interruptible on purpose: a Stop that waits out the rest of a sentence reads as
a frozen button.
"""
import subprocess
import threading

from .config import ENGLISH_RATE, VOICES

_lock = threading.Lock()
_current = None


def say(text, voice=None, rate=None):
    """Speak, blocking until finished or interrupted."""
    global _current
    if not text:
        return
    command = ["say"]
    if voice:
        command += ["-v", voice]
    if rate:
        command += ["-r", str(rate)]
    try:
        with _lock:
            _current = subprocess.Popen(command + [text],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        _current.wait()
    except (FileNotFoundError, OSError):
        pass                        # no `say` on this machine; run silently
    finally:
        with _lock:
            _current = None


def stop_speaking():
    """Cut off whatever is being spoken, so Stop feels immediate."""
    with _lock:
        process = _current
    if process is not None:
        try:
            process.terminate()
        except Exception:
            pass


def spanish_voice(dialect):
    return VOICES.get(dialect, VOICES["es-MX"])


def say_english(text):
    say(text, rate=ENGLISH_RATE)


def say_spanish(text, dialect):
    say(text, voice=spanish_voice(dialect))
