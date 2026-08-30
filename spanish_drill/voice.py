"""Pre-recorded prompts, played back through one long-lived process.

Every phrase the drill speaks is rendered to a file once and reused. That is
both faster and better sounding than synthesising live: rendering is not
running against a clock, so it can use a higher sample rate than real-time
synthesis, and the same phrase sounds identical every time.

The timings that drove this, measured on a one-second prompt:

    live `say` every time          1.96s
    spawn afplay on a cached file  2.05s      (no better: the cost is the spawn)
    long-lived player              1.02s      (exactly the audio, nothing more)
"""
import hashlib
import subprocess
import sys
import threading
from pathlib import Path

from .config import ENGLISH_RATE, PROMPT_CACHE, SAY_FORMAT, VOICES

_lock = threading.Lock()
_player = None


# -- rendering ------------------------------------------------------------
def _key(text, voice, rate):
    raw = f"{text}|{voice or 'default'}|{rate}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def path_for(text, voice=None, rate=ENGLISH_RATE):
    return Path(PROMPT_CACHE) / f"{_key(text, voice, rate)}.wav"


def render(text, voice=None, rate=ENGLISH_RATE):
    """Record the phrase if it is not already on disk. Returns the path."""
    target = path_for(text, voice, rate)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.wav")
    command = ["say", "-r", str(rate), "-o", str(partial),
               f"--data-format={SAY_FORMAT}"]
    if voice:
        command += ["-v", voice]
    try:
        subprocess.run(command + [text], check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        partial.replace(target)     # only appears once it is complete
        return target
    except Exception:
        partial.unlink(missing_ok=True)
        return None


# -- playback -------------------------------------------------------------
def _ensure_player():
    global _player
    if _player is not None and _player.poll() is None:
        return _player
    try:
        _player = subprocess.Popen(
            [sys.executable, "-m", "spanish_drill._player"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except Exception:
        _player = None
    return _player


def _play(path):
    """Send one file to the player and wait for it to finish. True on success."""
    with _lock:
        player = _ensure_player()
        if player is None:
            return False
        try:
            player.stdin.write(f"{path}\n")
            player.stdin.flush()
            return bool(player.stdout.readline())
        except Exception:
            return False


def speak(text, voice=None, rate=ENGLISH_RATE):
    """Say it, from the recording, falling back to live synthesis."""
    if not text:
        return
    recording = render(text, voice, rate)
    if recording is not None and _play(recording):
        return
    command = ["say", "-r", str(rate)] + (["-v", voice] if voice else []) + [text]
    try:
        subprocess.run(command, check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def stop():
    """Cut playback short. The player is respawned on the next phrase."""
    global _player
    with _lock:
        player, _player = _player, None
        if player is not None:
            try:
                player.kill()
            except Exception:
                pass


def prerecord(phrases, should_stop=None, on_progress=None):
    """Record a batch ahead of time. Returns how many were newly recorded."""
    made = 0
    for i, (text, voice, rate) in enumerate(phrases):
        if should_stop and should_stop():
            break
        if not path_for(text, voice, rate).exists():
            if render(text, voice, rate) is not None:
                made += 1
        if on_progress:
            on_progress(i + 1, len(phrases))
    return made


def phrases_for(deck, dialect, indexes=None):
    """Everything the drill will need to say for these cards."""
    voice = VOICES.get(dialect, VOICES["es-MX"])
    wanted = []
    for index in (indexes if indexes is not None else range(len(deck))):
        card = deck[index]
        wanted.append((card.spoken_prompt, None, ENGLISH_RATE))
        wanted.append((card.answers[0], voice, ENGLISH_RATE))
    return wanted
