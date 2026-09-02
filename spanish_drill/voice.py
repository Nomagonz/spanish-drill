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
import array
import hashlib
import io
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from concurrent import futures
from pathlib import Path

from .config import (ENGLISH_RATE, OPENAI_STYLE, OPENAI_TTS_MODEL,
                     OPENAI_VOICES, PROMPT_CACHE, SAY_FORMAT, SPANISH_RATE,
                     TTS_BASE_RATE, TTS_ENGINE, VOICES)

# Which style each stored voice name should be spoken in. The `say` voice names
# are kept as the identity so switching engines does not invalidate the cache
# key, only what gets recorded into it.
_KIND_OF = {None: "english", VOICES["es-MX"]: "es-MX", VOICES["es-ES"]: "es-ES"}

_lock = threading.Lock()
_player = None


# -- rendering ------------------------------------------------------------
def _slug(text):
    """A readable filename. These sit in the project, so they should be legible."""
    keep = [c.lower() if c.isalnum() else "-" for c in unicodedata.normalize(
        "NFD", text) if unicodedata.category(c) != "Mn"]
    slug = re.sub(r"-+", "-", "".join(keep)).strip("-")
    return slug[:48] or "phrase"


def _key(text, voice, rate):
    raw = f"{text}|{voice or 'default'}|{rate}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:6]


def path_for(text, voice=None, rate=ENGLISH_RATE):
    """prompts/english/to-happen-to-occur-1a2b3c.wav

    Named rather than hashed, because this directory lives in the project and
    someone will eventually open it. The short hash keeps two phrases that
    slugify the same from overwriting each other, and pins the voice and rate.
    """
    folder = voice or "english"
    return (Path(PROMPT_CACHE) / folder /
            f"{_slug(text)}-{_key(text, voice, rate)}.wav")


def _normalise(raw, destination):
    """Rewrite a WAV with an honest header.

    The API streams audio and leaves a placeholder length in the header: a
    two-second clip claims to be twenty-four hours long. Anything that trusts
    the header, our player included, then reads nonsense.
    """
    with wave.open(io.BytesIO(raw)) as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        chunks = []
        while True:
            block = source.readframes(4096)
            if not block:
                break
            chunks.append(block)
    frames = b"".join(chunks)
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return len(frames) // max(1, channels * width) / rate


def retempo(source, destination, tempo):
    """Rewrite audio at `tempo` times the speed, without moving the pitch.

    The API has no speed control, and asking for it in the style instructions
    only gets an approximation that varies phrase to phrase. Resampling would
    be exact but raises the pitch: 1.5x resampled is a chipmunk. ffmpeg's
    atempo stretches time and leaves the pitch alone, which is the only one of
    the three that is both exact and listenable.
    """
    if abs(tempo - 1.0) < 0.01:
        return False
    filters = []                    # atempo takes 0.5-2.0, so chain past that
    remaining = tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-filter:a", ",".join(filters), str(destination)],
        check=True, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _speed_up(path, rate):
    """Apply the requested rate to a file the API rendered at its own pace."""
    faster = path.with_suffix(".fast.wav")
    try:
        if retempo(path, faster, rate / TTS_BASE_RATE):
            faster.replace(path)
    except Exception:
        pass                    # keep the natural-speed audio rather than none
    finally:
        faster.unlink(missing_ok=True)


SILENT_RMS = 0.005


def is_audible(path):
    """Did anything actually get said?

    The API occasionally hands back a clip that is entirely silent: correct
    header, plausible length, no sound. Nothing downstream notices, because it
    is a perfectly valid WAV. The drill plays it, the card passes in silence,
    and there is no way to answer a cue you never heard. Ten of five hundred
    English prompts came out this way on the first pass.

    Measured over the whole deck the two populations are nowhere near each
    other: the quietest real prompt sits at 0.027 and the loudest dud at
    0.0012, so the threshold has more than an order of magnitude either side.
    """
    try:
        with wave.open(str(path)) as w:
            if w.getsampwidth() != 2 or not w.getnframes():
                return False
            samples = array.array("h", w.readframes(w.getnframes()))
    except Exception:
        return False
    if not samples:
        return False
    mean_square = sum(s * s for s in samples) / len(samples)
    return (mean_square ** 0.5) / 32768 >= SILENT_RMS


def _render_openai(text, voice, rate, destination, attempts=4):
    from openai import OpenAI
    kind = _KIND_OF.get(voice, "english")
    for attempt in range(attempts):
        try:
            reply = OpenAI().audio.speech.create(
                model=OPENAI_TTS_MODEL, voice=OPENAI_VOICES[kind],
                input=text, response_format="wav",
                instructions=OPENAI_STYLE[kind])
            _normalise(reply.content, destination)
            if not is_audible(destination):
                raise ValueError("the API returned silence")   # ask again
            _speed_up(destination, rate)
            return True
        except Exception:
            if attempt == attempts - 1:
                return False
            time.sleep(0.6 * (2 ** attempt))    # back off, mostly for rate limits
    return False


def _render_say(text, voice, rate, destination):
    command = ["say", "-r", str(rate), "-o", str(destination),
               f"--data-format={SAY_FORMAT}"]
    if voice:
        command += ["-v", voice]
    try:
        subprocess.run(command + [text], check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def render(text, voice=None, rate=ENGLISH_RATE, engine=None):
    """Record the phrase if it is not already on disk. Returns the path."""
    target = path_for(text, voice, rate)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(f".{os.getpid()}-{threading.get_ident()}.partial")
    engine = engine or TTS_ENGINE
    try:
        ok = (_render_openai(text, voice, rate, partial) if engine == "openai"
              else _render_say(text, voice, rate, partial))
        if ok and partial.exists() and is_audible(partial):
            partial.replace(target)     # only appears once it is complete
            return target
        return None
    except Exception:
        return None
    finally:
        partial.unlink(missing_ok=True)


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


def prerecord(phrases, should_stop=None, on_progress=None, workers=None):
    """Record a batch ahead of time. Returns how many were newly recorded.

    Concurrency pays very differently by engine. macOS `say` routes everything
    through one shared synthesis service, so it flattens at about 1.8 phrases a
    second no matter how many processes ask: measured, twelve at once and
    forty-eight at once are the same speed. API calls have no such choke point
    and scale with however many are in flight.
    """
    todo = [p for p in phrases if not path_for(*p).exists()]
    if not todo:
        if on_progress:
            on_progress(len(phrases), len(phrases))
        return 0

    if workers is None:
        workers = 32 if TTS_ENGINE == "openai" else min(12, os.cpu_count() or 4)
    made, done = 0, 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(render, *phrase): phrase for phrase in todo}
        for future in futures.as_completed(pending):
            done += 1
            try:
                if future.result() is not None:
                    made += 1
            except Exception:
                pass                # one bad phrase must not stop the batch
            if on_progress:
                on_progress(done, len(todo))
            if should_stop and should_stop():
                for other in pending:
                    other.cancel()
                break
    return made


def sentence_phrases(sentences, dialect):
    """Everything the sentence drill will say: the cue, and the answer.

    Recorded ahead like the deck's, and for the same reason. Rendering one
    costs about a second and a half, and paying that inside the card that
    needs it reads as the drill having gone quiet rather than as it working.
    """
    voice = VOICES.get(dialect, VOICES["es-ES"])
    wanted, seen = [], set()
    for item in sentences:
        for text, name, rate in ((item.en, None, ENGLISH_RATE),
                                 (item.es, voice, SPANISH_RATE)):
            key = (text, name, rate)
            if text and key not in seen:
                seen.add(key)
                wanted.append(key)
    return wanted


def phrases_for(deck, dialect, indexes=None, examples=True):
    """Everything the drill will ever say for these cards.

    Each phrase is stored under its own text, so a recording made during a
    placement run is the same file the normal drill plays later. That only
    holds if phrases are kept separate: the answer and its example sentence are
    spoken one after the other rather than joined, because a joined string is
    unique to one card and can never be reused.
    """
    voice = VOICES.get(dialect, VOICES["es-ES"])
    wanted, seen = [], set()

    def add(text, voice_name, rate):
        key = (text, voice_name, rate)
        if text and key not in seen:
            seen.add(key)
            wanted.append(key)

    for index in (indexes if indexes is not None else range(len(deck))):
        card = deck[index]
        add(card.spoken_prompt, None, ENGLISH_RATE)     # the English cue
        add(card.answers[0], voice, SPANISH_RATE)       # the answer, spoken back
        if examples:
            add(card.example, voice, SPANISH_RATE)      # read out after a miss
    return wanted
