"""Turning audio into text, locally and via the API.

The steer prompt is the important idea here. With no context, a bare uncommon
infinitive loses to a common phrase that sounds the same: "llevar" decodes as
"y el bar". Telling the recogniser to expect an isolated Spanish word collapses
that ambiguity. It also creates the failure this module spends most of its
effort guarding against, since these models echo the prompt back when the audio
holds nothing usable.
"""
from functools import lru_cache

from .config import SCOUT_MODEL, VERIFY_MODEL
from .deck import load_deck
from .text import normalize

# Every example word is verified absent from the deck by assert_steer_is_clean.
# Naming a word under test would make the recogniser agree with whatever was
# said, and the grade would stop meaning anything.
STEER = ("Palabras sueltas en español, a veces repetidas varias veces. "
         "Por ejemplo: pintar, nadar, saltar, bailar, fresa, morado, tijeras.")
STEER_WORDS = frozenset(
    {"pintar", "nadar", "saltar", "bailar", "fresa", "morado", "tijeras"})


def assert_steer_is_clean():
    """A steer word that is also a deck answer would give that card away."""
    answers = {normalize(a) for c in load_deck() for a in c.answers}
    clash = sorted(STEER_WORDS & answers)
    if clash:
        raise AssertionError(f"steer leaks deck answers: {clash}")


@lru_cache(maxsize=1)
def _steer_vocabulary():
    return frozenset(normalize(STEER).split())


def is_steer_echo(text):
    """True when the recogniser handed our own prompt back instead of speech.

    Three shapes, all seen in practice: the example list, the opening sentence
    on its own, and a single example word. The last one matters because a lone
    "pintar" once cost a card a wrong answer, and none of the example words
    exist in the deck, so one on its own can never be a real answer.
    """
    n = normalize(text)
    if not n:
        return False
    words = n.split()
    if len(words) == 1 and words[0] in STEER_WORDS:
        return True
    if len(set(words) & STEER_WORDS) >= 3:
        return True
    if len(words) >= 3 and all(w in _steer_vocabulary() for w in words):
        return True
    return len(n) >= 12 and n in normalize(STEER)


class LocalTranscriber:
    """faster-whisper, running on this machine."""

    def __init__(self, model_name, scout_name=SCOUT_MODEL):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        # A second, much faster model answers only "have they said it yet".
        # The main model needs seconds to transcribe a one-second snippet,
        # which cannot finish inside an answer window, so using it to check
        # early meant the window always ran out first.
        self.scout = (self.model if scout_name == model_name
                      else WhisperModel(scout_name, device="cpu",
                                        compute_type="int8"))

    def transcribe(self, audio, scout=False):
        """Returns text, or None when nothing usable came back."""
        model = self.scout if scout else self.model
        segments, _ = model.transcribe(
            audio, language="es", beam_size=5, temperature=0,
            vad_filter=False, initial_prompt=STEER)
        text = " ".join(s.text for s in segments).strip()
        if not text or is_steer_echo(text):
            return None
        return text


def openai_transcribe(path, model=VERIFY_MODEL, prompt=STEER):
    """One clip through the API. Returns text, or None if the call fails."""
    from openai import OpenAI
    try:
        client = OpenAI()           # reads OPENAI_API_KEY from the environment
        extra = {"prompt": prompt} if prompt else {}
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model, file=f, language="es",
                response_format="json", **extra)
        return (result.text or "").strip()
    except Exception as exc:                    # network, auth, rate limit
        print(f"    [api error: {type(exc).__name__}: {str(exc)[:70]}]")
        return None


def second_opinion(path, model=VERIFY_MODEL, transcriber=openai_transcribe):
    """Ask once with the steer; if the steer comes back, ask again without it.

    Returns (text, echoed). The retry is triggered only by the model echoing
    our prompt, never by it failing to say the expected answer. Retrying until
    the answer appears would be a slower way of leaking it.
    """
    text = transcriber(path, model, STEER)
    if text and not is_steer_echo(text):
        return text, False
    retry = transcriber(path, model, None)
    if retry and not is_steer_echo(retry):
        return retry, False
    return None, True
