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

# The sentence drill needs the opposite steer. Telling the recogniser to
# expect isolated words while somebody says a whole clause is worse than
# saying nothing: it chops the utterance into fragments and drops the small
# words, which is exactly what a sentence is graded on.
#
# No example words at all here, deliberately. Examples exist in the word
# steer to break a one-word ambiguity that context cannot resolve; a sentence
# carries its own context, so the examples would only be more text to leak.
SENTENCE_STEER = "Una frase corta y sencilla en español, dicha una sola vez."

# The conjugation drill needs its own, and for a measurable reason. Every
# example in STEER is an infinitive or a noun, so the recogniser is told to
# expect dictionary forms and duly produces them: in one real session it wrote
# "seguir" for `sigo`, "poner" for `pongo`, "salir" for `sales` and "oir" for
# `oigo`. Four different verbs, all pulled the same way.
#
# No example words in the text, for the reason SENTENCE_STEER gives and one
# measured here. Examples are the part a model hands back instead of a
# transcript, and on nine real clips a four-example version echoed on two of
# them while the instruction alone never did. The instruction is what fixes
# the infinitive bias; the examples were only ever costing echoes.
CONJUGATION_STEER = "Un verbo español conjugado, una sola palabra."


def _conjugations_of(verbs):
    """Every regular form of these verbs, minus anything the deck teaches.

    Not examples — the steer names none. This is the net underneath it: the
    word steer does list `pintar` and friends, and a model asked for a
    conjugated verb hands them back conjugated. `pinto` cannot be a real
    answer, because none of these verbs is in the deck.

    The subtraction is the important half. `nada` is what nadar does and also
    an ordinary word this deck teaches, and dismissing a real answer as an
    echo would mark a correct card wrong. Anything the deck can account for
    is left out and judged on its merits.
    """
    from .conjugation import CONDITIONAL_ENDINGS, FUTURE_ENDINGS, REGULAR
    answers = {normalize(a) for c in load_deck() for a in c.answers}
    out = set()
    for verb in verbs:
        stem, ending = verb[:-2], verb[-2:]
        for tense in ("pres", "pret", "imp"):
            out.update(normalize(stem + e) for e in REGULAR[tense][ending])
        out.update(normalize(verb + e) for e in FUTURE_ENDINGS)
        out.update(normalize(verb + e) for e in CONDITIONAL_ENDINGS)
    return frozenset(w for w in out if w and w not in answers)


CONJUGATION_STEER_WORDS = _conjugations_of(
    tuple(w for w in sorted(STEER_WORDS) if w.endswith(("ar", "er", "ir"))))


def assert_steer_is_clean():
    """A steer word that is also a deck answer would give that card away.

    Every steer's examples are checked, not just the word drill's. A
    conjugation example that happened to be a form in the deck would be the
    same failure wearing a different tense.
    """
    answers = {normalize(a) for c in load_deck() for a in c.answers}
    clash = sorted((STEER_WORDS | CONJUGATION_STEER_WORDS) & answers)
    if clash:
        raise AssertionError(f"steer leaks deck answers: {clash}")


def _fold(text):
    """Normalise for comparing a transcript against a prompt.

    `ñ` goes here, where it survives everywhere else. A recogniser echoing
    "un verbo espanol" without the tilde is still echoing, and letting one
    character decide otherwise meant a prompt came back looking like an
    answer.
    """
    return normalize(text).replace("ñ", "n")


@lru_cache(maxsize=8)
def _steer_vocabulary(steer):
    return frozenset(_fold(steer).split())


def is_steer_echo(text, steer=None):
    """True when the recogniser handed our own prompt back instead of speech.

    Four shapes, all seen in practice: the whole prompt, an opening fragment
    of it, an utterance built from nothing but its words, and a single
    example word. The last matters because a lone "pintar" once cost a card a
    wrong answer, and no example word exists in the deck, so one on its own
    can never be a real answer.

    Every check runs against whichever steer was actually used. The fallback
    used to compare with the word steer no matter what was passed, so a
    conjugation prompt coming back was measured against the wrong text and
    got through as a transcript.
    """
    n = _fold(text)
    if not n:
        return False
    used = steer or STEER
    clean = _fold(used)
    if n == clean or (len(n) >= 12 and n in clean):
        return True
    words = n.split()
    examples = STEER_WORDS | CONJUGATION_STEER_WORDS
    if len(words) == 1 and words[0] in examples:
        return True
    if len(set(words) & examples) >= 3:
        return True
    # Every word came out of the prompt, and there is enough of it that the
    # coincidence is not credible. Three words is the floor because every
    # answer this drill grades is one word.
    if len(words) >= 3 and all(w in _steer_vocabulary(used) for w in words):
        return True
    return False


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

    def _decode(self, model, audio, prompt):
        segments, _ = model.transcribe(
            audio, language="es", beam_size=5, temperature=0,
            vad_filter=False, initial_prompt=prompt)
        return " ".join(s.text for s in segments).strip()

    def transcribe(self, audio, scout=False, steer=None, retry=None):
        """Returns text, or None when nothing usable came back.

        On an echo the clip is decoded again with no prompt, exactly as the
        second opinion has always done, and for the same reason: the prompt
        is what caused the echo. Until this was here, an echo made the local
        model return nothing at all, the answer was recorded as silence, and
        a card you had shouted at came back marked as unattempted. Measured
        on nine real clips, the steer came back instead of a transcript on
        five of them.

        The retry is never triggered by the answer failing to appear. Asking
        again until it does would be a slower way of leaking it.

        `retry` is about whether this decode is the final answer, not about
        which model is doing it. The early-accept poll turns it off: it runs
        at every pause and only answers "have they said it yet", so a second
        decode there costs more than it buys and the next pause tries again
        anyway. Whichever model produces the answer that gets graded keeps
        the retry, because that is the one whose silence becomes a mark.
        """
        model = self.scout if scout else self.model
        prompt = STEER if steer is None else steer
        if retry is None:
            # The scout only ever answers "have they said it yet", at every
            # pause, and the next pause asks again a moment later. A second
            # decode there buys nothing and costs the early accept.
            retry = not scout
        text = self._decode(model, audio, prompt)
        if text and not is_steer_echo(text, prompt):
            return text
        if not retry:
            return None
        again = self._decode(model, audio, None)
        if again and not is_steer_echo(again, prompt):
            return again
        return None


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


def second_opinion(path, model=VERIFY_MODEL, transcriber=openai_transcribe,
                   steer=STEER):
    """Ask once with the steer; if the steer comes back, ask again without it.

    Returns (text, echoed). The retry is triggered only by the model echoing
    our prompt, never by it failing to say the expected answer. Retrying until
    the answer appears would be a slower way of leaking it.

    `steer` is a parameter because the sentence drill needs a different one:
    the word steer tells the recogniser to expect isolated words, which is
    the wrong instruction when somebody has just said a whole clause.
    """
    text = transcriber(path, model, steer)
    if text and not is_steer_echo(text, steer):
        return text, False
    retry = transcriber(path, model, None)
    if retry and not is_steer_echo(retry, steer):
        return retry, False
    return None, True


def sentence_second_opinion(path, model=VERIFY_MODEL,
                            transcriber=openai_transcribe):
    """The same re-check, steered for a whole sentence rather than a word."""
    return second_opinion(path, model, transcriber, steer=SENTENCE_STEER)


def conjugation_second_opinion(path, model=VERIFY_MODEL,
                               transcriber=openai_transcribe):
    """The same re-check, steered for a conjugated form rather than a lemma.

    Worth its own steer for the reason CONJUGATION_STEER gives: told to
    expect infinitives, both recognisers reach for one.
    """
    return second_opinion(path, model, transcriber, steer=CONJUGATION_STEER)
