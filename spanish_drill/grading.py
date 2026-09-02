"""Deciding whether what you said counts as the answer.

The bias throughout is toward rejecting. A false reject costs one extra review
of a word you know. A false accept banks a mistake as correct and hides the
word for weeks, which is the failure this whole exercise exists to prevent.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

from .deck import load_deck
from .text import lev, normalize, sounds_as, strip_article, tolerance


@dataclass(frozen=True)
class Match:
    answer: str         # which accepted answer it matched
    close: bool         # matched only by tolerating a transcription slip

    def __bool__(self):
        return True


@lru_cache(maxsize=1)
def _all_deck_answers():
    return frozenset(normalize(a) for c in load_deck() for a in c.answers)


def check(said, card):
    """Return a Match, or None.

    Three passes, strictest first:
      1. the answer, exactly
      2. the answer somewhere inside a longer utterance (repeats, filler)
      3. the answer within a small edit distance, but never when the transcript
         is itself another word in the deck

    The first two passes compare how the words sound as well as how they are
    spelled. Spanish `h` is silent, so a recogniser writes `ago` for `hago`
    and `e` for `he` — it is transcribing what was actually said. Marking
    that wrong told you off for pronouncing the word correctly, and on a
    four-letter answer the edit-distance pass cannot save it either, because
    the tolerance for a short word is zero on purpose.

    A sound match is a full match, not a close one. Nothing was fumbled: the
    letter was never spoken in the first place.
    """
    heard = normalize(said)
    if not heard:
        return None

    variants = [heard]
    stripped = strip_article(heard)
    if stripped != heard:
        variants.append(stripped)

    for variant in variants:
        spoken = sounds_as(variant)
        for answer in card.answers:
            target = normalize(answer)
            sounded = sounds_as(target)
            if variant == target or spoken == sounded:
                return Match(answer, close=False)
            # Some entries carry a trailing preposition ("cerca de"); allow the
            # bare form too.
            if variant == re.sub(r" (de|a|que)$", "", target):
                return Match(answer, close=False)
            if re.search(rf"(^| ){re.escape(target)}( |$)", variant):
                return Match(answer, close=False)
            if re.search(rf"(^| ){re.escape(sounded)}( |$)", spoken):
                return Match(answer, close=False)

    for variant in variants:
        # A transcript that is itself a real deck answer is a different word,
        # not a mangled version of this one. llevar and llegar differ by one
        # character, and accepting one for the other would silently mark a
        # wrong answer correct on the hardest pair in the deck.
        if variant in _all_deck_answers():
            continue
        for answer in card.answers:
            target = normalize(answer)
            if lev(variant, target) <= tolerance(target):
                return Match(answer, close=True)

    return None


# Typed answers are graded exactly like spoken ones, accents included in
# neither. A keyboard could be held to a higher standard, and was: a missing
# accent scored a 3 rather than a 5. It was turned off deliberately. Reaching
# for the option key on every other word is a tax on the mode that exists for
# answering quickly and quietly, and `hablo` for `habló` is a spelling slip,
# not a failure to know the word.


# What a confident typed answer should cost, in seconds. The spoken window is
# the wrong yardstick: saying "ir" and saying "encontrar" take almost the same
# time, and typing them does not. Without a length allowance every long word
# would score as laboured and drift its own schedule, which is a scheduling
# bug wearing a measurement's clothes.
TYPING_BASE_SECONDS = 1.5
TYPING_PER_CHARACTER = 0.25


def typing_budget(answer):
    return TYPING_BASE_SECONDS + TYPING_PER_CHARACTER * len(answer or "")


def typed_quality(ok, close, blank, elapsed, answer):
    """`quality`, with typing's own sense of what counts as quick."""
    if blank:
        return 0
    if not ok:
        return 1
    if close:
        return 3
    return 5 if elapsed <= typing_budget(answer) else 4


COMMANDS = {
    "repeat": ("repite", "repetir", "otra vez", "repeat", "again"),
    "skip": ("salta", "saltar", "skip", "pasa", "siguiente", "next"),
    "stop": ("para", "parar", "alto", "stop", "pausa", "pause"),
    "reveal": ("no se", "no lo se", "dime", "i dont know", "tell me", "pass"),
}


@lru_cache(maxsize=1)
def _command_lookup():
    return {normalize(phrase): name
            for name, phrases in COMMANDS.items() for phrase in phrases}


def command_of(said):
    """Which control word this is, or None.

    Deliberately an exact match on the whole utterance: "no sé" is a command,
    but a card whose answer contains it should not be hijacked.
    """
    return _command_lookup().get(normalize(said))


def quality(ok, close, silent, elapsed, window):
    """Map an answer onto SM-2's 0-5 scale.

    Only four things are observable: whether it matched, whether it needed
    tolerance, whether anything was said at all, and how long it took.
    Hesitation is real evidence, so an instant recall outranks a laboured one.
    """
    if silent:
        return 0            # no attempt is worse than a wrong attempt
    if not ok:
        return 1
    if close:
        return 3
    return 5 if elapsed <= window * 0.45 else 4
