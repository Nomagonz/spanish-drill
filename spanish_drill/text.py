"""Text normalisation and edit distance.

Pure functions, no state. Kept apart from grading so the comparison rules can
be tested without a deck.
"""
import re
import unicodedata

_LEADING_ARTICLE = re.compile(r"^(el|la|los|las|un|una|unos|unas)\s+")
_NON_LETTERS = re.compile(r"[^a-z0-9ñ ]")
_NON_LETTERS_ACCENTED = re.compile(r"[^a-z0-9ñáéíóúü ]")
_SPACES = re.compile(r"\s+")


def normalize(s, accents=False):
    """Lowercase, strip punctuation, collapse whitespace.

    Accents go by default because the recogniser is inconsistent about them and
    a missing tilde is not a vocabulary mistake. 'ñ' survives either way: it is
    a distinct letter, not an accented 'n'.

    `accents=True` keeps them, for the one caller that can fairly be strict.
    Speech has an excuse for losing an accent and a keyboard does not, so a
    typed answer is compared against a form that still carries them.
    """
    if not s:
        return ""
    if accents:
        recomposed = unicodedata.normalize("NFC", s.lower())
        return _SPACES.sub(" ", _NON_LETTERS_ACCENTED.sub(" ", recomposed)).strip()
    decomposed = unicodedata.normalize("NFD", s.lower())
    without_marks = "".join(
        c for c in decomposed if unicodedata.category(c) != "Mn" or c == "̃"
    )
    recomposed = unicodedata.normalize("NFC", without_marks)
    return _SPACES.sub(" ", _NON_LETTERS.sub(" ", recomposed)).strip()


# Three spellings that Spanish pronounces identically, so a recogniser
# writing down what it heard can land on either one and be right.
#
#   `h` is silent everywhere except in the digraph `ch`, so `hago` is `ago`,
#       `he` is `e`, `ha` is `a`, `hay` is `ay`.
#   `b` and `v` are one phoneme, /b/, with no distinction whatsoever. This is
#       not a dialect: no Spanish speaker anywhere separates them. `voy` and
#       the English `boy` come out of a recogniser interchangeably.
#   `ll` and `y` merged for all but a few speakers (yeísmo), so `llamar` is
#       often written `yamar`.
#
# Applied to both sides of a comparison, never to anything stored or shown.
_SILENT_H = re.compile(r"(?<!c)h")


def sounds_as(s):
    """The word reduced to how it is actually pronounced.

    Only for comparing. Never for display or storage: the deck's spelling is
    the right spelling, and this exists so a transcript that spells a silent
    letter out of existence, or picks the other letter for the same sound, is
    still recognised as the same word.

    Measured against all 1579 deck answers, every rule here together merges
    exactly one pair, `o` and `oh`, an interjection and a conjunction that no
    card asks for in a way the other could satisfy. That is the test these
    rules have to keep passing: they may fix a spelling, never merge two
    words the deck actually distinguishes.

    Note what is deliberately absent. `p` and `b` are separate phonemes and
    stay separate, so `paz` is never `vas`. Tolerating a single edit on short
    words would cover far more transcription slips and is refused for the
    same reason: 612 pairs of real deck answers sit one edit apart, `anda`
    and `andan` and `amiga` and `amigo` among them, and a conjugation drill
    whose whole subject is the last letter of a word cannot afford that.
    """
    return _SILENT_H.sub("", s).replace("v", "b").replace("ll", "y")


def strip_article(s):
    """'la casa' -> 'casa'. Saying the article is not an error."""
    return _LEADING_ARTICLE.sub("", s)


def lev(a, b):
    """Levenshtein distance, iterative with two rows."""
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def tolerance(word):
    """How far off a transcript may be and still count as the same word.

    Scaled by length: one character wrong in "ir" is a different word, while
    one wrong in "encontrar" is a recogniser artefact.
    """
    n = len(word)
    if n <= 5:
        return 0
    return 1 if n <= 8 else 2
