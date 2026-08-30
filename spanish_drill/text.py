"""Text normalisation and edit distance.

Pure functions, no state. Kept apart from grading so the comparison rules can
be tested without a deck.
"""
import re
import unicodedata

_LEADING_ARTICLE = re.compile(r"^(el|la|los|las|un|una|unos|unas)\s+")
_NON_LETTERS = re.compile(r"[^a-z0-9ñ ]")
_SPACES = re.compile(r"\s+")


def normalize(s):
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Accents go because the recogniser is inconsistent about them and a missing
    tilde is not a vocabulary mistake. 'ñ' survives: it is a distinct letter,
    not an accented 'n'.
    """
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFD", s.lower())
    without_marks = "".join(
        c for c in decomposed if unicodedata.category(c) != "Mn" or c == "̃"
    )
    recomposed = unicodedata.normalize("NFC", without_marks)
    return _SPACES.sub(" ", _NON_LETTERS.sub(" ", recomposed)).strip()


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
