"""The vocabulary deck."""
import json
from dataclasses import dataclass
from functools import lru_cache

from .config import DECK_PATH
from .conjugation import PRONOUN

# "you all" before "you", so the longer subject is never read as the short one.
_SUBJECTS = tuple(sorted(set(PRONOUN.values()), key=len, reverse=True))


@dataclass(frozen=True)
class Card:
    prompt: str             # the English cue
    answers: tuple          # accepted Spanish answers, best first
    example: str            # a sentence using it
    gloss: str              # that sentence in English
    pos: str = "other"      # verb, noun, adjective, ... for filtering

    # Saved progress is keyed by this, never by position. Editing the deck
    # moves cards around, and a position-keyed save silently hands one word's
    # history to whichever word landed in its slot.
    id: str = ""
    # Set on conjugated forms: which infinitive they belong to, and which one
    # they are. Empty on ordinary vocabulary.
    lemma: str = ""
    form: str = ""          # "pres-yo", "pret-ellos", ...

    @property
    def subject(self):
        """Who the cue is about, on a conjugated form: "you" out of "you hear".

        It is the whole difference between one form and the next, and in the
        ordinary drill it arrives buried in a stream of vocabulary cues that
        have no subject at all. Handed back separately so the screen can mark
        it, rather than leaving the reader to spot the one word that decides
        the answer.

        Empty on ordinary vocabulary, including a card whose cue happens to
        open with the same word: "you (informal)" is the pronoun itself being
        taught, not a paradigm.
        """
        if not self.form:
            return ""
        for who in _SUBJECTS:
            if self.prompt == who or self.prompt.startswith(who + " "):
                return who
        return ""

    @property
    def spoken_prompt(self):
        """The cue as it should be read aloud, parenthetical included.

        The parenthetical is the whole point on cards like "to know (a fact)"
        and "to know (a person or place)": drop it and both prompts sound
        identical, and there is no way to tell which answer is wanted. The
        brackets become a comma so the voice pauses instead of reading them.
        """
        spoken = self.prompt.replace("(", ", ").replace(")", "")
        spoken = spoken.replace(" ,", ",").replace(",,", ",")
        return " ".join(spoken.split()).strip(" ,")


def index_by_id(deck=None):
    """id -> position. Built fresh; the deck is loaded once and cached anyway."""
    return {c.id: i for i, c in enumerate(deck or load_deck())}


def categories(deck=None):
    """Every part of speech present, most common first."""
    from collections import Counter
    counts = Counter(c.pos for c in (deck or load_deck()))
    return [pos for pos, _ in counts.most_common()]


@lru_cache(maxsize=1)
def load_deck(path=None):
    with open(path or DECK_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return tuple(
        Card(prompt=c["en"], answers=tuple(c["es"]), example=c["ex"],
             gloss=c["gl"], pos=c.get("pos", "other"),
             # A deck written before ids existed keys off its first answer,
             # which is unique across the vocabulary.
             id=c.get("id") or c["es"][0],
             lemma=c.get("lemma", ""), form=c.get("form", ""))
        for c in raw
    )
