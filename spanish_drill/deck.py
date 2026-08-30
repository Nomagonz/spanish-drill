"""The vocabulary deck."""
import json
from dataclasses import dataclass
from functools import lru_cache

from .config import DECK_PATH


@dataclass(frozen=True)
class Card:
    prompt: str             # the English cue
    answers: tuple          # accepted Spanish answers, best first
    example: str            # a sentence using it
    gloss: str              # that sentence in English
    pos: str = "other"      # verb, noun, adjective, ... for filtering

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
             gloss=c["gl"], pos=c.get("pos", "other"))
        for c in raw
    )
