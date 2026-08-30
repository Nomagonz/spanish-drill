"""SM-2 spaced repetition.

Every card carries its own ease factor, so difficulty is per word rather than
one ladder for everything: a word you keep fumbling grows slower permanently,
and an easy one accelerates.
"""
import time
from dataclasses import asdict, dataclass, field

from .config import DAY_SECONDS

EASE_START = 2.5
EASE_MIN = 1.3          # SM-2's floor; below this intervals stop growing
FIRST_INTERVAL = 1      # days, after the first successful recall
SECOND_INTERVAL = 6     # days, after the second
LEECH_AT = 8            # lapses before a word is called out as a problem
MATURE_AT = 21          # days; Anki's threshold for "this one has stuck"
PASSING_QUALITY = 3     # below this is a failure

_LEITNER_LADDER = (0, 1, 3, 7, 16, 35, 90, 180)     # the original scheduler


def today():
    return int(time.time() // DAY_SECONDS)


@dataclass
class Card:
    ease: float = EASE_START
    interval: int = 0       # days until it comes back
    reps: int = 0           # consecutive successes
    lapses: int = 0         # times it has been forgotten
    due: int = 0            # day number

    @classmethod
    def new(cls, today=None):
        return cls(due=today if today is not None else globals()["today"]())

    @classmethod
    def from_dict(cls, d):
        return cls(ease=d["ease"], interval=d["interval"], reps=d["reps"],
                   lapses=d["lapses"], due=d["due"])

    def to_dict(self):
        return asdict(self)


def migrate(d, today=None):
    """Accept a card written by either scheduler.

    The original used Leitner boxes. Those saves keep their place on the ladder
    and their lapse history, and start from the default ease.
    """
    if d is None:
        return None
    if "ease" in d:
        return Card.from_dict(d)
    box = d.get("b", 0)
    now = today if today is not None else globals()["today"]()
    return Card(ease=EASE_START,
                interval=_LEITNER_LADDER[min(box, len(_LEITNER_LADDER) - 1)],
                reps=box,
                lapses=d.get("l", 0),
                due=d.get("d", now))


def ease_delta(q):
    """SM-2's ease adjustment for a grade.

    Separated out so a penalty can be reversed exactly when a miss turns out to
    have been the recogniser's fault rather than yours.
    """
    return 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)


def schedule(card, q, today=None):
    """Apply a grade. Mutates and returns the card."""
    now = today if today is not None else globals()["today"]()
    if q >= PASSING_QUALITY:
        if card.reps == 0:
            card.interval = FIRST_INTERVAL
        elif card.reps == 1:
            card.interval = SECOND_INTERVAL
        else:
            card.interval = max(1, round(card.interval * card.ease))
        card.reps += 1
    else:
        card.reps = 0
        card.interval = 0       # due again today, not in a week
        card.lapses += 1
    card.ease = max(EASE_MIN, card.ease + ease_delta(q))
    card.due = now + card.interval
    return card


def unschedule_penalty(card, old_q, new_q):
    """Undo a grade that should never have been applied.

    The ease penalty is reversed arithmetically rather than by restoring a
    snapshot, because a missed word is requeued and may have been answered
    correctly since. That later repetition is real and has to survive.
    """
    card.ease = max(EASE_MIN, card.ease - ease_delta(old_q) + ease_delta(new_q))
    card.lapses = max(0, card.lapses - 1)
    return card


def is_leech(card):
    return card.lapses >= LEECH_AT


def is_mature(card):
    return card.interval >= MATURE_AT


def describe_interval(interval):
    if interval == 0:
        return "again this session"
    if interval == 1:
        return "again tomorrow"
    if interval < 30:
        return f"again in {interval} days"
    months = round(interval / 30)
    return f"again in {months} month" + ("" if months == 1 else "s")


def describe_state(card):
    if card is None:
        return "New word"
    if is_leech(card):
        return f"Leech · missed {card.lapses}x"
    if card.reps == 0:
        return "Relearning"
    return f"Review · {card.interval}d · ease {card.ease:.2f}"
