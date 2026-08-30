"""Rapid placement test.

Sorts the deck into words you already know and words you need to learn, as
fast as you can answer. Get one right twice and it is treated as known; get it
wrong once and it goes straight into the learning pile.

This exists because the normal drill introduces 20 new words a day, which takes
thirteen days to even show you all 253. If you already know a third of them,
that is a lot of time spent proving it one day at a time.

The trade is deliberate and worth stating: two quick correct answers is a much
weaker signal than SM-2's four recalls spread over three weeks. A word passed
here is assumed known rather than demonstrated known, so the first real review
lands three weeks out. If that turns out to be too generous, lower
`passes_needed` expectations rather than trusting the interval.
"""
from collections import defaultdict

from .scheduler import EASE_START, MATURE_AT, Card, today
from .session import DrillSession, Outcome

PASSES_TO_PASS = 2      # correct answers needed to count a word as known


def mark_known(progress, index, passes, today_=None):
    """Treat the word as already learned: park it three weeks out."""
    now = today_ if today_ is not None else today()
    progress.cards[index] = Card(ease=EASE_START, interval=MATURE_AT,
                                 reps=passes, lapses=0, due=now + MATURE_AT)
    return progress.cards[index]


def mark_to_learn(progress, index, today_=None):
    """Put the word at the front of the normal drill: due now, no progress."""
    now = today_ if today_ is not None else today()
    progress.cards[index] = Card(ease=EASE_START, interval=0, reps=0,
                                 lapses=0, due=now)
    return progress.cards[index]


class PlacementSession(DrillSession):
    """A DrillSession that triages instead of scheduling.

    Everything about asking, listening, grading and the second opinion is
    inherited unchanged. Only what happens to the card afterwards differs.
    """

    def __init__(self, *args, passes_needed=PASSES_TO_PASS, limit=None, **kw):
        super().__init__(*args, **kw)
        self.passes_needed = passes_needed
        self.limit = limit                  # cards to try, None for the deck
        self.correct_so_far = defaultdict(int)
        self.known = []
        self.to_learn = []

    # -- what to ask ------------------------------------------------------
    def next_queue(self):
        """Every unclassified card, ignoring the daily new-word cap.

        The cap exists to keep tomorrow's review load survivable. Nothing here
        creates review load, so it does not apply.
        """
        if self.known or self.to_learn:
            return []                       # one pass; requeues happen inline
        if self.progress.queue_override is not None:
            return list(self.progress.queue_override)
        pending = [i for i in range(len(self.deck))
                   if i not in self.progress.cards
                   and self.progress.in_category(i, self.deck)]
        self.rng.shuffle(pending)
        return pending[: self.limit] if self.limit else pending

    # -- what happens afterwards -----------------------------------------
    def _apply(self, index, q):
        from .scheduler import PASSING_QUALITY
        if q >= PASSING_QUALITY:
            self.correct_so_far[index] += 1
            if self.correct_so_far[index] >= self.passes_needed:
                state = mark_known(self.progress, index, self.passes_needed)
                self.known.append(index)
            else:
                # Ask again later in this run for the second confirmation.
                state = Card.new()
                gap = min(len(self.queue), 5 + self.rng.randint(0, 4))
                self.queue.insert(gap, index)
        else:
            # One miss is enough. It goes to the learning pile and is not
            # asked again here; the normal drill will take it from there.
            state = mark_to_learn(self.progress, index)
            self.to_learn.append(index)
            self.progress.missed_today += 1
        self.progress.save()
        return state

    def _speak_verdict(self, card, correct):
        """Rapid fire: no spoken answer, no example sentence, no pause.

        The normal drill reads the answer and a sentence after a miss, which is
        how you learn a word. Here you are only being sorted, and that takes
        several seconds per card that buys nothing.
        """
        self._emit("on_status", "Correct" if correct else "To learn")

    def summary(self):
        return {"known": list(self.known), "to_learn": list(self.to_learn),
                "tested": len(self.known) + len(self.to_learn)}
