"""Saved progress and settings.

Where it is kept is `store.py`'s problem, not this file's. That used to be one
atomically written file and can now be a database every client shares, and
nothing below this line can tell the difference: a store hands over a dict in
the shape `save()` writes, and takes one back.

What stays here is the part that was never about files. Two copies of this
object exist whenever two clients are open, so a write that finds the store
changed underneath it merges rather than overwrites. Losing weeks of
scheduling is not an acceptable failure mode for something that runs daily.
"""
import random
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import (MAIN_MODEL, PERSON_ORDER, PROGRESS_PATH, SPAIN_DIALECT,
                      SPAIN_ONLY, TENSE_ORDER, UNLOCK_REPS)
from .deck import index_by_id, load_deck
from . import scheduler
from .scheduler import MATURE_AT, Card, is_mature, migrate
from .store import CONFLICT, default_store

# How many times a save will fold in somebody else's work and offer its own
# again before giving up. Bounded so that a store refusing every write cannot
# stall the drill between two cards; giving up here is not data loss, because
# the agreement is left where it was and the next save carries everything.
SAVE_ATTEMPTS = 3


@lru_cache(maxsize=4)
def _ladder(deck):
    """lemma -> its forms as one chain, in the order they are taught.

    One form at a time, never a tense at a time: I, then you, then we, then
    he, then they, and only once all five of those are in does the next tense
    start. Opening five near-identical forms together is the worst case for
    interference, since they compete to be the answer to nearly the same cue.
    """
    rank = {(tense, person): (ti, pi)
            for ti, tense in enumerate(TENSE_ORDER)
            for pi, person in enumerate(PERSON_ORDER)}
    out = {}
    for index, card in enumerate(deck):
        if card.lemma:
            tense, person = card.form.split("-", 1)
            out.setdefault(card.lemma, []).append((rank[(tense, person)], index))
    return {lemma: [i for _, i in sorted(rows)] for lemma, rows in out.items()}


def _read_cards(raw, now):
    """Rebuild the index -> Card map from a saved file.

    Accepts both shapes. Current saves are keyed by the card's stable id;
    files written before ids existed are keyed by deck position, which is only
    correct as long as nobody has touched the deck since. An id that no longer
    exists is dropped rather than guessed at.
    """
    ids = index_by_id()
    cards = {}
    for key, value in raw.items():
        index = int(key) if key.lstrip("-").isdigit() else ids.get(key)
        if index is None:
            continue
        cards[index] = migrate(value, today=now)
    return cards


@dataclass
class Progress:
    path: Path = PROGRESS_PATH
    cards: dict = field(default_factory=dict)      # deck index -> Card

    # settings
    dialect: str = "es-ES"       # Castilian: the c/z "th", and vosotros
    input_device: str = ""      # by NAME; indices shift as devices connect
    model: str = MAIN_MODEL
    new_per: int = 20
    window: float = 6.0
    hints: bool = True
    verify_live: bool = True
    # Whether the English cue is spoken. Off makes the drill quiet on the
    # way in and leaves it exactly as loud on the way out: a missed card
    # still says the answer and its example, which is the part that teaches.
    #
    # The cost of turning it off is that the drill stops being hands-free,
    # since the cue then has to be read off the screen. That is a fair trade
    # in the conjugation drill, where the cues are long and the answers are
    # short, and a bad one in the ordinary drill, which is the whole reason
    # the desktop app exists. So it is a setting rather than a decision.
    speak_cue: bool = True
    category: str = "all"       # restrict the drill to one part of speech

    # daily counters
    day: int = 0
    new_done: int = 0           # words met for the first time today
    reviews_done: int = 0       # repetitions of words already known today
    missed_today: int = 0

    # lifetime second-opinion tally
    kept: int = 0
    overturned: int = 0

    # Sentences answered perfectly, by id, for good. A sentence is not a card
    # and has no schedule: the rule is that one you have produced correctly
    # is finished with, and that has to survive closing the app. Kept here
    # rather than in its own file so there is still one thing to back up.
    sentences_done: set = field(default_factory=set)

    # test hook: force a specific queue instead of building one
    queue_override: list = None

    # What the file held when we last read or wrote it, and the state we had
    # at that moment. Together they are how a save can tell what somebody
    # else changed from what it changed itself.
    _seen: tuple = field(default=None, repr=False, compare=False)
    _baseline: dict = field(default=None, repr=False, compare=False)
    _writing: object = field(default_factory=threading.RLock, repr=False,
                             compare=False)

    # Where this schedule is kept. Left unset it follows the configuration,
    # which is a plain file unless a shared database has been switched on.
    # Passed explicitly by tests, and by anything that needs a specific one.
    store: object = field(default=None, repr=False, compare=False)
    _backend: object = field(default=None, repr=False, compare=False)
    _backend_for: object = field(default=None, repr=False, compare=False)

    # (lemma, dialect) -> teaching order. Every conjugation card asks for its
    # verb's chain while the queue is built, and the lru_cache behind it is
    # keyed on the deck, so each of those 1170 lookups was re-hashing all 1670
    # frozen cards. That put a fifth of a second on the UI thread every time
    # the panel counted what was due.
    _chains: dict = field(default_factory=dict, repr=False, compare=False)

    _SETTINGS = ("dialect", "input_device", "model", "new_per", "window",
                 "hints", "verify_live", "category", "speak_cue")
    _COUNTERS = ("day", "new_done", "reviews_done", "missed_today",
                 "kept", "overturned")


    # -- persistence ------------------------------------------------------
    @classmethod
    def load(cls, path=None, today=None, store=None):
        path = Path(path or PROGRESS_PATH)
        now = today if today is not None else scheduler.today()
        p = cls(path=path, store=store)
        raw = p._store().read()
        for key in cls._SETTINGS + cls._COUNTERS:
            if key in raw:
                setattr(p, key, raw[key])
        p.sentences_done = set(raw.get("sentences_done", []) or [])
        p.cards = _read_cards(raw.get("cards", {}), now)
        p.roll_over(now)                # a new day resets the daily caps
        p._remember()
        return p

    def _store(self):
        """The backend, built on first use rather than in `__init__`.

        Late because `path` is how every caller and every test builds one of
        these, and one that has its path changed afterwards has to end up
        talking to the file it asked for rather than the one it was born with.
        """
        if self.store is not None:
            return self.store
        if self._backend is None or self._backend_for != self.path:
            self._backend = default_store(self.path)
            self._backend_for = self.path
        return self._backend

    def _stamp(self):
        """What the store looks like from outside, cheaply."""
        return self._store().stamp()

    _EMPTY = None

    def _base(self):
        """What we knew when we last touched the file.

        Nothing, for an object that was built rather than loaded: it has read
        no file, so everything it holds is its own work and wins the merge.
        Without this, a Progress constructed directly never merged at all and
        the phone quietly stopped seeing the desk.
        """
        if self._baseline is None:
            return {"cards": {}, "done": set(),
                    "counters": {k: 0 for k in self._COUNTERS}}
        return self._baseline

    def _snapshot(self, deck=None):
        deck = deck or load_deck()
        return {"cards": {deck[k].id: v.to_dict()
                          for k, v in self.cards.items() if 0 <= k < len(deck)},
                "counters": {k: getattr(self, k) for k in self._COUNTERS},
                "done": set(self.sentences_done)}

    def _remember(self, deck=None):
        # `settled`, not `stamp`: a write has just been told the new version
        # by the store itself, and asking again would put a second round trip
        # in the path of every answered card.
        store = self._store()
        self._seen = store.settled()
        # A read the store served locally is not evidence of what the shared
        # copy holds, so we claim no baseline at all. The merge then counts
        # every card here as touched and keeps it, which is the safe way to
        # be wrong: the cost is that today's counters can be added twice and
        # come out high until the day turns over, and the alternative is
        # handing a whole offline session to whatever the database still had.
        self._baseline = None if store.stale() else self._snapshot(deck)

    _DAILY = ("new_done", "reviews_done", "missed_today")

    def _absorb(self, raw, deck):
        """Fold somebody else's file into this one without losing either.

        Two copies of this object exist whenever the app and the phone are
        both open, and a plain write from one wipes whatever the other did:
        measured, the desktop erased a card, five reviews and a finished
        sentence. So a save that finds the file changed underneath it merges
        instead of overwriting.

        Cards go by who touched them: anything changed here wins, anything
        untouched here keeps their version. Counters are added as deltas
        rather than taken whole, because both sides really did do that work.
        Settings are the exception and are simply ours, since they are
        preferences and the last person to change one meant it.
        """
        base = self._base()["cards"]
        mine = self._snapshot(deck)["cards"]
        merged = dict(raw.get("cards", {}))
        for card_id, card in mine.items():
            if base.get(card_id) != card:       # changed here since we read
                merged[card_id] = card
        for card_id in base:
            if card_id not in mine:             # deleted here since we read
                merged.pop(card_id, None)
        self.cards = _read_cards(merged, scheduler.today())

        same_day = raw.get("day") == self.day
        for key in self._COUNTERS:
            if key == "day":
                continue
            if key in self._DAILY and not same_day:
                continue        # their tally belongs to a day that is not ours
            theirs = raw.get(key, 0)
            delta = getattr(self, key) - self._base()["counters"][key]
            setattr(self, key, theirs + delta)

        self.sentences_done |= set(raw.get("sentences_done", []) or [])

    def refresh(self):
        """Pick up what the other side has done. True if anything arrived.

        Only reads; whatever is unsaved here survives, because the merge
        keeps every card this copy has touched.
        """
        store = self._store()
        with self._writing:
            if store.stamp() == self._seen:
                return False
            raw = store.read()
            if not raw:         # nothing saved yet, or the store is unreachable
                return False
            deck = load_deck()
            self._absorb(raw, deck)
            self._remember(deck)
            return True

    def save(self):
        # Keyed by card id, never by position: the deck gets edited, and a
        # position-keyed file hands one word's history to whatever word landed
        # in its slot afterwards.
        deck = load_deck()
        store = self._store()
        with self._writing:
            # Retried rather than attempted once, because a shared store can
            # refuse a write that somebody else got in ahead of. Each turn
            # folds in what they did and offers the result again; the loop is
            # bounded so a store that refuses everything cannot hang a drill.
            for _ in range(SAVE_ATTEMPTS):
                # Somebody else wrote while we were thinking. Fold their work
                # in rather than over it.
                if store.stamp() != self._seen:
                    raw = store.read()
                    # Empty means a store with nothing in it yet, or one we
                    # could not reach. Neither is somebody else's work, and
                    # absorbing it would drop every card this copy has not
                    # touched.
                    if raw:
                        self._absorb(raw, deck)
                    # What we are now building on. Only this moves: the
                    # baseline stays at what the store is known to hold, so
                    # anything done here still counts as ours in a later
                    # merge if this save turns out not to land.
                    self._seen = store.settled()

                data = {"cards": {deck[k].id: v.to_dict()
                                  for k, v in sorted(self.cards.items())
                                  if 0 <= k < len(deck)}}
                for key in self._SETTINGS + self._COUNTERS:
                    data[key] = getattr(self, key)
                # Sorted so the file diffs readably and a set survives JSON.
                data["sentences_done"] = sorted(self.sentences_done)

                outcome = store.write(data, base=self._seen)
                if outcome is CONFLICT:
                    continue        # they landed first; take another turn
                # Only once it has actually landed does this become what we
                # and the store agree on. A push that failed leaves the
                # agreement where it was, so everything done since still
                # reads as ours and still wins the merge when it comes back.
                if outcome is not False:
                    self._remember(deck)
                return

    # -- queries ----------------------------------------------------------
    def card(self, index):
        return self.cards.get(index)

    def card_or_new(self, index, today=None):
        return self.cards.get(index) or Card.new(today)

    def due_indexes(self, today=None):
        now = today if today is not None else scheduler.today()
        return [i for i, c in self.cards.items() if c.due <= now]

    def in_category(self, index, deck=None):
        if self.category in ("all", "", None):
            return True
        return (deck or load_deck())[index].pos == self.category

    def roll_over(self, today=None):
        """Begin a new day if one has begun. True if it just did.

        This used to happen only while loading, so an app left open across
        midnight kept yesterday's tallies and its spent new-word allowance
        until it was restarted. Drilling into the small hours is exactly when
        that is least obvious and most annoying.
        """
        now = today if today is not None else scheduler.today()
        if self.day == now:
            return False
        self.day = now
        self.new_done = self.reviews_done = self.missed_today = 0
        return True

    def learned(self, index):
        """Answered right and now in the review rotation.

        One correct answer is the bar on purpose. Requiring maturity instead
        would mean the present tense took three weeks to finish and the
        conditional arrived the better part of a year later; the point is for
        the next variation to start arriving while the last one is still
        being reviewed.

        A lapse resets `reps` to zero, so forgetting a form closes what it had
        opened until you get it back.
        """
        card = self.cards.get(index)
        return bool(card and card.reps >= UNLOCK_REPS)

    def unlocked(self, index, deck=None):
        """May this card be introduced yet?

        Vocabulary always may. A conjugated form waits for exactly one thing:
        the form immediately before it in its verb's chain, with the
        infinitive itself standing at the head of that chain. So `tengo` opens
        when `tener` is learnt, `tienes` when `tengo` is, and the preterite
        only after every person of the present is in. Never more than one new
        form of a verb waiting at a time.
        """
        deck = deck or load_deck()
        card = deck[index]
        if not card.lemma:
            return True
        chain = self._chain(card.lemma, deck)
        if index not in chain:
            return False
        step = chain.index(index)
        previous = (index_by_id(deck).get(card.lemma) if step == 0
                    else chain[step - 1])
        return previous is not None and self.learned(previous)

    def _chain(self, lemma, deck):
        """This verb's forms in teaching order, for the dialect in use.

        On Latin American Spanish the vosotros forms are dropped rather than
        skipped over, so the chain closes up and `tenéis` never becomes a
        step you have to get past to reach the preterite.
        """
        key = (lemma, self.dialect)
        chain = self._chains.get(key)
        if chain is None:
            chain = _ladder(deck).get(lemma, [])
            if self.dialect != SPAIN_DIALECT:
                chain = [i for i in chain
                         if deck[i].form.split("-", 1)[1] not in SPAIN_ONLY]
            # Keyed by dialect as well as verb, so switching accents rebuilds
            # rather than handing back a chain that still has vosotros in it.
            self._chains[key] = chain
        return chain

    def unseen_indexes(self):
        deck = load_deck()
        return [i for i in range(len(deck))
                if i not in self.cards and self.in_category(i, deck)
                and self.unlocked(i, deck)]

    def new_remaining(self):
        return max(0, self.new_per - self.new_done)

    def due_by_stage(self):
        """How many known cards sit on each step of the ladder.

        The interval is the step: a card answered right twice is on six days,
        and one that has survived a few rounds after that is out at three
        weeks or more. Without this the only visible number is a total, which
        says nothing about whether the deck is actually maturing.
        """
        from collections import Counter
        stages = Counter()
        for card in self.cards.values():
            if card.reps == 0:
                stages["relearning" if card.lapses else "new"] += 1
            elif card.interval >= MATURE_AT:
                stages["mature"] += 1
            else:
                stages[f"{card.interval}d"] += 1
        return stages

    def learning_count(self):
        """Answered right and scheduled ahead. Moves the moment you get one."""
        return sum(1 for c in self.cards.values() if c.interval >= 1)

    def mature_count(self):
        """Anki's sense of stuck: three weeks or more between reviews."""
        return sum(1 for c in self.cards.values() if is_mature(c))

    def queue_parts(self, today=None):
        """What a session would ask: due reviews, and new words, separately.

        The dashboard and the drill read this same method so they cannot
        disagree. The counter used to call due_indexes directly, which skips
        the category filter, so "verbs only" still reported every due card in
        the deck and promised four words the drill was never going to ask.
        """
        deck = load_deck()
        due = [i for i in self.due_indexes(today) if self.in_category(i, deck)]
        fresh = self.unseen_indexes()[: self.new_remaining()]
        return due, fresh

    def ladder_steps(self):
        """[(label, count)] soonest first: the shape of the deck, not a total.

        A single "learning" number hides everything worth knowing. Ninety-nine
        cards in progress reads like steady work whether they are all still on
        one day or spread out to three weeks.

        Shared rather than written twice. Both the window and the phone show
        this, and two copies of the ordering rule would eventually disagree
        about the same deck.
        """
        stages = self.due_by_stage()

        def position(step):
            if step == "relearning":
                return -1
            # dict.get would evaluate int("mature") before finding the key
            return 10 ** 6 if step == "mature" else int(step.rstrip("d"))

        label = {"relearning": "RELEARNING", "mature": "21d+"}
        return [(label.get(k, k), stages[k])
                for k in sorted((k for k in stages if k != "new"), key=position)]

    def why_nothing_is_due(self):
        """Why the queue is empty, in words. Shared by the window and phone.

        "Nothing to drill" and "nothing is scheduled" are different answers,
        and a blank panel cannot tell them apart.
        """
        reasons = []
        if self.new_per and not self.new_remaining():
            word = "word" if self.new_per == 1 else "words"
            reasons.append(f"today's {self.new_per} new {word} are done")
        elif not self.new_per:
            reasons.append("new words are switched off")
        if self.category not in ("all", "", None):
            reasons.append(f"the drill is limited to {self.category}s")
        if not reasons:
            reasons.append("nothing is scheduled for review yet")
        return " and ".join(reasons).capitalize() + "."

    def placement_scope(self, deck=None):
        """(never sorted, everything) for the category in play.

        Exactly what `PlacementSession.next_queue` will offer, conjugation
        cards included. An earlier version of this counted vocabulary only,
        which read as a tidier number and was a lie: the panel said nothing
        was left to sort while pressing the button started a run of eleven
        hundred cards.
        """
        deck = deck or load_deck()
        in_scope = [i for i in range(len(deck)) if self.in_category(i, deck)]
        return (sum(1 for i in in_scope if i not in self.cards), len(in_scope))

    def build_queue(self, today=None, rng=None):
        """Due reviews, shuffled, with the day's new words spread through."""
        if self.queue_override is not None:
            return list(self.queue_override)
        rng = rng or random
        due, fresh = self.queue_parts(today)
        rng.shuffle(due)
        queue = list(due)
        if fresh:
            step = max(1, (len(queue) + len(fresh)) // len(fresh))
            at = 0
            for index in fresh:
                queue.insert(min(at, len(queue)), index)
                at += step + 1
        return queue
