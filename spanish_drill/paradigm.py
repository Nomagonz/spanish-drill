"""Conjugations on their own, for verbs the main drill has already taught.

A second way to practise, running beside the normal drill rather than inside
it. Nothing but conjugated forms, a few verbs at a time, worked through in
frequency order.

**It cannot touch the vocabulary schedule.** That is the point of the mode and
the reason for nearly every decision here. `progress.json` is opened once, read
for a single fact — which infinitives are learned — and never written. The
schedule this drill moves lives in its own file, and so does its answer log,
because `--review` repairs cards in whichever tracker it is handed and a shared
log would let a conjugation re-check reach into the words.

The conjugation cards themselves are the same cards the main drill introduces
through `Progress.unlocked`. Both may drill them, each on its own schedule, and
neither can see the other's. That is deliberate: this mode was asked for as an
addition, not a replacement, and taking the forms out of the normal drill to
make the two exclusive would be changing something nobody asked to change.

What is inherited, and it is almost everything: asking, listening, the early
accept, grading, the second opinion, SM-2, the session learning ladder. A
paradigm is drilled exactly the way a word is. Only the answers to "which
cards, and whose schedule" differ, and those are the two things overridden.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import CONJUGATION_BATCH, CONJUGATION_PROGRESS_PATH
from .deck import index_by_id, load_deck
from .progress import Progress
from .session import DrillSession
from .transcribe import CONJUGATION_STEER, conjugation_second_opinion


def _conjugation_verifier(path):
    return conjugation_second_opinion(str(path))


@lru_cache(maxsize=4)
def verb_order(deck):
    """Every verb with conjugation cards, most frequent first.

    `deck.json` is already written in frequency order — ser, estar, haber,
    tener, ir, hacer — so a verb's position in the deck is its frequency rank
    and nothing else has to be kept in step with it. A hand-written ranking
    beside the deck would be one more list to drift.

    Read off the deck rather than off `conjugation.VERBS`, so a verb whose
    cards exist is drillable here whatever the tables happen to hold.
    """
    lemmas = {card.lemma for card in deck if card.lemma}
    return tuple(card.answers[0] for card in deck
                 if not card.lemma and card.answers
                 and card.answers[0] in lemmas)


def known_verbs(progress, deck=None):
    """The verbs whose infinitive the main drill counts as learned.

    The same bar the main unlock chain uses, `Progress.learned`: answered
    right and in the review rotation. So "verbs I already know" means here
    exactly what it means everywhere else in the app, rather than being a
    second definition that can disagree with the first.
    """
    deck = deck or load_deck()
    ids = index_by_id(deck)
    return tuple(lemma for lemma in verb_order(deck)
                 if (i := ids.get(lemma)) is not None and progress.learned(i))


@dataclass
class ConjugationProgress(Progress):
    """The conjugation drill's own schedule.

    A `Progress` in every respect except which cards it will introduce and
    where it saves. Subclassed rather than rewritten because the queue
    building, the day roll-over, the atomic save and the id-keyed format are
    all worth having and none of them care what kind of card they hold.
    """

    path: Path = CONJUGATION_PROGRESS_PATH
    # Silent on the way in. The cues here are long — "you all would be able
    # to" — the answers are one or two syllables, and the cue is on screen
    # anyway, so reading it aloud is several seconds a card buying nothing.
    # The way out is untouched: a missed form still says the answer and its
    # example, which is the part that teaches.
    speak_cue: bool = False
    # Lemmas the main drill has taught. Read across once when this tracker is
    # opened and never again: a snapshot, not a live link, so a long session
    # cannot be reshaped halfway through by the other tracker moving.
    known: tuple = ()

    @classmethod
    def over(cls, main):
        """The conjugation view of a schedule already in memory.

        No reading of anything. `open` loads, which was the right thing when
        this had a file of its own and is wasteful now that it shares one:
        the panel asks how many forms are due every time it refreshes, and
        with a shared database behind it that was a network round trip and a
        file write per answered card, sitting between you and the next one.

        The cards are the same objects, not a copy. There is one schedule and
        this is another way of looking at it.
        """
        p = cls(path=main.path, store=main._store())
        for key in cls._SETTINGS:
            if key not in ("category", "speak_cue"):
                setattr(p, key, getattr(main, key))
        p.category = "all"
        p.day = main.day
        p.cards = main.cards
        p.known = known_verbs(main)
        return p

    @classmethod
    def open(cls, main, path=None, today=None):
        """Load the conjugation schedule, borrowing the main drill's settings.

        Settings are mirrored rather than stored twice over. Dialect decides
        whether vosotros is in the chain at all, the answer window decides
        what counts as a quick recall, and the device and model decide whether
        anything is heard: two copies of those would be two ways to be wrong.
        They are copied in on open and whatever the file held is overwritten.

        Two are not mirrored. `category` is forced, because scope in this
        mode is the verb batch and a main drill restricted to nouns would
        otherwise silently leave the conjugation queue empty. `speak_cue`
        keeps whatever this tracker's own file says, defaulting to off, so
        the ordinary drill stays hands-free while this one stays quiet.
        """
        p = cls.load(path or CONJUGATION_PROGRESS_PATH, today)
        for key in cls._SETTINGS:
            if key not in ("category", "speak_cue"):
                setattr(p, key, getattr(main, key))
        p.category = "all"
        p.known = known_verbs(main)
        return p

    # -- which cards ------------------------------------------------------
    def in_category(self, index, deck=None):
        """Forms only, whatever else the schedule happens to hold.

        `unlocked` already refuses vocabulary, but it only governs what gets
        introduced. Reviews are drawn from whatever is already due, and they
        never pass through it. That was harmless while this tracker had a
        file to itself and became wrong the moment the two shared one:
        measured on a real schedule, all 167 cards this drill offered were
        ordinary vocabulary and not one was a conjugated form.

        Enforced here rather than in `queue_parts` because this is the
        question that method already asks about every due card, so the
        counter on the panel and the queue in the drill cannot disagree.
        """
        return bool((deck or load_deck())[index].lemma)

    def unlocked(self, index, deck=None):
        """May this form be introduced yet?

        The same one-form-at-a-time chain the main drill walks, with one
        difference at the head of it. There, step zero waits on the
        infinitive's own card in the same tracker. Here that card does not
        exist and never will, because this drill does not teach infinitives,
        so step zero asks the main tracker instead. That single question is
        the whole of the link between the two files.

        Vocabulary is refused outright rather than merely never offered. A
        word reaching this queue would be a word whose schedule is being kept
        in the wrong file.
        """
        deck = deck or load_deck()
        card = deck[index]
        if not card.lemma:
            return False
        chain = self._chain(card.lemma, deck)
        if index not in chain:
            return False            # a form the dialect drops, e.g. vosotros
        step = chain.index(index)
        if step == 0:
            return card.lemma in self.known
        return self.learned(chain[step - 1])

    def finished(self, lemma, deck=None):
        """True once every form of this verb is in the rotation.

        What retires a verb from the batch. Learned, not mature: the forms
        carry on coming back for review on their own schedule long after the
        verb has stopped being one of the ten, which is what review is for.
        """
        deck = deck or load_deck()
        chain = self._chain(lemma, deck)
        return bool(chain) and all(self.learned(i) for i in chain)

    def batch(self, deck=None):
        """The verbs being drilled right now: the most frequent unfinished ones.

        Rolling rather than a block that has to be cleared. When a verb's last
        form joins the rotation it drops out and the next one down the
        frequency list takes the slot, so the run never narrows to a single
        stubborn paradigm while nine finished ones hold their places.
        """
        deck = deck or load_deck()
        out = []
        for lemma in self.known:
            if self.finished(lemma, deck):
                continue
            out.append(lemma)
            if len(out) >= CONJUGATION_BATCH:
                break
        return tuple(out)

    def unseen_indexes(self):
        """New forms this drill may introduce: the batch's, and only the batch's.

        The batch gates what is *met*, never what is *reviewed*. Forms of a
        verb that has already retired stay due on their own schedule and come
        back through `due_indexes` like anything else, which is the whole
        difference between finishing a verb and forgetting it.
        """
        deck = load_deck()
        wanted = set(self.batch(deck))
        return [i for i, card in enumerate(deck)
                if card.lemma in wanted and i not in self.cards
                and self.unlocked(i, deck)]

    # -- for saying where you are -----------------------------------------
    def verb_progress(self, lemma, deck=None):
        """(forms learned, forms in the chain) for one verb."""
        deck = deck or load_deck()
        chain = self._chain(lemma, deck)
        return sum(1 for i in chain if self.learned(i)), len(chain)

    def remaining(self, deck=None):
        """How many verbs are still to come, the current batch included."""
        deck = deck or load_deck()
        return sum(1 for lemma in self.known if not self.finished(lemma, deck))


class ConjugationSession(DrillSession):
    """A drill over conjugated forms, scheduled in its own file.

    Asking, grading, the learning ladder and SM-2 are all inherited: a
    paradigm is practised exactly the way a word is. What is queued differs,
    and that is settled by `ConjugationProgress` rather than here.

    The one thing genuinely different is what the recognisers are told to
    expect. Every example in the ordinary steer is an infinitive or a noun,
    and both models follow it: in one real session it wrote "seguir" for
    `sigo`, "poner" for `pongo`, "salir" for `sales` and "oir" for `oigo`.
    Four verbs, all pulled the same way. So the local model gets
    `CONJUGATION_STEER` through `steer`, and the second opinion gets it
    through its own verifier.
    """

    # Read by DrillSession._ask and handed to the local recogniser.
    steer = CONJUGATION_STEER

    # The quick model decides, as it does in a placement run, and for the
    # same reason: the main model costs the better part of ten seconds on a
    # long window here and only ever changes misses, which go to the second
    # opinion regardless. Anything it would have accepted the scout has
    # already accepted, and the second opinion is both faster and better at
    # the rest. On a mode whose answers are one or two syllables and whose
    # cards come thirty to a verb, that difference is the mode being usable.
    fast_recognition = True

    # Affordable precisely because of the line above: the extra decode is the
    # quick model, about a second. It is what makes one and two syllable
    # answers gradeable at all — `he` and `tengo` both decode correctly with
    # no prompt and are lost with one.
    second_pass = True

    def __init__(self, *args, verifier=_conjugation_verifier, **kw):
        super().__init__(*args, verifier=verifier, **kw)

    def run(self):
        """Refuse to start on a deck with no verbs learned, and say why.

        The queue would otherwise come back empty and the drill would report
        "Queue clear.", which is what it says when you have finished the day's
        work. Those are opposite situations and they must not read the same:
        one means well done, the other means this mode has nothing to work
        with until the ordinary drill has taught some infinitives.
        """
        if not self.progress.known:
            self._emit("on_status", "No verbs learned yet — the ordinary "
                                    "drill teaches the infinitives this "
                                    "mode builds on.")
            self._emit("on_finished")
            return
        super().run()

    def next_queue(self):
        queue = super().next_queue()
        if queue:
            self._announce()
        return queue

    def _announce(self):
        """Say which verbs are in hand, since the cues never name them.

        The cue for a conjugation card deliberately hides its infinitive —
        working out which verb is wanted is most of the card. That leaves no
        way to tell what a session is actually about, so the batch is named
        once, up front, where it cannot help with any particular answer.
        """
        batch = self.progress.batch(self.deck)
        if not batch:
            return
        left = self.progress.remaining(self.deck)
        self._emit("on_status",
                   f"{', '.join(batch)} · {left} verbs to go")

    def summary(self):
        deck = self.deck
        batch = self.progress.batch(deck)
        return {"verbs": list(batch),
                "done": [lemma for lemma in self.progress.known
                         if self.progress.finished(lemma, deck)],
                "remaining": self.progress.remaining(deck)}
