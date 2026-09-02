"""The drill loop.

Deliberately free of any UI framework. It takes a listener and a verifier as
plain callables and reports through callbacks, so the whole thing runs in a
test with no audio hardware, no network, and no Qt.
"""
import random
import time
from dataclasses import dataclass
from enum import Enum

from .answers import AnswerLog, AnswerRecord
from .deck import load_deck
from .grading import check, command_of, quality, typed_quality
from .config import LEARNING_STEPS
from .scheduler import (FIRST_INTERVAL, PASSING_QUALITY, Card,
                        describe_interval, is_leech, schedule, today)
from .speech import say_english, say_spanish
from .transcribe import second_opinion


# Short pauses that only pace the spoken feedback. Named so tests can
# neutralise them without patching time.sleep globally, which would also
# silence the timing that audio capture depends on.
def pace(seconds):
    time.sleep(seconds)


class Outcome(Enum):
    CORRECT = "correct"
    OVERTURNED = "overturned"       # graded wrong locally, reversed on re-check
    MISS = "miss"


@dataclass
class Result:
    card: object
    card_index: int
    outcome: Outcome
    said: str
    quality: int
    close: bool
    silent: bool
    api_text: str
    state: Card
    next_review: str

    @property
    def correct(self):
        return self.outcome in (Outcome.CORRECT, Outcome.OVERTURNED)

    @property
    def overturned(self):
        return self.outcome is Outcome.OVERTURNED


def _api_verifier(path):
    return second_opinion(str(path))


class DrillSession:
    """One sitting. Call run(); call stop() from anywhere to end it."""

    # Callbacks, all optional. Assigned by whoever is driving.
    on_prompt = None        # (card, state_label)
    on_status = None        # (str)
    on_heard = None         # (str)
    on_result = None        # (Result)
    on_counts = None        # ()
    on_progress = None      # (done, total) for modes that have an end in sight
    on_verify = None        # (kept, overturned)
    on_finished = None      # ()

    # Modes that value speed over the main model's edge on misses set this.
    fast_recognition = False

    # What the recogniser is told to expect. None is the ordinary steer, for
    # isolated dictionary words. A mode whose answers are not dictionary
    # words overrides it: told to expect infinitives, the model writes them
    # down whatever was actually said.
    steer = None

    # Whether a reading that is not the answer gets a second look with no
    # prompt at all. Costs one extra decode per miss, so it is off where that
    # decode is the slow model and on where it is the quick one.
    second_pass = False

    # Correct answers to put one card to bed: every rung, then the top.
    LADDER_PASSES = len(LEARNING_STEPS) + 1

    def __init__(self, progress, listener, verifier=_api_verifier,
                 answer_log=None, deck=None, rng=None, typed=False,
                 hold_on_miss=None):
        self.progress = progress
        self.listener = listener
        # Silent throughout: the cue is read off the screen and the answer
        # comes from a keyboard. Nothing here speaks, and there is no clip to
        # send for a second opinion.
        self.typed = typed
        # Called after a miss, and blocks until the reader is done looking at
        # the answer. Only ever supplied by the typing mode: the spoken drill
        # is hands-free on purpose, and making it wait for a keypress on every
        # miss would take away the thing it is for.
        self.hold_on_miss = hold_on_miss
        self.verifier = verifier
        self.log = answer_log if answer_log is not None else AnswerLog()
        self.deck = deck or load_deck()
        self.rng = rng or random
        self.queue = []
        # deck index -> how many rungs of the learning ladder it has
        # cleared in this session. Session-scoped on purpose: the ladder
        # is about one sitting, and tomorrow starts it again.
        self.rungs = {}
        # deck index -> passes this card is expected to cost today, for
        # the bar. A review costs one; anything being learned costs the
        # whole ladder.
        self.weights = {}
        self.running = False
        self.current = None         # deck index being asked, for observers
        self._stop_requested = False

    # -- control ----------------------------------------------------------
    def stop(self):
        # Sticky, because stop can arrive before run() has started: the worker
        # calibrates first, and a stop during that used to be forgotten when
        # run() then set running back to True and drilled anyway.
        self._stop_requested = True
        self.running = False
        from .speech import stop_speaking
        stop_speaking()             # do not sit through the rest of a sentence

    def _emit(self, name, *args):
        callback = getattr(self, name, None)
        if callback:
            callback(*args)

    # -- the loop ---------------------------------------------------------
    @property
    def stop_requested(self):
        return self._stop_requested

    def run(self):
        """Runs until the queue empties or stop() is called."""
        if self._stop_requested:
            self._emit("on_finished")
            return
        self.running = True
        try:
            self._loop()
        except Exception:
            import traceback
            traceback.print_exc()
            self._emit("on_status", "Crashed — see terminal")
        finally:
            self.running = False
            self.progress.save()
            self._emit("on_finished")

    def _loop(self):
        while self.running:
            if not self.queue:
                self.queue = self.next_queue()
            if not self.queue:
                self._emit("on_status", "Queue clear.")
                return

            index = self.queue.pop(0)
            self.current = index
            card = self.deck[index]
            self._emit("on_prompt", card, self._state_label(index))
            self._emit("on_heard", "")
            self._emit("on_counts")

            said, elapsed, stopped = self._ask(card)
            if not self.running:
                return
            if stopped == "skip":
                self.queue.append(index)
                continue
            if stopped == "stop":
                return

            self._judge(index, card, said, elapsed, silent=said is None)

    def next_queue(self):
        """What to ask next. Overridden by other modes."""
        queue = self.progress.build_queue(rng=self.rng)
        self._plan(queue)
        self._report_progress()
        return queue

    def _plan(self, queue):
        """Record what each card in the queue is going to cost today.

        Fixed when the card is first planned rather than recomputed, because a
        card's cost drops the moment it graduates and a total that shrinks
        under the bar makes it lurch backwards.
        """
        for index in queue:
            if index in self.weights:
                continue
            card = self.progress.cards.get(index)
            learning = card is None or card.reps == 0
            self.weights[index] = self.LADDER_PASSES if learning else 1

    def _report_progress(self):
        """How much of the sitting is behind you, in rungs rather than cards.

        Counting finished cards looks right and is useless: with the ladder
        nothing finishes until it has been answered five times, so a bar over
        a hundred-card queue reads zero for hundreds of answers and then moves
        in a rush at the end. Rungs cleared is the same journey measured
        somewhere it actually changes.

        The total can grow, because a review that gets missed drops into the
        ladder and now costs five passes instead of one. That is honest: the
        work really did get bigger.
        """
        if not self.weights:
            return
        done = 0
        for index, weight in self.weights.items():
            if self._is_done_today(index):
                done += weight
            else:
                done += min(self.rungs.get(index, 0), weight)
        self._emit("on_progress", done, sum(self.weights.values()))

    def _is_done_today(self, index):
        card = self.progress.cards.get(index)
        return bool(card and card.due > today())

    def _ask(self, card):
        """Speak the cue and listen. Returns (transcript, elapsed, control)."""
        if self.typed:
            # The cue is already on screen; saying it out loud is the one
            # thing this mode exists to avoid. Anything typed while the last
            # card was still being graded belongs to that card, not this one.
            # A spoken session with the cue turned off reads it off the
            # screen the same way, but still answers out loud, so only the
            # typed branch has a queue to clear.
            clear = getattr(self.listener, "clear", None)
            if clear:
                clear()
        elif self.progress.speak_cue:
            self._emit("on_status", "Speaking")
            say_english(card.spoken_prompt)
        if not self.running:
            return None, 0.0, "stop"

        while self.running:
            # One place decides how the wait is described, so a typed session
            # cannot end up reporting that it is listening to a microphone it
            # never opened.
            self._emit("on_status", "Type it" if self.typed else "Listening")
            started = time.time()
            said = self.listener.listen(
                self.progress.window,
                should_stop=lambda: not self.running,
                accept=lambda t: check(t, card) is not None,
                fast=self.fast_recognition, steer=self.steer,
                second_pass=self.second_pass)
            elapsed = time.time() - started
            if not self.running:
                return None, elapsed, "stop"
            if said is None:
                return None, elapsed, None

            self._emit("on_heard", said)
            # The expected answer outranks the command vocabulary. "para" is
            # the answer to "for, to", "parar" to "to stop", "alto" to "tall"
            # and "siguiente" to "following". Reading those as commands meant
            # answering correctly ended or derailed the session.
            command = None if self._check(said, card) else command_of(said)
            if command == "stop":
                self._emit("on_status", "Paused")
                self.running = False
                return None, elapsed, "stop"
            if command == "skip":
                return None, elapsed, "skip"
            if command == "reveal":
                return None, elapsed, None      # treated as a miss
            if command == "repeat":
                if self.typed or not self.progress.speak_cue:
                    continue        # nothing to replay; just ask again
                self._emit("on_status", "Speaking")
                say_english(card.spoken_prompt)
                continue
            return said, elapsed, None
        return None, 0.0, "stop"

    def _check(self, said, card):
        """Grade one answer. Typed and spoken are held to the same standard."""
        return check(said, card)

    def _judge(self, index, card, said, elapsed, silent):
        match = self._check(said, card) if said else None
        correct = match is not None
        close = bool(match and match.close)

        # Keep the clip before anything else: the re-check needs a file, and a
        # verdict should always be traceable to the audio behind it.
        stamp, wav = self.log.save_audio(
            index, getattr(self.listener, "last_audio", None),
            word=card.answers[0] if card.answers else None)

        api_text, checked, echoed, overturned = None, False, False, False
        # Nothing was recorded when the answer was typed, so there is nothing
        # to re-check and nobody to blame for mishearing it.
        if (not correct and wav and not self.typed
                and self.progress.verify_live and self.verifier):
            self._emit("on_status", "Double-checking…")
            api_text, echoed = self.verifier(wav)
            checked = api_text is not None
            if api_text:
                again = check(api_text, card)
                if again:
                    correct, close, silent, overturned = True, again.close, False, True
            if checked:
                # Only count a verdict we actually got. An echoed prompt is not
                # evidence either way and must not be scored as a miss the
                # second opinion agreed with.
                if overturned:
                    self.progress.overturned += 1
                else:
                    self.progress.kept += 1
                self._emit("on_verify", self.progress.kept, self.progress.overturned)

        if self.typed:
            q = typed_quality(correct, close, silent, elapsed,
                              card.answers[0] if card.answers else "")
        else:
            q = quality(correct, close, silent, elapsed, self.progress.window)
        before = self.progress.card_or_new(index)
        before_dict = before.to_dict()
        state = self._apply(index, q)

        self.log.append(AnswerRecord(
            id=stamp, card_index=index, card_id=card.id, prompt=card.prompt,
            expected=list(card.answers), heard="" if silent else (said or ""),
            quality=q, correct=correct, silent=silent, elapsed=round(elapsed, 2),
            audio=wav.name if wav else None, before=before_dict,
            api_text=api_text, api_checked=checked, api_echoed=echoed,
            overturned=overturned,
            verdict=("overturned-live" if overturned
                     else "kept-live" if checked else None),
            mode="typed" if self.typed else "voice"))

        outcome = (Outcome.OVERTURNED if overturned
                   else Outcome.CORRECT if correct else Outcome.MISS)
        self._emit("on_result", Result(
            card=card, card_index=index, outcome=outcome,
            said="" if silent else (said or ""), quality=q, close=close,
            silent=silent, api_text=api_text, state=state,
            next_review=describe_interval(state.interval)))
        self._emit("on_counts")
        self._report_progress()
        self._speak_verdict(card, correct)
        if not correct and self.hold_on_miss:
            # The answer is on screen and nothing else happens until it has
            # been read. Emitted before blocking so the window can say what it
            # is waiting for rather than looking frozen.
            self._emit("on_status", "Press Enter to continue")
            self.hold_on_miss(lambda: not self.running)

    def _apply(self, index, q):
        """Grade the card and walk it up or down the session's learning ladder.

        A word being learned has to come back at every rung in LEARNING_STEPS,
        each one further away than the last, before it counts as learned for
        the day. Until it clears the top it stays due today whatever SM-2
        would otherwise have given it, because a single correct answer five
        cards after being told the word is recognition, not recall.

        Words already in the review rotation are untouched: answer one right
        and it schedules ahead exactly as before. A lapse drops it into the
        ladder, which is where relearning belongs.
        """
        self.progress.roll_over()   # a session can run past local midnight
        is_new = index not in self.progress.cards
        card_state = self.progress.card_or_new(index)
        passed = q >= PASSING_QUALITY
        # New, mid-ladder, or previously lapsed: all still being learned.
        learning = (is_new or index in self.rungs or card_state.reps == 0)
        lapses_before = card_state.lapses

        schedule(card_state, q)
        if not passed:
            self.progress.missed_today += 1
            if learning and index in self.rungs:
                # Stepping down a rung is not a fresh lapse. Without this a
                # hard word could rack up eight of them in one sitting and be
                # called a leech for the crime of being learned.
                card_state.lapses = lapses_before

        if learning:
            graduated = self._step_ladder(index, passed, card_state)
            if graduated:
                # Clearing the ladder is one successful day-scale repetition,
                # not the four or five it took to get there. Without this the
                # rep count carried the climb into tomorrow and the next
                # interval was computed from it.
                card_state.reps = 1
                card_state.interval = FIRST_INTERVAL
                card_state.due = today() + FIRST_INTERVAL
            else:
                # Whatever SM-2 handed out, it is not finished today.
                card_state.interval = 0
                card_state.due = today()
        elif not passed:
            self._step_ladder(index, passed, card_state)    # a lapse re-enters

        self.progress.cards[index] = card_state
        if is_new:
            self.progress.new_done += 1
        else:
            self.progress.reviews_done += 1
        self.progress.save()
        return card_state

    def _step_ladder(self, index, passed, card_state):
        """Move the card a rung and put it back in the queue. True if learned.

        Right moves up, wrong moves down one rung rather than to the bottom,
        so one slip near the top does not throw away the whole climb.
        """
        rung = self.rungs.get(index, 0)
        if passed:
            if rung >= len(LEARNING_STEPS):
                self.rungs.pop(index, None)
                return True                     # every rung cleared
        else:
            rung = max(0, rung - 1)
        if self.weights.get(index) == 1:
            self.weights[index] = self.LADDER_PASSES     # it lapsed; it costs more now
        gap = LEARNING_STEPS[min(rung, len(LEARNING_STEPS) - 1)]
        if is_leech(card_state):
            gap = LEARNING_STEPS[0]             # a word that keeps biting
        self.rungs[index] = rung + 1 if passed else rung
        self.queue.insert(min(len(self.queue), gap), index)
        return False

    def _speak_verdict(self, card, correct):
        if self.typed:
            # The slip on screen carries the answer, the example and the
            # gloss. Saying any of it defeats the point of the mode.
            self._emit("on_status", "Correct" if correct else "Missed")
            return
        if correct:
            self._emit("on_status", "Correct")
            if self.progress.hints:
                say_spanish(card.answers[0], self.progress.dialect)
            pace(0.25)
        else:
            self._emit("on_status", "Missed")
            # Two phrases, not one joined string: each is recorded once and
            # reused everywhere it appears. Joining them would make a unique
            # phrase per card that no other mode can share.
            say_spanish(card.answers[0], self.progress.dialect)
            say_spanish(card.example, self.progress.dialect)
            pace(0.3)

    def _state_label(self, index):
        """The card's schedule, and nothing about the answer.

        The infinitive used to be shown here on conjugation cards. Naming the
        verb is most of the recall: seeing "volver" next to "I return" leaves
        only the ending to produce. Working out which verb the cue wants is
        part of the card, so the cue has to carry that on its own.
        """
        from .scheduler import describe_state
        return describe_state(self.progress.card(index))
