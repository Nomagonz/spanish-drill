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
from .grading import check, command_of, quality
from .scheduler import (PASSING_QUALITY, Card, describe_interval, is_leech,
                        schedule, today)
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
    on_verify = None        # (kept, overturned)
    on_finished = None      # ()

    def __init__(self, progress, listener, verifier=_api_verifier,
                 answer_log=None, deck=None, rng=None):
        self.progress = progress
        self.listener = listener
        self.verifier = verifier
        self.log = answer_log if answer_log is not None else AnswerLog()
        self.deck = deck or load_deck()
        self.rng = rng or random
        self.queue = []
        self.running = False

    # -- control ----------------------------------------------------------
    def stop(self):
        self.running = False
        from .speech import stop_speaking
        stop_speaking()             # do not sit through the rest of a sentence

    def _emit(self, name, *args):
        callback = getattr(self, name, None)
        if callback:
            callback(*args)

    # -- the loop ---------------------------------------------------------
    def run(self):
        """Runs until the queue empties or stop() is called."""
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
                self.queue = self.progress.build_queue(rng=self.rng)
            if not self.queue:
                self._emit("on_status", "Queue clear.")
                return

            index = self.queue.pop(0)
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

    def _ask(self, card):
        """Speak the cue and listen. Returns (transcript, elapsed, control)."""
        self._emit("on_status", "Speaking")
        say_english(card.spoken_prompt)
        if not self.running:
            return None, 0.0, "stop"

        while self.running:
            self._emit("on_status", "Listening")
            started = time.time()
            said = self.listener.listen(
                self.progress.window,
                should_stop=lambda: not self.running,
                accept=lambda t: check(t, card) is not None)
            elapsed = time.time() - started
            if not self.running:
                return None, elapsed, "stop"
            if said is None:
                return None, elapsed, None

            self._emit("on_heard", said)
            command = command_of(said)
            if command == "stop":
                self._emit("on_status", "Paused")
                self.running = False
                return None, elapsed, "stop"
            if command == "skip":
                return None, elapsed, "skip"
            if command == "reveal":
                return None, elapsed, None      # treated as a miss
            if command == "repeat":
                self._emit("on_status", "Speaking")
                say_english(card.spoken_prompt)
                continue
            return said, elapsed, None
        return None, 0.0, "stop"

    def _judge(self, index, card, said, elapsed, silent):
        match = check(said, card) if said else None
        correct = match is not None
        close = bool(match and match.close)

        # Keep the clip before anything else: the re-check needs a file, and a
        # verdict should always be traceable to the audio behind it.
        stamp, wav = self.log.save_audio(
            index, getattr(self.listener, "last_audio", None))

        api_text, checked, echoed, overturned = None, False, False, False
        if not correct and wav and self.progress.verify_live and self.verifier:
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

        q = quality(correct, close, silent, elapsed, self.progress.window)
        before = self.progress.card_or_new(index)
        before_dict = before.to_dict()
        state = self._apply(index, q)

        self.log.append(AnswerRecord(
            id=stamp, card_index=index, prompt=card.prompt,
            expected=list(card.answers), heard="" if silent else (said or ""),
            quality=q, correct=correct, silent=silent, elapsed=round(elapsed, 2),
            audio=wav.name if wav else None, before=before_dict,
            api_text=api_text, api_checked=checked, api_echoed=echoed,
            overturned=overturned,
            verdict=("overturned-live" if overturned
                     else "kept-live" if checked else None)))

        outcome = (Outcome.OVERTURNED if overturned
                   else Outcome.CORRECT if correct else Outcome.MISS)
        self._emit("on_result", Result(
            card=card, card_index=index, outcome=outcome,
            said="" if silent else (said or ""), quality=q, close=close,
            silent=silent, api_text=api_text, state=state,
            next_review=describe_interval(state.interval)))
        self._emit("on_counts")
        self._speak_verdict(card, correct)

    def _apply(self, index, q):
        """Grade the card and requeue it if it was missed."""
        is_new = index not in self.progress.cards
        card_state = self.progress.card_or_new(index)
        schedule(card_state, q)
        if q < PASSING_QUALITY:
            self.progress.missed_today += 1
            # Come back inside this session too, sooner for a word with a
            # history of biting, which is what the lapse count is for.
            gap = 2 if is_leech(card_state) else 4 + self.rng.randint(0, 2)
            self.queue.insert(min(len(self.queue), gap), index)
        self.progress.cards[index] = card_state
        if is_new:
            self.progress.new_done += 1
        self.progress.save()
        return card_state

    def _speak_verdict(self, card, correct):
        if correct:
            self._emit("on_status", "Correct")
            if self.progress.hints:
                say_spanish(card.answers[0], self.progress.dialect)
            pace(0.25)
        else:
            self._emit("on_status", "Missed")
            say_spanish(f"{card.answers[0]}. {card.example}", self.progress.dialect)
            pace(0.3)

    def _state_label(self, index):
        from .scheduler import describe_state
        return describe_state(self.progress.card(index))
