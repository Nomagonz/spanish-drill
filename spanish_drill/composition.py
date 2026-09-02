"""The sentence drill: read the English, write the Spanish, see the mistakes.

Shaped like `DrillSession` from the outside — the same callbacks, the same
stop(), the same `typed` flag — so the window's worker thread drives it
without knowing which of the two it is holding. Inside it is much simpler,
because a sentence is not a card: there is no schedule to move, no audio to
keep and no second opinion to ask.

It deliberately changes nothing about SM-2. The words in a sentence are
already being reviewed on their own account, and grading them a second time
here would count one answer twice and quietly distort the schedule the rest
of the app depends on. This is practice at putting known words together, and
the only thing it writes is what it puts on screen.
"""
import random
import threading
import time
from dataclasses import dataclass

from . import cues
from .config import (SENTENCE_SPEECH_BASE, SENTENCE_SPEECH_MAX,
                     SENTENCE_SPEECH_PER_WORD)
from .grading import command_of
from .sentences import (Grade, Sentence, blocked_by, grade, known_ids,
                        tokens, unfinished)
from .speech import say_english, say_spanish
from .transcribe import sentence_second_opinion


def speech_window(sentence):
    """How long to listen for one spoken sentence.

    Sized off the sentence rather than the answer-wait dial, which is set for
    a single word. Ten words spoken at a natural pace do not fit in six
    seconds, and a window that runs out mid-clause scores silence.
    """
    return min(SENTENCE_SPEECH_MAX,
               SENTENCE_SPEECH_BASE
               + SENTENCE_SPEECH_PER_WORD * len(tokens(sentence.es)))


def _api_verifier(path):
    return sentence_second_opinion(str(path))


@dataclass
class SentenceResult:
    sentence: Sentence
    grade: Grade
    elapsed: float
    said: str = ""              # what the recogniser heard, spoken mode only
    api_text: str = None        # what the second opinion heard
    overturned: bool = False    # graded wrong locally, reversed on re-check


class SentenceDrill:
    """One sitting of sentence composition."""

    # Callbacks, all optional, assigned by whoever drives it. The names match
    # DrillSession's so SessionWorker can wire either one up the same way.
    on_prompt = None        # (sentence, label)
    on_status = None        # (str)
    on_heard = None         # (str)
    on_result = None        # (SentenceResult)
    on_counts = None        # ()
    on_progress = None      # (done, total)
    on_verify = None        # (kept, overturned) — never fired here
    on_finished = None      # ()

    # Typed unless told otherwise. Spoken, the worry is that a whole sentence
    # dictated to a recogniser that mishears single words gets graded on the
    # recogniser's mistakes rather than yours. The second opinion is the
    # answer to that, exactly as it is in the word drill: anything judged
    # wrong locally goes to a stronger model before it counts.
    typed = True

    # How far ahead a missed sentence comes back, in sentences.
    RETRY_GAP = 5

    # How long to sit waiting for the first written sentence when nothing is
    # unlocked yet and the generator is still working. Only ever reached on a
    # deck with no curated sentences available, since the bank covers the
    # normal case instantly.
    WAIT_SECONDS = 40

    def __init__(self, progress, listener, sentences=None, rng=None,
                 hold_on_miss=None, limit=None, expecting_more=False,
                 typed=True, verifier=None, answer_log=None):
        self.progress = progress
        self.listener = listener
        self.rng = rng or random
        self.hold_on_miss = hold_on_miss
        self.limit = limit
        self.pool = sentences         # None means "whatever is unlocked"
        # Set when a generator is running behind this drill. Without it an
        # empty pool ends the run at once, which is right: waiting for
        # sentences nobody is writing is just a slower way of saying no.
        self.expecting_more = expecting_more
        self.typed = typed
        # Only ever consulted on a spoken miss. Typed answers leave no clip
        # and there is nobody to blame for mishearing a keyboard.
        self.verifier = _api_verifier if verifier is None and not typed \
            else verifier
        self.log = answer_log
        self.queue = []
        self.done = []                # sentences answered perfectly
        self.asked = set()            # ids seen at least once, for the label
        self._ids = set()             # everything queued, so a late arrival
        self._lock = threading.Lock() # cannot be added twice
        self.attempts = 0
        self.total = 0
        self.running = False
        self.current = None
        self._stop_requested = False

    # -- control ----------------------------------------------------------
    def stop(self):
        self._stop_requested = True
        self.running = False

    @property
    def stop_requested(self):
        return self._stop_requested

    def _emit(self, name, *args):
        callback = getattr(self, name, None)
        if callback:
            callback(*args)

    # -- what to ask ------------------------------------------------------
    def build_queue(self):
        pool = (list(self.pool) if self.pool is not None
                else unfinished(self.progress))
        self.rng.shuffle(pool)
        if self.limit:
            pool = pool[: self.limit]
        with self._lock:
            # Anything the generator handed over before the run got going.
            # Assigning over the queue instead of merging would throw away a
            # sentence that has already been paid for.
            early = list(self.queue)
            queued = {sentence.id for sentence in pool}
            merged = pool + [s for s in early if s.id not in queued]
            self._ids = {s.id for s in merged}
            self.queue = merged
            self.total = len(merged)
        return merged

    def add(self, arriving):
        """Take sentences written after the run started.

        Called from the generator's thread while the drill is mid-card, so
        the queue is only ever touched under the lock. Anything already
        queued is ignored rather than asked twice.
        """
        with self._lock:
            fresh = [s for s in arriving if s.id not in self._ids]
            for sentence in fresh:
                self._ids.add(sentence.id)
            self.queue.extend(fresh)
            self.total += len(fresh)
        if fresh:
            self._emit("on_progress", len(self.done), self.total)
        return len(fresh)

    def _wait_for_arrivals(self):
        """Sit still until the generator hands over something, or time runs out."""
        deadline = time.time() + self.WAIT_SECONDS
        while self.running and time.time() < deadline:
            with self._lock:
                if self.queue:
                    self.total = max(self.total, len(self.queue))
                    return True
            time.sleep(0.1)
        return False

    def why_empty(self):
        """Something useful to say when nothing is unlocked yet.

        An empty screen that says "nothing to do" is indistinguishable from a
        broken mode. Naming the words that are holding the bank back says
        which of the two it is, and what would fix it.
        """
        from .sentences import all_sentences, available
        known = known_ids(self.progress)
        bank = all_sentences()
        unlocked = available(self.progress)
        done = set(getattr(self.progress, "sentences_done", None) or ())
        if unlocked and all(s.id in done for s in unlocked):
            return (f"All {len(unlocked)} unlocked sentences are done. "
                    "More open up as you learn more words.")
        blockers = {}
        for sentence in bank:
            for word in blocked_by(sentence, known):
                blockers[word] = blockers.get(word, 0) + 1
        if not bank:
            return "No sentences on file."
        top = sorted(blockers.items(), key=lambda kv: -kv[1])[:4]
        names = ", ".join(word for word, _ in top)
        return (f"None of the {len(bank)} sentences are unlocked yet. "
                f"Learn these next and they open up: {names}.")

    # -- the loop ---------------------------------------------------------
    def run(self):
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
            self._emit("on_finished")

    def _loop(self):
        self.build_queue()
        if not self.total:
            self._emit("on_status", "Writing the first sentence…")
            self._emit("on_prompt", None, "")
            if not (self.expecting_more and self._wait_for_arrivals()):
                self._emit("on_status", "Nothing unlocked")
                return
        self._emit("on_progress", 0, self.total)

        while self.running and self.queue:
            with self._lock:
                sentence = self.queue.pop(0)
            self.current = sentence
            label = self._label(sentence)
            self.asked.add(sentence.id)
            self._emit("on_prompt", sentence, label)
            self._emit("on_heard", "")

            said, elapsed, control = self._ask(sentence)
            if not self.running or control == "stop":
                return
            if control == "skip":
                self.queue.append(sentence)
                continue
            self._judge(sentence, said, elapsed)

        if self.running:
            self._emit("on_status", "Every sentence answered.")

    def _ask(self, sentence):
        """Wait for one sentence, typed or spoken. (text, elapsed, control)."""
        if self.typed:
            clear = getattr(self.listener, "clear", None)
            if clear:
                clear()
        else:
            self._emit("on_status", "Speaking")
            say_english(sentence.en)
        if not self.running:
            return None, 0.0, "stop"

        while self.running:
            # Typed gets no countdown: a whole sentence is not a single word,
            # and the answer window that keeps the spoken drill moving would
            # mark you wrong for typing carefully. Spoken needs one, because
            # silence has to end the turn somehow.
            self._emit("on_status", "Write it" if self.typed else "Say it")
            window = None if self.typed else speech_window(sentence)
            started = time.time()
            said = self.listener.listen(
                window, should_stop=lambda: not self.running,
                accept=None if self.typed
                else (lambda text: grade(text, sentence).perfect))
            elapsed = time.time() - started
            if not self.running:
                return None, elapsed, "stop"
            if said is None:
                return None, elapsed, None

            # A correct answer outranks the command words, the same way it
            # does in the word drill: "para" is an answer before it is a stop.
            command = (None if grade(said, sentence).perfect
                       else command_of(said))
            if command == "stop":
                self._emit("on_status", "Paused")
                self.running = False
                return None, elapsed, "stop"
            if command == "skip":
                return None, elapsed, "skip"
            if command == "reveal":
                return None, elapsed, None      # graded as a blank answer
            if command == "repeat":
                if self.typed:
                    continue                    # the cue never left the screen
                self._emit("on_status", "Speaking")
                say_english(sentence.en)
                continue
            return said, elapsed, None
        return None, 0.0, "stop"

    def _judge(self, sentence, said, elapsed):
        result = grade(said, sentence)
        self.attempts += 1
        api_text, overturned = None, False

        # Keep the clip before anything else: the re-check needs a file, and
        # a verdict should be traceable to the audio behind it.
        clip = None
        if not self.typed and self.log is not None:
            clip = self._save_clip(sentence)

        if (not result.perfect and clip is not None and not self.typed
                and self.progress.verify_live and self.verifier):
            # The same bargain as the word drill. The local model decides
            # misses and is worst at exactly that, so a miss is never final
            # until a stronger model has heard the same audio.
            self._emit("on_status", "Double-checking…")
            api_text, echoed = self.verifier(clip)
            if api_text:
                again = grade(api_text, sentence)
                if again.perfect:
                    result, overturned = again, True
            if api_text is not None:
                # An echoed prompt is not evidence either way and must not be
                # scored as a miss the second opinion agreed with.
                if overturned:
                    self.progress.overturned += 1
                else:
                    self.progress.kept += 1
                self._emit("on_verify", self.progress.kept,
                           self.progress.overturned)

        if result.perfect:
            self.done.append(sentence)
            # Finished with, and it has to stay finished with after the app
            # is closed. Saved immediately rather than at the end of the run,
            # because a session that is stopped half way through still got
            # these right.
            self.progress.sentences_done.add(sentence.id)
            self.progress.save()
        else:
            # Back into the queue, the same as a missed card. Being shown the
            # answer is not the same as being able to produce it.
            at = min(len(self.queue), self.RETRY_GAP)
            with self._lock:
                self.queue.insert(at, sentence)

        self._emit("on_heard", said or "")
        self._emit("on_result", SentenceResult(
            sentence=sentence, grade=result, elapsed=elapsed,
            said=said or "", api_text=api_text, overturned=overturned))
        self._emit("on_progress", len(self.done), self.total)
        self._emit("on_counts")
        self._speak_verdict(sentence, result.perfect)
        if not result.perfect and self.hold_on_miss:
            self._emit("on_status", "Press Enter to continue")
            self.hold_on_miss(lambda: not self.running)

    def _save_clip(self, sentence):
        """The audio behind the verdict, or None when nothing was captured."""
        audio = getattr(self.listener, "last_audio", None)
        if audio is None:
            return None
        _, path = self.log.save_audio(0, audio, word=sentence.es[:40])
        return path

    def _speak_verdict(self, sentence, perfect):
        """Immediate, and on a miss the sentence read back properly.

        A miss is the one moment the sentence is worth hearing, and that is
        as true typed as spoken: you have just failed to produce it, and the
        slip on screen shows the spelling but not the sound. So typed reads
        it back too, unless SAY THE ANSWER BACK is off, which is what keeps
        the mode usable at a dinner table.

        A correct answer stays silent when typed. There is nothing to teach,
        and speaking every right answer is what would make the quiet mode
        stop being quiet.
        """
        if perfect:
            self._emit("on_status", "Perfect")
            if not self.typed:
                cues.correct()
            return
        self._emit("on_status", "Not quite")
        if not self.typed:
            cues.wrong()
        if not self.typed or self.progress.hints:
            say_spanish(sentence.es, self.progress.dialect)

    def _label(self, sentence):
        """Says whether this one has come round before.

        Read before the card is marked as asked, not after: the sentence has
        already been taken off the queue by the time this runs, so asking the
        queue whether it is still in there says "again" for everything.
        """
        return "SENTENCE · AGAIN" if sentence.id in self.asked else "SENTENCE"

    def summary(self):
        return {"perfect": len(self.done), "asked": self.attempts,
                "total": self.total}
