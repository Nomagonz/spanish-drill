"""Typed answers, for the times you cannot speak.

This stands in for the microphone and nothing else changes. `DrillSession`
takes its listener as a plain dependency precisely so it can be handed
something that is not a microphone, and this is that: the same shape, with no
audio, no models and no network behind it.
"""
import queue
import threading
import time


class Gate:
    """Holds the drill still until somebody lets it through.

    A missed card is the one moment there is something to read, and the drill
    used to speak the answer and move on while you were still looking at it.
    This keeps it on screen until you say you are done with it.

    Shaped as a plain callable so `DrillSession` never learns what a keypress
    is: a test releases it from another thread, the window releases it from a
    key event, and neither is visible from the loop.
    """

    POLL_SECONDS = 0.05

    def __init__(self):
        self._open = threading.Event()
        self._waiting = threading.Event()

    @property
    def waiting(self):
        """True while the drill is actually held, so a key can know to release."""
        return self._waiting.is_set()

    def release(self):
        self._open.set()

    def __call__(self, should_stop=None):
        """Block until released, or until the session is stopping.

        The release is consumed on the way out rather than cleared on the way
        in, so an Enter that lands between the result being shown and this
        being reached still counts. Clearing first threw that keypress away
        and the drill sat there waiting for a second one.
        """
        self._waiting.set()
        try:
            while not self._open.wait(self.POLL_SECONDS):
                if should_stop and should_stop():
                    return
        finally:
            self._open.clear()      # spend it; the next hold waits afresh
            self._waiting.clear()


class TypedListener:
    """Waits for one typed answer at a time.

    The drill runs on a worker thread and whatever collects the answer (a text
    field, a test) runs somewhere else, so the handoff is a queue rather than a
    shared attribute. Blocking on the queue rather than polling a variable is
    also what lets the loop sit still between cards without burning a core.
    """

    POLL_SECONDS = 0.05         # how soon a waiting card notices a stop

    def __init__(self):
        self._answers = queue.Queue()
        # Set means running. A pause is held here rather than in the drill
        # because this is the only place that waits, and the wait is the
        # whole thing that has to stop: paused anywhere else, the answer
        # window would keep running and come back marked as silence.
        self._running = threading.Event()
        self._running.set()
        # Nothing is recorded, so the answer log skips the clip and the live
        # second opinion has nothing to send. Both already handle None.
        self.last_audio = None

    # -- pausing ----------------------------------------------------------
    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    @property
    def paused(self):
        return not self._running.is_set()

    # -- the writing end --------------------------------------------------
    def submit(self, text):
        """Hand an answer to whoever is waiting. Safe from any thread."""
        self._answers.put(text)

    def clear(self):
        """Drop anything typed before the current card was asked.

        Without this, a stray Enter pressed while the previous card was being
        graded would be spent on the next one, which reads as the drill
        skipping a card on its own.
        """
        while True:
            try:
                self._answers.get_nowait()
            except queue.Empty:
                return

    # -- the reading end, shaped like a microphone ------------------------
    def listen(self, window=None, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
        """Wait for one answer. None means silence, or a stop.

        The signature matches the microphone's so the session never has to ask
        which one it is holding, and `window` means what it means there: how
        long you get before it counts as no answer. A window of None waits
        forever, which is what a bare call in a test wants. `accept`, `fast`,
        `steer` and `second_pass` are recogniser concerns and there is no
        recogniser here.
        """
        # Wait out a pause before the clock starts, so the card on screen
        # keeps its full window when you come back to it.
        while not self._running.wait(self.POLL_SECONDS):
            if should_stop and should_stop():
                return None
        deadline = None if window is None else time.monotonic() + window
        while True:
            if should_stop and should_stop():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None             # no answer, the same as saying nothing
            wait = self.POLL_SECONDS
            if deadline is not None:
                wait = max(0.0, min(wait, deadline - time.monotonic()))
            if not wait:
                continue                # the deadline check above will fire
            try:
                text = self._answers.get(timeout=wait)
            except queue.Empty:
                continue
            return (text or "").strip() or None

    def calibrate(self):
        """Nothing to measure. Returned high so no dead-microphone warning fires."""
        return 1.0

    def set_device(self, device_name):
        pass

    def close(self):
        pass
