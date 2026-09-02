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


class SharedListener:
    """One card, however many screens are looking at it.

    `DrillSession` takes its listener as a plain dependency, so this is
    another one. It holds whatever this machine listens with — a microphone,
    or nothing at all — and a queue that any screen can drop an answer into.
    Whichever arrives first is the answer, and the other is told to give up.

    That is the whole of what makes two screens one drill rather than two.
    Without it the window and the phone each had to own a session to have
    somewhere to put an answer, and two sessions cannot be on the same card.

    The inner listener is run on a thread of its own because a microphone
    blocks while it records: there is no way to wait on audio and on a queue
    at once without one of them being somewhere else.
    """

    POLL_SECONDS = 0.05

    def __init__(self, inner=None):
        self.inner = inner
        self._typed = queue.Queue()
        self._paused = threading.Event()

    # -- what any screen calls ---------------------------------------------
    def submit(self, text):
        """An answer, from whichever screen was in front of you."""
        self._typed.put(text or "")

    def clear(self):
        while True:
            try:
                self._typed.get_nowait()
            except queue.Empty:
                break
        if hasattr(self.inner, "clear"):
            self.inner.clear()

    # -- pausing ------------------------------------------------------------
    def pause(self):
        self._paused.set()
        if hasattr(self.inner, "pause"):
            self.inner.pause()

    def resume(self):
        self._paused.clear()
        if hasattr(self.inner, "resume"):
            self.inner.resume()

    @property
    def paused(self):
        return self._paused.is_set()

    # -- the microphone's shape ---------------------------------------------
    def listen(self, window=None, should_stop=None, **rest):
        """Wait for one answer from anywhere. None means silence, or a stop."""
        done = threading.Event()
        box = {}

        def enough():
            return done.is_set() or bool(should_stop and should_stop())

        def run_inner():
            try:
                box["said"] = self.inner.listen(window, should_stop=enough, **rest)
            except Exception:
                box["said"] = None
            finally:
                # Assigned before the flag, so a reader that sees the flag can
                # rely on the answer being there.
                done.set()

        worker = None
        if self.inner is not None:
            worker = threading.Thread(target=run_inner, daemon=True)
            worker.start()

        while True:
            if done.is_set() and "said" in box:
                return box["said"]          # this machine's own input won
            try:
                text = self._typed.get(timeout=self.POLL_SECONDS)
            except queue.Empty:
                if should_stop and should_stop():
                    done.set()
                    if worker:
                        worker.join(timeout=1)
                    return None
                continue
            if self.paused:
                continue        # a held drill takes nothing from any screen
            done.set()          # tell this machine's own input to give up
            if worker:
                worker.join(timeout=1)
            return (text or "").strip() or None

    # -- the rest of the microphone's surface -------------------------------
    def calibrate(self):
        return self.inner.calibrate() if hasattr(self.inner, "calibrate") else 1.0

    def set_device(self, device_name):
        if hasattr(self.inner, "set_device"):
            self.inner.set_device(device_name)

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()

    @property
    def last_audio(self):
        return getattr(self.inner, "last_audio", None)
