"""Recorder plus transcriber: audio in, words out."""
from .audio import Recorder
from .config import MAIN_MODEL, SCOUT_MODEL
from .transcribe import LocalTranscriber


class Listener:
    """Listens for one answer.

    Re-checks at each pause with the fast scout model and returns the moment
    what you said contains the answer, so a correct first try moves on
    immediately. If it does not, it keeps listening for the rest of the window,
    so extra repeats still land inside the same turn.
    """

    def __init__(self, model_name=MAIN_MODEL, device_name="",
                 scout_name=SCOUT_MODEL, transcriber=None, recorder=None):
        self.transcriber = transcriber or LocalTranscriber(model_name, scout_name)
        self.recorder = recorder or Recorder(device_name)
        self.last_audio = None      # the full window, for the record

    @property
    def floor(self):
        return self.recorder.floor

    def set_device(self, device_name):
        self.recorder.set_device(device_name)

    def calibrate(self):
        return self.recorder.calibrate()

    def listen(self, window, should_stop=None, accept=None):
        """Returns the transcript, or None if nothing usable was said."""
        self.last_audio = None

        def on_pause(partial):
            if accept is None:
                return False
            text = self.transcriber.transcribe(partial, scout=True)
            return bool(text and accept(text))

        speech, full, heard, early = self.recorder.record(
            window, should_stop=should_stop,
            on_pause=on_pause if accept else None)
        self.last_audio = full

        if speech is None or not heard:
            return None
        if early:
            # The scout already matched; re-running the main model would only
            # add latency to an answer that is settled.
            return self.transcriber.transcribe(speech, scout=True)
        return self.transcriber.transcribe(speech)
