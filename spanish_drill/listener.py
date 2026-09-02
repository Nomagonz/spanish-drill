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

    def close(self):
        """Release the microphone. The stream is held open all session."""
        self.recorder.close()

    def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
        """Returns the transcript, or None if nothing usable was said.

        `steer` is the prompt the recogniser is given about what to expect.
        None means the ordinary one, for isolated dictionary words; the
        conjugation drill passes its own, because told to expect infinitives
        the model produces them.

        `fast` keeps the quick scout model for the final answer too. The main
        model takes about six seconds on a two-second clip, which is most of a
        card, and it only decides misses: anything it accepts the scout has
        already accepted, and anything it rejects goes to the second opinion,
        which is both quicker and better.

        `second_pass` decodes once more with no prompt at all when the steered
        reading is not the answer. A steer is a prior, and on a one or two
        syllable word the prior can swamp the signal: measured on clean
        recordings of known forms, the steer turned `he` into "Hi." and
        `tengo` into a miss, both of which decode perfectly with no prompt.

        This is not "ask again until the answer appears". The second prompt is
        empty, so it carries nothing about what was expected and cannot pull
        the reading toward it. Measured on fourteen clean forms with the
        answer known, six of them deliberately the wrong person or tense: the
        steer alone scored 12, no prompt scored 14, both together scored 14,
        and no wrong form was ever accepted by any of the three. On the
        owner's own recordings it took accepted answers from 18 to 22 of 42.
        """
        self.last_audio = None

        def on_pause(partial):
            if accept is None:
                return False
            # No retry: this runs at every pause and only decides whether to
            # stop early. Paying for a second decode here would eat the time
            # the early accept exists to save.
            text = self.transcriber.transcribe(partial, scout=True,
                                               steer=steer, retry=False)
            return bool(text and accept(text))

        speech, full, heard, early = self.recorder.record(
            window, should_stop=should_stop,
            on_pause=on_pause if accept else None)
        self.last_audio = full

        if speech is None or not heard:
            return None
        # `retry=True` on both, explicitly. This decode is the one that gets
        # graded, whichever model produces it, and an echo here becomes a
        # mark against a card you did answer. The transcriber's own default
        # turns the retry off for the scout, which is right for the poll
        # above and wrong here: in a fast mode the scout IS the answer.
        quick = bool(early or fast)
        text = self.transcriber.transcribe(speech, scout=quick, steer=steer,
                                           retry=True)
        if not second_pass or accept is None or (text and accept(text)):
            return text
        # The steer did not produce the answer. Read it once more with no
        # prompt, so nothing at all is being suggested, and prefer that only
        # if it does. A blank prompt cannot leak what was expected.
        blind = self.transcriber.transcribe(speech, scout=quick, steer="",
                                            retry=False)
        return blind if blind and accept(blind) else text
