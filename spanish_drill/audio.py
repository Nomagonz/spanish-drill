"""Microphone capture and device selection.

Capture only. What the audio means is somebody else's problem.
"""
import queue
import time
import wave

import numpy as np
import sounddevice as sd

from .config import (BLOCK_SECONDS, CALIBRATION_SECONDS, MIN_SPEECH_SECONDS,
                     SAMPLE_RATE)


def input_devices():
    """[(index, name)] for everything that can record."""
    try:
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


def resolve_device(name):
    """Name -> index, or None for the system default.

    Matching by name matters: connecting or unplugging a device renumbers the
    rest, so a stored index quietly starts pointing at a different microphone.
    An unknown name falls back to the default rather than failing.
    """
    if not name:
        return None
    for index, device_name in input_devices():
        if device_name == name:
            return index
    return None


def save_wav(path, audio):
    """float32 mono -> 16-bit PCM, so anything can open it."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class Recorder:
    """Records a window of audio and reports where speech was.

    Keeps the whole window, including when it decides you were silent. A miss
    blamed on silence may be speech that fell under the noise floor, and that
    is only ever provable from the audio that produced the verdict.
    """

    def __init__(self, device_name=""):
        self.device = resolve_device(device_name)
        self.floor = 0.004
        self.level = 0.0

    def set_device(self, device_name):
        self.device = resolve_device(device_name)

    def calibrate(self, seconds=CALIBRATION_SECONDS):
        """Measure the room, every session rather than once per process.

        A floor measured while music was playing stays wrong all session
        otherwise.
        """
        buf = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=1, dtype="float32", device=self.device)
        sd.wait()
        rms = float(np.sqrt(np.mean(buf ** 2)))
        self.floor = max(0.0025, rms * 2.5)
        return self.floor

    def record(self, seconds, should_stop=None, on_pause=None):
        """Capture a window.

        `on_pause(audio_so_far)` is called each time you seem to have finished
        a try. Returning True ends the recording early, which is how a correct
        answer moves on without waiting out the window.

        Returns (speech_audio, full_window, heard_speech, stopped_early).
        `speech_audio` starts just before the first sound; the leading silence
        is dropped because feeding it to a recogniser invites hallucination.
        """
        inbox = queue.Queue()
        blocks = []
        heard_speech, first_block, stopped_early = False, None, False
        pause_run, last_check = 0.0, 0.0
        block_frames = int(SAMPLE_RATE * BLOCK_SECONDS)

        def callback(indata, frames, time_info, status):
            inbox.put(indata.copy())

        def drain():
            # A pause callback can block this loop for a second or more while a
            # model runs. The audio callback keeps filling the queue throughout,
            # so whatever arrived meanwhile has to be pulled in, or the saved
            # clip is missing exactly the part recorded during the check.
            while True:
                try:
                    blocks.append(inbox.get_nowait())
                except queue.Empty:
                    return

        started = time.time()
        with sd.InputStream(callback=callback, blocksize=block_frames,
                            dtype="float32", device=self.device,
                            samplerate=SAMPLE_RATE, channels=1):
            while time.time() - started < seconds:
                if should_stop and should_stop():
                    drain()
                    return None, self._join(blocks), heard_speech, False
                try:
                    block = inbox.get(timeout=0.3)
                except queue.Empty:
                    continue
                blocks.append(block)

                self.level = float(np.sqrt(np.mean(block ** 2)))
                if self.level > self.floor:
                    if not heard_speech:
                        heard_speech = True
                        first_block = max(0, len(blocks) - 4)   # a little lead-in
                    pause_run = 0.0
                elif heard_speech:
                    pause_run += BLOCK_SECONDS

                if (on_pause is not None and heard_speech
                        and pause_run >= self._pause_threshold
                        and time.time() - last_check > self._check_interval):
                    last_check = time.time()
                    partial = self._join(blocks[first_block:])
                    if partial is not None and len(partial) >= SAMPLE_RATE * MIN_SPEECH_SECONDS:
                        accepted = on_pause(partial)
                        drain()     # everything recorded while it ran
                        if accepted:
                            return partial, self._join(blocks), True, True
            drain()                 # and anything still queued at the end

        window = self._join(blocks)
        if not heard_speech or window is None:
            return None, window, False, False
        speech = self._join(blocks[first_block:])
        if speech is None or len(speech) < SAMPLE_RATE * MIN_SPEECH_SECONDS:
            return None, window, heard_speech, False
        return speech, window, True, False

    _pause_threshold = 0.25
    _check_interval = 0.4

    @staticmethod
    def _join(blocks):
        return np.concatenate(blocks).flatten() if blocks else None
