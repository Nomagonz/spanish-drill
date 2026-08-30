"""A tiny audio player that stays alive.

Run as `python -m spanish_drill._player`. It reads WAV paths on stdin, plays
each one, and prints a line when it is done.

Why a separate process at all: playing from the drill's own process means
opening CoreAudio output while the microphone stream is live, which is how this
app deadlocked twice. Why long-lived: spawning afplay per prompt costs about a
second of process startup on top of the audio, which is a third of the time
budget for a card. Holding one process with one open output stream costs
exactly the length of the sound.
"""
import sys
import wave

import numpy as np
import sounddevice as sd


def main():
    stream = None
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            with wave.open(path) as w:
                rate = w.getframerate()
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            audio = (audio.astype(np.float32) / 32768.0).reshape(-1, 1)
            if stream is None or int(stream.samplerate) != rate:
                if stream is not None:
                    stream.stop()
                    stream.close()
                stream = sd.OutputStream(samplerate=rate, channels=1,
                                         dtype="float32")
                stream.start()
            stream.write(audio)
        except Exception as exc:                # a bad file must not kill it
            print(f"error {exc}", file=sys.stderr, flush=True)
        print("done", flush=True)


if __name__ == "__main__":
    main()
