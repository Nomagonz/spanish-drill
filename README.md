# spanish-drill

A hands-free Spanish vocabulary drill. It says an English word, listens for the
Spanish, grades what you said, and moves on. 253 cards, spaced repetition.

There are two versions. The **desktop app** is the one that works properly; the
**web page** came first and is kept because it is deployed.

## Desktop app

    ./run.sh                  drill
    ./run.sh --model small    ~3x faster per word, measurably less accurate
    ./run.sh --review         re-check past answers against a stronger model
    ./run.sh --devices        list microphones

Everything is local except the optional second opinion. Speech models are cached
in `~/.cache/huggingface/hub` after the first run and never fetched again, so a
session works with the network off.

- **Recognition**: faster-whisper on the CPU.
- **Speech**: the macOS `say` voice (Paulina for Latin America, Mónica for Spain).
- **Progress**: `progress.json`, written atomically.
- **Answer log**: every answer's audio in `answers/`, with `answers.jsonl`
  recording how each one was judged.

### How an answer is judged

1. The recogniser re-checks at every pause and accepts the moment what you said
   contains the answer, so a correct first try moves on immediately. If it does
   not hear it, it keeps listening for the rest of the window, so repeating
   yourself still lands in the same turn.
2. A miss goes straight to `gpt-4o-transcribe` for a second opinion. If the clip
   really did contain the answer, the miss is reversed on the spot and the card
   is never penalised for the local model mishearing you.
3. Anything still judged a miss comes back later in the session.

The bias is toward rejecting. A false reject costs one extra review of a word
you know; a false accept banks a mistake as correct and hides the word for
weeks. So `llevar` is never accepted for `llegar` even though they differ by one
character, and a conjugation is never accepted for an infinitive.

### The steer prompt

With no context a bare uncommon infinitive loses to a common phrase that sounds
the same: `llevar` decodes as "y el bar". The recogniser is told to expect an
isolated Spanish word, which collapses that ambiguity.

The prompt must never name a word in the deck. Prompting with the word under
test makes the recogniser agree with whatever was said, and the grade stops
meaning anything. `assert_steer_is_clean()` enforces this and the review refuses
to run otherwise.

That same prompt causes the failure most of `transcribe.py` guards against:
these models hand the prompt back when the audio holds nothing usable. That is
not a transcript and is never graded as one. On an echo the clip is sent again
with no prompt, since the prompt is what caused it. A retry is never triggered
by "it didn't say the answer" — asking again until the answer appears is a
slower way of leaking it.

### Scheduling

SM-2. Every card carries its own ease factor, so a word you keep fumbling grows
slower permanently while an easy one accelerates. Answer quality comes from what
is observable: silence scores below a wrong guess, a mangled-but-right answer
below a clean one, and an instant recall above a laboured one.

`LEARNING` counts cards scheduled at least a day out. `MATURE` is Anki's
threshold of a 21-day interval, which takes four correct reviews over about
three weeks, so it reads zero on the first day by design.

### Layout

    spanish_drill/
      config.py       paths and tuning values
      text.py         normalisation, edit distance      (no deck knowledge)
      deck.py         the cards
      grading.py      what counts as a correct answer
      scheduler.py    SM-2
      progress.py     saved progress and settings
      answers.py      the answer log and its audio
      audio.py        microphone capture and devices    (no opinion on meaning)
      speech.py       the spoken voice
      transcribe.py   local and API recognition, the steer, echo detection
      listener.py     capture + recognition, with early accept
      session.py      the drill loop                    (no Qt, no hardware)
      review.py       re-judging past answers
      ui.py           the window                        (no drilling logic)
      cli.py          entry point

`session.py` holds the loop and takes its listener and verifier as plain
callables, so the whole thing runs in a test with no microphone, no network and
no Qt. `ui.py` is an adapter over it and contains no rules about what an answer
means.

### Tests

    .venv/bin/python -m pytest tests/ -q

The audio tests are worth knowing about. Checking mid-window blocks the capture
loop while a model runs, and the audio callback keeps filling its queue
throughout. Nothing pulled those blocks back out, so a five-second window was
saving as little as one second, and that truncated clip was what got sent for a
second opinion. Both `drain()` calls in `Recorder.record` are pinned by tests
that fail if either is removed.

A related trap: patching `time.sleep` globally in a fixture silently disabled
the timing those tests depend on and turned them into tests that could not fail.
The pacing pauses in `session.py` go through `pace()` so tests can neutralise
them without touching the clock everything else depends on.

## Web version

`index.html` is the original single-file version, live at
https://nomagonz.github.io/spanish-drill/

It works, but iOS Safari cannot do hands-free drilling: Apple requires a user
gesture to start each recognition, and the loop starts listening after speaking
a word, where no gesture exists. It also still has the older Leitner scheduler
rather than SM-2. The desktop app exists because no amount of restructuring
fixes the first problem.

Progress there is stored in `localStorage` under `esdrill:v2`.

## Unrelated repo — do not touch

`Nomagonz/boerne-site` is a live production business site serving
`boernephotoboothco.com` via Netlify continuous deployment. This project is
deliberately kept in a separate repo.
