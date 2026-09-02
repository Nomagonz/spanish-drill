# spanish-drill

A hands-free Spanish vocabulary drill. It says an English word, listens for the
Spanish, grades what you said, and moves on. 253 cards, spaced repetition.

There are two versions. The **desktop app** is the one that works properly; the
**web page** came first and is kept because it is deployed.

## Desktop app

    ./run.sh                  drill
    ./run.sh --serve          drill from a phone, typed
    ./run.sh --model small    ~3x faster per word, measurably less accurate
    ./run.sh --review         re-check past answers against a stronger model
    ./run.sh --review --conjugations
                              the same, for the conjugation drill's own log
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

### Spelling that is not a mistake

Three Spanish spellings say the same sound, so a recogniser writing down what
it heard can land on either one and be right. `h` is silent outside the digraph
`ch`, so `hago` is `ago` and `he` is `e`. `b` and `v` are one phoneme with no
distinction anywhere in the language, so `voy` comes back as `boy`. `ll` and
`y` merged for all but a few speakers. Answers are compared with those folded
away, and a match found that way is a full match, not a near one: nothing was
fumbled, the letter was never spoken.

Measured across all 1579 deck answers, the three rules together merge exactly
one pair, `o` and `oh`. That is the test any further rule has to pass: it may
fix a spelling, never merge two words the deck distinguishes.

Note what is deliberately absent. `p` and `b` stay separate, because Spanish
really does contrast them, so `paz` is never `vas`. Tolerating a single edit on
short words would cover far more transcription slips and is refused: 612 pairs
of real deck answers sit one edit apart, `anda`/`andan` and `amiga`/`amigo`
among them, and a drill whose whole subject is the last letter of a word cannot
afford that.

### Why the prompt never names the answer

The steer biases decoding. Name the expected answer in it and the recogniser
writes that answer down whether or not you said it, so the grade stops meaning
anything. This was measured rather than assumed, on twelve clips the drill had
correctly marked wrong: with the answer written into the prompt, **nine came
back correct**. "Arco." decoded as "Hago." and "Pas, pas, pas." as "Vas, vas,
vas."

`assert_steer_is_clean()` enforces it and the review refuses to run otherwise.
A steer is a constant per mode, never built per card, because a per-card steer
is exactly how the answer gets in.

The tempting version is a prompt naming the form under test: "this is a form of
haber, it should be `he`, do not accept others." On clean recordings it scores
well. On real ones it destroys correct answers, and the reason is worth writing
down, because it is not the obvious one. Once the answer is in the prompt, a
transcript of the answer and an echo of the prompt are the same string, so echo
detection stops working: `soy`, `voy`, `eres` and `estás` — all answered
correctly — came back as silence.

### Reading it again with nothing suggested

A steer is a prior, and on a one or two syllable word the prior swamps the
signal. Measured on clean recordings of known forms, the conjugation steer
turned `he` into "Hi." and lost `tengo` outright; both decode perfectly with no
prompt at all.

So when a steered reading is not the answer, the conjugation drill decodes once
more with an empty prompt and prefers that only if it matches. This is not
asking again until the answer appears. The second prompt is empty, so it
suggests nothing and cannot pull the reading anywhere, which is exactly what
naming the answer does do.

Measured on fourteen clean forms with the answer known, six of them
deliberately the wrong person or tense: steer alone 12, no prompt 14, both 14,
and no wrong form was accepted by any of the three. On real recordings it took
accepted answers from 18 to 22 of 42. It costs one extra decode per miss, which
is why it is on where that decode is the quick model and off in the ordinary
drill, where a miss goes to the API second opinion anyway.

An echoed prompt is not a transcript. When one comes back the clip is decoded
again with no prompt, on the local model as well as the API, because until that
was true an echo made the local model return nothing and a card you had
shouted at was recorded as unattempted. Measured on nine real clips, the steer
came back instead of a transcript on five. The retry fires only on an echo,
never on the answer failing to appear: asking again until it shows up is a
slower way of leaking it.

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

### Typing instead

TYPE INSTEAD drills the same cards in silence: the cue stays on screen, the
answer comes from the keyboard, nothing is spoken and nothing is recorded. It
does not wait for the speech models, because the whole point is to be usable
the moment you open it somewhere you cannot talk.

It is also the only mode that can grade accents. Speech cannot be: the
recogniser is inconsistent about them, so `normalize` strips them, and 748 of
the deck's answers lose a distinction the drill can otherwise never test. 692
of those are conjugated forms, which is where the accent is doing grammatical
work rather than decorating a vowel. Spoken, `hablo` is accepted for `habló`
and always will be. Typed, it is a near miss: the right word recalled
imperfectly, which is what `close` already means everywhere else, and worth a
3 rather than a 5. That also keeps the mode usable without reaching for the
option key at a dinner table.

Typing gets its own sense of what counts as a quick answer. Saying `ir` and
saying `encontrar` take about the same time and typing them does not, so the
speed bonus is measured against a budget that grows with the word. Without
that every long word would score as laboured and drift its own schedule, which
would be a scheduling bug wearing a measurement's clothes.

The answer window applies here too, so silence costs the same either way and
the two modes grade on one clock. At three seconds that is brutal for a long
word; the dial is in Settings under ANSWER WAIT, and the countdown next to the
status is there because a deadline you cannot see is one that marks you wrong
without warning.

A missed card stays on screen until you press Enter. Only here: the spoken
drill is hands-free on purpose and would stop being so the moment it started
waiting for a keypress.

Every answer records which way it was given, because a typed answer and a
spoken one are not the same evidence about the same word, and a log that does
not say which is which can never be separated afterwards.

### Sentences

SENTENCES shows an English sentence and takes the Spanish typed back. A
perfect answer is perfect; anything else comes back marked word by word, with
what you wrote next to what was wanted, missing words in amber and invented
ones listed underneath. It is typed only, so it never waits for the speech
models and works the moment the window opens.

The rule that makes it worth doing: **a sentence is only ever offered once
every content word in it is already in your rotation.** Composing is hard
enough without it also being a vocabulary test. "In the rotation" means
answered right at least once and scheduled a day or more ahead, which is what
the panel calls LEARNING, so it takes in both the words still being learned
and the ones that have matured past them. A word you have never seen can
never qualify, because it is not in `progress.json` at all.

Function words are the one exception and have to be. `el`, `la`, `un`, `una`,
`no` and `a` have no card in the deck, and `de`, `en`, `con` and `que` are
cards most people never drill. Gating on articles would mean no sentence is
ever available however much vocabulary has been learned.

Requirements are read off the Spanish rather than written beside it. A list
kept by hand drifts from the sentence it describes and the drift is
invisible, because the sentence still reads fine while the gate has quietly
stopped checking a word. `sentences.py` resolves every word back to a deck
card, conjugations and plurals and attached pronouns included, and a word it
cannot account for fails the tests rather than shipping.

Where a word is two cards the strict direction wins, with two exceptions
that are corrections rather than leniency. A word spelled exactly as the deck
writes it is that word and not an inflection of some other one: `lista` is
the noun, not the feminine of `listo`, and reading it as both held back
sentences over an adjective they never used. Verb readings are never dropped
that way, because an unlearned conjugated form is the one thing that must not
slip through, so `vino` still answers for `venir` as well as for wine.

The conjugated form is gated too, not just the infinitive behind it: knowing
`dormir` is not knowing `duermo`. Where a written form is two cards — Spanish
spells the nosotros present and preterite of an -ir verb identically, so
`salimos` is both — any one of them satisfies it, because you only ever
produce one. Verbs with no conjugation cards at all, which is most of them,
fall back to the infinitive since there is nothing else to check.

Subject pronouns are optional, because in Spanish they are: "hablo español"
and "yo hablo español" are the same sentence. Only at the front, though, and
`él` and `tú` are only forgiven when added, never when missing, since with
the accents stripped they are indistinguishable from `el` and `tu` and
forgiving those would forgive a dropped article.

It can be spoken as well as typed, following the same TYPE INSTEAD checkbox
as the other two modes. Spoken, a miss goes to `gpt-4o-transcribe` for a
second opinion exactly as a missed word does, because the local model decides
misses and is worst at precisely that. The steer prompt is different: the word
one tells the recogniser to expect isolated Spanish words, which is the wrong
instruction for a whole clause and chops it into fragments. The answer window
grows with the sentence, since the dial that suits one word does not suit ten.
A spoken miss never waits for a keypress; only typing holds the card on screen.

A sentence is finished with the moment you get it right, and stays finished
with after the app is closed: the id goes into `sentences_done` in
`progress.json` the instant it is earned, not at the end of the run, because
a session stopped half way through still got those right. A miss brings it
back five sentences later and nothing else. There is no ease factor and no
2/8/30/47 ladder here; that is the word drill's.

Nothing here touches SM-2. The words are already being reviewed on their own
account and grading them twice would distort the schedule. What a sentence
writes is which ones are done, the second-opinion tally, and, when spoken,
the clip behind its verdict.

The bank is deliberately larger than what is unlocked. Sentences are held
back by the gate rather than thrown away, so the same file keeps yielding as
more words and more conjugated forms are learned.

#### Where the sentences come from

`sentences.json` is a curated bank of 324 that ships with the repo. It needs
no key, no network and no spend, and it is the floor: the mode works on it
alone.

On top of that, opening the mode asks the API for more, built from the words
currently available. One first so there is something fresh straight away,
then batches in the background that join the queue as they arrive. They are
cached in `sentences-generated.json` and capped at 500 for good — the file is
the tally, so reopening the app cannot start the bill again.

**The model proposes and the gate disposes.** Every sentence that comes back
goes through exactly the same resolver the curated bank is held to, and one
word outside your rotation throws the whole sentence away. This is not
belt-and-braces. Asked to stay inside a word list, the model wanders out of
it reliably, and the wandering is invisible because the sentence still reads
perfectly well. Two failures worth recording, both silent and both total:

- With the JSON fields called `en` and `es`, 58 of 82 sentences came back
  with **Spanish in the English field**. The cue then is the answer, and the
  card asks you to copy what is already on screen. Naming the fields
  `spanish` and `english`, and generating the Spanish first so the English is
  a translation of something, fixed it. `is_english()` catches the rest.
- It writes subject pronouns however firmly it is told not to, so a stored
  "Él tiene un perro" would mark the ordinary "Tiene un perro" as a missing
  word. They are stripped on the way in, matched with their accents on so
  that "El perro corre" keeps its article.

Roughly four in five survive. The rest cost a fraction of a cent and never
reach you.

### Drilling from a phone

`--serve` puts a typed drill on a web page. The browser is a keyboard and a
screen and nothing else: the deck, the schedule, the grading and the sentence
bank never leave this machine, and what crosses the wire is a cue going out
and a typed answer coming back. Both the sentence drill and the ordinary
word drill are there, and the word drill moves the real SM-2 schedule, so a
session on the sofa counts the same as one at the desk.

No web framework. Server-sent events one way and a form POST the other is the
whole protocol, and both are in the standard library, so drilling remotely
adds no dependency to a project whose point is that it works offline.

The port is only reachable from this machine. Put a tunnel in front of it to
get at it from outside, and note that the tunnel is public the moment it
exists: the token in the printed URL is the only thing between a stranger who
finds it and your deck.

Speech is not wired up yet. The architecture allows it — `Recorder` reads
blocks out of a deque that only one line fills, so a remote source is a
sixty-line subclass rather than a rewrite — but the browser side is the real
work: 48kHz Opus has to become 16kHz mono PCM, and the phone's speaker feeds
its own microphone.

### Conjugations on their own

CONJUGATIONS drills nothing but conjugated forms, for verbs the ordinary drill
has already taught, ten verbs at a time in frequency order.

**It keeps its own books, and that is the whole point.** The schedule it moves
lives in `conjugation-progress.json`, and its answers land in
`answers/conjugations.jsonl`. `progress.json` is opened once, read for a single
fact, and never written: which infinitives are learned. So however long a
sitting of paradigms runs, not one vocabulary card's review date moves.

The separate answer log is part of the same rule rather than tidiness.
`--review` repairs cards in whichever tracker it is handed, so a shared log
would let a conjugation re-check reach into the vocabulary schedule. Use
`./run.sh --review --conjugations` to re-check that log against its own.

Frequency order costs nothing to maintain, because `deck.json` is already
written in it: ser, estar, haber, tener, ir, hacer. A verb's position in the
deck is its rank, so there is no second list to drift out of step.

The batch rolls rather than emptying. A verb whose every form has joined the
rotation drops out and the next one down the list takes the slot, so a run
never narrows to one stubborn paradigm while nine finished verbs hold their
places. Retiring is not forgetting: those forms keep coming back for review
on their own schedule. The batch gates what is *met*, never what is *reviewed*.

Ten verbs is 300 cards on the Spain dialect, which sounds worse than it is.
The chain is the same one-form-at-a-time ladder the main drill walks, so each
verb has exactly one card unlocked and unlearned at any moment: ten new forms
in flight, never three hundred.

The cue is not read aloud here, and the answer after a miss always is. The
English cues in this mode are long ("you all would be able to"), the answers
are a syllable or two, and the cue is on screen anyway, so speaking it was
several seconds a card buying nothing. Turning it off costs the hands-free
property, which is a fair trade here and a bad one in the ordinary drill, so
it is a setting: READ THE ENGLISH CUE ALOUD, in Settings. It only ever governs
the way in. A missed card speaks the answer and its example whatever it says,
because that is the part that teaches.

Both recognisers get their own steer. Every example in the word steer is an
infinitive or a noun, and both models follow it: in one real session it wrote
"seguir" for `sigo`, "poner" for `pongo`, "salir" for `sales` and "oir" for
`oigo`. Four verbs, all pulled the same way. `CONJUGATION_STEER` says to expect
a conjugated form and carries no example words at all, because examples are the
part a model hands back instead of a transcript.

The quick model decides, as in a placement run and for the same reason: the
main model costs the better part of ten seconds on a long window and only ever
changes misses, which go to the second opinion regardless.

Everything else is inherited unchanged. Asking, grading, SM-2, the session
learning ladder. A paradigm is drilled exactly the way a word is, and the cue
still refuses to name the verb, because working out which verb is wanted is
most of the card. The batch is named once at the start of the run instead,
where it cannot help with any particular answer.

It runs *beside* the ordinary drill rather than replacing it. The main drill
still introduces conjugations through its own unlock chain, on its own
schedule, and neither tracker can see the other's. That is deliberate: this
was asked for as an addition, and taking the forms out of the normal drill to
make the two exclusive would be changing something nobody asked to change.

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
      typed.py        the keyboard, shaped like a microphone
      conjugation.py  the verb tables and the cues they generate
      paradigm.py     the conjugation-only drill and its separate tracker
      sentences.py    the sentence bank, the gate, and the diff
      composition.py  the sentence drill loop
      generate.py     sentences written by the API, judged locally
      session.py      the drill loop                    (no Qt, no hardware)
      review.py       re-judging past answers
      ui.py           the window                        (no drilling logic)
      composition.py  the sentence drill loop
      serve.py        the same drill, over HTTP, for a phone
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

## The interface

Both the window and the phone are organised the same way, into three tabs:

    DRILL      the card, what you type, and how it was judged
    PROGRESS   today's work, the ladder, the double-check tally
    SETTINGS   everything set once

What stays above the tabs is the pair that has to be readable whichever one is
open: what the drill is doing, and how far into it you are.

Before this, both were a single column of eight stacked zones, so the card, the
deck's whole history and a preference touched once a month competed for the
same eye. Only the first of those is wanted while a word is on screen.

### Things the reorganisation turned up

None of these were visible from reading the code. All three came from driving
the real interface and looking at what it did.

- **The vocabulary drill was called GO.** Beside three buttons named SENTENCES,
  PLACEMENT and CONJUGATIONS, that reads as a generic start button rather than
  as one of the four modes, and the ordinary drill looked missing. It is called
  WORDS in both interfaces now, and the name is a constant so the four places
  that put the button back cannot drift apart.
- **The phone's CONJUGATIONS button did nothing at all.** It had markup and no
  click handler: every other mode was wired and that one was never connected.
- **STOP could not stop a drill.** The panel reports `running` from whether the
  session thread is alive, nothing published a panel after starting one, and
  `on_finished` is raised from inside that thread, so the answer was stale on
  the way in and wrong on the way out. PAUSE and STOP are driven off it and sat
  disabled for whole sessions. Starting now announces at once, and the hub
  tracks whether a drill is meant to be going separately from whether its
  thread has finished unwinding.

Each mode also says what it is and how much of it is waiting, which meant the
panel had to learn one number it never reported: how many conjugated forms are
due. Asking costs about two milliseconds, so it is asked every time rather than
cached into something that can go stale.

## One database

The desktop app and the phone page have always shared `progress.json`, because
they run on the same machine. The web page could not: it is a static file on
GitHub Pages, and static hosting cannot write anything, so it kept its own
schedule in `localStorage` and drifted. Two counters that were never going to
agree.

`store.py` is what fixes that. `Progress` no longer knows where it is kept: it
asks a store for a dict and hands one back, and the store is either the file it
always was or a database every client shares. The merge that already existed
for two processes on one file did not have to change at all.

    stamp()     has it moved since we last looked
    read()      the saved state
    write()     commit what we built from a known version

Switched on with two local files, both gitignored:

    .sync-url       https://<the worker>
    .sync-token     the same value as the worker's SYNC_TOKEN secret

`run.sh` exports them if both exist. With either missing the drill runs exactly
as it did before, on `progress.json` alone, which is why the whole test suite
passes untouched.

### What keeps it honest

Two failure modes, both with tests that fail without the fix:

- **A push that did not land is not recorded as agreed.** Otherwise reconnecting
  after a session offline would see that work as already shared and let the
  database's older copy overwrite it.
- **A write carries the version it was built on**, and the worker refuses it if
  that version has moved. Without it, the phone and the desktop saving inside
  one round trip means whichever lands second throws the other's cards away.
  A refused save folds in what the other client did and offers its own again.

One known rough edge, taken deliberately: drilling with the network down, then
quitting and coming back, keeps every card but can count today's numbers twice
until the day rolls over. The alternative was risking the whole session.

### The worker

`worker/` holds it. One row, not one row per card: a card per row would rewrite
all 1670 on every save, a few seconds apart, and pass D1's daily allowance
inside one sitting. The per-card merging lives on the client, where it already
worked.

    cd worker
    npx wrangler d1 execute spanish-drill-progress --remote --file=schema.sql
    npx wrangler deploy
    npx wrangler secret put SYNC_TOKEN     # the contents of .sync-token

The account id is pinned in `wrangler.toml` on purpose. This login can see a
second account holding live business data, and an unpinned command is one
prompt away from landing there.

## Web version

`index.html`, live at https://nomagonz.github.io/spanish-drill/

It is no longer a separate program. It reads the same `deck.json`, runs the
same SM-2 schedule keyed by the same card ids, grades by the same rules, and
keeps its progress in the same database the window and the phone use. Answer a
word here and it is answered everywhere.

What it does not do is the spoken drill, or sentences, placement and
conjugations. Those stay in the desktop app, and the page says so rather than
pretending otherwise.

### Why the rules are written twice

The page cannot ask this machine whether an answer was right: the whole point
of it is to work when this machine is asleep. So `web/drill.js` is a port of
`text.py`, `grading.py`, `scheduler.py` and the queue-building half of
`progress.py`.

Two copies of a rule drift. `tests/test_web_port.py` is what stops it: it puts
the same inputs through both, over the real deck, and fails on any difference.
Not a spot check — the cases are generated from the deck, and the suite is
checked by breaking the port on purpose and confirming each break is caught:

    the half-to-even rounding SM-2 intervals land on
    the silent h, and its exception in "ch"
    the full list of articles that get stripped
    the unlock rule, which asks for a right answer and not merely a card
    the head of a verb's chain, which is the infinitive in the other tracker
    the order of the persons, and of the tenses
    the dialect filter that drops vosotros

Each of those was found by a break slipping through a first draft of the tests
that looked thorough and was not: every state had reps of 1, so "answered
right" and "seen at all" gave the same answer, and no sequence ever landed on
a rounding boundary.

### The key

The page is public and the database is not. With a key it drills the shared
schedule; without one it keeps its own progress in that browser and says so.
The key is typed in once, kept on the device, and never written into the page.

Progress from the old Leitner version is carried forward rather than dropped:
those saves keep their rung on the ladder and their lapse history.

## Unrelated repo — do not touch

`Nomagonz/boerne-site` is a live production business site serving
`boernephotoboothco.com` via Netlify continuous deployment. This project is
deliberately kept in a separate repo.
