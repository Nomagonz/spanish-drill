"""Paths and tuning constants.

Everything here is a value someone might reasonably want to change. Anything
that encodes a rule rather than a preference lives with the code that uses it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DECK_PATH = ROOT / "deck.json"
PROGRESS_PATH = ROOT / "progress.json"
ANSWERS_DIR = ROOT / "answers"
ANSWER_LOG = ANSWERS_DIR / "answers.jsonl"

# The shared database, if there is one. Unset means the original behaviour and
# nothing else: progress.json on this disk, this machine, nobody else reading
# it. Switching it on is what makes the desktop app and the web page two views
# of one schedule rather than two schedules that happen to look alike.
#
# Kept in the environment rather than in progress.json because the token is a
# credential and progress.json is a file people copy around and back up.
SYNC_URL = os.environ.get("SPANISH_DRILL_SYNC_URL", "").rstrip("/")
SYNC_TOKEN = os.environ.get("SPANISH_DRILL_SYNC_TOKEN", "")
# Short on purpose. This sits in the path of a drill that answers a card every
# few seconds, and a slow database must degrade to the local cache quickly
# rather than hold up the next word.
SYNC_TIMEOUT = float(os.environ.get("SPANISH_DRILL_SYNC_TIMEOUT", "4"))

# Audio
SAMPLE_RATE = 16_000            # what Whisper expects: 16 kHz mono
BLOCK_SECONDS = 0.05            # capture granularity
MIN_SPEECH_SECONDS = 0.25       # shorter than this cannot be a word
PAUSE_BEFORE_CHECK = 0.25       # silence that suggests you finished a try
MIN_SECONDS_BETWEEN_CHECKS = 0.4
CALIBRATION_SECONDS = 0.7
MAX_BUFFERED_SECONDS = 30       # cap on captured audio held between cards
SILENT_FLOOR = 0.0026           # a floor at or below this means a dead input

# Models. Accuracy over speed on purpose: measured on 30 spoken cards, "small"
# graded 24/30 at 0.8s per word and "medium" 29/30 at 2.5s. A word misheard as
# a miss lowers that card's ease and poisons its schedule, which costs more
# than the extra seconds. The scout only answers "have they said it yet", where
# speed is the whole point.
MAIN_MODEL = "medium"
SCOUT_MODEL = "small"
VERIFY_MODEL = "gpt-4o-transcribe"

# Speech out. Prompts are recorded once and replayed: rendering is not racing a
# clock, so it can use a higher sample rate than live synthesis, and the same
# phrase sounds identical every time.
VOICES = {"es-MX": "Paulina", "es-ES": "Mónica"}

# How fast each side is spoken, in `say` words per minute. The English cue is
# the part you already understand, so it is read fast; the Spanish is the thing
# being learned and stays at a natural pace. The two are separate constants
# because they are separate decisions, and because the rate is part of a
# recording's identity: changing one must not invalidate the other's files.
TTS_BASE_RATE = 180             # the API voice's own natural speed
ENGLISH_RATE = 270              # 1.5x
SPANISH_RATE = 180
PROMPT_CACHE = ROOT / "prompts"
SAY_FORMAT = "LEI16@22050"      # 16-bit mono, what `say -o` writes natively

# Which engine records the prompts. macOS `say` funnels every request through
# one shared synthesis service, so concurrency buys about 1.7x and then flattens
# completely: twelve at once and forty-eight at once render at the same rate.
# The API has no such bottleneck and sounds better, at roughly a dollar for the
# whole deck. "say" remains the offline fallback.
TTS_ENGINE = "openai"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_VOICES = {"english": "alloy", "es-MX": "nova", "es-ES": "shimmer"}
OPENAI_STYLE = {
    "english": "Speak clearly and plainly in American English, unhurried.",
    "es-MX": "Speak clearly in neutral Latin American Spanish, unhurried.",
    "es-ES": "Speak clearly in Castilian Spanish from Spain, unhurried.",
}

# Phase two: conjugated forms, unlocked one tense at a time once the
# infinitive itself has survived real spaced review.
#
# Five persons, not seven. Spanish collapses he/she/usted onto one form and
# they/ustedes onto another, so drilling seven English pronouns would mean two
# pairs of cards with identical answers and no way to tell them apart.
PERSON_ORDER = ("yo", "tu", "nos", "el", "vos", "ellos")

# `vosotros` is everyday speech in Spain and does not exist in Latin America,
# where `ustedes` covers every plural "you". So it is taught only on the
# Spain dialect rather than being in or out of the deck: switching dialects
# adds or removes it without rebuilding anything.
SPAIN_ONLY = ("vos",)
SPAIN_DIALECT = "es-ES"

# Present, then the two pasts adjacent to each other because telling them
# apart is the whole difficulty, then will, then would. The near future
# ("voy a hablar") is left out: it is the present of `ir` plus an infinitive,
# a pattern rather than a set of words to memorise.
TENSE_ORDER = ("pres", "pret", "imp", "fut", "cond")

# How many correct answers a form needs before the next one opens up. One:
# as soon as a word is learned and joins the review rotation, the next
# variation should start coming. Waiting for it to mature first would put the
# second tense three weeks out and the fifth one most of a year away.
UNLOCK_REPS = 1


# -- the conjugation-only drill -------------------------------------------
# A second way to practise, running beside the normal drill rather than
# inside it: nothing but conjugated forms, for verbs the main drill has
# already taught, worked through a few verbs at a time.
#
# Its own file, and that is the whole point. The vocabulary schedule in
# progress.json is never opened for writing by this mode, so a heavy session
# of paradigms cannot move a single word's review date. The two trackers do
# not know about each other beyond one read: which infinitives are learned.
# One database, deliberately. These were separate files, and the split was
# invisible until you looked at two screens at once: the window showed the
# conjugation tracker's counters while the phone showed the vocabulary one,
# and neither was wrong. Nothing could be picked up where the other left off.
#
# They can share safely because the ids never collide — vocabulary is `tener`
# and a form is `tener:pres-yo` — so one file holds both without either
# reading the other's cards. The names are kept as aliases so every caller
# lands on the same file without knowing about the change.
CONJUGATION_PROGRESS_PATH = PROGRESS_PATH

# Answers land in their own log for the same reason. `--review` repairs cards
# in whichever tracker it was handed, and pointing it at the shared log would
# let a conjugation re-check write into the vocabulary schedule.
# Likewise one log. Two existed so that `--review` could not repair a
# conjugation into the vocabulary schedule, and with one schedule there is
# nothing left to protect: every record carries a card id, and an id points
# at exactly one card in one deck.
CONJUGATION_LOG = ANSWER_LOG

# How many verbs are being drilled at once. Ten paradigms is 300 cards on the
# Spain dialect, which is a lot of forms but only ever ten of them waiting to
# be met: the one-form-at-a-time chain means each verb has exactly one card
# unlocked and unlearned at any moment.
#
# The batch rolls rather than emptying. A verb whose every form is in the
# rotation drops out and the next most frequent one takes the slot, so the
# run never ends up grinding the last verb of a block on its own.
CONJUGATION_BATCH = 10

# The learning ladder, in cards rather than days: how far ahead a word is put
# each time you get it right, inside the one session. Clear every rung and it
# is learned for the day and joins the day-scale schedule below.
#
# It exists because one correct answer is not evidence. A missed word used to
# come back about five cards later, and getting it right that once sent it
# away for a day: you were repeating something you had been told moments
# earlier, which is recognition rather than recall. Each rung puts more
# between you and the last time you saw it.
#
# A miss steps down one rung instead of back to the bottom, so a single slip
# late in the ladder does not throw away the whole climb.
LEARNING_STEPS = (2, 8, 30, 47)

DAY_SECONDS = 86_400


# -- sentence composition -------------------------------------------------
SENTENCES_PATH = ROOT / "sentences.json"

# What a word has to have reached before a sentence may use it. One means
# answered right at least once and scheduled a day or more ahead, which is
# the same bar the panel calls LEARNING. Raise it to MATURE_AT (21) to
# compose only from words that have survived three weeks of review.
#
# Measured on the current save: at 1 the pool is 337 cards, at 21 it is 183
# and every conjugated form drops out, because none has matured yet.
SENTENCE_KNOWN_INTERVAL = 1

# Whether a sentence also needs the exact conjugated form it uses, not just
# the infinitive behind it.
#
# On. Knowing `dormir` is not knowing `duermo`, and a sentence that asks for
# a form never drilled is asking you to guess an ending. Where the deck has
# no conjugation cards for a verb — only 39 verbs have them — there is
# nothing to check and the infinitive stands in.
#
# This holds sentences back rather than throwing them out: the bank keeps
# them and they unlock as the forms are learned.
SENTENCE_REQUIRES_KNOWN_FORM = True

# Sentences written by the API, on top of the curated bank that ships with
# the repo. The bank is the floor: it needs no key, no network and no spend,
# and the generator only ever adds to it.
# Whether to ask the API for new sentences at all.
#
# Off. The gate can check vocabulary mechanically and cannot check meaning,
# and three separate defects reached a card on screen before this switch
# existed: Spanish in the English field, words with no card in the deck, and
# a cue glossing `deber` as "must" when the deck teaches it as "should".
# The first two are now caught by tests. The third cannot be, because no
# local check can tell whether an English sentence means what the Spanish
# says. The curated bank in sentences.json was written against the deck's own
# glosses and has none of these problems.
#
# Set this True to turn generation back on. Everything it produces still has
# to survive the same gate.
SENTENCE_GENERATION = False

SENTENCE_MODEL = "gpt-4o-mini"
GENERATED_SENTENCES_PATH = ROOT / "sentences-generated.json"

# A hard ceiling on how many are ever paid for. Once the store holds this
# many the generator stops calling out, for good rather than for the session:
# the file is the tally, so restarting the app cannot start the bill again.
MAX_GENERATED_SENTENCES = 500

# One first, so the drill starts against something fresh straight away, then
# batches in the background. A batch of one costs a whole round trip per
# sentence, which is why only the first is asked for alone.
FIRST_BATCH = 1
SENTENCE_BATCH = 20

# How many of the known words to put in front of the model per call. All of
# them every time is dearer and, worse, less varied: the same list produces
# the same handful of obvious sentences. A fresh sample each call is what
# keeps the bank from converging on twenty ways to say "I have a dog".
WORDS_PER_CALL = 55

# How long you get to say a whole sentence. The word drill's ANSWER WAIT is
# the wrong yardstick: it is sized for one word, and a ten-word sentence read
# off the screen and spoken takes several times that. Grows with the
# sentence, because that is what actually varies.
SENTENCE_SPEECH_BASE = 5.0
SENTENCE_SPEECH_PER_WORD = 0.9
SENTENCE_SPEECH_MAX = 22.0

# Where the phone's key lives. Written once and reused, so the URL on the
# phone keeps working: a token regenerated every launch means a new link
# every launch, which is the same as having no permanent address at all.
# Kept out of git for the obvious reason.
SERVE_TOKEN_PATH = ROOT / ".serve-token"
