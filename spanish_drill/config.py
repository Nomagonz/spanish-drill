"""Paths and tuning constants.

Everything here is a value someone might reasonably want to change. Anything
that encodes a rule rather than a preference lives with the code that uses it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DECK_PATH = ROOT / "deck.json"
PROGRESS_PATH = ROOT / "progress.json"
ANSWERS_DIR = ROOT / "answers"
ANSWER_LOG = ANSWERS_DIR / "answers.jsonl"

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
ENGLISH_RATE = 180
PROMPT_CACHE = ROOT / "prompts"
SAY_FORMAT = "LEI16@22050"      # 16-bit mono, what `say -o` writes natively

DAY_SECONDS = 86_400
