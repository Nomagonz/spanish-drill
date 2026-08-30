#!/usr/bin/env python3
"""
Spanish voice drill, local edition.

Hands free: it says an English word, listens, grades what you said, and moves on.
Speech in is Whisper running on this machine. Speech out is the macOS `say` voice.
Neither one is a browser, so none of the Web Speech API's rules apply.

    ./run.sh                  normal session
    ./run.sh --model small    ~3x faster per word, measurably less accurate

Model choice is accuracy over speed on purpose. Measured on 30 spoken cards,
"small" graded 24/30 at 0.8s per word and "medium" 29/30 at 2.5s. Since the
scheduler now adapts to right and wrong, a word misheard as a miss lowers that
card's ease and poisons its schedule, which costs more than the two seconds.
"""
import json, os, queue, re, subprocess, sys, threading, time, unicodedata
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFrame, QComboBox,
                             QSpinBox, QCheckBox, QSizePolicy)

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = json.load(open(os.path.join(HERE, "deck.json"), encoding="utf-8"))
STATE_PATH = os.path.join(HERE, "progress.json")
MISS_DIR = os.path.join(HERE, "misses")
MISS_LOG = os.path.join(MISS_DIR, "misses.jsonl")

SR = 16000                      # Whisper wants 16 kHz mono
DAY = 86400
# SM-2. Every card carries its own ease factor, so difficulty is per word
# rather than one ladder for everything.
EASE_START = 2.5
EASE_MIN = 1.3          # SM-2's floor; below this intervals stop growing
FIRST_IVL = 1           # days, after the first successful recall
SECOND_IVL = 6          # days, after the second
LEECH_AT = 8            # lapses before a word is called out as a problem
MATURE_AT = 21          # days; Anki's threshold for "this one has stuck"

VOICES = {"es-MX": "Paulina", "es-ES": "Mónica"}


# ---------------------------------------------------------------- persistence
@dataclass
class State:
    cards: dict = field(default_factory=dict)   # index -> {b, d, r, l}
    dialect: str = "es-MX"
    input_device: str = ""          # by NAME; indices shift as devices connect
    new_per: int = 20
    window: float = 6.0
    hints: bool = True
    day: int = 0
    new_done: int = 0
    miss_today: int = 0

    @staticmethod
    def load():
        s = State()
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            s.cards = {int(k): migrate(v) for k, v in s.cards.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if s.day != today():                    # a new day resets the daily caps
            s.day, s.new_done, s.miss_today = today(), 0, 0
        return s

    def save(self):
        tmp = STATE_PATH + ".tmp"               # write-then-rename so a crash
        with open(tmp, "w", encoding="utf-8") as f:   # can't shred progress
            json.dump(self.__dict__, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_PATH)


def today():
    return int(time.time() // DAY)


def new_card():
    return {"ef": EASE_START, "ivl": 0, "reps": 0, "lapses": 0, "due": today()}


def migrate(c):
    """Old Leitner cards carried {b,d,r,l}. Keep their progress, give them an ease."""
    if "ef" in c:
        return c
    box = c.get("b", 0)
    old_ladder = [0, 1, 3, 7, 16, 35, 90, 180]
    return {"ef": EASE_START,
            "ivl": old_ladder[min(box, len(old_ladder) - 1)],
            "reps": box,
            "lapses": c.get("l", 0),
            "due": c.get("d", today())}


def quality(ok, close, silent, elapsed, window):
    """
    Map what we can actually observe onto SM-2's 0-5 scale.

    Hesitation is real evidence: recalling a word instantly is not the same as
    dragging it up after four seconds, and SM-2 wants that distinction.
    """
    if silent:
        return 0                        # no answer at all
    if not ok:
        return 1                        # said something, it was wrong
    if close:
        return 3                        # right word, mangled
    return 5 if elapsed <= window * 0.45 else 4


def ease_delta(q):
    """SM-2's ease adjustment for a grade. Pulled out so it can be undone."""
    return 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)


def schedule(c, q):
    """Textbook SM-2, with a failure sending the card back to relearning."""
    if q >= 3:
        if c["reps"] == 0:
            c["ivl"] = FIRST_IVL
        elif c["reps"] == 1:
            c["ivl"] = SECOND_IVL
        else:
            c["ivl"] = max(1, round(c["ivl"] * c["ef"]))
        c["reps"] += 1
    else:
        c["reps"] = 0
        c["ivl"] = 0                    # due again today, not in a week
        c["lapses"] += 1
    # The ease itself moves, so a word that keeps biting you grows slower
    # forever after, and an easy one accelerates.
    c["ef"] = max(EASE_MIN, c["ef"] + ease_delta(q))
    c["due"] = today() + c["ivl"]
    return c


def is_leech(c):
    return c["lapses"] >= LEECH_AT


# ------------------------------------------------------- recording the misses
def save_wav(path, audio):
    """float32 mono -> 16-bit PCM, so anything can open it."""
    import wave
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def record_answer(cid, card, heard, q, silent, elapsed, before, audio, live_ok):
    """
    Keep everything needed to re-judge this later: the audio that produced the
    verdict, and the card's exact state BEFORE grading, so a bad call can be
    reversed rather than approximated.

    Every answer is kept, not just the misses. Grading you correct on a word you
    fluffed is just as wrong as the reverse, and only shows up by re-checking
    the ones it accepted too.
    """
    os.makedirs(MISS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{cid:03d}"
    wav = None
    if audio is not None and len(audio) > SR * 0.1:
        wav = os.path.join(MISS_DIR, stamp + ".wav")
        try:
            save_wav(wav, audio)
        except Exception:
            wav = None
    rec = {"id": stamp, "cid": cid, "en": card["en"], "expected": card["es"],
           "heard": heard or "", "quality": q, "live_ok": bool(live_ok),
           "silent": bool(silent),
           "elapsed": round(elapsed, 2), "audio": os.path.basename(wav) if wav else None,
           "before": before, "verdict": None, "at": time.time()}
    with open(MISS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_misses():
    if not os.path.exists(MISS_LOG):
        return []
    out = []
    for line in open(MISS_LOG, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def repair_false_miss(S, rec, new_q):
    """
    Undo a miss the recogniser invented, without trampling anything that
    happened afterwards.

    The ease penalty is reversed by arithmetic rather than by restoring the old
    card, because the word is requeued after a miss and may well have been
    answered correctly later in the same session. That later rep is real and
    must survive.
    """
    cid = rec["cid"]
    c = migrate(S.cards.get(cid, new_card()))
    old_q = rec["quality"]

    # swap the penalty for what the grade should have been
    c["ef"] = max(EASE_MIN, c["ef"] - ease_delta(old_q) + ease_delta(new_q))
    c["lapses"] = max(0, c["lapses"] - 1)

    # Only touch the interval if the card is still sitting in the state the bad
    # miss put it in. If reps > 0 it has since been answered correctly and that
    # scheduling is already right.
    if c["reps"] == 0 and c["ivl"] == 0:
        restored = dict(rec["before"])
        restored["ef"] = c["ef"]
        restored["lapses"] = c["lapses"]
        schedule(restored, new_q)
        c = restored
        touched_interval = True
    else:
        touched_interval = False

    S.cards[cid] = c
    S.miss_today = max(0, S.miss_today - 1)
    return c, touched_interval


# A fixed, generic steer. It tells the recogniser to expect isolated Spanish
# dictionary words, possibly repeated, which is what collapses "y el bar" back
# to "llevar". It must NEVER contain the expected answer: prompting with the
# word you are being tested on makes the recogniser agree with you and the
# grade becomes meaningless.
# Every example word here is verified absent from the deck (see the assertion
# below), so the steer can never hand over the answer being tested.
STEER = ("Palabras sueltas en español, a veces repetidas varias veces. "
         "Por ejemplo: pintar, nadar, saltar, bailar, fresa, morado, tijeras.")
_STEER_WORDS = {"pintar", "nadar", "saltar", "bailar", "fresa", "morado", "tijeras"}


def is_steer_echo(text):
    """
    These models echo the prompt back when the audio has no usable speech.
    That is not a transcript and must never be graded as one.

    The echo is not always the example words: it is often just the steer's
    opening sentence, so the whole prompt has to be matched, not the examples.
    """
    n = norm(text)
    if not n:
        return False
    words = n.split()
    if len(set(words) & _STEER_WORDS) >= 3:
        return True
    steer_words = set(norm(STEER).split())
    # A run of words that all come from the prompt is the prompt talking.
    if len(words) >= 3 and all(w in steer_words for w in words):
        return True
    # Or a verbatim chunk of it.
    return len(n) >= 12 and n in norm(STEER)


def _assert_steer_is_clean():
    """A steer word that is also a deck answer would give that card away."""
    clash = sorted({a for c in DECK for a in c["es"] if norm(a) in _STEER_WORDS})
    if clash:
        raise AssertionError(f"steer leaks deck answers: {clash}")


def transcribe_openai(path, model="gpt-4o-transcribe"):
    """One clip through OpenAI. Returns text, or None if the call fails."""
    from openai import OpenAI
    try:
        client = OpenAI()               # reads OPENAI_API_KEY from the environment
        with open(path, "rb") as f:
            r = client.audio.transcriptions.create(
                model=model, file=f, language="es",
                prompt=STEER, response_format="json")
        return (r.text or "").strip()
    except Exception as e:
        print(f"    [api error: {type(e).__name__}: {str(e)[:70]}]")
        return None


def review(verify_model="gpt-4o-transcribe", play=False):
    """Re-judge every recorded miss with a stronger model and repair the schedule."""
    recs = load_misses()
    pending = [r for r in recs if r.get("verdict") is None]
    if not pending:
        print(f"{len(recs)} miss(es) on file, none awaiting review.")
        return
    _assert_steer_is_clean()
    print(f"{len(pending)} answer(s) to re-check with '{verify_model}'.")
    print("Local models download on first use; gpt-* models call the API.\n")

    import wave
    use_api = verify_model.startswith("gpt-")
    m = None
    if not use_api:
        from faster_whisper import WhisperModel
        m = WhisperModel(verify_model, device="cpu", compute_type="int8")
    S = State.load()
    false_misses, confirmed, unverifiable, accepted = [], [], [], []

    for r in pending:
        card = DECK[r["cid"]]
        if not r.get("audio"):
            r["verdict"] = "no-audio"
            unverifiable.append(r)
            continue
        path = os.path.join(MISS_DIR, r["audio"])
        if not os.path.exists(path):
            r["verdict"] = "no-audio"
            unverifiable.append(r)
            continue
        if use_api:
            second = transcribe_openai(path, verify_model)
            if second is None:
                r["verdict"] = None      # leave it pending, not wrongly judged
                unverifiable.append(r)
                continue
        else:
            with wave.open(path) as w:
                a = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                     .astype(np.float32) / 32768.0)
            # Deliberately harder decoding than the live pass: this runs once,
            # offline, with no one waiting on it.
            segs, _ = m.transcribe(a, language="es", beam_size=10, best_of=5,
                                   temperature=[0.0, 0.2, 0.4], vad_filter=True,
                                   initial_prompt=STEER)
            second = " ".join(x.text for x in segs).strip()
        # Guard against the steer feeding itself back as a "transcript".
        if is_steer_echo(second) or norm(second) in _STEER_WORDS:
            r["verdict"] = "no-signal"
            r["second_opinion"] = second
            unverifiable.append(r)
            continue
        r["second_opinion"] = second
        v = check(second, card)
        was_miss = r["quality"] < 3
        if v and was_miss:
            # graded wrong live, right on review: the recogniser's fault
            new_q = quality(True, bool(v["close"]), False, r["elapsed"], S.window)
            r["verdict"] = "false-miss"
            r["corrected_quality"] = new_q
            _, touched = repair_false_miss(S, r, new_q)
            r["interval_restored"] = touched
            false_misses.append(r)
        elif was_miss:
            r["verdict"] = "confirmed"
            confirmed.append(r)
        else:
            # graded correct live; nothing to repair
            r["verdict"] = "accepted"
            accepted.append(r)
        if play and os.path.exists(path):
            subprocess.run(["afplay", path], check=False)

    S.save()
    rewrite_misses(recs)

    print(f"{'-'*72}")
    if accepted:
        print(f"MARKED CORRECT live ({len(accepted)}):")
        for r in accepted:
            print(f"  {r['en'][:30]:<32} expected {r['expected'][0]!r:<18} "
                  f"heard {r['heard']!r}")
        print()
    if false_misses:
        print(f"OVERTURNED — you were right, the recogniser was wrong ({len(false_misses)}):")
        for r in false_misses:
            note = "" if r.get("interval_restored") else "  [ease only; already re-answered]"
            print(f"  {r['en'][:30]:<32} expected {r['expected'][0]!r}")
            print(f"     live heard {r['heard']!r:<24} -> recheck {r['second_opinion']!r}{note}")
    if confirmed:
        print(f"\nCONFIRMED misses ({len(confirmed)}):")
        for r in confirmed:
            print(f"  {r['en'][:30]:<32} expected {r['expected'][0]!r:<18} "
                  f"heard {r['second_opinion']!r}")
    if unverifiable:
        ns = [r for r in unverifiable if r.get("verdict") == "no-signal"]
        na = [r for r in unverifiable if r.get("verdict") == "no-audio"]
        if ns:
            print(f"\nNO USABLE SPEECH in the recording ({len(ns)}) — "
                  f"the recogniser returned nothing, so these were graded on "
                  f"nothing and are left alone:")
            for r in ns:
                print(f"  {r['en'][:30]:<32} expected {r['expected'][0]!r}")
        if na:
            print(f"\nNo audio kept, left alone ({len(na)}).")
    print(f"{'-'*72}")
    graded_wrong = len(false_misses)
    print(f"Local model got {len(pending) - graded_wrong} of {len(pending)} right.")
    if graded_wrong:
        print(f"{graded_wrong} were the recogniser's fault, and those cards have "
              f"been repaired.")


def rewrite_misses(recs):
    os.makedirs(MISS_DIR, exist_ok=True)
    tmp = MISS_LOG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, MISS_LOG)


# -------------------------------------------------------------------- grading
def norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_lead(s):
    return re.sub(r"^(el|la|los|las|un|una|unos|unas)\s+", "", s)


_REAL = None
def real_words():
    global _REAL
    if _REAL is None:
        _REAL = {norm(a) for c in DECK for a in c["es"]}
    return _REAL


def lev(a, b):
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tolerance(w):
    return 0 if len(w) <= 5 else (1 if len(w) <= 8 else 2)


def check(said, card):
    """Same acceptance rules as the web version, including the near-miss pass."""
    raw = norm(said)
    if not raw:
        return None
    tries = [raw] if raw == strip_lead(raw) else [raw, strip_lead(raw)]
    for heard in tries:
        for ans in card["es"]:
            a = norm(ans)
            if heard == a or heard == re.sub(r" (de|a|que)$", "", a):
                return {"matched": ans, "close": False}
            if re.search(r"(^| )" + re.escape(a) + r"( |$)", heard):
                return {"matched": ans, "close": False}
    # only then allow a typo-level miss, and never one that is itself a real word
    for heard in tries:
        if heard in real_words():
            continue
        for ans in card["es"]:
            a = norm(ans)
            if lev(heard, a) <= tolerance(a):
                return {"matched": ans, "close": True}
    return None


CMD = {
    "repeat": ["repite", "repetir", "otra vez", "repeat", "again"],
    "skip":   ["salta", "saltar", "skip", "pasa", "siguiente", "next"],
    "stop":   ["para", "parar", "alto", "stop", "pausa", "pause"],
    "reveal": ["no se", "no lo se", "dime", "i dont know", "tell me", "pass"],
}


def command_in(said):
    n = norm(said)
    for k, phrases in CMD.items():
        if any(n == norm(p) for p in phrases):
            return k
    return None


# ---------------------------------------------------------------- input device
def input_devices():
    """[(index, name)] for everything that can record."""
    try:
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


def resolve_device(name):
    """
    Name -> index, or None for the system default.

    Matching by name matters: unplugging a device renumbers the rest, so a
    stored index silently starts pointing at the wrong microphone.
    """
    if not name:
        return None
    for i, n in input_devices():
        if n == name:
            return i
    return None


# ------------------------------------------------------------------- speech out
def say(text, voice=None, rate=None):
    """macOS `say`. Blocks until the phrase finishes, which is what we want."""
    if not text:
        return
    cmd = ["say"]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(rate)]
    try:
        subprocess.run(cmd + [text], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


# -------------------------------------------------------------------- listening
class Listener:
    """
    Records until you stop talking, then hands the audio to Whisper.

    Endpointing is plain RMS against an ambient floor measured at startup, which
    is enough for single words and keeps the whole thing dependency free.
    """

    def __init__(self, model_name="medium", device_name=""):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.device = resolve_device(device_name)
        self.floor = 0.004
        self.level = 0.0
        self.last_audio = None

    def _transcribe(self, audio):
        # The steer tells it to expect isolated Spanish words, which is what
        # stops "llevar" being decoded as the far more common "y el bar".
        segs, _ = self.model.transcribe(audio, language="es", beam_size=5,
                                        temperature=0, vad_filter=False,
                                        initial_prompt=STEER)
        txt = " ".join(s.text for s in segs).strip()
        # On unusable audio these models echo the prompt back. That is not a
        # transcript, and grading it as one turns silence into a wrong answer.
        if not txt or is_steer_echo(txt):
            return None
        return txt

    def _keep(self, blocks):
        self.last_audio = (np.concatenate(blocks).flatten()
                           if blocks else None)

    def calibrate(self, seconds=0.7):
        buf = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                     dtype="float32", device=self.device)
        sd.wait()
        rms = float(np.sqrt(np.mean(buf ** 2)))
        self.floor = max(0.0025, rms * 2.5)

    def listen(self, max_seconds, should_stop=None, accept=None):
        """
        Returns the transcript, or None if you never said anything.

        The FULL window is always kept on self.last_audio, including when this
        decides you were silent. A miss blamed on silence may really be speech
        that fell under the noise floor, and that is only provable if the audio
        that produced the verdict was kept.
        """
        q = queue.Queue()
        sd.default.samplerate, sd.default.channels = SR, 1
        self.last_audio = None

        def cb(indata, frames, t, status):
            q.put(indata.copy())

        everything = []
        heard_speech, first_idx, started = False, None, time.time()
        hop = 0.05
        pause_run, last_try = 0.0, 0.0
        with sd.InputStream(callback=cb, blocksize=int(SR * hop),
                            dtype="float32", device=self.device,
                            samplerate=SR, channels=1):
            # Listen for the WHOLE window. Ending the turn shortly after the
            # first pause used to clip the second attempt whenever a word was
            # repeated, which is exactly when someone is struggling and repeats
            # themselves several times.
            while time.time() - started < max_seconds:
                if should_stop and should_stop():
                    self._keep(everything)
                    return None
                try:
                    block = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                everything.append(block)
                rms = float(np.sqrt(np.mean(block ** 2)))
                self.level = rms
                if rms > self.floor:
                    if not heard_speech:
                        heard_speech, first_idx = True, max(0, len(everything) - 4)
                    pause_run = 0.0
                elif heard_speech:
                    pause_run += hop

                # As soon as a pause suggests you finished a try, check whether
                # what you said already contains the answer. If it does there is
                # nothing to wait for. If it does not, keep listening, so extra
                # repeats still land inside the same window.
                if (accept is not None and heard_speech and pause_run >= 0.35
                        and time.time() - last_try > 0.7):
                    last_try = time.time()
                    partial = np.concatenate(everything[first_idx:]).flatten()
                    if len(partial) >= SR * 0.25:
                        txt = self._transcribe(partial)
                        if txt and accept(txt):
                            self._keep(everything)
                            return txt

        self._keep(everything)
        if not heard_speech or not everything:
            return None                     # genuinely nothing said
        audio = np.concatenate(everything[first_idx:]).flatten()
        if len(audio) < SR * 0.25:
            return None
        return self._transcribe(audio)


# ------------------------------------------------------------------ drill loop
class Drill(QObject):
    prompt = pyqtSignal(str, str)      # english, card state label
    status = pyqtSignal(str)
    heard = pyqtSignal(str)
    result = pyqtSignal(dict)
    counts = pyqtSignal(int, int, int, int, int)
    finished = pyqtSignal()

    def __init__(self, state, listener):
        super().__init__()
        self.S, self.listener = state, listener
        self.queue, self.running = [], False

    def stop(self):
        self.running = False

    # -- scheduling, ported from the web version ---------------------------
    def due(self):
        t = today()
        return [i for i, c in self.S.cards.items() if c["due"] <= t]

    def fresh(self):
        return [i for i in range(len(DECK)) if i not in self.S.cards]

    def new_left(self):
        return max(0, self.S.new_per - self.S.new_done)

    def learning(self):
        """Answered right and scheduled for a future day. Moves immediately."""
        return sum(1 for c in self.S.cards.values() if c["ivl"] >= 1)

    def mature(self):
        """Anki's sense of stuck: the interval has stretched past three weeks."""
        return sum(1 for c in self.S.cards.values() if c["ivl"] >= MATURE_AT)

    def build_queue(self):
        import random
        due = self.due()
        random.shuffle(due)
        new = self.fresh()[: self.new_left()]
        q = due[:]
        if new:                                  # spread new words through reviews
            step = max(1, (len(q) + len(new)) // len(new))
            pos = 0
            for cid in new:
                q.insert(min(pos, len(q)), cid)
                pos += step + 1
        self.queue = q

    def grade(self, cid, q):
        import random
        is_new = cid not in self.S.cards
        c = migrate(self.S.cards.get(cid, new_card()))
        schedule(c, q)
        if q < 3:
            self.S.miss_today += 1
            # Come back inside this session too, sooner for a word that has a
            # history of biting, since that is what the lapse count is for.
            gap = 2 if is_leech(c) else 4 + random.randint(0, 2)
            self.queue.insert(min(len(self.queue), gap), cid)
        self.S.cards[cid] = c
        if is_new:
            self.S.new_done += 1
        self.S.save()
        return c

    def emit_counts(self):
        self.counts.emit(self.new_left(), len(self.queue),
                         self.S.miss_today, self.learning(), self.mature())

    # -- the loop ----------------------------------------------------------
    def run(self):
        """
        Wrapper so the thread ALWAYS finishes. Several paths below return early,
        and an unhandled exception in a slot makes PyQt abort the whole process
        rather than raise, so neither is allowed to escape.
        """
        try:
            self._run()
        except Exception:
            import traceback
            traceback.print_exc()
            self.status.emit("Crashed — see terminal")
        finally:
            self.running = False
            self.finished.emit()

    def _run(self):
        self.running = True
        voice = VOICES.get(self.S.dialect, "Paulina")
        while self.running:
            if not self.queue:
                self.build_queue()
            if not self.queue:
                self.status.emit("Queue clear.")
                break

            cid = self.queue.pop(0)
            card = DECK[cid]
            c = self.S.cards.get(cid)
            if not c:
                label = "New word"
            elif is_leech(c):
                label = f"Leech · missed {c['lapses']}x"
            elif c["reps"] == 0:
                label = "Relearning"
            else:
                label = f"Review · {c['ivl']}d · ease {c['ef']:.2f}"
            self.prompt.emit(card["en"], label)
            self.heard.emit("")
            self.emit_counts()

            self.status.emit("Speaking")
            say(re.sub(r"\s*\(.*?\)\s*", " ", card["en"]).strip(), rate=180)
            if not self.running:
                break

            said, verdict, silent, elapsed = None, None, False, 0.0
            while self.running:
                self.status.emit("Listening")
                t_ask = time.time()
                said = self.listener.listen(
                    self.S.window,
                    should_stop=lambda: not self.running,
                    accept=lambda t: check(t, card) is not None)
                elapsed = time.time() - t_ask
                if not self.running:
                    return
                if said is None:
                    silent = True
                    break
                self.heard.emit(said)
                cmd = command_in(said)
                if cmd == "stop":
                    self.running = False
                    self.status.emit("Paused")
                    return
                if cmd == "skip":
                    self.queue.append(cid)
                    said = "__skip__"
                    break
                if cmd == "reveal":
                    silent = True
                    break
                if cmd == "repeat":
                    self.status.emit("Speaking")
                    say(re.sub(r"\s*\(.*?\)\s*", " ", card["en"]).strip(), rate=180)
                    continue
                verdict = check(said, card)
                break

            if said == "__skip__":
                continue
            if not self.running:
                break

            ok = verdict is not None
            close = bool(verdict and verdict["close"])
            q = quality(ok, close, silent, elapsed, self.S.window)
            before = dict(migrate(self.S.cards.get(cid, new_card())))
            c = self.grade(cid, q)
            try:
                record_answer(cid, card, said if not silent else "", q, silent,
                              elapsed, before,
                              getattr(self.listener, "last_audio", None), ok)
            except Exception:
                pass            # never let bookkeeping kill a drill session
            self.result.emit({
                "ok": ok, "said": "" if silent else (said or ""), "card": card,
                "box": c, "close": close, "quality": q,
                "silent": silent, "next": next_interval(c),
                "leech": is_leech(c), "ease": c["ef"], "lapses": c["lapses"],
            })
            self.emit_counts()

            if ok:
                self.status.emit("Correct")
                if self.S.hints:
                    say(card["es"][0], voice)
                time.sleep(0.25)
            else:
                self.status.emit("Missed")
                say(card["es"][0] + ". " + card["ex"], voice)
                time.sleep(0.3)


def next_interval(c):
    d = c["ivl"]
    if d == 0:
        return "again this session"
    if d == 1:
        return "again tomorrow"
    if d < 30:
        return f"again in {d} days"
    m = round(d / 30)
    return f"again in {m} month" + ("" if m == 1 else "s")


# -------------------------------------------------------------------------- UI
CH = "#1E2B26"; CH2 = "#2B3B34"; RULE = "#3C4F47"; SCREEN = "#ECE5D3"
INK = "#141A17"; SOFT = "#5A665F"; SIG = "#E19B33"; MISS = "#C24B36"
HIT = "#6E9A6B"; MUTE = "#8FA096"


class Window(QWidget):
    def __init__(self, model_name):
        super().__init__()
        self.S = State.load()
        self.listener = None
        self.model_name = model_name
        self.thread = None
        self._starting = False
        self.setWindowTitle("Spanish Drill")
        self.resize(560, 720)
        self.setStyleSheet(f"background:{CH};color:{SCREEN};")
        self.build()

    # -- widgets -----------------------------------------------------------
    def build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        mark = QLabel("DRILL · ES")
        mark.setStyleSheet(f"color:{SCREEN};font:800 15px 'Helvetica Neue';"
                           "letter-spacing:2px;")
        root.addWidget(mark)

        # counters
        row = QHBoxLayout(); row.setSpacing(0)
        self.cnt = {}
        self.sublabels = {}
        for key, label, col in (("new", "NEW LEFT", SCREEN), ("due", "IN QUEUE", SIG),
                                ("miss", "MISSED TODAY", MISS), ("known", "LEARNING", HIT)):
            box = QVBoxLayout()
            n = QLabel("0"); n.setStyleSheet(f"color:{col};font:700 22px 'Helvetica Neue';")
            t = QLabel(label); t.setStyleSheet(f"color:{MUTE};font:9px 'Menlo';letter-spacing:1px;")
            self.sublabels[key] = t
            box.addWidget(n); box.addWidget(t)
            w = QWidget(); w.setLayout(box)
            w.setStyleSheet(f"border-right:1px solid {RULE};")
            row.addWidget(w)
            self.cnt[key] = n
        wrap = QWidget(); wrap.setLayout(row)
        wrap.setStyleSheet(f"border-top:1px solid {RULE};border-bottom:1px solid {RULE};")
        root.addWidget(wrap)

        # the screen
        self.screen = QFrame()
        self.screen.setStyleSheet(f"background:{SCREEN};border-radius:3px;")
        sl = QVBoxLayout(self.screen); sl.setContentsMargins(20, 18, 20, 16)
        tag = QHBoxLayout()
        self.state_lbl = QLabel(""); self.state_lbl.setStyleSheet(
            f"color:{SOFT};font:9px 'Menlo';letter-spacing:2px;")
        self.status_lbl = QLabel("Ready"); self.status_lbl.setStyleSheet(
            f"color:{SOFT};font:9px 'Menlo';letter-spacing:2px;")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        tag.addWidget(self.state_lbl); tag.addWidget(self.status_lbl)
        sl.addLayout(tag)

        self.prompt_lbl = QLabel("Press Go.")
        self.prompt_lbl.setWordWrap(True)
        self.prompt_lbl.setStyleSheet(f"color:{INK};font:700 34px 'Helvetica Neue';")
        self.prompt_lbl.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Expanding)
        sl.addWidget(self.prompt_lbl)

        self.heard_lbl = QLabel("")
        self.heard_lbl.setStyleSheet(f"color:{SOFT};font:13px 'Menlo';")
        self.heard_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        sl.addWidget(self.heard_lbl)
        self.screen.setMinimumHeight(210)
        root.addWidget(self.screen)

        # result slip
        self.slip = QFrame(); self.slip.setVisible(False)
        self.slip.setStyleSheet(f"background:{CH2};border-radius:2px;")
        pl = QVBoxLayout(self.slip); pl.setContentsMargins(14, 12, 14, 12)
        self.verdict_lbl = QLabel(""); self.verdict_lbl.setStyleSheet(
            f"color:{MUTE};font:10px 'Menlo';letter-spacing:2px;")
        self.answer_lbl = QLabel(""); self.answer_lbl.setStyleSheet(
            f"color:{SCREEN};font:700 24px 'Helvetica Neue';")
        self.ex_lbl = QLabel(""); self.ex_lbl.setWordWrap(True)
        self.ex_lbl.setStyleSheet(f"color:{SCREEN};font:13px 'Helvetica Neue';")
        self.gl_lbl = QLabel(""); self.gl_lbl.setWordWrap(True)
        self.gl_lbl.setStyleSheet(f"color:{MUTE};font:12px 'Helvetica Neue';")
        self.sched_lbl = QLabel(""); self.sched_lbl.setStyleSheet(
            f"color:{MUTE};font:10px 'Menlo';letter-spacing:1px;")
        for w in (self.verdict_lbl, self.answer_lbl, self.ex_lbl,
                  self.gl_lbl, self.sched_lbl):
            pl.addWidget(w)
        root.addWidget(self.slip)

        root.addStretch(1)

        # settings
        # Microphone picker on its own row: the names are long, and which mic is
        # used matters more to accuracy than anything else here.
        mrow = QHBoxLayout()
        mlab = QLabel("mic"); mlab.setStyleSheet(f"color:{MUTE};font:10px 'Menlo';")
        self.mic = QComboBox()
        self.mic.addItem("System default", "")
        for _, name in input_devices():
            self.mic.addItem(name, name)
        want = self.S.input_device
        if want and self.mic.findData(want) >= 0:
            self.mic.setCurrentIndex(self.mic.findData(want))
        self.mic.setStyleSheet(
            f"background:{CH2};color:{SCREEN};border:1px solid {RULE};padding:3px;")
        mrow.addWidget(mlab); mrow.addWidget(self.mic, 1)
        root.addLayout(mrow)

        srow = QHBoxLayout()
        self.dialect = QComboBox(); self.dialect.addItems(["es-MX", "es-ES"])
        self.dialect.setCurrentText(self.S.dialect)
        self.newper = QSpinBox(); self.newper.setRange(0, 100)
        self.newper.setValue(self.S.new_per)
        self.win = QSpinBox(); self.win.setRange(3, 20)
        self.win.setValue(int(self.S.window))
        self.hints = QCheckBox("say it back"); self.hints.setChecked(self.S.hints)
        for lbl, w in (("accent", self.dialect), ("new/day", self.newper),
                       ("wait s", self.win)):
            t = QLabel(lbl); t.setStyleSheet(f"color:{MUTE};font:10px 'Menlo';")
            srow.addWidget(t); srow.addWidget(w)
        self.hints.setStyleSheet(f"color:{MUTE};font:10px 'Menlo';")
        srow.addWidget(self.hints)
        for w in (self.dialect, self.newper, self.win):
            w.setStyleSheet(f"background:{CH2};color:{SCREEN};border:1px solid {RULE};padding:3px;")
        root.addLayout(srow)

        self.go = QPushButton("GO")
        self.go.setStyleSheet(
            f"background:{SIG};color:#20160A;font:600 14px 'Menlo';letter-spacing:2px;"
            f"padding:18px;border:none;border-radius:3px;")
        self.go.clicked.connect(self.toggle)
        root.addWidget(self.go)

        self.hint_lbl = QLabel("Say “skip”, “repeat”, “no sé”, or “stop” at any time. "
                               "Headphones stop the mic hearing the answer.")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setStyleSheet(f"color:{MUTE};font:10px 'Menlo';")
        root.addWidget(self.hint_lbl)

        self.refresh_counts()

    # -- wiring ------------------------------------------------------------
    def refresh_counts(self):
        cards = self.S.cards
        t = today()
        due = sum(1 for c in cards.values() if c["due"] <= t)
        self.cnt["new"].setText(str(max(0, self.S.new_per - self.S.new_done)))
        self.cnt["due"].setText(str(due))
        self.cnt["miss"].setText(str(self.S.miss_today))
        self.cnt["known"].setText(
            str(sum(1 for c in cards.values() if c["ivl"] >= 1)))
        self.sublabels["known"].setText(
            f"LEARNING · {sum(1 for c in cards.values() if c['ivl'] >= MATURE_AT)} MATURE")

    def toggle(self):
        # processEvents() below pumps the event queue while the model loads, so
        # a second click re-enters here. Without this guard the re-entrant call
        # reassigns self.thread and destroys a running QThread, which is a
        # qFatal abort, not an exception.
        if self._starting:
            return
        if self.thread and self.thread.isRunning():
            self.drill.stop()
            self.go.setText("GO")
            return
        chosen = self.mic.currentData() or ""
        if chosen != self.S.input_device and self.listener is not None:
            self.listener.device = resolve_device(chosen)   # switch without reloading
        self.S.input_device = chosen
        self.S.dialect = self.dialect.currentText()
        self.S.new_per = self.newper.value()
        self.S.window = float(self.win.value())
        self.S.hints = self.hints.isChecked()
        self.S.save()

        self._starting = True
        self.go.setEnabled(False)
        try:
            if self.listener is None:
                # First run also downloads the model, which is slow and silent.
                self.go.setText("LOADING…")
                self.status_lbl.setText(f"Loading Whisper ({self.model_name})")
                self.prompt_lbl.setText("Loading the speech model.\n"
                                        "First run downloads it, which takes a minute.")
                QApplication.processEvents()
                self.listener = Listener(self.model_name, self.S.input_device)
            self.go.setText("CALIBRATING…")
            self.status_lbl.setText("Measuring room noise — stay quiet")
            QApplication.processEvents()
            self.listener.calibrate()      # every session, not just the first
            if self.listener.floor <= 0.0026:
                self.status_lbl.setText("Warning: mic reads near silence")
        except Exception as e:
            self.prompt_lbl.setText("Mic or model failed")
            self.status_lbl.setText(str(e)[:60])
            return
        finally:
            self._starting = False
            self.go.setEnabled(True)

        self.go.setText("STOP")
        self.slip.setVisible(False)
        self.drill = Drill(self.S, self.listener)
        self.thread = QThread()
        self.drill.moveToThread(self.thread)
        self.thread.started.connect(self.drill.run)
        self.drill.prompt.connect(self.on_prompt)
        self.drill.status.connect(self.status_lbl.setText)
        self.drill.heard.connect(lambda s: self.heard_lbl.setText(s))
        self.drill.result.connect(self.on_result)
        self.drill.counts.connect(self.on_counts)
        self.drill.finished.connect(self.on_finished)
        self.thread.start()

    def on_prompt(self, en, label):
        self.prompt_lbl.setText(en)
        self.state_lbl.setText(label.upper())
        self.slip.setVisible(False)

    def on_counts(self, new, due, miss, learning, mature):
        self.cnt["new"].setText(str(new)); self.cnt["due"].setText(str(due))
        self.cnt["miss"].setText(str(miss)); self.cnt["known"].setText(str(learning))
        self.sublabels["known"].setText(f"LEARNING · {mature} MATURE")

    def on_result(self, r):
        col = HIT if r["ok"] else MISS
        v = ("Correct" if r["ok"] else
             ("No answer — counted as a miss" if r["silent"] else "Missed"))
        if r["ok"] and r["close"]:
            v = "Correct — check the spelling"
        self.slip.setStyleSheet(
            f"background:{CH2};border-radius:2px;border-left:3px solid {col};")
        self.verdict_lbl.setStyleSheet(f"color:{col};font:10px 'Menlo';letter-spacing:2px;")
        self.verdict_lbl.setText(v.upper())
        alt = r["card"]["es"]
        self.answer_lbl.setText(alt[0] + ("   (also: " + ", ".join(alt[1:]) + ")" if len(alt) > 1 else ""))
        self.ex_lbl.setText(r["card"]["ex"])
        self.gl_lbl.setText(r["card"]["gl"])
        said = f"  ·  you said: {r['said']}" if r["said"] else ""
        detail = f"  ·  ease {r['ease']:.2f}"
        if r["lapses"]:
            detail += f"  ·  missed {r['lapses']}x"
        if r["leech"]:
            detail += "  ·  LEECH"
        self.sched_lbl.setText(r["next"] + said + detail)
        self.slip.setVisible(True)

    def on_finished(self):
        self.go.setText("GO")
        if self.thread:
            self.thread.quit(); self.thread.wait()
        self.refresh_counts()

    def closeEvent(self, e):
        if self.thread and self.thread.isRunning():
            self.drill.stop(); self.thread.quit(); self.thread.wait(2000)
        self.S.save()
        e.accept()


def main():
    model = "medium"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--review" in sys.argv:
        vm = "gpt-4o-transcribe"
        if "--verify-model" in sys.argv:
            vm = sys.argv[sys.argv.index("--verify-model") + 1]
        review(vm, play="--play" in sys.argv)
        return
    app = QApplication(sys.argv)
    w = Window(model)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
