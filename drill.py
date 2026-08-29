#!/usr/bin/env python3
"""
Spanish voice drill, local edition.

Hands free: it says an English word, listens, grades what you said, and moves on.
Speech in is Whisper running on this machine. Speech out is the macOS `say` voice.
Neither one is a browser, so none of the Web Speech API's rules apply.

    ./run.sh              normal session
    ./run.sh --model tiny faster start, less accurate
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

SR = 16000                      # Whisper wants 16 kHz mono
DAY = 86400
IVL = [0, 1, 3, 7, 16, 35, 90, 180]     # same boxes as the web version

VOICES = {"es-MX": "Paulina", "es-ES": "Mónica"}


# ---------------------------------------------------------------- persistence
@dataclass
class State:
    cards: dict = field(default_factory=dict)   # index -> {b, d, r, l}
    dialect: str = "es-MX"
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
            s.cards = {int(k): v for k, v in s.cards.items()}
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

    def __init__(self, model_name="small"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.floor = 0.004
        self.level = 0.0

    def calibrate(self, seconds=0.7):
        buf = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(buf ** 2)))
        self.floor = max(0.0025, rms * 2.5)

    def listen(self, max_seconds, should_stop=None):
        """Returns the transcript, or None if you never said anything."""
        q = queue.Queue()
        sd.default.samplerate, sd.default.channels = SR, 1

        def cb(indata, frames, t, status):
            q.put(indata.copy())

        frames, speaking, silence_run, started = [], False, 0.0, time.time()
        hop = 0.05
        with sd.InputStream(callback=cb, blocksize=int(SR * hop), dtype="float32"):
            while True:
                if should_stop and should_stop():
                    return None
                if time.time() - started > max_seconds and not speaking:
                    return None
                try:
                    block = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(block ** 2)))
                self.level = rms
                if rms > self.floor:
                    speaking, silence_run = True, 0.0
                    frames.append(block)
                elif speaking:
                    silence_run += hop
                    frames.append(block)
                    if silence_run > 0.7:           # trailing pause ends the turn
                        break
                if speaking and time.time() - started > max_seconds + 6:
                    break                            # hard cap on a rambler

        if not frames:
            return None
        audio = np.concatenate(frames).flatten()
        if len(audio) < SR * 0.25:
            return None
        # beam_size=5 measurably beats greedy on single words: greedy turned
        # "querer" into "Quieres". A biasing initial_prompt was tried and made
        # it worse, so there deliberately isn't one.
        segs, _ = self.model.transcribe(audio, language="es", beam_size=5,
                                        temperature=0, vad_filter=False)
        return " ".join(s.text for s in segs).strip() or None


# ------------------------------------------------------------------ drill loop
class Drill(QObject):
    prompt = pyqtSignal(str, str)      # english, card state label
    status = pyqtSignal(str)
    heard = pyqtSignal(str)
    result = pyqtSignal(dict)
    counts = pyqtSignal(int, int, int, int)
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
        return [i for i, c in self.S.cards.items() if c["d"] <= t]

    def fresh(self):
        return [i for i in range(len(DECK)) if i not in self.S.cards]

    def new_left(self):
        return max(0, self.S.new_per - self.S.new_done)

    def learned(self):
        return sum(1 for c in self.S.cards.values() if c["b"] >= 3)

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

    def grade(self, cid, ok):
        import random
        is_new = cid not in self.S.cards
        c = self.S.cards.get(cid, {"b": 0, "d": today(), "r": 0, "l": 0})
        c["r"] += 1
        if ok:
            c["b"] = min(len(IVL) - 1, max(1, c["b"] + 1))
            c["d"] = today() + IVL[c["b"]]
        else:
            c["b"], c["d"] = 0, today()
            c["l"] += 1
            self.S.miss_today += 1
            self.queue.insert(min(len(self.queue), 4 + random.randint(0, 2)), cid)
        self.S.cards[cid] = c
        if is_new:
            self.S.new_done += 1
        self.S.save()
        return c

    def emit_counts(self):
        self.counts.emit(self.new_left(), len(self.queue),
                         self.S.miss_today, self.learned())

    # -- the loop ----------------------------------------------------------
    def run(self):
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
            label = ("New word" if not c or c["r"] < 1
                     else ("Relearning" if c["b"] == 0 else f"Review · box {c['b']}"))
            self.prompt.emit(card["en"], label)
            self.heard.emit("")
            self.emit_counts()

            self.status.emit("Speaking")
            say(re.sub(r"\s*\(.*?\)\s*", " ", card["en"]).strip(), rate=180)
            if not self.running:
                break

            said, verdict, silent = None, None, False
            while self.running:
                self.status.emit("Listening")
                said = self.listener.listen(self.S.window,
                                            should_stop=lambda: not self.running)
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
                    self.finished.emit()
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
            c = self.grade(cid, ok)
            self.result.emit({
                "ok": ok, "said": "" if silent else (said or ""), "card": card,
                "box": c, "close": bool(verdict and verdict["close"]),
                "silent": silent, "next": next_interval(c),
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

        self.finished.emit()


def next_interval(c):
    if c["b"] == 0:
        return "again this session"
    d = IVL[c["b"]]
    if d == 1:
        return "again tomorrow"
    return f"again in {d} days" if d < 30 else f"again in {round(d/30)} months"


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
        for key, label, col in (("new", "NEW LEFT", SCREEN), ("due", "IN QUEUE", SIG),
                                ("miss", "MISSED TODAY", MISS), ("known", "LEARNED", SCREEN)):
            box = QVBoxLayout()
            n = QLabel("0"); n.setStyleSheet(f"color:{col};font:700 22px 'Helvetica Neue';")
            t = QLabel(label); t.setStyleSheet(f"color:{MUTE};font:9px 'Menlo';letter-spacing:1px;")
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
        due = sum(1 for c in cards.values() if c["d"] <= t)
        self.cnt["new"].setText(str(max(0, self.S.new_per - self.S.new_done)))
        self.cnt["due"].setText(str(due))
        self.cnt["miss"].setText(str(self.S.miss_today))
        self.cnt["known"].setText(str(sum(1 for c in cards.values() if c["b"] >= 3)))

    def toggle(self):
        if self.thread and self.thread.isRunning():
            self.drill.stop()
            self.go.setText("GO")
            return
        self.S.dialect = self.dialect.currentText()
        self.S.new_per = self.newper.value()
        self.S.window = float(self.win.value())
        self.S.hints = self.hints.isChecked()
        self.S.save()

        if self.listener is None:
            self.status_lbl.setText("Loading Whisper…")
            QApplication.processEvents()
            try:
                self.listener = Listener(self.model_name)
                self.listener.calibrate()
            except Exception as e:
                self.prompt_lbl.setText("Mic or model failed")
                self.status_lbl.setText(str(e)[:60])
                return

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

    def on_counts(self, new, due, miss, known):
        self.cnt["new"].setText(str(new)); self.cnt["due"].setText(str(due))
        self.cnt["miss"].setText(str(miss)); self.cnt["known"].setText(str(known))

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
        self.sched_lbl.setText(r["next"] + said)
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
    model = "small"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    app = QApplication(sys.argv)
    w = Window(model)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
