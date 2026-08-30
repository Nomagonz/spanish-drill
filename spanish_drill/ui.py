"""The window.

A thin adapter over DrillSession: it owns widgets and threads, and no drilling
logic. Anything that decides what an answer means belongs in session.py.
"""
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QSpinBox, QVBoxLayout, QWidget)

from .audio import input_devices
from .config import MAIN_MODEL, SILENT_FLOOR
from .listener import Listener
from .progress import Progress
from .scheduler import MATURE_AT
from .session import DrillSession

CHASSIS = "#1E2B26"; PANEL = "#2B3B34"; RULE = "#3C4F47"; SCREEN = "#ECE5D3"
INK = "#141A17"; INK_SOFT = "#5A665F"; SIGNAL = "#E19B33"; MISS = "#C24B36"
HIT = "#6E9A6B"; MUTE = "#8FA096"

MONO = "'Menlo','IBM Plex Mono',monospace"
DISPLAY = "'Helvetica Neue',sans-serif"


def model_is_cached(name):
    """True when the weights are already on disk, so nothing will be fetched."""
    import glob
    import os
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for hit in glob.glob(os.path.join(hub, f"models--*faster-whisper-{name}")):
        if glob.glob(os.path.join(hit, "snapshots", "*", "model.bin")):
            return True
    return False


class ModelLoader(QObject):
    """Loads the models off the main thread.

    They are cached on disk and never re-downloaded, but reading them into
    memory still costs a couple of seconds. Doing that on the first GO froze
    the window and made it look like something was being fetched every run.
    """
    done = pyqtSignal(object, str)

    def __init__(self, model_name, device_name):
        super().__init__()
        self.model_name, self.device_name = model_name, device_name

    def run(self):
        try:
            self.done.emit(Listener(self.model_name, self.device_name), "")
        except Exception as exc:
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


class SessionWorker(QObject):
    """Runs a DrillSession on a worker thread and republishes its callbacks."""
    prompt = pyqtSignal(object, str)
    status = pyqtSignal(str)
    heard = pyqtSignal(str)
    result = pyqtSignal(object)
    counts = pyqtSignal()
    verified = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, session):
        super().__init__()
        self.session = session
        session.on_prompt = self.prompt.emit
        session.on_status = self.status.emit
        session.on_heard = self.heard.emit
        session.on_result = self.result.emit
        session.on_counts = self.counts.emit
        session.on_verify = self.verified.emit
        session.on_finished = self.finished.emit

    def run(self):
        self.session.run()

    def stop(self):
        self.session.stop()


def _label(text, style):
    w = QLabel(text)
    w.setStyleSheet(style)
    return w


class Window(QWidget):
    def __init__(self, model_name=MAIN_MODEL):
        super().__init__()
        self.progress = Progress.load()
        self.model_name = model_name or self.progress.model
        self.listener = None
        self.worker = self.thread = None
        self._starting = False

        self.setWindowTitle("Spanish Drill")
        self.resize(560, 760)
        self.setStyleSheet(f"background:{CHASSIS};color:{SCREEN};")
        self._build()
        self._start_preload()

    # -- construction -----------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        root.addWidget(_label("DRILL · ES",
                              f"color:{SCREEN};font:800 15px {DISPLAY};letter-spacing:2px;"))
        root.addWidget(self._counters())
        root.addWidget(self._screen())
        root.addWidget(self._slip())
        root.addStretch(1)
        root.addLayout(self._microphone_row())
        root.addLayout(self._settings_row())

        self.tally = _label("", f"color:{MUTE};font:10px {MONO};letter-spacing:1px;")
        root.addWidget(self.tally)

        self.go = QPushButton("GO")
        self.go.setStyleSheet(
            f"background:{SIGNAL};color:#20160A;font:600 14px {MONO};"
            f"letter-spacing:2px;padding:18px;border:none;border-radius:3px;")
        self.go.clicked.connect(self.toggle)
        root.addWidget(self.go)

        root.addWidget(_label(
            "Say “skip”, “repeat”, “no sé”, or “stop” at any time.",
            f"color:{MUTE};font:10px {MONO};"))
        self._refresh_counters()
        self._refresh_tally()

    def _counters(self):
        row = QHBoxLayout()
        row.setSpacing(0)
        self.counters, self.counter_labels = {}, {}
        for key, text, colour in (("new", "NEW LEFT", SCREEN),
                                  ("due", "IN QUEUE", SIGNAL),
                                  ("missed", "MISSED TODAY", MISS),
                                  ("learning", "LEARNING", HIT)):
            value = _label("0", f"color:{colour};font:700 22px {DISPLAY};")
            caption = _label(text, f"color:{MUTE};font:9px {MONO};letter-spacing:1px;")
            box = QVBoxLayout()
            box.addWidget(value)
            box.addWidget(caption)
            holder = QWidget()
            holder.setLayout(box)
            holder.setStyleSheet(f"border-right:1px solid {RULE};")
            row.addWidget(holder)
            self.counters[key], self.counter_labels[key] = value, caption
        wrapper = QWidget()
        wrapper.setLayout(row)
        wrapper.setStyleSheet(f"border-top:1px solid {RULE};border-bottom:1px solid {RULE};")
        return wrapper

    def _screen(self):
        frame = QFrame()
        frame.setStyleSheet(f"background:{SCREEN};border-radius:3px;")
        frame.setMinimumHeight(210)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 16)

        header = QHBoxLayout()
        self.state_label = _label("", f"color:{INK_SOFT};font:9px {MONO};letter-spacing:2px;")
        self.status_label = _label("Ready", f"color:{INK_SOFT};font:9px {MONO};letter-spacing:2px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.state_label)
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.prompt_label = _label("Press Go.", f"color:{INK};font:700 34px {DISPLAY};")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Expanding)
        layout.addWidget(self.prompt_label)

        self.heard_label = _label("", f"color:{INK_SOFT};font:13px {MONO};")
        self.heard_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.heard_label.setWordWrap(True)
        layout.addWidget(self.heard_label)
        return frame

    def _slip(self):
        self.slip = QFrame()
        self.slip.setVisible(False)
        layout = QVBoxLayout(self.slip)
        layout.setContentsMargins(14, 12, 14, 12)
        self.verdict_label = _label("", f"color:{MUTE};font:10px {MONO};letter-spacing:2px;")
        self.answer_label = _label("", f"color:{SCREEN};font:700 24px {DISPLAY};")
        self.example_label = _label("", f"color:{SCREEN};font:13px {DISPLAY};")
        self.gloss_label = _label("", f"color:{MUTE};font:12px {DISPLAY};")
        self.detail_label = _label("", f"color:{MUTE};font:10px {MONO};letter-spacing:1px;")
        for w in (self.verdict_label, self.answer_label, self.example_label,
                  self.gloss_label, self.detail_label):
            w.setWordWrap(True)
            layout.addWidget(w)
        return self.slip

    def _microphone_row(self):
        row = QHBoxLayout()
        row.addWidget(_label("mic", f"color:{MUTE};font:10px {MONO};"))
        self.mic = QComboBox()
        self.mic.addItem("System default", "")
        for _, name in input_devices():
            self.mic.addItem(name, name)
        stored = self.progress.input_device
        if stored and self.mic.findData(stored) >= 0:
            self.mic.setCurrentIndex(self.mic.findData(stored))
        self.mic.setStyleSheet(
            f"background:{PANEL};color:{SCREEN};border:1px solid {RULE};padding:3px;")
        row.addWidget(self.mic, 1)
        return row

    def _settings_row(self):
        row = QHBoxLayout()
        self.dialect = QComboBox()
        self.dialect.addItems(["es-MX", "es-ES"])
        self.dialect.setCurrentText(self.progress.dialect)
        self.new_per = QSpinBox()
        self.new_per.setRange(0, 100)
        self.new_per.setValue(self.progress.new_per)
        self.window_seconds = QSpinBox()
        self.window_seconds.setRange(3, 20)
        self.window_seconds.setValue(int(self.progress.window))
        for caption, widget in (("accent", self.dialect), ("new/day", self.new_per),
                                ("wait s", self.window_seconds)):
            row.addWidget(_label(caption, f"color:{MUTE};font:10px {MONO};"))
            widget.setStyleSheet(
                f"background:{PANEL};color:{SCREEN};border:1px solid {RULE};padding:3px;")
            row.addWidget(widget)
        self.hints = QCheckBox("say it back")
        self.hints.setChecked(self.progress.hints)
        self.double_check = QCheckBox("double-check misses")
        self.double_check.setChecked(self.progress.verify_live)
        for box in (self.hints, self.double_check):
            box.setStyleSheet(f"color:{MUTE};font:10px {MONO};")
            row.addWidget(box)
        return row

    # -- model loading ----------------------------------------------------
    def _start_preload(self):
        cached = model_is_cached(self.model_name)
        self.go.setEnabled(False)
        self.go.setText("LOADING MODELS…")
        self.status_label.setText("Loading from disk" if cached else "Downloading models")
        if not cached:
            self.prompt_label.setText("Fetching the speech models.\n"
                                      "This happens once; then they load from disk.")
        self._loader = ModelLoader(self.model_name, self.progress.input_device)
        self._loader_thread = QThread()
        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.done.connect(self._on_models_ready)
        self._loader_thread.start()

    def _on_models_ready(self, listener, error):
        self._loader_thread.quit()
        self._loader_thread.wait()
        if listener is None:
            self.prompt_label.setText("Model failed to load")
            self.status_label.setText(error[:70])
            return
        self.listener = listener
        self.go.setEnabled(True)
        self.go.setText("GO")
        self.status_label.setText("Ready")
        self.prompt_label.setText("Press Go.")

    # -- running ----------------------------------------------------------
    def toggle(self):
        # The button is disabled while loading or stopping, but do not depend
        # on the widget alone: a re-entrant call here would reassign the thread
        # and destroy a running QThread, which aborts the process.
        if self._starting or not self.go.isEnabled():
            return
        if self.thread and self.thread.isRunning():
            self._request_stop()
            return
        self._apply_settings()
        if self.listener is None:
            self.status_label.setText("Models still loading…")
            return
        if not self._calibrate():
            return
        self._start_session()

    def _apply_settings(self):
        chosen = self.mic.currentData() or ""
        if chosen != self.progress.input_device and self.listener:
            self.listener.set_device(chosen)
        self.progress.input_device = chosen
        self.progress.dialect = self.dialect.currentText()
        self.progress.new_per = self.new_per.value()
        self.progress.window = float(self.window_seconds.value())
        self.progress.hints = self.hints.isChecked()
        self.progress.verify_live = self.double_check.isChecked()
        self.progress.model = self.model_name
        self.progress.save()

    def _calibrate(self):
        self._starting = True
        self.go.setEnabled(False)
        self.go.setText("CALIBRATING…")
        self.status_label.setText("Measuring room noise — stay quiet")
        QApplication.processEvents()
        try:
            floor = self.listener.calibrate()
        except Exception as exc:
            self.prompt_label.setText("Microphone failed")
            self.status_label.setText(str(exc)[:70])
            return False
        finally:
            self._starting = False
            self.go.setEnabled(True)
        if floor <= SILENT_FLOOR:
            self.status_label.setText("Warning: mic reads near silence")
        return True

    def _start_session(self):
        self.go.setText("STOP")
        self.slip.setVisible(False)
        session = DrillSession(self.progress, self.listener)
        self.worker = SessionWorker(session)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.prompt.connect(self._on_prompt)
        self.worker.status.connect(self.status_label.setText)
        self.worker.heard.connect(self.heard_label.setText)
        self.worker.result.connect(self._on_result)
        self.worker.counts.connect(self._refresh_counters)
        self.worker.verified.connect(self._on_verified)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _request_stop(self):
        # The worker can be mid-speech or mid-transcription and takes a moment
        # to notice. Say so and refuse clicks until it has really finished,
        # rather than showing GO and silently ignoring presses.
        self.worker.stop()
        self.go.setText("STOPPING…")
        self.go.setEnabled(False)
        self.status_label.setText("Stopping")

    # -- updates ----------------------------------------------------------
    def _on_prompt(self, card, state):
        self.prompt_label.setText(card.prompt)
        self.state_label.setText(state.upper())
        self.slip.setVisible(False)

    def _on_result(self, result):
        # When a miss is overturned the transcript on screen is the wrong one.
        # Leaving it up makes a correct answer look like a bad grade.
        if result.overturned and result.api_text:
            self.heard_label.setText(
                f"{result.api_text}   (local heard “{result.said}”)")

        colour = HIT if result.correct else MISS
        if result.overturned:
            verdict = "Correct — local model misheard you"
        elif result.correct:
            verdict = "Correct — check the spelling" if result.close else "Correct"
        else:
            verdict = "No answer — counted as a miss" if result.silent else "Missed"

        self.slip.setStyleSheet(
            f"background:{PANEL};border-radius:2px;border-left:3px solid {colour};")
        self.verdict_label.setStyleSheet(
            f"color:{colour};font:10px {MONO};letter-spacing:2px;")
        self.verdict_label.setText(verdict.upper())

        answers = result.card.answers
        extra = f"   (also: {', '.join(answers[1:])})" if len(answers) > 1 else ""
        self.answer_label.setText(answers[0] + extra)
        self.example_label.setText(result.card.example)
        self.gloss_label.setText(result.card.gloss)

        detail = [result.next_review, f"ease {result.state.ease:.2f}"]
        if result.said:
            detail.append(f"you said: {result.said}")
        if result.state.lapses:
            detail.append(f"missed {result.state.lapses}x")
        if result.api_text:
            detail.append(f"re-check heard “{result.api_text}”")
        self.detail_label.setText("  ·  ".join(detail))
        self.slip.setVisible(True)

    def _on_verified(self, kept, overturned):
        self._refresh_tally()

    def _on_finished(self):
        self.go.setText("GO")
        self.go.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self._refresh_counters()
        self._refresh_tally()

    def _refresh_counters(self):
        p = self.progress
        self.counters["new"].setText(str(p.new_remaining()))
        self.counters["due"].setText(str(len(p.due_indexes())))
        self.counters["missed"].setText(str(p.missed_today))
        self.counters["learning"].setText(str(p.learning_count()))
        self.counter_labels["learning"].setText(
            f"LEARNING · {p.mature_count()} MATURE")

    def _refresh_tally(self):
        kept, overturned = self.progress.kept, self.progress.overturned
        total = kept + overturned
        share = (f"  ({100 * overturned / total:.0f}% of misses were the "
                 f"local model's fault)" if total else "")
        self.tally.setText(
            f"DOUBLE-CHECK:  {overturned} OVERTURNED  ·  {kept} KEPT{share}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(3000)
        self.progress.save()
        event.accept()
