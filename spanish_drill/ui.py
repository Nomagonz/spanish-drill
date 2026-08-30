"""The window.

A thin adapter over DrillSession: it owns widgets and threads, and no drilling
logic. Anything that decides what an answer means belongs in session.py.
"""
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                             QHBoxLayout, QLabel, QProgressBar, QPushButton,
                             QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from . import voice
from .audio import input_devices
from .deck import categories, load_deck
from .config import MAIN_MODEL, SILENT_FLOOR
from .listener import Listener
from .progress import Progress
from .scheduler import MATURE_AT
from .placement import PlacementSession
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


class Prerecorder(QObject):
    """Records the prompts for a category ahead of time, off the main thread.

    Recording a phrase takes about nine tenths of a second. Doing it on demand
    means paying that inside the first card that needs it; doing it in the
    background means the drill only ever plays a file that already exists.
    """
    progress = pyqtSignal(int, int)
    done = pyqtSignal(int)

    def __init__(self, phrases):
        super().__init__()
        self.phrases = phrases
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        made = voice.prerecord(self.phrases,
                               should_stop=lambda: self._stop,
                               on_progress=self.progress.emit)
        self.done.emit(made)


class SessionWorker(QObject):
    """Runs a DrillSession on a worker thread and republishes its callbacks."""
    prompt = pyqtSignal(object, str)
    status = pyqtSignal(str)
    heard = pyqtSignal(str)
    result = pyqtSignal(object)
    counts = pyqtSignal()
    progress = pyqtSignal(int, int)
    verified = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, session, silent_floor=SILENT_FLOOR):
        super().__init__()
        self.session = session
        self.silent_floor = silent_floor
        session.on_prompt = self.prompt.emit
        session.on_status = self.status.emit
        session.on_heard = self.heard.emit
        session.on_result = self.result.emit
        session.on_counts = self.counts.emit
        session.on_progress = self.progress.emit
        session.on_verify = self.verified.emit
        session.on_finished = self.finished.emit

    def run(self):
        # Calibration reads the microphone for about a second. On the main
        # thread that froze the window on every press, which is what made the
        # button feel broken.
        if self.session.stop_requested:
            self.finished.emit()
            return
        try:
            self.status.emit("Measuring room noise — stay quiet")
            floor = self.session.listener.calibrate()
        except Exception as exc:
            self.status.emit(f"Microphone failed: {str(exc)[:50]}")
            self.finished.emit()
            return
        if floor <= self.silent_floor:
            self.status.emit("Warning: mic reads near silence")
        self.session.run()      # returns at once if a stop arrived meanwhile

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
        self._recorder = self._recorder_thread = None
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

        header = QHBoxLayout()
        header.addWidget(_label(
            "DRILL · ES",
            f"color:{SCREEN};font:800 15px {DISPLAY};letter-spacing:2px;"))
        self.mode_label = _label("", f"color:{SIGNAL};font:700 11px {MONO};"
                                     "letter-spacing:3px;")
        header.addSpacing(12)
        header.addWidget(self.mode_label)
        header.addStretch(1)
        self.progress_note = _label("", f"color:{MUTE};font:10px {MONO};"
                                        "letter-spacing:1px;")
        header.addWidget(self.progress_note)
        root.addLayout(header)

        # Only shown while a run has a known end: the placement test knows how
        # many words it set out to classify, an open-ended drill does not.
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setVisible(False)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:{PANEL};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{SIGNAL};border-radius:3px;}}")
        root.addWidget(self.bar)

        root.addWidget(self._counters())
        root.addWidget(self._screen())
        root.addWidget(self._slip())
        root.addStretch(1)
        root.addLayout(self._microphone_row())
        root.addLayout(self._category_row())
        root.addLayout(self._settings_row())

        self.tally = _label("", f"color:{MUTE};font:10px {MONO};letter-spacing:1px;")
        root.addWidget(self.tally)

        self.go = QPushButton("GO")
        self.go.setStyleSheet(
            f"background:{SIGNAL};color:#20160A;font:600 14px {MONO};"
            f"letter-spacing:2px;padding:18px;border:none;border-radius:3px;")
        self.go.clicked.connect(self.toggle)
        root.addWidget(self.go)

        prow = QHBoxLayout()
        prow.addWidget(_label("test", f"color:{MUTE};font:10px {MONO};"))
        self.scope = QComboBox()
        self.scope.setStyleSheet(
            f"background:{PANEL};color:{SCREEN};border:1px solid {RULE};padding:3px;")
        prow.addWidget(self.scope, 1)
        root.addLayout(prow)
        self._refresh_scope()

        self.placement = QPushButton("PLACEMENT TEST")
        self.placement.setStyleSheet(
            f"background:transparent;color:{MUTE};font:600 11px {MONO};"
            f"letter-spacing:2px;padding:10px;border:1px solid {RULE};"
            f"border-radius:3px;")
        self.placement.clicked.connect(self.toggle_placement)
        root.addWidget(self.placement)

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
        frame = self.screen = QFrame()
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

    def _category_row(self):
        row = QHBoxLayout()
        row.addWidget(_label("drill", f"color:{MUTE};font:10px {MONO};"))
        self.category = QComboBox()
        deck = load_deck()
        self.category.addItem(f"everything ({len(deck)})", "all")
        for pos in categories(deck):
            n = sum(1 for c in deck if c.pos == pos)
            self.category.addItem(f"{pos}s only ({n})", pos)
        chosen = self.progress.category or "all"
        if self.category.findData(chosen) >= 0:
            self.category.setCurrentIndex(self.category.findData(chosen))
        self.category.setStyleSheet(
            f"background:{PANEL};color:{SCREEN};border:1px solid {RULE};padding:3px;")
        self.category.currentIndexChanged.connect(self._on_category_changed)
        row.addWidget(self.category, 1)
        return row

    def _on_category_changed(self):
        """The scope counts are per category, so they move with it."""
        self.progress.category = self.category.currentData() or "all"
        self._refresh_scope()
        self._refresh_counters()
        if self.listener is not None:
            self._start_prerecording()      # a new category needs new prompts

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
        self.placement.setEnabled(False)
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
        self.placement.setEnabled(True)
        self.go.setText("GO")
        self.status_label.setText("Ready")
        self.prompt_label.setText("Press Go.")
        self._start_prerecording()

    def _start_prerecording(self):
        """Record this category's prompts quietly in the background."""
        deck = load_deck()
        wanted = [i for i in range(len(deck))
                  if self.progress.in_category(i, deck)]
        phrases = voice.phrases_for(deck, self.progress.dialect, wanted)
        missing = [p for p in phrases if not voice.path_for(*p).exists()]
        if not missing:
            return
        self._recorder = Prerecorder(missing)
        self._recorder_thread = QThread()
        self._recorder.moveToThread(self._recorder_thread)
        self._recorder_thread.started.connect(self._recorder.run)
        self._recorder.progress.connect(self._on_prerecord_progress)
        self._recorder.done.connect(self._on_prerecord_done)
        self._recorder_thread.start()

    def _on_prerecord_progress(self, done, total):
        if self.thread is None or not self.thread.isRunning():
            self.status_label.setText(f"Recording prompts {done}/{total}")

    def _on_prerecord_done(self, made):
        self._recorder_thread.quit()
        self._recorder_thread.wait()
        if self.thread is None or not self.thread.isRunning():
            self.status_label.setText("Ready")

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
        self.progress.category = self.category.currentData() or "all"
        self.progress.model = self.model_name
        self.progress.save()

    def toggle_placement(self):
        """Rapid triage instead of a drill: right twice is known, wrong is not."""
        if self._starting or not self.placement.isEnabled():
            return
        if self.thread and self.thread.isRunning():
            self._request_stop()
            return
        self._apply_settings()
        if self.listener is None:
            self.status_label.setText("Models still loading…")
            return
        self._start_session(placement=True)

    def _start_session(self, placement=False):
        self.go.setText("STOP")
        self.slip.setVisible(False)
        self._hide_progress()
        if placement:
            session = PlacementSession(self.progress, self.listener,
                                       retest=self.scope.currentData() == "all")
            self.mode_label.setText("PLACEMENT TEST")
            self.placement.setText("STOP TEST")
            self.go.setEnabled(False)
            self.screen.setStyleSheet(
                f"background:{SCREEN};border-radius:3px;"
                f"border:2px solid {SIGNAL};")
        else:
            session = DrillSession(self.progress, self.listener)
            self.mode_label.setText("")
            self.screen.setStyleSheet(f"background:{SCREEN};border-radius:3px;")
        self.worker = SessionWorker(session)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.prompt.connect(self._on_prompt)
        self.worker.status.connect(self.status_label.setText)
        self.worker.heard.connect(self.heard_label.setText)
        self.worker.result.connect(self._on_result)
        self.worker.counts.connect(self._refresh_counters)
        self.worker.progress.connect(self._on_progress)
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
        self.placement.setEnabled(False)
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

    def _on_progress(self, done, total):
        if not total:
            self._hide_progress()
            return
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        self.bar.setVisible(True)
        session = getattr(self.worker, "session", None)
        skipped = getattr(session, "skipped", 0)
        extra = f"  ({skipped} skipped)" if skipped else ""
        self.progress_note.setText(f"{done} / {total}{extra}")

    def _hide_progress(self):
        self.bar.setVisible(False)
        self.progress_note.setText("")

    def _on_verified(self, kept, overturned):
        self._refresh_tally()

    def _on_finished(self):
        session = getattr(self.worker, "session", None)
        if isinstance(session, PlacementSession):
            s = session.summary()
            if s["tested"]:
                self.prompt_label.setText(
                    f"{len(s['known'])} known · {len(s['to_learn'])} to learn")
                self.progress_note.setText(f"{s['tested']} / {session.total} done")
        self.go.setText("GO")
        self.go.setEnabled(True)
        self.placement.setEnabled(True)
        self.placement.setText("PLACEMENT TEST")
        self.mode_label.setText("")
        self.screen.setStyleSheet(f"background:{SCREEN};border-radius:3px;")
        self._refresh_scope()
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

    def _refresh_scope(self):
        """Spell out what each option would actually ask, with live counts.

        "re-test words already sorted" said nothing about how many that was,
        or how to start over when the sorting turned out to be wrong.
        """
        deck = load_deck()
        in_scope = [i for i in range(len(deck))
                    if self.progress.in_category(i, deck)]
        unsorted = [i for i in in_scope if i not in self.progress.cards]
        current = self.scope.currentData() if self.scope.count() else "new"
        self.scope.blockSignals(True)
        self.scope.clear()
        self.scope.addItem(f"only words never sorted ({len(unsorted)})", "new")
        self.scope.addItem(f"start over, all of them ({len(in_scope)})", "all")
        index = self.scope.findData(current)
        self.scope.setCurrentIndex(max(0, index))
        self.scope.blockSignals(False)

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
        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            recorder.stop()
            self._recorder_thread.quit()
            self._recorder_thread.wait(3000)
        voice.stop()
        release = getattr(self.listener, "close", None)
        if release:
            try:
                release()       # let go of the microphone
            except Exception:
                pass
        self.progress.save()
        event.accept()
