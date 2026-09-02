"""The window.

A thin adapter over DrillSession: it owns widgets and threads, and no drilling
logic. Anything that decides what an answer means belongs in session.py.
"""
import time
from html import escape

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QSizePolicy,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from . import voice
from .audio import input_devices
from .deck import categories, load_deck
from .config import CONJUGATION_LOG, MAIN_MODEL, SILENT_FLOOR
from .listener import Listener
from .progress import Progress
from .scheduler import MATURE_AT
from .answers import AnswerLog
from .composition import SentenceDrill
from .paradigm import (ConjugationProgress, ConjugationSession,
                       _conjugation_verifier)
from .placement import PlacementSession
from .session import DrillSession, _api_verifier
from .typed import Gate, TypedListener

CHASSIS = "#1E2B26"; PANEL = "#2B3B34"; RULE = "#3C4F47"; SCREEN = "#ECE5D3"
INK = "#141A17"; INK_SOFT = "#5A665F"; SIGNAL = "#E19B33"; MISS = "#C24B36"
HIT = "#6E9A6B"; MUTE = "#8FA096"
# Interaction states. A control that looks identical whether it is hovered,
# pressed or dead is the cheapest thing to leave out and the most obvious once
# you have seen it.
PANEL_HI = "#33463E"; SIGNAL_HI = "#EFAA44"; SIGNAL_LO = "#C9862A"

MONO = "'Menlo','IBM Plex Mono',monospace"
DISPLAY = "'Helvetica Neue',sans-serif"

# One spacing step and one corner, used everywhere. Gaps that are all slightly
# different from each other read as accidental, and that is most of what an
# unfinished window actually looks like.
GAP = 10
RADIUS = 4


def _input_style():
    """One look for every control that holds a value.

    The drop-down arrow and the spinner arrows are deliberately left native.
    Styling them cost their indicators, and a combo box with no arrow is
    indistinguishable from a text field.
    """
    return (
        f"QComboBox,QSpinBox{{background:{PANEL};color:{SCREEN};"
        f"border:1px solid {RULE};border-radius:{RADIUS}px;padding:6px 8px;"
        f"font:12px {MONO};}}"
        f"QComboBox:hover,QSpinBox:hover{{border-color:{MUTE};}}"
        f"QComboBox:focus,QSpinBox:focus{{border-color:{SIGNAL};}}"
        f"QComboBox QAbstractItemView{{background:{PANEL};color:{SCREEN};"
        f"border:1px solid {RULE};selection-background-color:{RULE};"
        f"outline:none;}}")


def _button_style(kind="secondary"):
    """Three weights, so which control to reach for is never a guess.

    Primary is the one thing this window is for. Secondary are the other two
    ways to start. Ghost is not really a button at all, it just opens a
    drawer, and giving it a border would put it in the same class as the two
    that begin a session.
    """
    if kind == "primary":
        return (f"QPushButton{{background:{SIGNAL};color:#20160A;"
                f"font:700 14px {MONO};letter-spacing:2px;padding:16px;"
                f"border:none;border-radius:{RADIUS}px;}}"
                f"QPushButton:hover{{background:{SIGNAL_HI};}}"
                f"QPushButton:pressed{{background:{SIGNAL_LO};}}"
                f"QPushButton:disabled{{background:{RULE};color:{MUTE};}}")
    if kind == "ghost":
        return (f"QPushButton{{background:transparent;color:{MUTE};"
                f"font:600 10px {MONO};letter-spacing:2px;padding:8px 2px;"
                f"border:none;text-align:left;}}"
                f"QPushButton:hover{{color:{SCREEN};}}")
    return (f"QPushButton{{background:{PANEL};color:{SCREEN};"
            f"font:600 11px {MONO};letter-spacing:2px;padding:13px;"
            f"border:1px solid {RULE};border-radius:{RADIUS}px;}}"
            f"QPushButton:hover{{background:{PANEL_HI};border-color:{MUTE};}}"
            f"QPushButton:pressed{{background:{CHASSIS};}}"
            f"QPushButton:disabled{{background:transparent;color:{RULE};"
            f"border-color:{RULE};}}")


def _caption(text):
    """The small muted label that names a control."""
    return _label(text, f"color:{MUTE};font:9px {MONO};letter-spacing:1px;")


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


class SentenceMaker(QObject):
    """Writes sentences through the API, off the main thread.

    One first so the drill has something fresh almost immediately, then
    batches, because a batch of one costs a whole round trip per sentence.
    It stops for good once the store is full: the file is the tally, so
    reopening the app cannot start the bill over.

    Every batch is judged locally before it counts, and a batch where nothing
    survives is not retried forever — three barren rounds and it gives up,
    which is what an unreachable API or a word list too small to build
    sentences from actually looks like.
    """
    made = pyqtSignal(list)         # Sentence objects that survived the gate
    note = pyqtSignal(str)
    done = pyqtSignal(int)

    BARREN_ROUNDS = 3

    def __init__(self, progress):
        super().__init__()
        self.progress = progress
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from .generate import (as_sentences, generate_batch, load_generated,
                               remaining_allowance, save_generated)
        from .config import FIRST_BATCH, SENTENCE_BATCH
        stored = load_generated()
        written, barren = 0, 0
        wanted = FIRST_BATCH
        while not self._stop and remaining_allowance(stored) > 0:
            kept, dropped = generate_batch(self.progress, wanted, stored=stored)
            wanted = SENTENCE_BATCH
            if not kept:
                barren += 1
                if barren >= self.BARREN_ROUNDS or not dropped:
                    break       # nothing usable, or nothing coming back at all
                continue
            barren = 0
            stored.extend(kept)
            save_generated(stored)      # paid for, so keep it before anything else
            written += len(kept)
            self.made.emit(as_sentences(kept))
            self.note.emit(f"{len(stored)} written · {remaining_allowance(stored)} left")
        self.done.emit(written)


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
        if getattr(self.session, "typed", False):
            self.session.run()      # no microphone is opened, so nothing to measure
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
    """A label that paints only its text.

    The window sets `background` on itself, and a Qt stylesheet cascades that
    to every child, so each label was filling its own box with the chassis
    colour. Invisible against a dark panel, and a row of dark rectangles once
    one is placed on the cream card.
    """
    w = QLabel(text)
    w.setStyleSheet("background:transparent;" + style)
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
        self._typed_input = None    # set only while a typing session runs
        # (maker, thread) pairs still alive. Held rather than dropped: a
        # QThread garbage collected while running aborts the process, and the
        # writer can be inside an API call that quit() cannot interrupt.
        self._makers = []
        self._gate = None           # holds a missed card until Enter
        # Counts the answer window down while typing. A deadline you
        # cannot see is one that marks you wrong without warning.
        self._countdown = QTimer(self)
        self._countdown.timeout.connect(self._tick_countdown)
        self._deadline = 0.0

        self.setWindowTitle("Spanish Drill")
        self.resize(680, 780)
        self.setStyleSheet(f"background:{CHASSIS};color:{SCREEN};")
        self._build()
        self._start_preload()

    # -- construction -----------------------------------------------------
    WORDS = "WORDS"
    IDLE_CUE = "Pick a drill to start."

    def _build(self):
        """A header that always shows, then three tabs.

        One column of eight stacked zones put the card, the deck's whole
        history and a preference set once a month in front of the same eye at
        the same time. Only the first of those is wanted while a word is on
        screen, so the other two took a tab each and the drill got the window
        to itself. What stays above the tabs is the pair that has to be
        readable from any of them: what the drill is doing, and how far in.
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 20)
        root.setSpacing(GAP)

        root.addLayout(self._header())
        root.addWidget(self._progress_bar())

        # Built first because it constructs every stored control, including
        # the three the session bar borrows straight back onto the drill tab.
        settings = self._settings_panel()

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self._tab_style())
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tabs.addTab(self._drill_tab(), "DRILL")
        self.tabs.addTab(self._progress_tab(), "PROGRESS")
        self.tabs.addTab(self._padded(settings), "SETTINGS")
        root.addWidget(self.tabs, 1)

        self._refresh_scope()
        self._refresh_counters()
        self._refresh_tally()
        self._watch_for_a_new_day()

    def _tab_style(self):
        return (f"QTabWidget::pane{{border:none;background:transparent;}}"
                f"QTabBar{{qproperty-drawBase:0;}}"
                f"QTabBar::tab{{background:{PANEL};color:{MUTE};"
                f"border:1px solid {RULE};border-radius:{RADIUS}px;"
                f"padding:9px 20px;margin-right:6px;"
                f"font:700 10px {MONO};letter-spacing:2px;}}"
                f"QTabBar::tab:selected{{background:{RULE};color:{SCREEN};}}")

    def _padded(self, widget):
        """A tab page with the tab bar's gap under it and nothing else."""
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, GAP, 0, 0)
        box.setSpacing(GAP)
        box.addWidget(widget)
        box.addStretch(1)
        return holder

    def _drill_tab(self):
        """The card and everything used to answer it. Nothing else."""
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, GAP, 0, 0)
        box.setSpacing(GAP)
        box.addWidget(self._stage(), 1)
        box.addWidget(self._session_bar())
        box.addWidget(self._actions())
        return holder

    def _progress_tab(self):
        """Today's work, then the shape of the deck behind it."""
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, GAP, 0, 0)
        box.setSpacing(GAP)
        box.addWidget(self._counters())
        box.addWidget(self._deck_state())
        box.addStretch(1)
        return holder

    # -- zones ------------------------------------------------------------
    def _header(self):
        """Identity on the left, what the drill is doing on the right.

        The status used to sit inside the card, competing with the word for
        the one spot the eye is already aimed at.
        """
        row = QHBoxLayout()
        row.setSpacing(GAP)
        row.addWidget(_label(
            "DRILL · ES",
            f"color:{SCREEN};font:800 15px {DISPLAY};letter-spacing:2px;"))
        self.mode_label = _label("", f"color:{SIGNAL};font:700 10px {MONO};"
                                     "letter-spacing:3px;")
        row.addWidget(self.mode_label)
        row.addStretch(1)
        self.progress_note = _label("", f"color:{MUTE};font:10px {MONO};"
                                        "letter-spacing:1px;")
        row.addWidget(self.progress_note)
        self.status_label = _label("Ready", f"color:{MUTE};font:10px {MONO};"
                                            "letter-spacing:2px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status_label.setMinimumWidth(190)
        row.addWidget(self.status_label)
        return row

    def _progress_bar(self):
        """Only shown while a run has a known end. An open drill has none."""
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.setVisible(False)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:{PANEL};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{SIGNAL};border-radius:2px;}}")
        return self.bar

    def _counters(self):
        """Today's work first, then the state of the deck behind it.

        The panel used to open with what was left rather than what had been
        done, so a session could go well and show nothing for it.

        Three across rather than five. Five cells in the width available left
        every caption wrapping mid-phrase and shrunk to eight pixels to fit at
        all, which is a lot of work to read a number.
        """
        frame = QFrame()
        frame.setObjectName("counters")
        frame.setStyleSheet(f"#counters{{background:{PANEL};"
                            f"border:1px solid {RULE};"
                            f"border-radius:{RADIUS}px;}}")
        grid = QGridLayout(frame)
        grid.setContentsMargins(4, 14, 4, 13)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(16)
        self.counters, self.counter_labels = {}, {}
        fields = (("learned", "LEARNED", SCREEN), ("reviews", "REVIEWED", SCREEN),
                  ("due", "IN QUEUE", SIGNAL), ("missed", "MISSED", MISS),
                  ("learning", "LEARNING", HIT))
        across = 3
        for at, (key, text, colour) in enumerate(fields):
            row, column = divmod(at, across)
            value = _label("0", f"color:{colour};font:700 26px {DISPLAY};")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption = _label(text, f"color:{MUTE};font:9px {MONO};"
                                   "letter-spacing:1px;")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setWordWrap(True)
            cell = QVBoxLayout()
            cell.setSpacing(4)
            cell.setContentsMargins(10, 0, 10, 0)
            cell.addWidget(value)
            cell.addWidget(caption)
            holder = QWidget()
            holder.setLayout(cell)
            grid.addWidget(holder, row, column * 2)
            # A rule between cells, never against the frame edge and never
            # trailing the last one on a row.
            if column < across - 1 and at < len(fields) - 1:
                grid.addWidget(self._rule(), row, column * 2 + 1)
            self.counters[key], self.counter_labels[key] = value, caption
        for column in range(across):
            grid.setColumnStretch(column * 2, 1)
        return frame

    def _rule(self):
        line = QFrame()
        line.setFixedWidth(1)
        line.setStyleSheet(f"background:{RULE};")
        return line

    def _stage(self):
        """The card and everything that answers it, in one column.

        Given the stretch, so a taller window grows the card rather than
        spreading the controls apart.
        """
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(GAP)
        # The card takes the slack rather than a trailing spacer, so a taller
        # window grows the thing being read instead of opening a dead gap
        # between it and the controls.
        box.addWidget(self._screen(), 1)
        box.addWidget(self._answer_box())
        box.addWidget(self._slip())
        return holder

    def _screen(self):
        frame = self.screen = QFrame()
        frame.setObjectName("screen")
        frame.setStyleSheet(self._screen_style())
        frame.setMinimumHeight(190)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(GAP)

        self.state_label = _label("", f"color:{INK_SOFT};font:9px {MONO};"
                                      "letter-spacing:2px;")
        layout.addWidget(self.state_label)

        self.prompt_label = _label(self.IDLE_CUE,
                                   f"color:{INK};font:700 34px {DISPLAY};")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Expanding)
        layout.addWidget(self.prompt_label)

        self.heard_label = _label("", f"color:{INK_SOFT};font:13px {MONO};")
        self.heard_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.heard_label.setWordWrap(True)
        layout.addWidget(self.heard_label)
        return frame

    def _screen_style(self, accent=None):
        """The card, optionally ringed to say which mode is running."""
        ring = f"border:2px solid {accent};" if accent else ""
        return f"#screen{{background:{SCREEN};border-radius:{RADIUS}px;{ring}}}"

    def _answer_box(self):
        """Where a typed answer goes. Hidden unless the drill is typing.

        Directly under the cue rather than down by the buttons: in that mode
        the eye never leaves the pair, and the controls are not part of
        answering.
        """
        self.answer_box = QLineEdit()
        self.answer_box.setPlaceholderText("type the Spanish, then Enter")
        self.answer_box.setVisible(False)
        self.answer_box.setStyleSheet(
            f"QLineEdit{{background:{PANEL};color:{SCREEN};"
            f"border:1px solid {RULE};border-radius:{RADIUS}px;padding:14px;"
            f"font:600 18px {MONO};}}"
            f"QLineEdit:focus{{border-color:{SIGNAL};}}")
        self.answer_box.returnPressed.connect(self._submit_typed)
        return self.answer_box

    def _submit_typed(self):
        """Enter: let a held miss go, or hand over what was typed.

        One key does both because at the point of a miss there is nothing to
        answer, and asking for a different key to dismiss it would be a second
        thing to know for no gain.
        """
        if self._gate is not None and self._gate.waiting:
            # Whatever was typed while the miss was up was aimed at a card
            # that is already graded. Leaving it in the field carried it onto
            # the next word, so the next card opened half answered.
            self.answer_box.clear()
            self._gate.release()
            return
        text = self.answer_box.text()
        self.answer_box.clear()
        if self._typed_input is not None:
            self._typed_input.submit(text)

    def keyPressEvent(self, event):
        """Enter releases a held miss even when the field is not focused."""
        enter = (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if event.key() in enter and self._gate is not None and self._gate.waiting:
            self.answer_box.clear()
            self._gate.release()
            return
        super().keyPressEvent(event)

    def _slip(self):
        """What the answer was, and what happens to the card now."""
        self.slip = QFrame()
        self.slip.setObjectName("slip")
        self.slip.setVisible(False)
        layout = QVBoxLayout(self.slip)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        self.verdict_label = _label("", f"color:{MUTE};font:10px {MONO};"
                                        "letter-spacing:2px;")
        self.answer_label = _label("", f"color:{SCREEN};font:700 25px {DISPLAY};")
        self.example_label = _label("", f"color:{SCREEN};font:13px {DISPLAY};")
        self.gloss_label = _label("", f"color:{MUTE};font:12px {DISPLAY};")
        self.detail_label = _label("", f"color:{MUTE};font:10px {MONO};"
                                       "letter-spacing:1px;")
        # The marked-up sentence: right words plain, wrong ones carrying what
        # was written instead. Hidden outside the sentence mode.
        self.diff_label = _label("", f"color:{SCREEN};font:15px {MONO};")
        self.diff_label.setVisible(False)
        self.diff_label.setTextFormat(Qt.TextFormat.RichText)
        # Shown only while the drill is actually waiting, so it never claims
        # a keypress is needed when nothing is listening for one.
        self.continue_label = _label(
            "PRESS ENTER TO CONTINUE",
            f"color:{SIGNAL};font:700 10px {MONO};letter-spacing:2px;")
        self.continue_label.setVisible(False)
        for w in (self.verdict_label, self.answer_label, self.diff_label,
                  self.example_label, self.gloss_label, self.detail_label,
                  self.continue_label):
            w.setWordWrap(True)
            layout.addWidget(w)
        return self.slip

    def _deck_state(self):
        """The shape of the deck, and how the second opinion has been doing.

        Two lines that used to float loose at opposite ends of the window,
        both answering the same question: how much of this is working.
        """
        frame = QFrame()
        frame.setObjectName("deckState")
        frame.setStyleSheet(f"#deckState{{background:{PANEL};"
                            f"border-radius:{RADIUS}px;}}")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 10, 14, 10)
        box.setSpacing(5)
        self.ladder = _label("", f"color:{MUTE};font:10px {MONO};"
                                 "letter-spacing:1px;")
        self.tally = _label("", f"color:{MUTE};font:10px {MONO};"
                                "letter-spacing:1px;")
        for w in (self.ladder, self.tally):
            w.setWordWrap(True)
            box.addWidget(w)
        return frame

    def _session_bar(self):
        """The three dials that decide what the next session asks.

        These are not preferences. They are what you reach for when the queue
        runs dry and you want more words, or when the answer window is too
        short to type a long one. Filing them with the microphone and the
        accent put a daily decision behind a disclosure nobody opens.
        """
        frame = QFrame()
        frame.setObjectName("sessionBar")
        frame.setStyleSheet(f"#sessionBar{{background:{PANEL};"
                            f"border:1px solid {RULE};"
                            f"border-radius:{RADIUS}px;}}")
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)
        for at, (caption, widget, stretch) in enumerate(
                (("DRILL", self.category, 3),
                 ("NEW TODAY", self.new_per, 1),
                 ("ANSWER WAIT", self.window_seconds, 1))):
            if at:
                row.addSpacing(4)
            cell = QVBoxLayout()
            cell.setSpacing(4)
            cell.addWidget(_caption(caption))
            cell.addWidget(widget)
            holder = QWidget()
            holder.setLayout(cell)
            row.addWidget(holder, stretch)
        return frame

    def _actions(self):
        """One thing to press, then the other two ways to start.

        These were three buttons at three different weights scattered down the
        window with a dropdown loose between them. All three begin a session,
        so they belong together, and the dropdown that only ever configured
        the placement run went into the settings with the rest of the setup.
        """
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)

        self.go = QPushButton(self.WORDS)
        self.go.setStyleSheet(_button_style("primary"))
        self.go.clicked.connect(self.toggle)
        box.addWidget(self.go)

        self.placement = QPushButton("PLACEMENT TEST")
        self.placement.setStyleSheet(_button_style())
        self.placement.clicked.connect(self.toggle_placement)
        box.addWidget(self.placement)

        # Always typed, so it never waits for the speech models and is the one
        # button that works the moment the window opens.
        self.sentences = QPushButton("SENTENCES")
        self.sentences.setStyleSheet(_button_style())
        self.sentences.clicked.connect(self.toggle_sentences)
        box.addWidget(self.sentences)

        # Conjugations only, on their own schedule. Beside the others
        # rather than folded into GO, because it is a different sitting with
        # a different tracker behind it, not a filter on this one.
        self.conjugations = QPushButton("CONJUGATIONS")
        self.conjugations.setStyleSheet(_button_style())
        self.conjugations.clicked.connect(self.toggle_conjugations)
        box.addWidget(self.conjugations)

        # Typing is how you answer, not a third thing to start. Every button
        # above honours it, so a placement run can be typed too.
        self.typing = QCheckBox("type instead of speaking")
        self.typing.setStyleSheet(f"color:{MUTE};font:10px {MONO};")
        self.typing.toggled.connect(self._on_typing_toggled)
        box.addWidget(self.typing)

        # Controls must not hold focus, or Enter would re-fire whichever one
        # was clicked last instead of dismissing a held miss.
        for widget in (self.go, self.placement, self.sentences,
                       self.conjugations, self.typing):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            widget.setCursor(Qt.CursorShape.PointingHandCursor)

        self.hint = _label("", f"color:{MUTE};font:10px {MONO};")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.hint)
        self._set_hint()
        return holder

    def _set_hint(self, typed=None):
        """The commands work either way; only how you give them changes."""
        if typed is None:
            typed = self.typing.isChecked()
        self.hint.setText(f"{'Type' if typed else 'Say'} "
                          "“skip”, “repeat”, “no sé”, or “stop” at any time.")

    def _on_typing_toggled(self, on):
        """Typing needs no microphone, so it needs no models either."""
        self._set_hint(on)
        self.go.setEnabled(on or self.listener is not None)
        self.placement.setEnabled(on or self.listener is not None)
        self.conjugations.setEnabled(on or self.listener is not None)

    def _settings_panel(self):
        """Everything set once, in a tab of its own.

        These were folded behind a disclosure to keep them out of the drill's
        way. That worked, and left them somewhere nobody ever opened. A tab
        keeps them out of the way just as well without hiding them.
        """
        self.settings_body = QFrame()
        self.settings_body.setObjectName("settings")
        self.settings_body.setStyleSheet(f"#settings{{background:{PANEL};"
                                         f"border:1px solid {RULE};"
                                         f"border-radius:{RADIUS}px;}}")
        grid = QGridLayout(self.settings_body)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        rows = self._setting_rows()
        for at, (caption, widget) in enumerate(rows):
            grid.addWidget(_caption(caption), at, 0, right)
            grid.addWidget(widget, at, 1)

        checks = QHBoxLayout()
        checks.setSpacing(16)
        for check in (self.hints, self.double_check, self.speak_cue):
            check.setStyleSheet(f"color:{MUTE};font:10px {MONO};")
            check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            checks.addWidget(check)
        checks.addStretch(1)
        grid.addLayout(checks, len(rows), 1)
        return self.settings_body

    def _setting_rows(self):
        """Every stored preference, built once, in the order you meet them."""
        deck = load_deck()

        self.mic = QComboBox()
        self.mic.addItem("System default", "")
        for _, name in input_devices():
            self.mic.addItem(name, name)
        stored = self.progress.input_device
        if stored and self.mic.findData(stored) >= 0:
            self.mic.setCurrentIndex(self.mic.findData(stored))

        self.category = QComboBox()
        self.category.addItem(f"everything ({len(deck)})", "all")
        for pos in categories(deck):
            n = sum(1 for c in deck if c.pos == pos)
            self.category.addItem(f"{pos}s only ({n})", pos)
        chosen = self.progress.category or "all"
        if self.category.findData(chosen) >= 0:
            self.category.setCurrentIndex(self.category.findData(chosen))
        # Connected after the stored value is set, so restoring it does not
        # read as the user changing it.
        self.category.currentIndexChanged.connect(self._on_category_changed)

        self.dialect = QComboBox()
        self.dialect.addItems(["es-ES", "es-MX"])   # Spain first: the default
        self.dialect.setCurrentText(self.progress.dialect)

        self.new_per = QSpinBox()
        self.new_per.setRange(0, 100)
        self.new_per.setValue(self.progress.new_per)
        # Refreshes the panel, not just the stored value. Turning this dial
        # used to change nothing you could see: IN QUEUE kept its old number
        # until something else happened to redraw it, so raising the
        # allowance after finishing the day's words looked like a dial that
        # did nothing at all.
        self.new_per.valueChanged.connect(self._on_new_per_changed)

        self.window_seconds = QSpinBox()
        self.window_seconds.setRange(3, 20)
        self.window_seconds.setValue(int(self.progress.window))
        self.window_seconds.setSuffix(" s")
        # The drill re-reads the window before every card, so this one can
        # take effect on the next card rather than the next session.
        self.window_seconds.valueChanged.connect(
            lambda seconds: setattr(self.progress, "window", float(seconds)))

        self.scope = QComboBox()        # filled in by _refresh_scope

        self.hints = QCheckBox("say the answer back")
        self.hints.setChecked(self.progress.hints)
        self.double_check = QCheckBox("double-check misses")
        self.double_check.setChecked(self.progress.verify_live)
        # The cue only. A miss always speaks the answer and its example
        # whatever this says, because that is the part that teaches.
        self.speak_cue = QCheckBox("read the English cue aloud")
        self.speak_cue.setChecked(self.progress.speak_cue)

        for widget in (self.mic, self.category, self.dialect, self.new_per,
                       self.window_seconds, self.scope):
            widget.setStyleSheet(_input_style())
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # What the app is, rather than what this sitting is. The three dials
        # that decide what a session will actually ask live in the session bar
        # instead, in the open, next to the button that starts it.
        return (("MICROPHONE", self.mic), ("ACCENT", self.dialect),
                ("PLACEMENT", self.scope))

    def _on_new_per_changed(self, count):
        self.progress.new_per = int(count)
        self._refresh_counters()
        if not (self.thread and self.thread.isRunning()):
            self._show_idle_state()     # the empty-queue notice may now be stale

    def _on_category_changed(self):
        """The scope counts are per category, so they move with it."""
        self.progress.category = self.category.currentData() or "all"
        self._refresh_scope()
        self._refresh_counters()
        if self.listener is not None:
            self._start_prerecording()      # a new category needs new prompts

    def _watch_for_a_new_day(self):
        """Notice local midnight even while the app sits idle.

        The counters roll over whenever they are read, but nothing reads them
        between sessions, so an app left open overnight would still be showing
        yesterday's tallies in the morning.
        """
        self._day_timer = QTimer(self)
        self._day_timer.timeout.connect(self._refresh_counters)
        self._day_timer.start(60_000)

    def _start_preload(self):
        cached = model_is_cached(self.model_name)
        self.go.setEnabled(False)
        self.placement.setEnabled(False)
        self.sentences.setEnabled(True)     # typed only, so it is ready now
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
        self._start_prerecording()
        # A typing session can already be under way: it does not wait for the
        # models. Re-enabling GO and overwriting the cue mid-card would be a
        # loading routine reaching into a running drill.
        if self.thread and self.thread.isRunning():
            return
        self.go.setEnabled(True)
        self.placement.setEnabled(True)
        self.go.setText(self.WORDS)
        self.status_label.setText("Ready")
        self.prompt_label.setText(self.IDLE_CUE)

    def _start_prerecording(self):
        """Record this category's prompts quietly in the background."""
        deck = load_deck()
        # Locked conjugations are most of the deck and cannot be asked yet.
        # Recording them now would mean thousands of clips for words the drill
        # will not reach for weeks.
        wanted = [i for i in range(len(deck))
                  if self.progress.in_category(i, deck)
                  and (i in self.progress.cards
                       or self.progress.unlocked(i, deck))]
        phrases = voice.phrases_for(deck, self.progress.dialect, wanted)
        # The sentence drill speaks too, and none of its phrases are in the
        # deck. Only the unlocked ones: recording the whole bank would mean
        # paying for sentences the gate will not offer for weeks.
        from .sentences import unfinished
        phrases += voice.sentence_phrases(
            unfinished(self.progress, deck), self.progress.dialect)
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
    def _nothing_to_ask(self):
        """Would a drill start with an empty queue?

        Pressing GO on an empty queue used to start a session, find nothing,
        and end inside a millisecond. The window flashed into the running
        state and straight back out, which reads as the button being broken
        rather than as the day being finished.
        """
        return not self.progress.build_queue()

    def _why_nothing_is_due(self):
        p = self.progress
        reasons = []
        if p.new_per and not p.new_remaining():
            word = "word" if p.new_per == 1 else "words"
            reasons.append(f"today's {p.new_per} new {word} are done")
        elif not p.new_per:
            reasons.append("new words are switched off")
        if p.category not in ("all", "", None):
            reasons.append(f"the drill is limited to {p.category}s")
        if not reasons:
            reasons.append("nothing is scheduled for review yet")
        return " and ".join(reasons).capitalize() + "."

    def _show_idle_state(self):
        """Say where the drill stands while nothing is running."""
        if self.thread and self.thread.isRunning():
            return
        self.slip.setVisible(False)
        self.continue_label.setVisible(False)
        self.heard_label.setText("")
        if self._nothing_to_ask():
            self.state_label.setText("ALL CAUGHT UP")
            self.prompt_label.setText("Nothing to drill.")
            # Reading as a sentence under the headline, not tucked into the
            # bottom corner where a transcript goes.
            self.heard_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.heard_label.setText(
                self._why_nothing_is_due()
                + "  Raise NEW TODAY to keep going.")
            self.status_label.setText("Nothing due")
        else:
            self.state_label.setText("")
            self.prompt_label.setText(self.IDLE_CUE)
            self.heard_label.setText("")
            if self.listener is not None:
                self.status_label.setText("Ready")

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
        if self._nothing_to_ask():
            self._show_idle_state()
            return
        self._start_session(typed=self.typing.isChecked())

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
        self.progress.speak_cue = self.speak_cue.isChecked()
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
        typed = self.typing.isChecked()
        if self.listener is None and not typed:
            self.status_label.setText("Models still loading…")
            return
        self._start_session(placement=True, typed=typed)

    def toggle_sentences(self):
        """Compose whole sentences out of words already in the rotation.

        Typed or spoken, following the same checkbox as the other two. Spoken
        needs the models, so it waits for them; typed never does, which is
        why this button is live from the moment the window opens.
        """
        if self._starting or not self.sentences.isEnabled():
            return
        if self.thread and self.thread.isRunning():
            self._request_stop()
            return
        self._apply_settings()
        typed = self.typing.isChecked()
        if self.listener is None and not typed:
            self.status_label.setText("Models still loading…")
            return
        self._start_session(sentences=True, typed=typed)
        self._start_writing_sentences()

    def toggle_conjugations(self):
        """Drill nothing but conjugated forms, on a schedule of their own.

        A separate sitting rather than a filter on the ordinary drill: it
        keeps its own file, so however long you spend on paradigms, not one
        vocabulary card's review date moves.
        """
        if self._starting or not self.conjugations.isEnabled():
            return
        if self.thread and self.thread.isRunning():
            self._request_stop()
            return
        self._apply_settings()
        typed = self.typing.isChecked()
        if self.listener is None and not typed:
            self.status_label.setText("Models still loading…")
            return
        self._start_session(conjugations=True, typed=typed)

    def _start_session(self, placement=False, typed=False, sentences=False,
                       conjugations=False):
        # Held while the thread is being built. Every toggle reads this,
        # and nothing ever set it, so the guard against a second click
        # reassigning a live QThread has never actually been armed.
        self._starting = True
        try:
            self._build_session(placement, typed, sentences, conjugations)
        finally:
            self._starting = False

    def _build_session(self, placement=False, typed=False, sentences=False,
                       conjugations=False):
        # GO only becomes the stop control for the run it started.
        self.go.setText(self.WORDS if placement or sentences or conjugations
                        else "STOP")
        self.slip.setVisible(False)
        self._hide_progress()
        self.typing.setEnabled(False)       # not mid-session

        # Same reason as the phone: the queue has to be sized from the file
        # as it stands, not from the snapshot this process started with, or
        # the two screens count to different totals for the same deck.
        self.progress.refresh()
        self.progress.roll_over()

        listener = self.listener
        if typed:
            self._typed_input = listener = TypedListener()
            # Only the typing mode holds a miss on screen. The spoken drill is
            # hands-free by design and would stop being so the moment it
            # started waiting for a keypress.
            self._gate = Gate()
            self.answer_box.setVisible(True)
            self.answer_box.setFocus()
        if sentences:
            session = SentenceDrill(
                self.progress, listener, typed=typed,
                # Only the typed mode holds a miss on screen. Spoken is
                # hands-free by design and would stop being so the moment it
                # started waiting for a keypress.
                hold_on_miss=self._gate if typed else None,
                answer_log=None if typed else AnswerLog())
            self.mode_label.setText("SENTENCES" if typed
                                    else "SENTENCES · SPOKEN")
            self.sentences.setText("STOP")
            self.go.setEnabled(False)
            self.placement.setEnabled(False)
            self.conjugations.setEnabled(False)
            self.screen.setStyleSheet(self._screen_style(HIT))
        elif conjugations:
            # Its own tracker and its own log. `--review` repairs cards in
            # whichever pair it is handed, so sharing either one would let a
            # conjugation re-check write into the vocabulary schedule.
            session = ConjugationSession(
                ConjugationProgress.open(self.progress), listener, typed=typed,
                # Its own second opinion, steered for a conjugated form.
                verifier=None if typed else _conjugation_verifier,
                hold_on_miss=self._gate if typed else None,
                answer_log=AnswerLog(path=CONJUGATION_LOG))
            self.mode_label.setText("CONJUGATIONS · TYPING" if typed
                                    else "CONJUGATIONS")
            self.conjugations.setText("STOP")
            self.go.setEnabled(False)
            self.placement.setEnabled(False)
            self.sentences.setEnabled(False)
            self.screen.setStyleSheet(self._screen_style(HIT))
        elif placement:
            session = PlacementSession(
                self.progress, listener, typed=typed,
                verifier=None if typed else _api_verifier,
                hold_on_miss=self._gate if typed else None,
                retest=self.scope.currentData() == "all")
            self.mode_label.setText("PLACEMENT · TYPING" if typed
                                    else "PLACEMENT TEST")
            self.placement.setText("STOP TEST")
            self.go.setEnabled(False)
            self.sentences.setEnabled(False)
            self.conjugations.setEnabled(False)
            self.screen.setStyleSheet(self._screen_style(SIGNAL))
        else:
            session = DrillSession(self.progress, listener, typed=typed,
                                   verifier=None if typed else _api_verifier,
                                   hold_on_miss=self._gate if typed else None)
            self.mode_label.setText("TYPING" if typed else "")
            self.placement.setEnabled(False)
            self.sentences.setEnabled(False)
            self.conjugations.setEnabled(False)
            self.screen.setStyleSheet(self._screen_style())
        self._set_hint(typed)
        self.worker = SessionWorker(session)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.prompt.connect(
            self._on_sentence_prompt if sentences else self._on_prompt)
        self.worker.status.connect(self._on_status)
        self.worker.heard.connect(self.heard_label.setText)
        self.worker.result.connect(
            self._on_sentence_result if sentences else self._on_result)
        self.worker.counts.connect(self._refresh_counters)
        self.worker.progress.connect(self._on_progress)
        self.worker.verified.connect(self._on_verified)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _start_writing_sentences(self):
        """Ask the API for more, in the background, while the drill runs.

        Skipped without a key, which is the offline case: the curated bank
        needs no network and the mode works perfectly well on it alone.
        """
        import os
        from .config import SENTENCE_GENERATION
        if not SENTENCE_GENERATION:
            return              # curated bank only; see config for why
        if not os.environ.get("OPENAI_API_KEY"):
            return
        from .generate import remaining_allowance
        if remaining_allowance() <= 0:
            return              # the ceiling has been reached; never spend again
        if self._makers:
            return              # one is already writing
        session = getattr(self.worker, "session", None)
        if session is not None:
            session.expecting_more = True
        maker, thread = SentenceMaker(self.progress), QThread()
        maker.moveToThread(thread)
        thread.started.connect(maker.run)
        maker.made.connect(self._on_sentences_made)
        maker.note.connect(self._on_sentences_note)
        maker.done.connect(thread.quit)
        pair = (maker, thread)
        self._makers.append(pair)
        # Dropped only once the thread has really ended, never on request.
        thread.finished.connect(
            lambda: pair in self._makers and self._makers.remove(pair))
        thread.start()

    def _on_sentences_made(self, arriving):
        session = getattr(self.worker, "session", None)
        if isinstance(session, SentenceDrill):
            session.add(arriving)

    def _on_sentences_note(self, text):
        self.tally.setText(f"SENTENCES WRITTEN:  {text.upper()}")

    def _stop_writing_sentences(self, wait=False):
        """Ask the writer to stop, without blocking on a call in flight.

        It notices between batches. Waiting here would freeze the window for
        as long as an API round trip, so the thread is left to finish on its
        own and only the reference is kept, which is what stops Qt from
        destroying a running thread underneath itself.
        """
        for maker, thread in list(self._makers):
            maker.stop()
            thread.quit()
            if wait:
                thread.wait(8000)       # only at shutdown, where it is fine

    def _request_stop(self):
        # The worker can be mid-speech or mid-transcription and takes a moment
        # to notice. Say so and refuse clicks until it has really finished,
        # rather than showing GO and silently ignoring presses.
        if not (self.thread and self.thread.isRunning()):
            return          # already finished; nothing to wind down
        self.worker.stop()
        self._stop_writing_sentences()
        if self._gate is not None:
            self._gate.release()    # a card held on screen must not block the stop
        if self.placement.text() == "STOP TEST":
            stopping = self.placement
        elif self.sentences.text() == "STOP":
            stopping = self.sentences
        elif self.conjugations.text() == "STOP":
            stopping = self.conjugations
        else:
            stopping = self.go
        stopping.setText("STOPPING…")
        self.go.setEnabled(False)
        self.placement.setEnabled(False)
        self.sentences.setEnabled(False)
        self.conjugations.setEnabled(False)
        self.status_label.setText("Stopping")

    # -- updates ----------------------------------------------------------
    def _on_status(self, text):
        """Show what the drill is doing, and how long is left to answer."""
        self.status_label.setText(text)
        if text == "Type it" and self._typed_input is not None:
            self._deadline = time.monotonic() + self.progress.window
            self._countdown.start(100)
        else:
            self._countdown.stop()

    def _tick_countdown(self):
        left = self._deadline - time.monotonic()
        if left <= 0:
            self._countdown.stop()
            return                  # the worker decides it ran out, not this
        self.status_label.setText(f"Type it · {left:.1f}s")

    def _on_prompt(self, card, state):
        # The subject is marked on a conjugated cue, but only here. In the
        # conjugations drill every card has one, so colouring it says nothing
        # and the screen just turns red; mixed in with vocabulary it is the
        # one word that decides the answer.
        subject = "" if isinstance(getattr(self.worker, "session", None),
                                   ConjugationSession) else card.subject
        if subject:
            self.prompt_label.setTextFormat(Qt.TextFormat.RichText)
            self.prompt_label.setText(
                f"<span style='color:{MISS}'>{escape(subject)}</span>"
                f"{escape(card.prompt[len(subject):])}")
        else:
            self.prompt_label.setTextFormat(Qt.TextFormat.PlainText)
            self.prompt_label.setText(card.prompt)
        self.state_label.setText(state.upper())
        self.slip.setVisible(False)
        self.continue_label.setVisible(False)
        self.answer_box.clear()     # no card starts with the last one's text
        # Back to the transcript corner; the idle notice borrows this label.
        self.heard_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        if self.answer_box.isVisible():
            # Clicking anywhere in the window would otherwise leave the next
            # card with nothing to type into.
            self.answer_box.setFocus()

    def _on_sentence_prompt(self, sentence, state):
        """The English cue, with nothing about the Spanish on screen."""
        self.slip.setVisible(False)
        self.continue_label.setVisible(False)
        self.answer_box.clear()
        self.heard_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.heard_label.setText("")
        if sentence is None:
            # Nothing is unlocked. Say which words would open the bank rather
            # than showing an empty screen, which reads as a broken mode.
            session = getattr(self.worker, "session", None)
            self.state_label.setText("NOTHING UNLOCKED")
            self.prompt_label.setText("No sentences yet.")
            self.heard_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.heard_label.setText(
                session.why_empty() if session else "")
            return
        self.state_label.setText(state.upper())
        self.prompt_label.setText(sentence.en)
        self.answer_box.setFocus()

    def _on_sentence_result(self, result):
        """Perfect, or the sentence marked up word by word."""
        grade = result.grade
        # When a miss is overturned the transcript on screen is the wrong
        # one. Leaving it up makes a correct answer look like a bad grade.
        if result.overturned and result.api_text:
            self.heard_label.setText(
                f"{result.api_text}   (local heard “{result.said}”)")
        colour = HIT if grade.perfect else MISS
        self.slip.setStyleSheet(
            f"#slip{{background:{PANEL};border-radius:{RADIUS}px;"
            f"border-left:3px solid {colour};}}")
        self.verdict_label.setStyleSheet(
            f"background:transparent;color:{colour};font:10px {MONO};"
            "letter-spacing:2px;")
        self.verdict_label.setText(
            "PERFECT — LOCAL MODEL MISHEARD YOU" if result.overturned
            else "PERFECT" if grade.perfect else "NOT QUITE")

        # The whole correct sentence, always. Even on a perfect answer it is
        # worth seeing written out with its accents, which are not graded.
        self.answer_label.setText(grade.expected)
        self.diff_label.setText(self._diff_html(grade))
        self.diff_label.setVisible(not grade.perfect)
        for label in (self.example_label, self.gloss_label):
            label.setVisible(False)

        detail = [f"{grade.mistakes} to fix"] if not grade.perfect else []
        detail.append(f"{result.elapsed:.0f}s")
        if result.api_text and not result.overturned:
            detail.append(f"re-check heard “{result.api_text}”")
        self.detail_label.setText("  ·  ".join(detail))
        self.continue_label.setVisible(not grade.perfect and self._gate is not None)
        self.slip.setVisible(True)

    @staticmethod
    def _diff_html(grade):
        """Right words plain, wrong ones carrying what was written instead.

        Three states worth telling apart and they are not the same mistake: a
        word written wrong, a word left out, and a word invented. Colouring
        them the same would say "you got this wrong" without saying how.
        """
        from html import escape
        parts = []
        for token in grade.marked:
            if token.state == "right":
                parts.append(f'<span style="color:{SCREEN}">'
                             f'{escape(token.text)}</span>')
            elif token.state == "wrong":
                parts.append(
                    f'<span style="color:{MISS};font-weight:700">'
                    f'{escape(token.text)}</span>'
                    f'<span style="color:{MUTE}">&#8202;('
                    f'{escape(token.typed)})</span>')
            else:
                parts.append(f'<span style="color:{SIGNAL};font-weight:700">'
                             f'{escape(token.text)}</span>')
        line = " ".join(parts)
        if grade.extra:
            line += (f'<br><span style="color:{MUTE}">not in it: </span>'
                     f'<span style="color:{MISS}">'
                     f'{escape(" ".join(grade.extra))}</span>')
        return line

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
            f"#slip{{background:{PANEL};border-radius:{RADIUS}px;"
            f"border-left:3px solid {colour};}}")
        self.verdict_label.setStyleSheet(
            f"background:transparent;color:{colour};font:10px {MONO};"
            "letter-spacing:2px;")
        self.verdict_label.setText(verdict.upper())

        answers = result.card.answers
        extra = f"   (also: {', '.join(answers[1:])})" if len(answers) > 1 else ""
        self.answer_label.setText(answers[0] + extra)
        # Every card carries a sentence now, conjugations included. Hidden
        # rather than blank anyway: an empty label still takes its line and
        # leaves a hole in the middle of the slip.
        for label, text in ((self.example_label, result.card.example),
                            (self.gloss_label, result.card.gloss)):
            label.setText(text)
            label.setVisible(bool(text))

        detail = [result.next_review, f"ease {result.state.ease:.2f}"]
        if result.said:
            detail.append(f"you said: {result.said}")
        if result.state.lapses:
            detail.append(f"missed {result.state.lapses}x")
        if result.api_text:
            detail.append(f"re-check heard “{result.api_text}”")
        self.detail_label.setText("  ·  ".join(detail))
        # A missed card is held on screen until it has been read, but only in
        # the mode that has a gate to hold it. Shown from here rather than
        # when the wait begins, so it is already up by the time the drill
        # stops rather than appearing a beat later.
        self.continue_label.setVisible(not result.correct and self._gate is not None)
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
        if isinstance(session, SentenceDrill):
            done = session.summary()
            if done["asked"]:
                self.prompt_label.setText(
                    f"{done['perfect']} of {done['total']} sentences perfect")
        if isinstance(session, ConjugationSession):
            # Verbs finished, not forms answered: the unit the batch moves in.
            s = session.summary()
            self.prompt_label.setText(
                f"{len(s['done'])} verbs complete · {s['remaining']} to go")
        self.go.setText(self.WORDS)
        self.go.setEnabled(self.listener is not None)
        self.placement.setEnabled(self.listener is not None)
        self.placement.setText("PLACEMENT TEST")
        self.sentences.setText("SENTENCES")
        self.sentences.setEnabled(True)     # never needs the speech models
        self.conjugations.setText("CONJUGATIONS")
        self.conjugations.setEnabled(self.listener is not None
                                     or self.typing.isChecked())
        self.diff_label.setVisible(False)
        self._stop_writing_sentences()      # no point paying for a finished run
        self.typing.setEnabled(True)
        self.answer_box.setVisible(False)
        self.answer_box.clear()
        self.continue_label.setVisible(False)
        self._typed_input = None
        self._gate = None
        self.mode_label.setText("")
        self._set_hint()
        self.screen.setStyleSheet(self._screen_style())
        self._refresh_scope()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self._refresh_counters()
        self._refresh_tally()

    def _refresh_counters(self):
        """Both today's tallies and the shape of the deck behind them.

        Each caption carries the number its headline needs to be read against:
        words learned means little without how many are left in the day's
        allowance, and a count of cards in progress means little without how
        many of them have actually stuck.
        """
        p = self._active_progress()
        # Anything answered on the phone. Both sides hold their own copy of
        # progress.json, so without this the panel here shows a day's work
        # missing whatever was done on the sofa.
        p.refresh()
        if p.roll_over():           # the panel must not show a stale day
            p.save()
        self.counters["learned"].setText(str(p.new_done))
        # The allowance is named, not just what is left of it. Lowering new/day
        # below what has already been done leaves nothing left, and "40 · 0
        # LEFT" next to a dial reading 10 looks like the count is broken.
        self.counter_labels["learned"].setText(
            f"LEARNED · {p.new_remaining()}/{p.new_per} LEFT")
        self.counters["reviews"].setText(str(p.reviews_done))
        self.counter_labels["reviews"].setText("REVIEWED TODAY")
        due, fresh = p.queue_parts()
        self.counters["due"].setText(str(len(due) + len(fresh)))
        self.counter_labels["due"].setText(
            f"IN QUEUE · {len(fresh)} NEW" if fresh else "IN QUEUE")
        self.counters["missed"].setText(str(p.missed_today))
        self.counters["learning"].setText(str(p.learning_count()))
        self.counter_labels["learning"].setText(
            f"LEARNING · {p.mature_count()} MATURE")
        self.ladder.setText(self._ladder_text())

    def _active_progress(self):
        """Whichever schedule is being moved right now.

        The conjugation drill keeps its own, and the panel has to follow it
        or the counters spend the whole run describing a deck nobody is
        drilling: due totals from the vocabulary tracker while every answer
        moves the other one.
        """
        session = getattr(self.worker, "session", None)
        if self.thread and self.thread.isRunning() and session is not None:
            return getattr(session, "progress", self.progress)
        return self.progress

    def _ladder_text(self):
        """Where the deck actually sits on the ladder, step by step.

        A single "learning" total hides the shape: ninety-nine cards in
        progress reads like steady work whether they are all still on one day
        or spread out to three weeks. Ordered by interval so the deck is read
        left to right, soonest to furthest out.
        """
        stages = self._active_progress().due_by_stage()

        def position(step):
            # dict.get would evaluate int("mature") before finding the key
            if step == "relearning":
                return -1
            return 10 ** 6 if step == "mature" else int(step.rstrip("d"))

        steps = sorted((k for k in stages if k != "new"), key=position)
        if not steps:
            return "LADDER  nothing scheduled yet"
        label = {"relearning": "RELEARNING", "mature": "21d+"}
        return "LADDER  " + "  ".join(
            f"{label.get(k, k)} {stages[k]}" for k in steps)

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
        self._stop_writing_sentences(wait=True)
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            if self._gate is not None:
                self._gate.release()    # never wait on a keypress to shut down
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
