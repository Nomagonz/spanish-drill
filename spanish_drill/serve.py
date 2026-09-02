"""Drill from a phone, with everything that matters still on this machine.

The browser is a keyboard and a screen and nothing else. The deck, the
schedule, the grading and the sentence bank never leave the Mac; what crosses
the wire is a cue going out and a typed answer coming back.

No web framework, on purpose. Server-sent events plus a form POST is the whole
protocol, and both are in the standard library, so running the drill remotely
adds no dependency to a project whose selling point is that it works offline.

    ./run.sh --serve            then open the printed URL on the phone

Reaching it from outside the house is deliberately not this module's problem.
Point a tunnel at the port; the token in the URL is what keeps a stranger who
finds the tunnel from drilling your deck.
"""
import json
import pathlib
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import SERVE_TOKEN_PATH
from .composition import SentenceDrill
from .deck import categories, load_deck
from .config import CONJUGATION_LOG
from .answers import AnswerLog
from .paradigm import ConjugationProgress, ConjugationSession
from .placement import PlacementSession
from .progress import Progress
from .session import DrillSession
from .typed import Gate, TypedListener

# How long a poll waits for something to happen before answering "nothing
# yet". Long enough that an idle drill is not a busy loop, short enough that
# no proxy anywhere decides the request has stalled.
POLL_SECONDS = 20
BACKLOG = 200               # events kept for a browser that joins late


class Hub:
    """One drill, however many browsers are watching it.

    Everything the session emits is fanned out to every connected client, so
    opening the page on a second device shows the same card rather than
    starting a competing session. Only one drill runs at a time.
    """

    def __init__(self, progress=None):
        self.progress = progress or Progress.load()
        self.deck = load_deck()
        # A numbered log rather than a stream per browser. Server-sent events
        # worked perfectly on this machine and died behind a Cloudflare quick
        # tunnel: a 200, the right headers, and no body however long you
        # waited, because a proxy cannot forward a response whose end it
        # cannot see. Every answer here is a complete, bounded response, and
        # those go through anything.
        self.events = []                # (sequence number, json string)
        self.seq = 0
        self.changed = threading.Condition()
        self.session = None
        self.thread = None
        # Whether a drill is meant to be going, as distinct from whether the
        # thread has finished unwinding. `on_finished` is raised from inside
        # that thread, so asking `is_alive()` there always says yes and the
        # panel published a beat later contradicted the finish it had just
        # sent. That is what left PAUSE and STOP showing after a stop.
        self.live = False
        self.typed = None
        self.gate = None
        self.scope = "new"              # placement: only words never sorted
        # A miss is on screen and the next Enter belongs to it, not to the
        # answer box. Set before the result is published, because the phone
        # can answer faster than the drill reaches its own hold. A correct
        # answer never sets it: that card is already gone.
        self._hold_pending = False

    # -- the log ----------------------------------------------------------
    def publish(self, event, **data):
        message = json.dumps({"event": event, **data})
        with self.changed:
            self.seq += 1
            self.events.append((self.seq, message))
            del self.events[:-BACKLOG]
            self.changed.notify_all()

    def since(self, mark, timeout=POLL_SECONDS):
        """Everything after `mark`, waiting up to `timeout` for something.

        A browser that joins late passes 0 and is caught up from the backlog,
        so opening the page mid-card shows the card rather than a blank
        screen until the next one.
        """
        deadline = time.time() + timeout
        with self.changed:
            while True:
                fresh = [m for n, m in self.events if n > mark]
                if fresh or time.time() >= deadline:
                    return self.seq, fresh
                self.changed.wait(min(1.0, max(0.0, deadline - time.time())))

    # -- what the browser can ask for -------------------------------------
    def state(self):
        """Everything the panel shows, from the same methods the window uses.

        Read off `Progress` rather than recomputed here, so the phone and the
        window cannot end up disagreeing about the same deck.
        """
        from . import sentences as S
        # Whichever schedule is being moved right now. The conjugation drill
        # keeps its own file, and a panel that always read the vocabulary one
        # would spend that whole mode describing a deck nobody is drilling —
        # which is exactly why the phone's numbers did not match the window's.
        p = self.active_progress()
        # Whatever the app on the desk has done since we last looked. Cheap:
        # a stat, and a read only when the file has actually moved.
        p.refresh()
        p.roll_over()
        due, fresh = p.queue_parts()
        deck = self.deck
        unsorted, in_scope = p.placement_scope(deck)
        total = p.kept + p.overturned
        # What the conjugation drill has waiting. It was the one mode whose
        # button could say nothing about what pressing it would do, because
        # this panel only ever described the schedule in hand. Measured at
        # about two milliseconds, so it is asked every time rather than
        # cached into something that can go stale.
        if isinstance(p, ConjugationProgress):
            forms_due = len(due) + len(fresh)   # already the one in hand
        else:
            paradigms = ConjugationProgress.open(p)
            waiting, opening = paradigms.queue_parts()
            forms_due = len(waiting) + len(opening)
        return {
            "sentences": len(S.unfinished(p, deck)),
            "bank": len(S.load_sentences()),
            "words_due": len(due) + len(fresh),
            "forms_due": forms_due,
            "new_in_queue": len(fresh),
            "learned": p.new_done,
            "new_left": p.new_remaining(),
            "new_per": p.new_per,
            "reviews": p.reviews_done,
            "missed": p.missed_today,
            "learning": p.learning_count(),
            "mature": p.mature_count(),
            "ladder": p.ladder_steps(),
            "kept": p.kept,
            "overturned": p.overturned,
            "overturned_share": (round(100 * p.overturned / total)
                                 if total else None),
            "category": p.category,
            "categories": [["all", f"everything ({len(deck)})"]] + [
                [pos, f"{pos}s only ({sum(1 for c in deck if c.pos == pos)})"]
                for pos in categories(deck)],
            "window": p.window,
            "dialect": p.dialect,
            "unsorted": unsorted,
            "in_scope": in_scope,
            "why_empty": p.why_nothing_is_due(),
            "nothing_due": not (due or fresh),
            "running": bool(self.live and self.thread and self.thread.is_alive()),
            "paused": self.paused,
        }

    def active_progress(self):
        """The schedule the running drill is actually moving."""
        if self.thread and self.thread.is_alive() and self.session is not None:
            return getattr(self.session, "progress", self.progress)
        return self.progress

    def configure(self, **settings):
        """The dials, changed from the phone. Saved at once.

        The answer window is re-read before every card, so it lands on the
        next one rather than the next session.
        """
        p = self.progress
        if "category" in settings:
            p.category = settings["category"] or "all"
        if "new_per" in settings:
            p.new_per = max(0, min(100, int(settings["new_per"])))
        if "window" in settings:
            p.window = float(max(3, min(20, int(settings["window"]))))
        if "dialect" in settings and settings["dialect"] in ("es-ES", "es-MX"):
            p.dialect = settings["dialect"]
        if "scope" in settings:
            self.scope = settings["scope"]
        p.save()
        self.announce()

    def announce(self):
        self.publish("ready", **self.state())

    def start_of_day(self):
        """The first thing a fresh browser should be told."""
        self.announce()

    def start(self, mode="sentences"):
        """Begin a mode, whatever is running now.

        Refusing while something else was going meant the only way out of the
        sentence drill was to find STOP first, and a session that stuck for
        any reason locked every button on the page. Pressing a mode is an
        unambiguous instruction to be in that mode.
        """
        if self.thread and self.thread.is_alive():
            self.stop()
            self.thread.join(timeout=5)
        self.typed = TypedListener()
        self.gate = Gate()
        # The queue is about to be built, and it must be built from what is
        # on disk right now. Both sides hold their own copy of progress.json
        # and only the panel ever re-read it, so a drill started here sized
        # itself from whatever this process happened to boot with: the phone
        # and the window would sit side by side counting to different totals.
        self.progress.refresh()
        self.progress.roll_over()
        if mode == "conjugations":
            # Its own tracker and its own log: `--review` repairs cards in
            # whichever pair it is handed, so sharing either would let a
            # conjugation re-check write into the vocabulary schedule.
            self.session = ConjugationSession(
                ConjugationProgress.open(self.progress), self.typed,
                typed=True, verifier=None, hold_on_miss=self.gate,
                answer_log=AnswerLog(path=CONJUGATION_LOG))
            self.session.on_result = self._word_result
        elif mode == "placement":
            self.session = PlacementSession(
                self.progress, self.typed, typed=True, verifier=None,
                hold_on_miss=self.gate, retest=self.scope == "all")
            self.session.on_result = self._word_result
        elif mode == "words":
            self.session = DrillSession(self.progress, self.typed, typed=True,
                                        verifier=None, hold_on_miss=self.gate)
            self.session.on_result = self._word_result
        else:
            self.session = SentenceDrill(self.progress, self.typed,
                                         hold_on_miss=self.gate)
            self.session.on_result = self._sentence_result
        self.session.on_prompt = self._prompt
        self.session.on_status = lambda text: self.publish("status", text=text)
        # Bound to this session rather than read off self, so a run that is
        # replaced mid-flight cannot report the next run's numbers.
        session = self.session
        self.session.on_progress = lambda done, total: self.publish(
            "progress", done=done, total=total,
            skipped=getattr(session, "skipped", 0))
        self.session.on_counts = self.announce
        self.session.on_finished = self._finished
        self.thread = threading.Thread(target=self.session.run, daemon=True)
        self.live = True
        self.thread.start()
        # Say so at once. The panel reports `running` from whether this thread
        # is alive, and nothing else here published one after starting it, so
        # the browser kept the answer it was given before the drill began:
        # not running. PAUSE and STOP are driven off that, which left them
        # disabled for the whole of a session and no way to end one.
        self.announce()
        return True

    def stop(self):
        if self.typed is not None:
            self.typed.resume()     # a paused drill must still be stoppable
        if self.session:
            self.session.stop()
        if self.gate:
            self.gate.release()     # a held card must not block the stop

    def pause(self, on=None):
        """Hold the drill where it is. The card stays, the clock stops.

        Not a stop: the session, its queue and the learning ladder it has
        built up all survive, so resuming carries on with the same card
        rather than picking a fresh one.
        """
        if self.typed is None:
            return False
        want = (not self.typed.paused) if on is None else bool(on)
        self.typed.pause() if want else self.typed.resume()
        self.publish("status", text="Paused" if want else "Carry on")
        self.publish("paused", paused=want)
        return want

    @property
    def paused(self):
        return bool(self.typed is not None and self.typed.paused)

    def answer(self, text):
        """One Enter. Releases a held card, or hands over what was typed."""
        if self.gate is not None and (self._hold_pending or self.gate.waiting):
            self._hold_pending = False
            self.gate.release()
            return
        if self.typed is not None:
            self.typed.submit(text)

    # -- session callbacks ------------------------------------------------
    def _prompt(self, item, label):
        self._hold_pending = False      # a new card: nothing is being held
        if item is None:            # the sentence drill found nothing unlocked
            why = getattr(self.session, "why_empty", lambda: "")()
            self.publish("prompt", text="Nothing unlocked.", label="",
                         note=why)
            return
        cue = getattr(item, "en", None) or getattr(item, "prompt", "")
        # Marked on screen: on a conjugated cue it is the one word that
        # decides the answer, and it arrives between vocabulary cards that
        # have no subject at all. Empty on everything else.
        self.publish("prompt", text=cue, label=label, note="",
                     subject=getattr(item, "subject", ""))

    def _sentence_result(self, result):
        g = result.grade
        # Only a miss waits, and the drill itself is what holds it. Set
        # before the result goes out because the phone can send the
        # releasing Enter before the drill has reached its own hold.
        self._hold_pending = not g.perfect
        self.publish("result", perfect=g.perfect, expected=g.expected,
                     mistakes=g.mistakes, extra=list(g.extra),
                     marked=[{"text": t.text, "state": t.state,
                              "typed": t.typed} for t in g.marked],
                     hold=not g.perfect)

    def _word_result(self, result):
        answers = list(result.card.answers)
        state = result.state
        detail = [result.next_review, f"ease {state.ease:.2f}"]
        if result.said:
            detail.append(f"you typed: {result.said}")
        if state.lapses:
            detail.append(f"missed {state.lapses}x")
        self._hold_pending = not result.correct
        self.publish("result", perfect=result.correct, expected=answers[0],
                     mistakes=0 if result.correct else 1, extra=[],
                     marked=[], hold=not result.correct,
                     example=result.card.example, gloss=result.card.gloss,
                     alternates=answers[1:], next_review=result.next_review,
                     detail="  ·  ".join(detail),
                     close=result.close, silent=result.silent)

    def _finished(self):
        # Before either goes out, so the panel that follows agrees with it.
        self.live = False
        self.publish("finished")
        self.announce()


PAGE = """<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#1E2B26">
<title>Drill ES</title><style>
/* One scale, one palette, named once. The page grew a handful of ad-hoc
   greens and paddings that were nearly but not quite the same, which is most
   of why it read as untidy rather than as anything specific. */
:root{
  --ink:#ECE5D3; --mute:#8FA096; --dim:#5A665F;
  --bg:#1E2B26; --panel:#2B3B34; --rule:#3C4F47; --raised:#33463E;
  --signal:#E19B33; --hit:#6E9A6B; --miss:#C24B36;
  --mono:ui-monospace,Menlo,monospace;
  --r:6px;
  /* A tap target under about 44px is a miss on a phone, so controls are
     sized from this rather than from whatever the text happened to need. */
  --tap:46px;
}
*{box-sizing:border-box}
html,body{margin:0}
body{
  background:var(--bg); color:var(--ink);
  font:16px -apple-system,'Helvetica Neue',sans-serif;
  -webkit-text-size-adjust:100%;
  padding:0 12px calc(16px + env(safe-area-inset-bottom));
  max-width:640px; margin:0 auto;
}
.mono{font-family:var(--mono)}

/* ---------- header: always visible, because a running drill has to be
     readable from whichever tab you are on ---------- */
header{
  position:sticky; top:0; z-index:5; background:var(--bg);
  padding:calc(10px + env(safe-area-inset-top)) 0 8px;
}
.brand{
  display:flex; align-items:baseline; gap:8px;
  font:10px var(--mono); letter-spacing:2px; color:var(--mute);
}
.brand .name{color:var(--ink); font-weight:700}
.brand .tail{margin-left:auto; display:flex; align-items:baseline; gap:10px;
  min-width:0}
#mode{color:var(--signal); font-weight:700}
#count{letter-spacing:1px; white-space:nowrap}
#status{white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  max-width:52vw; text-align:right}
#track{height:3px; background:var(--panel); border-radius:2px;
  margin-top:8px; overflow:hidden; display:none}
#fill{height:100%; width:0; background:var(--signal);
  transition:width .25s ease}

/* ---------- tabs ---------- */
.tabs{
  display:grid; grid-template-columns:repeat(3,1fr); gap:4px;
  background:var(--panel); border:1px solid var(--rule);
  border-radius:var(--r); padding:4px; margin-top:10px;
}
.tab{
  appearance:none; -webkit-appearance:none; border:none; background:none;
  color:var(--mute); font:700 10px var(--mono); letter-spacing:1.5px;
  padding:10px 4px; border-radius:4px; min-height:38px;
}
.tab[aria-selected="true"]{background:var(--raised); color:var(--ink)}
.tab .dot{
  display:inline-block; min-width:16px; margin-left:5px; padding:0 4px;
  border-radius:8px; background:var(--signal); color:#20160A;
  font-size:9px; line-height:14px;
}
.tab .dot:empty{display:none}
.page{display:none; padding-top:12px}
.page.on{display:block}

/* ---------- the card being asked ---------- */
.card{
  background:var(--ink); color:#141A17; border-radius:var(--r); padding:22px 20px;
  min-height:132px; display:flex; flex-direction:column; justify-content:center;
}
.label{font:9px var(--mono); letter-spacing:2px; color:var(--dim);
  margin-bottom:10px; min-height:11px}
.cue{font-size:25px; font-weight:700; line-height:1.25}
.note{font-size:13px; color:var(--dim); margin-top:10px}
.subject{color:var(--miss)}

input,select{
  width:100%; padding:13px; border-radius:var(--r);
  border:1px solid var(--rule); background:var(--panel); color:var(--ink);
  font:600 17px var(--mono); -webkit-appearance:none; min-height:var(--tap);
}
select{font-size:14px; padding-right:34px}
.sel{position:relative}
.sel::after{
  content:""; position:absolute; right:15px; top:50%; margin-top:-5px;
  width:7px; height:7px; pointer-events:none; transform:rotate(45deg);
  border-right:2px solid var(--mute); border-bottom:2px solid var(--mute);
}
input:focus,select:focus{outline:none; border-color:var(--signal)}
#box{margin-top:10px}

/* ---------- what the answer was worth ---------- */
.slip{
  background:var(--panel); border-radius:var(--r); padding:14px;
  margin-top:10px; border-left:3px solid var(--hit); display:none;
}
.slip.bad{border-left-color:var(--miss)}
.verdict{font:10px var(--mono); letter-spacing:2px; color:var(--hit)}
.slip.bad .verdict{color:var(--miss)}
.answer{font-size:20px; font-weight:700; margin:6px 0}
.diff{font:15px var(--mono); line-height:1.7}
.right{color:var(--ink)} .wrong{color:var(--miss); font-weight:700}
.missing{color:var(--signal); font-weight:700} .you{color:var(--mute)}
.ex{font-size:13px; margin-top:4px} .gl{font-size:12px; color:var(--mute)}
.detail{font:10.5px var(--mono); color:var(--mute); margin-top:8px;
  line-height:1.6}

/* ---------- controls ---------- */
.controls{margin-top:14px}
.legend{
  font:9px var(--mono); letter-spacing:2px; color:var(--mute);
  margin:0 0 8px 2px;
}
/* Each mode says what it is and how much of it is waiting. The old row of
   four bare words gave no way to tell what any of them would do, and the
   ordinary vocabulary drill was labelled GO, which read as a start button
   rather than as one of the four modes. */
.modes{display:grid; gap:8px}
.mode{
  display:flex; align-items:center; gap:12px; width:100%; text-align:left;
  padding:13px 14px; min-height:var(--tap);
  background:var(--panel); border:1px solid var(--rule);
  border-radius:var(--r); color:var(--ink);
}
.mode:active{background:var(--raised)}
.mode.lead{border-color:var(--signal)}
.mode .what{flex:1; min-width:0}
.mode .name{display:block; font:700 12px var(--mono); letter-spacing:1.5px}
.mode.lead .name{color:var(--signal)}
/* Sans, and its own line. Inheriting the button's monospace made a plain
   English sentence read like a status code. */
.mode .desc{display:block; margin-top:4px; line-height:1.4;
  font:12px -apple-system,'Helvetica Neue',sans-serif; color:var(--mute)}
.mode .many{
  font:700 15px var(--mono); color:var(--signal); white-space:nowrap;
}
.mode .many small{display:block; font:8px var(--mono); color:var(--mute);
  letter-spacing:1px; font-weight:400; text-align:right; margin-top:2px}
.running{display:grid; grid-template-columns:1fr 1fr; gap:8px;
  margin-bottom:16px}
button{
  padding:14px; border-radius:var(--r); background:var(--panel);
  color:var(--ink); border:1px solid var(--rule);
  font:700 11px var(--mono); letter-spacing:1.5px; min-height:var(--tap);
}
button:disabled{opacity:.35}
button.stop{border-color:var(--miss); color:var(--miss)}

/* ---------- progress ---------- */
/* Two rows of readable figures rather than five columns squeezed into a
   phone, where the captions had shrunk to 7.5px to fit at all. */
.figures{display:grid; grid-template-columns:repeat(2,1fr); gap:8px}
.figure{
  background:var(--panel); border:1px solid var(--rule);
  border-radius:var(--r); padding:13px 14px;
}
.figure.wide{grid-column:1 / -1}
.figure b{display:block; font-size:27px; font-weight:700; line-height:1.1}
.figure span{display:block; font:9px var(--mono); letter-spacing:1px;
  color:var(--mute); margin-top:5px}
.block{
  background:var(--panel); border:1px solid var(--rule);
  border-radius:var(--r); padding:14px; margin-top:10px;
}
.block h2{font:9px var(--mono); letter-spacing:2px; color:var(--mute);
  margin:0 0 8px; font-weight:400}
.block .body{font:12px var(--mono); color:var(--mute); line-height:1.9}
.rung{display:inline-block; margin-right:18px; white-space:nowrap}
.rung b{color:var(--ink); margin-left:6px}

/* ---------- settings ---------- */
/* Label above the control. Right-aligned labels in a fixed 96px column
   truncated the longer ones and left the inputs starting at a random x. */
.field{margin-bottom:14px}
.field label{display:block; font:9px var(--mono); letter-spacing:1.5px;
  color:var(--mute); margin-bottom:6px}
.field .hint{font-size:11px; color:var(--dim); margin-top:5px; line-height:1.4}
</style></head><body>

<header>
  <div class="brand">
    <span class="name">DRILL &middot; ES</span><span id="mode"></span>
    <span class="tail"><span id="count"></span><span id="status">connecting</span></span>
  </div>
  <div id="track"><div id="fill"></div></div>
</header>

<div class="tabs" role="tablist">
  <button class="tab" role="tab" id="tab_drill" aria-selected="true">DRILL</button>
  <button class="tab" role="tab" id="tab_progress" aria-selected="false">PROGRESS<span class="dot" id="dot_due"></span></button>
  <button class="tab" role="tab" id="tab_settings" aria-selected="false">SETTINGS</button>
</div>

<section class="page on" id="page_drill">
  <div class="card">
    <div class="label" id="label"></div>
    <div class="cue" id="cue">Pick a drill below to start.</div>
    <div class="note" id="note"></div>
  </div>

  <input id="box" placeholder="type the Spanish, then Enter" autocomplete="off"
   autocorrect="off" autocapitalize="off" spellcheck="false" enterkeyhint="go">

  <div class="slip" id="slip">
    <div class="verdict" id="verdict"></div>
    <div class="answer" id="answer"></div>
    <div class="diff" id="diff"></div>
    <div class="ex" id="ex"></div>
    <div class="gl" id="gl"></div>
    <div class="detail" id="detail"></div>
  </div>

  <div class="controls">
    <div class="running" id="running" style="display:none">
      <button id="pause">PAUSE</button>
      <button class="stop" id="stop">STOP</button>
    </div>
    <p class="legend" id="legend">CHOOSE A DRILL</p>
    <div class="modes" id="modes">
      <button class="mode lead" id="go">
        <span class="what"><span class="name">WORDS</span>
          <span class="desc">The ordinary vocabulary drill. English in, Spanish out.</span></span>
        <span class="many" id="n_words">0<small>DUE</small></span>
      </button>
      <button class="mode" id="sentences">
        <span class="what"><span class="name">SENTENCES</span>
          <span class="desc">Build whole sentences from words you already know.</span></span>
        <span class="many" id="n_sentences">0<small>READY</small></span>
      </button>
      <button class="mode" id="conjugations">
        <span class="what"><span class="name">CONJUGATIONS</span>
          <span class="desc">Verb forms only, for verbs the drill has taught.</span></span>
        <span class="many" id="n_conj">0<small>FORMS DUE</small></span>
      </button>
      <button class="mode" id="placement">
        <span class="what"><span class="name">PLACEMENT</span>
          <span class="desc">Sort words you have never been asked, fast.</span></span>
        <span class="many" id="n_place">0<small>UNSORTED</small></span>
      </button>
    </div>
  </div>
</section>

<section class="page" id="page_progress">
  <div class="figures">
    <div class="figure"><b id="c_learned">0</b><span id="l_learned">LEARNED</span></div>
    <div class="figure"><b id="c_reviews">0</b><span>REVIEWED TODAY</span></div>
    <div class="figure"><b id="c_due" style="color:var(--signal)">0</b><span id="l_due">IN QUEUE</span></div>
    <div class="figure"><b id="c_missed" style="color:var(--miss)">0</b><span>MISSED TODAY</span></div>
    <div class="figure wide"><b id="c_learning" style="color:var(--hit)">0</b><span id="l_learning">LEARNING</span></div>
  </div>
  <div class="block"><h2>LADDER</h2><div class="body" id="ladder"></div></div>
  <div class="block"><h2>DOUBLE-CHECK</h2><div class="body" id="tally"></div></div>
</section>

<section class="page" id="page_settings">
  <div class="field"><label for="category">WHICH WORDS</label>
    <div class="sel"><select id="category"></select></div></div>
  <div class="field"><label for="new_per">NEW WORDS PER DAY</label>
    <input id="new_per" type="number" inputmode="numeric" min="0" max="100">
    <div class="hint">How many unseen words the drill will introduce today.</div></div>
  <div class="field"><label for="window">ANSWER WAIT (SECONDS)</label>
    <input id="window" type="number" inputmode="numeric" min="3" max="20">
    <div class="hint">Answer inside this and the card counts as a quick recall.</div></div>
  <div class="field"><label for="dialect">ACCENT</label>
    <div class="sel"><select id="dialect">
      <option value="es-ES">es-ES &mdash; Spain</option>
      <option value="es-MX">es-MX &mdash; Latin America</option>
    </select></div></div>
  <div class="field"><label for="scope">PLACEMENT COVERS</label>
    <div class="sel"><select id="scope"></select></div></div>
</section>

<script>
const T = new URLSearchParams(location.search).get('t') || '';
const $ = i => document.getElementById(i);
let holding = false, running = false, paused = false, ready = null;

const post = (path, body) => fetch(path + '?t=' + T,
  {method:'POST', headers:{'Content-Type':'application/json'},
   body: JSON.stringify(body||{})});

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}

function fillSelect(el, pairs, chosen){
  if (el.dataset.filled === JSON.stringify(pairs) && el.value === chosen) return;
  el.dataset.filled = JSON.stringify(pairs);
  el.innerHTML = pairs.map(([v,t]) =>
    '<option value="'+esc(v)+'">'+esc(t)+'</option>').join('');
  el.value = chosen;
}

const TABS = ['drill','progress','settings'];
function openTab(which){
  TABS.forEach(t => {
    $('tab_'+t).setAttribute('aria-selected', String(t === which));
    $('page_'+t).classList.toggle('on', t === which);
  });
  if (which === 'drill' && running) $('box').focus();
}
TABS.forEach(t => $('tab_'+t).onclick = () => openTab(t));

function panel(m){
  ready = m;
  running = m.running;
  $('c_learned').textContent = m.learned;
  $('l_learned').innerHTML = 'LEARNED · ' + m.new_left + '/' + m.new_per + ' LEFT';
  $('c_reviews').textContent = m.reviews;
  $('c_due').textContent = m.words_due;
  $('l_due').textContent = m.new_in_queue ? 'IN QUEUE · ' + m.new_in_queue + ' NEW'
                                          : 'IN QUEUE';
  $('c_missed').textContent = m.missed;
  $('c_learning').textContent = m.learning;
  $('l_learning').textContent = 'LEARNING · ' + m.mature + ' MATURE';
  $('ladder').innerHTML = m.ladder.length
    ? m.ladder.map(([k,n]) =>
        '<span class="rung">' + esc(k) + '<b>' + esc(String(n)) + '</b></span>').join('')
    : 'nothing scheduled yet';
  $('tally').textContent = m.overturned + ' OVERTURNED  ·  ' + m.kept + ' KEPT'
    + (m.overturned_share !== null
        ? '  (' + m.overturned_share + '% of misses were the local model)' : '');
  // What each mode actually has waiting, so the choice is informed rather
  // than a guess at four similar-looking words.
  $('n_words').innerHTML = m.words_due + '<small>DUE</small>';
  $('n_sentences').innerHTML = m.sentences + '<small>READY</small>';
  $('n_place').innerHTML = m.unsorted + '<small>UNSORTED</small>';
  $('n_conj').innerHTML = m.forms_due + '<small>FORMS DUE</small>';
  $('dot_due').textContent = m.words_due ? m.words_due : '';
  fillSelect($('category'), m.categories, m.category);
  fillSelect($('scope'), [['new','only words never sorted ('+m.unsorted+')'],
                          ['all','start over, all of them ('+m.in_scope+')']],
             $('scope').value || 'new');
  if (document.activeElement !== $('new_per')) $('new_per').value = m.new_per;
  if (document.activeElement !== $('window')) $('window').value = Math.round(m.window);
  $('dialect').value = m.dialect;
  // Never disabled: pressing a mode switches to it. Greying them out while
  // a drill ran left no way from sentences back to words without finding
  // STOP first, and no way out at all if a session stuck.
  $('stop').disabled = !running;
  $('pause').disabled = !running;
  $('running').style.display = running ? 'grid' : 'none';
  $('legend').textContent = running ? 'SWITCH DRILL' : 'CHOOSE A DRILL';
  paused = !!m.paused;
  $('pause').textContent = paused ? 'RESUME' : 'PAUSE';
  if (!running){
    $('mode').textContent = '';
    $('status').textContent = m.sentences + ' sentences · ' + m.words_due + ' words';
  }
}

function show(m){
  if (m.event === 'ready') return panel(m);
  if (m.event === 'status') $('status').textContent = m.text;
  if (m.event === 'progress'){
    $('track').style.display = m.total ? 'block' : 'none';
    $('fill').style.width = m.total ? (100*m.done/m.total)+'%' : '0';
    $('count').textContent = m.total
      ? m.done + ' / ' + m.total + (m.skipped ? '  (' + m.skipped + ' skipped)' : '')
      : '';
  }
  if (m.event === 'prompt'){
    const who = m.subject || '';
    if (who) $('cue').innerHTML = '<span class="subject">' + esc(who) + '</span>'
                                  + esc(m.text.slice(who.length));
    else $('cue').textContent = m.text;
    $('label').textContent = m.label || '';
    $('note').textContent = m.note || '';
    $('slip').style.display = 'none';
    holding = false; $('box').value = ''; $('box').focus();
  }
  if (m.event === 'result'){
    const s = $('slip'); s.style.display = 'block';
    s.className = 'slip' + (m.perfect ? '' : ' bad');
    $('verdict').textContent = m.perfect ? 'CORRECT' : 'NOT QUITE';
    $('answer').textContent = m.expected +
      ((m.alternates && m.alternates.length) ? '   (also: '+m.alternates.join(', ')+')' : '');
    $('diff').innerHTML = (m.marked||[]).map(t =>
      t.state === 'right' ? '<span class="right">'+esc(t.text)+'</span>' :
      t.state === 'wrong' ? '<span class="wrong">'+esc(t.text)+'</span>'
                            +'<span class="you"> ('+esc(t.typed)+')</span>' :
      '<span class="missing">'+esc(t.text)+'</span>').join(' ')
      + ((m.extra||[]).length ? '<br><span class="you">not in it: </span>'
         +'<span class="wrong">'+esc(m.extra.join(' '))+'</span>' : '');
    $('ex').textContent = m.example || '';
    $('gl').textContent = m.gloss || '';
    let d = m.detail || (m.mistakes ? m.mistakes + ' to fix' : '');
    if (m.hold) d += (d ? '  ·  ' : '') + 'press Enter to continue';
    $('detail').textContent = d;
    holding = !!m.hold; $('box').value = ''; $('box').focus();
  }
  if (m.event === 'paused'){
    paused = m.paused;
    $('pause').textContent = paused ? 'RESUME' : 'PAUSE';
    if (!paused) $('box').focus();
  }
  if (m.event === 'finished'){
    $('mode').textContent = ''; holding = false; paused = false;
    $('pause').textContent = 'PAUSE';
    $('track').style.display = 'none'; $('count').textContent = '';
    $('running').style.display = 'none';
    $('legend').textContent = 'CHOOSE A DRILL';
  }
}

let since = 0;
async function poll(){
  try {
    const r = await fetch('/poll?t=' + T + '&since=' + since);
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    since = d.next;
    d.events.forEach(e => show(JSON.parse(e)));
  } catch (e) {
    $('status').textContent = 'reconnecting';
    await new Promise(r => setTimeout(r, 1500));
  }
  poll();
}
poll();

$('box').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  post('/answer', {text: $('box').value});
  $('box').value = '';
});
function begin(mode, name){
  $('mode').textContent = name;
  openTab('drill');
  $('box').focus();
  post('/start', {mode: mode});
}
$('go').onclick           = () => begin('words', 'WORDS');
$('sentences').onclick    = () => begin('sentences', 'SENTENCES');
$('placement').onclick    = () => begin('placement', 'PLACEMENT');
// Was never wired at all, so the button sat there doing nothing.
$('conjugations').onclick = () => begin('conjugations', 'CONJUGATIONS');
$('pause').onclick        = () => post('/pause', {});
$('stop').onclick         = () => post('/stop');
function settings(){
  post('/settings', {category: $('category').value,
                     new_per: +$('new_per').value,
                     window: +$('window').value,
                     dialect: $('dialect').value,
                     scope: $('scope').value});
}
['category','new_per','window','dialect','scope'].forEach(
  id => $(id).addEventListener('change', settings));
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    hub = None
    token = ""

    # HTTP/1.1, and the event stream is chunked by hand below. The default
    # here is HTTP/1.0, whose body means "read until the connection closes",
    # and a proxy cannot stream that: it has to buffer to find the end. Put
    # behind a Cloudflare tunnel, the phone got a 200 with the right headers
    # and no body at all, for as long as you cared to wait. Locally it worked
    # perfectly, which is what made it look like a tunnel problem.
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass                    # the drill's own output is the interesting one

    # -- plumbing ---------------------------------------------------------
    COOKIE = "drill"

    def _query_token(self):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(self.path).query).get("t", [""])[0]

    def _cookie_token(self):
        """The token as the browser remembers it.

        The bare hostname is the address worth bookmarking; a token pinned to
        the query string means the only usable link is one long enough that
        nobody types it, and one that sits in full in the address bar. Handed
        over once, it lives in the cookie and the base URL works on its own
        from then on.
        """
        from http.cookies import SimpleCookie
        raw = self.headers.get("Cookie")
        if not raw:
            return ""
        try:
            got = SimpleCookie(raw).get(self.COOKIE)
        except Exception:
            return ""
        return got.value if got else ""

    def _authorised(self):
        for given in (self._query_token(), self._cookie_token()):
            if given and secrets.compare_digest(given, self.token):
                return True
        return False

    def _send(self, code, body=b"", kind="text/plain; charset=utf-8",
              remember=False):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if remember:
            # A year, so the phone is not handed back to the long link every
            # time it is picked up. HttpOnly because no script needs to read
            # it, and Secure because the only way in from outside is the
            # tunnel, which is HTTPS.
            self.send_header("Set-Cookie",
                             f"{self.COOKIE}={self.token}; Path=/; Max-Age=31536000; "
                             "HttpOnly; SameSite=Lax; Secure")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _path(self):
        return self.path.split("?", 1)[0]

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = self._path()
        if path == "/" and self._authorised():
            # Only worth writing when the token arrived the long way; a
            # browser already holding it does not need telling again.
            return self._send(200, PAGE.encode("utf-8"),
                              "text/html; charset=utf-8",
                              remember=bool(self._query_token()))
        if not self._authorised():
            return self._send(403, b"bad or missing token")
        if path == "/poll":
            return self._poll()
        self._send(404, b"not here")

    def do_POST(self):
        if not self._authorised():
            return self._send(403, b"bad or missing token")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        path = self._path()
        if path == "/start":
            self.hub.start(body.get("mode", "sentences"))
        elif path == "/answer":
            self.hub.answer(body.get("text", ""))
        elif path == "/stop":
            self.hub.stop()
        elif path == "/pause":
            self.hub.pause(body.get("on"))
        elif path == "/settings":
            self.hub.configure(**body)
        else:
            return self._send(404, b"not here")
        self._send(200, b"ok")

    def _poll(self):
        """Everything that has happened since the browser's last mark.

        Bounded, with a Content-Length, so it survives any proxy between the
        phone and this machine. That is the whole reason it is not a stream.
        """
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        try:
            mark = int(query.get("since", ["0"])[0])
        except ValueError:
            mark = 0
        if mark == 0:
            # A browser starting fresh is told where things stand before it
            # is told to wait. Without this it sits on "connecting" until
            # something else happens to publish.
            self.hub.announce()
        try:
            nxt, events = self.hub.since(mark)
        except Exception:
            nxt, events = 0, []
        body = json.dumps({"next": nxt, "events": events}).encode("utf-8")
        self._send(200, body, "application/json")


def stored_token(path=None):
    """The phone's key, made once and kept.

    A fresh token every launch means a fresh link every launch, and a link
    that changes is not an address. Written owner-only, because anyone
    holding it can drill this deck.
    """
    path = pathlib.Path(path or SERVE_TOKEN_PATH)
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except (FileNotFoundError, OSError):
        pass
    token = secrets.token_urlsafe(9)
    try:
        path.write_text(token + "\n", encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        pass                    # unwritable: this run still works, next differs
    return token


def tailscale_host():
    """This machine's permanent name on the tailnet, if it has one.

    Worth printing because it is the address that does not change: no tunnel,
    nothing public, and the same URL tomorrow.
    """
    import json as _json
    import subprocess
    for command in (["tailscale", "status", "--json"],
                    ["/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                     "status", "--json"]):
        try:
            out = subprocess.run(command, capture_output=True, timeout=5,
                                 text=True)
            if out.returncode:
                continue
            me = _json.loads(out.stdout).get("Self") or {}
            name = (me.get("DNSName") or "").rstrip(".")
            return name or (me.get("TailscaleIPs") or [None])[0]
        except Exception:
            continue
    return None


def serve(port=8765, host="0.0.0.0", token=None, progress=None):
    """Run until interrupted. Returns nothing; prints where to point a phone."""
    token = token or stored_token()
    Handler.hub = Hub(progress)
    Handler.token = token
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    state = Handler.hub.state()
    # Flushed, because this is the only place the token is ever shown and
    # Python buffers stdout the moment it is not a terminal. Piped to a file
    # or a log, an unflushed banner means no URL and no way in.
    print(f"  {state['sentences']} sentences unlocked · "
          f"{state['words_due']} words due", flush=True)
    print(f"  http://localhost:{port}/?t={token}", flush=True)
    tailnet = tailscale_host()
    if tailnet:
        print(f"  http://{tailnet}:{port}/?t={token}"
              "   <- from any device on your tailnet", flush=True)
    else:
        print("  Put a tunnel in front of that port to reach it from a phone.",
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped", flush=True)
    finally:
        Handler.hub.stop()
        server.server_close()
    return 0
