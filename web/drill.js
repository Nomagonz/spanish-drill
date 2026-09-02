/**
 * The drill's rules, in the browser.
 *
 * A port, not a reimplementation. Every function here has a counterpart in the
 * Python and is meant to give the same answer for the same input; where the
 * two could drift, `tests/test_web_port.py` runs both over the real deck and
 * compares. The page could not simply ask the desktop for a verdict: the whole
 * point of it is to work when that machine is asleep.
 *
 *   text.py       normalize, soundsAs, stripArticle, lev, tolerance
 *   grading.py    check, quality, typedQuality, commandOf
 *   scheduler.py  today, schedule, migrate, describeState, describeInterval
 *   progress.py   queueParts, buildQueue, ladderSteps, and the counters
 *
 * Kept in one file with no imports so the same source runs in the page and
 * under node, which is what makes the comparison test possible at all.
 */
(function (root) {
  "use strict";

  // ---- scheduler.py ------------------------------------------------------
  var DAY_SECONDS = 86400;
  var EASE_START = 2.5;
  var EASE_MIN = 1.3;
  var FIRST_INTERVAL = 1;
  var SECOND_INTERVAL = 6;
  var LEECH_AT = 8;
  var MATURE_AT = 21;
  var PASSING_QUALITY = 3;
  var LEITNER_LADDER = [0, 1, 3, 7, 16, 35, 90, 180];

  /**
   * Days since the epoch in local time, matching scheduler.today().
   *
   * Dividing by a day alone turns the day over at UTC midnight, which in the
   * Americas is early evening: the new-word allowance would come back at 7pm.
   * getTimezoneOffset counts minutes west of UTC, so it is the negation of
   * Python's tm_gmtoff.
   */
  function today(now) {
    var ms = now === undefined ? Date.now() : now;
    var offset = -new Date(ms).getTimezoneOffset() * 60;
    return Math.floor((ms / 1000 + offset) / DAY_SECONDS);
  }

  function newCard(day) {
    return {ease: EASE_START, interval: 0, reps: 0, lapses: 0,
            due: day === undefined ? today() : day};
  }

  /** Accept a card written by either scheduler. Leitner saves keep their rung. */
  function migrate(d, day) {
    if (d === null || d === undefined) return null;
    if ("ease" in d) {
      return {ease: d.ease, interval: d.interval, reps: d.reps,
              lapses: d.lapses, due: d.due};
    }
    var box = d.b || 0;
    var now = day === undefined ? today() : day;
    return {
      ease: EASE_START,
      interval: LEITNER_LADDER[Math.min(box, LEITNER_LADDER.length - 1)],
      reps: box,
      lapses: d.l || 0,
      due: d.d === undefined ? now : d.d
    };
  }

  function easeDelta(q) {
    return 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02);
  }

  /**
   * Python rounds half to even; JavaScript's Math.round rounds half up, and
   * the two disagree on exactly the intervals SM-2 produces (2.5, 6.5, ...).
   */
  function roundHalfEven(x) {
    var floor = Math.floor(x);
    var rest = x - floor;
    if (rest > 0.5) return floor + 1;
    if (rest < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  }

  /** Apply a grade. Mutates and returns the card. */
  function schedule(card, q, day) {
    var now = day === undefined ? today() : day;
    if (q >= PASSING_QUALITY) {
      if (card.reps === 0) card.interval = FIRST_INTERVAL;
      else if (card.reps === 1) card.interval = SECOND_INTERVAL;
      else card.interval = Math.max(1, roundHalfEven(card.interval * card.ease));
      card.reps += 1;
    } else {
      card.reps = 0;
      card.interval = 0;        // due again today, not in a week
      card.lapses += 1;
    }
    card.ease = Math.max(EASE_MIN, card.ease + easeDelta(q));
    card.due = now + card.interval;
    return card;
  }

  function isLeech(card) { return card.lapses >= LEECH_AT; }
  function isMature(card) { return card.interval >= MATURE_AT; }

  function describeInterval(interval) {
    if (interval === 0) return "again this session";
    if (interval === 1) return "again tomorrow";
    if (interval < 30) return "again in " + interval + " days";
    var months = roundHalfEven(interval / 30);
    return "again in " + months + " month" + (months === 1 ? "" : "s");
  }

  function describeState(card) {
    if (!card) return "New word";
    if (isLeech(card)) return "Leech · missed " + card.lapses + "x";
    if (card.reps === 0) return "Relearning";
    var step = isMature(card) ? "day 21+" : "day " + card.interval;
    return "Review " + card.reps + " · " + step + " · ease " + card.ease.toFixed(2);
  }

  // ---- text.py -----------------------------------------------------------
  var LEADING_ARTICLE = /^(el|la|los|las|un|una|unos|unas)\s+/;

  /**
   * Lowercase, strip punctuation, collapse whitespace.
   *
   * Accents go by default. "ñ" survives either way: it is a distinct letter,
   * not an accented n, which is why the combining tilde is kept while every
   * other mark is dropped.
   */
  function normalize(s, accents) {
    if (!s) return "";
    var t = s.toLowerCase();
    if (accents) {
      t = t.normalize("NFC").replace(/[^a-z0-9ñáéíóúü ]/g, " ");
    } else {
      t = t.normalize("NFD")
           .replace(/\p{Mn}/gu, function (m) { return m === "\u0303" ? m : ""; })
           .normalize("NFC")
           .replace(/[^a-z0-9ñ ]/g, " ");
    }
    return t.replace(/\s+/g, " ").trim();
  }

  /**
   * The word reduced to how it is actually pronounced. Only for comparing.
   *
   * Silent h except in "ch", b and v being one phoneme, and ll and y having
   * merged for most speakers. The ch is parked rather than matched with a
   * lookbehind, which Safari only learned recently.
   */
  function soundsAs(s) {
    return s.replace(/ch/g, "\u0001")
            .replace(/h/g, "")
            .replace(/\u0001/g, "ch")
            .replace(/v/g, "b")
            .replace(/ll/g, "y");
  }

  function stripArticle(s) { return s.replace(LEADING_ARTICLE, ""); }

  function lev(a, b) {
    if (!a || !b) return a.length || b.length;
    var previous = [], i, j;
    for (j = 0; j <= b.length; j++) previous.push(j);
    for (i = 1; i <= a.length; i++) {
      var current = [i];
      for (j = 1; j <= b.length; j++) {
        current.push(Math.min(previous[j] + 1,
                              current[j - 1] + 1,
                              previous[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
      }
      previous = current;
    }
    return previous[previous.length - 1];
  }

  /** How far off a transcript may be and still count as the same word. */
  function tolerance(word) {
    var n = word.length;
    if (n <= 5) return 0;
    return n <= 8 ? 1 : 2;
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // ---- grading.py --------------------------------------------------------
  /**
   * Does what was said count as the answer? Returns {answer, close} or null.
   *
   * Three passes, strictest first: the answer exactly, the answer inside a
   * longer utterance, then the answer within a small edit distance but never
   * when the transcript is itself another word in the deck. `deckAnswers` is
   * that set of every normalized deck answer.
   *
   * The bias is toward rejecting. A false reject costs one extra review; a
   * false accept banks a mistake and hides the word for weeks.
   */
  function check(said, card, deckAnswers) {
    var heard = normalize(said);
    if (!heard) return null;

    var variants = [heard];
    var stripped = stripArticle(heard);
    if (stripped !== heard) variants.push(stripped);

    var i, j;
    for (i = 0; i < variants.length; i++) {
      var variant = variants[i];
      var spoken = soundsAs(variant);
      for (j = 0; j < card.answers.length; j++) {
        var answer = card.answers[j];
        var target = normalize(answer);
        var sounded = soundsAs(target);
        if (variant === target || spoken === sounded) {
          return {answer: answer, close: false};
        }
        // Some entries carry a trailing preposition ("cerca de"); allow the
        // bare form too.
        if (variant === target.replace(/ (de|a|que)$/, "")) {
          return {answer: answer, close: false};
        }
        if (new RegExp("(^| )" + escapeRegExp(target) + "( |$)").test(variant)) {
          return {answer: answer, close: false};
        }
        if (new RegExp("(^| )" + escapeRegExp(sounded) + "( |$)").test(spoken)) {
          return {answer: answer, close: false};
        }
      }
    }

    for (i = 0; i < variants.length; i++) {
      // A transcript that is itself a real deck answer is a different word,
      // not a mangled version of this one. llevar and llegar differ by one
      // character.
      if (deckAnswers.has(variants[i])) continue;
      for (j = 0; j < card.answers.length; j++) {
        var t = normalize(card.answers[j]);
        if (lev(variants[i], t) <= tolerance(t)) {
          return {answer: card.answers[j], close: true};
        }
      }
    }
    return null;
  }

  var TYPING_BASE_SECONDS = 1.5;
  var TYPING_PER_CHARACTER = 0.25;

  function typingBudget(answer) {
    return TYPING_BASE_SECONDS + TYPING_PER_CHARACTER * ((answer || "").length);
  }

  /** SM-2's 0-5 scale, with typing's own sense of what counts as quick. */
  function typedQuality(ok, close, blank, elapsed, answer) {
    if (blank) return 0;
    if (!ok) return 1;
    if (close) return 3;
    return elapsed <= typingBudget(answer) ? 5 : 4;
  }

  function quality(ok, close, silent, elapsed, window_) {
    if (silent) return 0;       // no attempt is worse than a wrong attempt
    if (!ok) return 1;
    if (close) return 3;
    return elapsed <= window_ * 0.45 ? 5 : 4;
  }

  var COMMANDS = {
    repeat: ["repite", "repetir", "otra vez", "repeat", "again"],
    skip: ["salta", "saltar", "skip", "pasa", "siguiente", "next"],
    stop: ["para", "parar", "alto", "stop", "pausa", "pause"],
    reveal: ["no se", "no lo se", "dime", "i dont know", "tell me", "pass"]
  };

  var commandLookup = null;
  function commandOf(said) {
    if (!commandLookup) {
      commandLookup = {};
      Object.keys(COMMANDS).forEach(function (name) {
        COMMANDS[name].forEach(function (phrase) {
          commandLookup[normalize(phrase)] = name;
        });
      });
    }
    return commandLookup[normalize(said)] || null;
  }

  // ---- progress.py -------------------------------------------------------
  var PERSON_ORDER = ["yo", "tu", "nos", "el", "vos", "ellos"];
  var TENSE_ORDER = ["pres", "pret", "imp", "fut", "cond"];
  var SPAIN_ONLY = ["vos"];
  var SPAIN_DIALECT = "es-ES";
  var UNLOCK_REPS = 1;

  /** lemma -> its forms as one chain, in the order they are taught. */
  function ladder(deck) {
    var rank = {}, out = {};
    TENSE_ORDER.forEach(function (tense, ti) {
      PERSON_ORDER.forEach(function (person, pi) {
        rank[tense + "-" + person] = ti * 100 + pi;
      });
    });
    deck.forEach(function (card, index) {
      if (!card.lemma) return;
      (out[card.lemma] = out[card.lemma] || []).push([rank[card.form], index]);
    });
    Object.keys(out).forEach(function (lemma) {
      out[lemma] = out[lemma]
        .sort(function (a, b) { return a[0] - b[0]; })
        .map(function (pair) { return pair[1]; });
    });
    return out;
  }

  /**
   * The schedule, and every question the drill asks of it.
   *
   * `cards` is keyed by deck index, exactly as the desktop keeps it in memory,
   * and is written out keyed by card id, exactly as progress.json holds it.
   */
  function Progress(deck, state) {
    state = state || {};
    this.deck = deck;
    this.byId = {};
    var self = this;
    deck.forEach(function (c, i) { self.byId[c.id] = i; });

    this.dialect = state.dialect || SPAIN_DIALECT;
    this.category = state.category || "all";
    this.new_per = state.new_per === undefined ? 20 : state.new_per;
    this.window = state.window === undefined ? 6 : state.window;
    this.day = state.day || 0;
    this.new_done = state.new_done || 0;
    this.reviews_done = state.reviews_done || 0;
    this.missed_today = state.missed_today || 0;
    this.kept = state.kept || 0;
    this.overturned = state.overturned || 0;
    this.sentences_done = state.sentences_done || [];
    this.hints = state.hints === undefined ? true : state.hints;
    this.verify_live = state.verify_live === undefined ? true : state.verify_live;
    this.speak_cue = state.speak_cue === undefined ? true : state.speak_cue;
    this.model = state.model || "medium";
    this.input_device = state.input_device || "";

    // Accepts both shapes, the same as _read_cards: current saves are keyed by
    // the card's stable id, older ones by deck position. An id that no longer
    // exists is dropped rather than guessed at.
    this.cards = {};
    var raw = state.cards || {};
    var now = today();
    Object.keys(raw).forEach(function (key) {
      var index = /^-?\d+$/.test(key) ? parseInt(key, 10) : self.byId[key];
      if (index === undefined || index === null) return;
      self.cards[index] = migrate(raw[key], now);
    });
    this._chains = {};
    this._ladder = null;
  }

  Progress.prototype.chain = function (lemma) {
    var key = lemma + "|" + this.dialect;
    if (this._chains[key]) return this._chains[key];
    if (!this._ladder) this._ladder = ladder(this.deck);
    var chain = this._ladder[lemma] || [];
    var deck = this.deck;
    if (this.dialect !== SPAIN_DIALECT) {
      chain = chain.filter(function (i) {
        return SPAIN_ONLY.indexOf(deck[i].form.split("-")[1]) === -1;
      });
    }
    this._chains[key] = chain;
    return chain;
  };

  Progress.prototype.learned = function (index) {
    var card = this.cards[index];
    return !!(card && card.reps >= UNLOCK_REPS);
  };

  /** May this card be introduced yet? Vocabulary always may. */
  Progress.prototype.unlocked = function (index) {
    var card = this.deck[index];
    if (!card.lemma) return true;
    var chain = this.chain(card.lemma);
    var step = chain.indexOf(index);
    if (step === -1) return false;      // a form the dialect drops
    var previous = step === 0 ? this.byId[card.lemma] : chain[step - 1];
    return previous !== undefined && previous !== null && this.learned(previous);
  };

  Progress.prototype.inCategory = function (index) {
    if (!this.category || this.category === "all") return true;
    return this.deck[index].pos === this.category;
  };

  Progress.prototype.dueIndexes = function (day) {
    var now = day === undefined ? today() : day;
    var self = this, out = [];
    Object.keys(this.cards).forEach(function (k) {
      if (self.cards[k].due <= now) out.push(parseInt(k, 10));
    });
    return out.sort(function (a, b) { return a - b; });
  };

  Progress.prototype.unseenIndexes = function () {
    var out = [];
    for (var i = 0; i < this.deck.length; i++) {
      if (!(i in this.cards) && this.inCategory(i) && this.unlocked(i)) out.push(i);
    }
    return out;
  };

  Progress.prototype.newRemaining = function () {
    return Math.max(0, this.new_per - this.new_done);
  };

  Progress.prototype.queueParts = function (day) {
    var self = this;
    var due = this.dueIndexes(day).filter(function (i) {
      return self.inCategory(i);
    });
    var fresh = this.unseenIndexes().slice(0, this.newRemaining());
    return {due: due, fresh: fresh};
  };

  /** Due reviews, shuffled, with the day's new words spread through. */
  Progress.prototype.buildQueue = function (day, shuffle) {
    var parts = this.queueParts(day);
    var due = parts.due.slice();
    (shuffle || defaultShuffle)(due);
    var queue = due;
    var fresh = parts.fresh;
    if (fresh.length) {
      var step = Math.max(1, Math.floor((queue.length + fresh.length) / fresh.length));
      var at = 0;
      for (var i = 0; i < fresh.length; i++) {
        queue.splice(Math.min(at, queue.length), 0, fresh[i]);
        at += step + 1;
      }
    }
    return queue;
  };

  function defaultShuffle(a) {
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
  }

  Progress.prototype.rollOver = function (day) {
    var now = day === undefined ? today() : day;
    if (this.day === now) return false;
    this.day = now;
    this.new_done = this.reviews_done = this.missed_today = 0;
    return true;
  };

  Progress.prototype.dueByStage = function () {
    var stages = {}, self = this;
    Object.keys(this.cards).forEach(function (k) {
      var card = self.cards[k], key;
      if (card.reps === 0) key = card.lapses ? "relearning" : "new";
      else if (card.interval >= MATURE_AT) key = "mature";
      else key = card.interval + "d";
      stages[key] = (stages[key] || 0) + 1;
    });
    return stages;
  };

  Progress.prototype.ladderSteps = function () {
    var stages = this.dueByStage();
    var labels = {relearning: "RELEARNING", mature: "21d+"};
    function position(step) {
      if (step === "relearning") return -1;
      return step === "mature" ? 1e6 : parseInt(step, 10);
    }
    return Object.keys(stages)
      .filter(function (k) { return k !== "new"; })
      .sort(function (a, b) { return position(a) - position(b); })
      .map(function (k) { return [labels[k] || k, stages[k]]; });
  };

  Progress.prototype.learningCount = function () {
    var self = this;
    return Object.keys(this.cards)
      .filter(function (k) { return self.cards[k].interval >= 1; }).length;
  };

  Progress.prototype.matureCount = function () {
    var self = this;
    return Object.keys(this.cards)
      .filter(function (k) { return isMature(self.cards[k]); }).length;
  };

  Progress.prototype.placementScope = function () {
    var inScope = [], i;
    for (i = 0; i < this.deck.length; i++) if (this.inCategory(i)) inScope.push(i);
    var self = this;
    var unsorted = inScope.filter(function (j) { return !(j in self.cards); }).length;
    return {unsorted: unsorted, in_scope: inScope.length};
  };

  Progress.prototype.whyNothingIsDue = function () {
    var reasons = [];
    if (this.new_per && !this.newRemaining()) {
      reasons.push("today's " + this.new_per + " new " +
                   (this.new_per === 1 ? "word" : "words") + " are done");
    } else if (!this.new_per) {
      reasons.push("new words are switched off");
    }
    if (this.category && this.category !== "all") {
      reasons.push("the drill is limited to " + this.category + "s");
    }
    if (!reasons.length) reasons.push("nothing is scheduled for review yet");
    var text = reasons.join(" and ");
    return text.charAt(0).toUpperCase() + text.slice(1) + ".";
  };

  /** The state as progress.json holds it: cards keyed by id, never position. */
  Progress.prototype.toState = function () {
    var cards = {}, self = this;
    Object.keys(this.cards)
      .map(Number)
      .sort(function (a, b) { return a - b; })
      .forEach(function (i) {
        if (i >= 0 && i < self.deck.length) cards[self.deck[i].id] = self.cards[i];
      });
    return {
      cards: cards,
      dialect: this.dialect, input_device: this.input_device, model: this.model,
      new_per: this.new_per, window: this.window, hints: this.hints,
      verify_live: this.verify_live, category: this.category,
      speak_cue: this.speak_cue, day: this.day, new_done: this.new_done,
      reviews_done: this.reviews_done, missed_today: this.missed_today,
      kept: this.kept, overturned: this.overturned,
      sentences_done: this.sentences_done.slice().sort()
    };
  };

  var api = {
    today: today, newCard: newCard, migrate: migrate, schedule: schedule,
    easeDelta: easeDelta, isLeech: isLeech, isMature: isMature,
    describeInterval: describeInterval, describeState: describeState,
    normalize: normalize, soundsAs: soundsAs, stripArticle: stripArticle,
    lev: lev, tolerance: tolerance, check: check, quality: quality,
    typedQuality: typedQuality, typingBudget: typingBudget,
    commandOf: commandOf, Progress: Progress, ladder: ladder,
    MATURE_AT: MATURE_AT, EASE_START: EASE_START
  };

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.Drill = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
