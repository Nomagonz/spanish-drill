/**
 * Runs the browser port over a list of jobs and prints what it decided.
 *
 * Exists so the Python can put the same inputs through both implementations
 * and compare. Reads one JSON document on stdin, writes one on stdout, and
 * holds no opinions of its own: every answer here has to come from drill.js
 * or the comparison proves nothing.
 */
const path = require("path");
const D = require(path.join(__dirname, "..", "web", "drill.js"));

let raw = "";
process.stdin.on("data", d => (raw += d));
process.stdin.on("end", () => {
  const job = JSON.parse(raw);
  const deck = job.deck;
  const answers = new Set(job.deck_answers);
  const out = {};

  if (job.normalize) {
    out.normalize = job.normalize.map(s => D.normalize(s));
    out.normalize_accents = job.normalize.map(s => D.normalize(s, true));
    out.sounds_as = job.normalize.map(s => D.soundsAs(D.normalize(s)));
    out.strip_article = job.normalize.map(s => D.stripArticle(D.normalize(s)));
  }

  if (job.lev) {
    out.lev = job.lev.map(([a, b]) => D.lev(a, b));
    out.tolerance = job.lev.map(([a]) => D.tolerance(a));
  }

  if (job.check) {
    out.check = job.check.map(([said, index]) => {
      const card = deck[index];
      const m = D.check(said, card, answers);
      return m === null ? null : [m.answer, m.close];
    });
  }

  if (job.schedule) {
    out.schedule = job.schedule.map(([card, grades, day]) => {
      const c = Object.assign({}, card);
      grades.forEach(q => D.schedule(c, q, day));
      return c;
    });
  }

  if (job.quality) {
    out.quality = job.quality.map(
      ([ok, close, silent, elapsed, w]) => D.quality(ok, close, silent, elapsed, w));
  }
  if (job.typed_quality) {
    out.typed_quality = job.typed_quality.map(
      ([ok, close, blank, elapsed, a]) => D.typedQuality(ok, close, blank, elapsed, a));
  }
  if (job.command_of) out.command_of = job.command_of.map(s => D.commandOf(s));

  if (job.describe) {
    out.describe_interval = job.describe.map(n => D.describeInterval(n));
  }
  if (job.describe_state) {
    out.describe_state = job.describe_state.map(c => D.describeState(c));
  }

  if (job.progress) {
    out.progress = job.progress.map(spec => {
      const p = new D.Progress(deck, spec.state);
      const parts = p.queueParts(spec.day);
      return {
        due: parts.due,
        fresh: parts.fresh,
        unseen_head: p.unseenIndexes().slice(0, 40),
        // The conjugated forms specifically. The head of the unseen list is
        // all vocabulary, so comparing that alone said nothing about the
        // chain rule that decides which form opens next.
        unseen_forms: p.unseenIndexes().filter(function (i) {
          return !!deck[i].lemma;
        }),
        ladder: p.ladderSteps(),
        learning: p.learningCount(),
        mature: p.matureCount(),
        scope: p.placementScope(),
        why: p.whyNothingIsDue(),
        state: p.toState()
      };
    });
  }

  process.stdout.write(JSON.stringify(out));
});
