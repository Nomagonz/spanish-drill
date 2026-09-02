"""The browser port, checked against the Python it was ported from.

The web page cannot ask this machine whether an answer was right: the whole
point of it is to work when this machine is asleep. So the rules are written
twice, and two copies of a rule eventually disagree unless something makes
them prove otherwise every time the suite runs.

That is all this file does. It puts the same inputs through `web/drill.js` and
through the modules it was ported from, over the real deck, and fails on any
difference. A drift here is not cosmetic: grading decides what gets banked as
known, and a schedule is expensive to repair once it has been written wrong.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from spanish_drill import grading, scheduler, text
from spanish_drill.deck import load_deck
from spanish_drill.progress import Progress

BRIDGE = Path(__file__).parent / "web_port_bridge.js"
DECK = load_deck()

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is needed to run the browser port")


def run(job):
    """Hand the jobs to the port and read back what it decided."""
    job["deck"] = [
        {"id": c.id, "en": c.prompt, "answers": list(c.answers),
         "pos": c.pos, "lemma": c.lemma, "form": c.form,
         "ex": c.example, "gl": c.gloss}
        for c in DECK
    ]
    job["deck_answers"] = sorted({text.normalize(a) for c in DECK
                                  for a in c.answers})
    done = subprocess.run(["node", str(BRIDGE)], input=json.dumps(job),
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def some_words():
    """A spread of real answers, plus the shapes a recogniser produces."""
    words = []
    for card in DECK[:400]:
        words.append(card.answers[0])
    words += ["Habló", "  el   Niño ", "¿Qué?", "LLAMAR", "hago", "ago",
              "voy", "boy", "yamar", "la casa", "el hombre", "no sé",
              "", "   ", "ñu", "años", "esta bien", "cerca de", "ir",
              # Every article the rule strips, not just the two obvious ones:
              # dropping "un" quietly from the port passed a shorter list.
              "un libro", "una casa", "unos amigos", "unas cosas",
              "los hombres", "las casas", "el la los", "launa"]
    return words


class TestTheTextRules:
    def test_normalising_agrees(self):
        words = some_words()
        out = run({"normalize": words})
        assert out["normalize"] == [text.normalize(w) for w in words]

    def test_normalising_with_accents_agrees(self):
        words = some_words()
        out = run({"normalize": words})
        assert out["normalize_accents"] == [text.normalize(w, accents=True)
                                            for w in words]

    def test_how_a_word_sounds_agrees(self):
        """The silent h, b for v, ll for y. Where a port most easily drifts,
        because the h rule has an exception the other two do not."""
        words = some_words()
        out = run({"normalize": words})
        assert out["sounds_as"] == [text.sounds_as(text.normalize(w))
                                    for w in words]

    def test_stripping_the_article_agrees(self):
        words = some_words()
        out = run({"normalize": words})
        assert out["strip_article"] == [text.strip_article(text.normalize(w))
                                        for w in words]

    def test_edit_distance_and_tolerance_agree(self):
        pairs = []
        for card in DECK[:150]:
            a = text.normalize(card.answers[0])
            pairs.append([a, a])
            pairs.append([a, a[:-1]])
            pairs.append([a, a + "s"])
            pairs.append([a, text.normalize(DECK[0].answers[0])])
        out = run({"lev": pairs})
        assert out["lev"] == [text.lev(a, b) for a, b in pairs]
        assert out["tolerance"] == [text.tolerance(a) for a, _ in pairs]


class TestGrading:
    """The rule that decides what gets banked as known."""

    def cases(self):
        """Every card's own answer, and the near misses worth worrying about."""
        cases = []
        for index, card in enumerate(DECK[:250]):
            best = card.answers[0]
            cases.append([best, index])
            cases.append([best.upper(), index])
            cases.append([f"  {best}  ", index])
            cases.append([f"pues {best}", index])
            cases.append([best[:-1], index])
            cases.append([best + "n", index])
            cases.append([text.sounds_as(text.normalize(best)), index])
            cases.append([f"la {best}", index])
            cases.append(["", index])
            cases.append([DECK[7].answers[0], index])
        return cases

    def test_every_verdict_agrees(self):
        cases = self.cases()
        out = run({"check": cases})
        expected = []
        for said, index in cases:
            m = grading.check(said, DECK[index])
            expected.append(None if m is None else [m.answer, m.close])
        assert len(out["check"]) == len(expected)
        for at, (got, want) in enumerate(zip(out["check"], expected)):
            assert got == want, (
                f"case {cases[at]!r}: port said {got!r}, python said {want!r}")

    def test_the_commands_agree(self):
        said = ["no sé", "skip", "para", "repite", "NO LO SE", "hola",
                "otra vez", "next", ""]
        out = run({"command_of": said})
        assert out["command_of"] == [grading.command_of(s) for s in said]

    def test_the_grades_agree(self):
        rows = []
        for ok in (True, False):
            for close in (True, False):
                for silent in (True, False):
                    for elapsed in (0.5, 2.7, 6.0):
                        rows.append([ok, close, silent, elapsed, 6.0])
        out = run({"quality": rows})
        assert out["quality"] == [
            grading.quality(ok, close, silent, e, w) for ok, close, silent, e, w in rows]

    def test_the_typed_grades_agree(self):
        rows = []
        for ok in (True, False):
            for close in (True, False):
                for blank in (True, False):
                    for elapsed in (0.5, 2.0, 9.0):
                        rows.append([ok, close, blank, elapsed, "encontrar"])
        out = run({"typed_quality": rows})
        assert out["typed_quality"] == [
            grading.typed_quality(ok, c, b, e, a) for ok, c, b, e, a in rows]


class TestTheScheduler:
    """A schedule written wrong is expensive to repair, and invisible for
    weeks while it is being written."""

    def starts(self):
        """Fresh cards, and ones parked exactly on a rounding boundary.

        `interval * ease` landing on a half is not a curiosity: 1 x 2.5 and
        3 x 2.5 are ordinary SM-2 states. Python rounds a half to even and
        JavaScript's Math.round rounds it up, so a port that reaches for the
        obvious function is wrong on exactly these and nowhere else. Starting
        every sequence from a new card never reached one.
        """
        return [
            {"ease": 2.5, "interval": 0, "reps": 0, "lapses": 0, "due": 0},
            {"ease": 2.5, "interval": 1, "reps": 2, "lapses": 0, "due": 0},
            {"ease": 2.5, "interval": 3, "reps": 3, "lapses": 0, "due": 0},
            {"ease": 2.5, "interval": 5, "reps": 4, "lapses": 0, "due": 0},
            {"ease": 1.5, "interval": 5, "reps": 4, "lapses": 2, "due": 0},
            {"ease": 1.3, "interval": 15, "reps": 7, "lapses": 5, "due": 0},
            {"ease": 2.36, "interval": 6, "reps": 2, "lapses": 1, "due": 0},
        ]

    def sequences(self):
        return [
            [], [5], [4], [3], [1], [0],
            [5, 5], [5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4], [3, 3, 3, 3], [5, 1, 5, 5], [1, 1, 1],
            [5, 5, 1, 5, 5, 5], [0, 5, 5, 5, 5],
            [5] * 12, [4] * 12, [5, 4, 3, 5, 4, 3, 5, 4],
        ]

    def jobs(self):
        day = 20000
        return [(dict(start, due=day), grades, day)
                for start in self.starts() for grades in self.sequences()]

    def test_every_sequence_lands_on_the_same_card(self):
        jobs = self.jobs()
        out = run({"schedule": [[dict(c), list(g), d] for c, g, d in jobs]})
        for at, (start, grades, day) in enumerate(jobs):
            card = scheduler.Card(**start)
            for q in grades:
                scheduler.schedule(card, q, today=day)
            got = out["schedule"][at]
            where = f"start={start} grades={grades}"
            assert round(got["ease"], 9) == round(card.ease, 9), where
            assert got["interval"] == card.interval, where
            assert got["reps"] == card.reps, where
            assert got["lapses"] == card.lapses, where
            assert got["due"] == card.due, where

    def test_the_wording_agrees(self):
        days = [0, 1, 2, 6, 12, 21, 29, 30, 45, 60, 75, 180, 365]
        out = run({"describe": days})
        assert out["describe_interval"] == [
            scheduler.describe_interval(d) for d in days]

    def test_the_state_wording_agrees(self):
        cards = [
            {"ease": 2.5, "interval": 0, "reps": 0, "lapses": 0, "due": 0},
            {"ease": 2.5, "interval": 0, "reps": 0, "lapses": 9, "due": 0},
            {"ease": 2.36, "interval": 6, "reps": 2, "lapses": 1, "due": 0},
            {"ease": 2.5, "interval": 30, "reps": 5, "lapses": 0, "due": 0},
        ]
        out = run({"describe_state": cards})
        assert out["describe_state"] == [
            scheduler.describe_state(scheduler.Card(**c)) for c in cards]


class TestTheQueue:
    """What the page will ask, against what the desktop would have asked."""

    def states(self):
        day = scheduler.today()
        learned = {DECK[i].id: {"ease": 2.5, "interval": 1, "reps": 1,
                                "lapses": 0, "due": day - 1}
                   for i in range(0, 60)}
        return [
            {"day": day, "state": {"cards": {}, "new_per": 20, "day": day,
                                   "category": "all", "dialect": "es-ES"}},
            {"day": day, "state": {"cards": learned, "new_per": 20, "day": day,
                                   "category": "all", "dialect": "es-ES"}},
            {"day": day, "state": {"cards": learned, "new_per": 0, "day": day,
                                   "category": "verb", "dialect": "es-ES"}},
            {"day": day, "state": {"cards": learned, "new_per": 5, "day": day,
                                   "new_done": 5, "category": "noun",
                                   "dialect": "es-MX"}},
            # A verb that has been met and then forgotten. `learned` asks for
            # a successful repetition, not merely a card on file, and it is
            # what decides whether that verb's first conjugated form may be
            # introduced. Every state above has reps of 1, where "answered
            # right once" and "seen at all" give the same answer.
            {"day": day, "state": {
                "cards": {
                    "ser": {"ease": 2.5, "interval": 0, "reps": 0,
                            "lapses": 2, "due": day},
                    "estar": {"ease": 2.5, "interval": 4, "reps": 3,
                              "lapses": 0, "due": day - 1},
                },
                "new_per": 40, "day": day, "category": "all",
                "dialect": "es-ES"}},
            # The same, on the dialect that drops vosotros, so a chain that
            # still had it in would show up as a different next form.
            {"day": day, "state": {
                "cards": {
                    "ser": {"ease": 2.5, "interval": 0, "reps": 0,
                            "lapses": 2, "due": day},
                    "estar": {"ease": 2.5, "interval": 4, "reps": 3,
                              "lapses": 0, "due": day - 1},
                },
                "new_per": 40, "day": day, "category": "all",
                "dialect": "es-MX"}},
        ] + [
            # Four persons of the present already in, so the next form to open
            # is decided by the order of the chain rather than by there being
            # only one candidate. On Spain that next form is vosotros; on
            # Latin America it is dropped and the chain closes up over it.
            # With one form unlocked, neither the order nor the dialect
            # filter could show.
            {"day": day, "state": {
                "cards": dict({
                    "estar": {"ease": 2.5, "interval": 9, "reps": 4,
                              "lapses": 0, "due": day - 1}},
                    **{f"estar:pres-{who}": {"ease": 2.5, "interval": 3,
                                             "reps": 2, "lapses": 0,
                                             "due": day - 1}
                       for who in ("yo", "tu", "nos", "el")}),
                "new_per": 40, "day": day, "category": "all",
                "dialect": dialect}}
            for dialect in ("es-ES", "es-MX")
        ] + [
            # Two persons in, so the next one is decided by where the order
            # puts "nos" against "el". Learning the whole disputed run first
            # hid that: swapping two forms inside a set that is entirely
            # learned changes nothing about which comes next.
            {"day": day, "state": {
                "cards": dict({
                    "estar": {"ease": 2.5, "interval": 9, "reps": 4,
                              "lapses": 0, "due": day - 1}},
                    **{f"estar:pres-{who}": {"ease": 2.5, "interval": 3,
                                             "reps": 2, "lapses": 0,
                                             "due": day - 1}
                       for who in ("yo", "tu")}),
                "new_per": 40, "day": day, "category": "all",
                "dialect": "es-ES"}},
            # The whole present in, so the next form is the head of the next
            # tense and the order of the tenses is what decides which.
            {"day": day, "state": {
                "cards": dict({
                    "estar": {"ease": 2.5, "interval": 9, "reps": 4,
                              "lapses": 0, "due": day - 1}},
                    **{f"estar:pres-{who}": {"ease": 2.5, "interval": 3,
                                             "reps": 2, "lapses": 0,
                                             "due": day - 1}
                       for who in ("yo", "tu", "nos", "el", "vos", "ellos")}),
                "new_per": 40, "day": day, "category": "all",
                "dialect": "es-ES"}},
        ]

    def python_progress(self, spec):
        p = Progress(path=None)
        raw = spec["state"]
        for key in ("dialect", "category", "new_per", "day", "new_done"):
            if key in raw:
                setattr(p, key, raw[key])
        from spanish_drill.progress import _read_cards
        p.cards = _read_cards(raw.get("cards", {}), spec["day"])
        return p

    def test_the_same_cards_come_due(self):
        specs = self.states()
        out = run({"progress": specs})
        for at, spec in enumerate(specs):
            p = self.python_progress(spec)
            due, fresh = p.queue_parts(spec["day"])
            assert sorted(out["progress"][at]["due"]) == sorted(due), at
            assert out["progress"][at]["fresh"] == fresh, at

    def test_the_same_cards_are_offered_next(self):
        specs = self.states()
        out = run({"progress": specs})
        for at, spec in enumerate(specs):
            p = self.python_progress(spec)
            assert out["progress"][at]["unseen_head"] == p.unseen_indexes()[:40], at

    def test_the_same_conjugated_forms_are_unlocked(self):
        """Which form opens next, and whether it opens at all.

        Compared on the forms alone. The head of the unseen list is entirely
        vocabulary, so asserting on that said nothing about the chain: a port
        that unlocked every form of every verb at once passed it.
        """
        specs = self.states()
        out = run({"progress": specs})
        deck = DECK
        for at, spec in enumerate(specs):
            p = self.python_progress(spec)
            expected = [i for i in p.unseen_indexes() if deck[i].lemma]
            got = out["progress"][at]["unseen_forms"]
            assert got == expected, (
                f"state {at}: port opens {[deck[i].id for i in got][:6]}, "
                f"python opens {[deck[i].id for i in expected][:6]}")

    def test_the_counters_agree(self):
        specs = self.states()
        out = run({"progress": specs})
        for at, spec in enumerate(specs):
            p = self.python_progress(spec)
            got = out["progress"][at]
            assert got["learning"] == p.learning_count(), at
            assert got["mature"] == p.mature_count(), at
            assert [tuple(r) for r in got["ladder"]] == p.ladder_steps(), at
            unsorted, in_scope = p.placement_scope()
            assert got["scope"] == {"unsorted": unsorted, "in_scope": in_scope}, at
            assert got["why"] == p.why_nothing_is_due(), at

    def test_what_it_would_save_is_what_python_would_save(self):
        """The page writes into the same database the desktop reads."""
        specs = self.states()
        out = run({"progress": specs})
        for at, spec in enumerate(specs):
            p = self.python_progress(spec)
            saved = out["progress"][at]["state"]["cards"]
            expected = {DECK[i].id: c.to_dict() for i, c in sorted(p.cards.items())}
            assert saved.keys() == expected.keys(), at
            for cid in expected:
                assert saved[cid] == expected[cid], (at, cid)

    def test_a_leitner_save_is_migrated_the_same_way(self):
        """The page has years of its own localStorage in the old format."""
        day = scheduler.today()
        old = {DECK[i].id: {"b": i % 8, "d": day + i, "r": 0, "l": i % 3}
               for i in range(40)}
        spec = {"day": day, "state": {"cards": old, "new_per": 0, "day": day}}
        out = run({"progress": [spec]})
        p = self.python_progress(spec)
        saved = out["progress"][0]["state"]["cards"]
        expected = {DECK[i].id: c.to_dict() for i, c in sorted(p.cards.items())}
        assert saved == expected
