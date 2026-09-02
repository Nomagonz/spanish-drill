"""The conjugation-only drill: its batch, its gate, and its separate books.

The tests that matter most here are the ones about separation. The whole
justification for a second tracker is that a session of paradigms cannot move
a vocabulary card's review date, and that is a property no amount of reading
the code proves.
"""
import json

import pytest

from spanish_drill.config import CONJUGATION_BATCH
from spanish_drill.deck import load_deck
from spanish_drill.paradigm import (ConjugationProgress, ConjugationSession,
                                    known_verbs, verb_order)
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card, today

DECK = load_deck()


def index_of(word):
    """The vocabulary card for a word, never a conjugated form."""
    return next(i for i, c in enumerate(DECK)
                if not c.lemma and c.answers[0] == word)


def form_index(lemma, form):
    return next(i for i, c in enumerate(DECK)
                if c.lemma == lemma and c.form == form)


def learn(progress, index, reps=1, interval=1):
    """Put a card in the review rotation, the way a right answer would."""
    progress.cards[index] = Card(interval=interval, reps=reps,
                                 due=today() + interval)
    return progress.cards[index]


@pytest.fixture
def main(tmp_path):
    """A vocabulary tracker with nothing learned yet."""
    return Progress(path=tmp_path / "progress.json", verify_live=False)


@pytest.fixture
def conj(tmp_path, main):
    return ConjugationProgress.open(main, path=tmp_path / "conjugation.json")


class TestVerbOrder:
    def test_is_the_deck_order_which_is_frequency_order(self):
        assert verb_order(DECK)[:6] == (
            "ser", "estar", "haber", "tener", "ir", "hacer")

    def test_only_verbs_that_actually_have_forms(self):
        lemmas = {c.lemma for c in DECK if c.lemma}
        assert set(verb_order(DECK)) == lemmas

    def test_no_conjugated_form_is_ever_listed_as_a_verb(self):
        assert "tengo" not in verb_order(DECK)


class TestKnownVerbs:
    def test_nothing_is_known_before_anything_is_learned(self, main):
        assert known_verbs(main, DECK) == ()

    def test_a_learned_infinitive_makes_its_verb_available(self, main):
        learn(main, index_of("tener"))
        assert known_verbs(main, DECK) == ("tener",)

    def test_a_seen_but_unlearned_infinitive_does_not(self, main):
        # reps 0 is a card that has been met and missed, not one that is known.
        main.cards[index_of("tener")] = Card(interval=0, reps=0)
        assert known_verbs(main, DECK) == ()

    def test_they_come_back_in_frequency_order(self, main):
        for word in ("hacer", "ser", "tener"):
            learn(main, index_of(word))
        assert known_verbs(main, DECK) == ("ser", "tener", "hacer")


class TestBatch:
    def test_empty_until_a_verb_is_learned(self, conj):
        assert conj.batch(DECK) == ()

    def test_takes_the_ten_most_frequent(self, main, tmp_path):
        for lemma in verb_order(DECK)[:15]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert len(c.batch(DECK)) == CONJUGATION_BATCH
        assert c.batch(DECK) == verb_order(DECK)[:CONJUGATION_BATCH]

    def test_fewer_than_ten_learned_is_simply_fewer(self, main, tmp_path):
        for lemma in ("ser", "tener", "ir"):
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert c.batch(DECK) == ("ser", "tener", "ir")

    def test_a_finished_verb_hands_its_slot_to_the_next_one(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert "ser" in c.batch(DECK) and "saber" not in c.batch(DECK)
        for i in c._chain("ser", DECK):     # every form of ser now known
            learn(c, i)
        assert "ser" not in c.batch(DECK)
        assert "saber" in c.batch(DECK)     # the eleventh verb took the slot
        assert len(c.batch(DECK)) == CONJUGATION_BATCH

    def test_a_partly_learned_verb_keeps_its_slot(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        chain = c._chain("ser", DECK)
        for i in chain[:-1]:                # all but the last form
            learn(c, i)
        assert "ser" in c.batch(DECK)


class TestUnlocking:
    def test_the_first_form_waits_on_the_main_tracker(self, main, tmp_path):
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert not c.unlocked(form_index("tener", "pres-yo"), DECK)
        learn(main, index_of("tener"))
        c2 = ConjugationProgress.open(main, path=tmp_path / "c2.json")
        assert c2.unlocked(form_index("tener", "pres-yo"), DECK)

    def test_one_form_at_a_time_after_that(self, main, tmp_path):
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert not c.unlocked(form_index("tener", "pres-tu"), DECK)
        learn(c, form_index("tener", "pres-yo"))
        assert c.unlocked(form_index("tener", "pres-tu"), DECK)
        assert not c.unlocked(form_index("tener", "pres-nos"), DECK)

    def test_vocabulary_is_never_unlocked_here(self, main, tmp_path):
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert not c.unlocked(index_of("tener"), DECK)
        assert not c.unlocked(index_of("casa"), DECK)

    def test_learning_a_form_in_the_main_tracker_does_not_open_the_next_here(
            self, main, tmp_path):
        """The two chains are genuinely separate, not two views of one.

        The main drill unlocks conjugations too, on option (b). Progress it
        makes there must not silently advance this tracker's chain, or the
        separation is only half real.
        """
        learn(main, index_of("tener"))
        learn(main, form_index("tener", "pres-yo"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert not c.unlocked(form_index("tener", "pres-tu"), DECK)


class TestDialect:
    def test_spain_drills_vosotros_and_latin_america_does_not(self, main,
                                                              tmp_path):
        learn(main, index_of("tener"))
        main.dialect = "es-ES"
        spain = ConjugationProgress.open(main, path=tmp_path / "es.json")
        main.dialect = "es-MX"
        latin = ConjugationProgress.open(main, path=tmp_path / "mx.json")
        assert len(spain._chain("tener", DECK)) == 30       # 5 tenses x 6
        assert len(latin._chain("tener", DECK)) == 25       # 5 tenses x 5
        assert not latin.unlocked(form_index("tener", "pres-vos"), DECK)

    def test_a_verb_finishes_without_vosotros_on_latin_america(self, main,
                                                              tmp_path):
        learn(main, index_of("tener"))
        main.dialect = "es-MX"
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        for i in c._chain("tener", DECK):
            learn(c, i)
        assert c.finished("tener", DECK)


class TestQueue:
    def test_offers_one_new_form_per_verb_and_no_more(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        fresh = c.unseen_indexes()
        assert len(fresh) == CONJUGATION_BATCH
        assert {DECK[i].form for i in fresh} == {"pres-yo"}
        assert {DECK[i].lemma for i in fresh} == set(c.batch(DECK))

    def test_never_offers_a_vocabulary_card(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert all(DECK[i].lemma for i in c.unseen_indexes())

    def test_a_retired_verb_still_comes_back_for_review(self, main, tmp_path):
        """The batch gates what is met, never what is reviewed."""
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        chain = c._chain("ser", DECK)
        for i in chain:
            learn(c, i)
        assert "ser" not in c.batch(DECK)       # retired
        c.cards[chain[0]].due = today() - 1     # but one form falls due
        due, _ = c.queue_parts()
        assert chain[0] in due

    def test_a_main_drill_category_filter_does_not_leak_in(self, main,
                                                           tmp_path):
        learn(main, index_of("tener"))
        main.category = "noun"          # would empty a verb-only queue
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert c.category == "all"
        assert c.unseen_indexes()


class TestSettings:
    def test_are_borrowed_from_the_main_tracker(self, main, tmp_path):
        main.dialect = "es-MX"
        main.window = 9.5
        main.model = "small"
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert (c.dialect, c.window, c.model) == ("es-MX", 9.5, "small")

    def test_borrowed_rather_than_kept(self, tmp_path, main):
        """A stale copy on disk must lose to what the main tracker says now."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"cards": {}, "dialect": "es-MX"}))
        main.dialect = "es-ES"
        assert ConjugationProgress.open(main, path=path).dialect == "es-ES"


class Answers:
    """Answers every card correctly, or every card wrongly."""

    def __init__(self, correct=True):
        self.correct = correct
        self.asked = []
        self.last_audio = None

    def bind(self, session):
        self.session = session
        return self

    def listen(self, window, should_stop=None, accept=None, fast=False,
               steer=None, second_pass=False):
        index = self.session.current
        self.asked.append(index)
        if not self.correct:
            return "xxxxxxxx"
        return DECK[index].answers[0]

    def calibrate(self):
        pass


def run(conj, correct=True, limit=None):
    listener = Answers(correct)
    session = ConjugationSession(conj, listener, verifier=None)
    listener.bind(session)
    if limit is not None:
        conj.queue_override = limit
    session.run()
    return session


class TestSeparation:
    """The reason the mode has its own tracker at all."""

    def test_a_whole_session_leaves_the_vocabulary_file_alone(self, main,
                                                              tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        main.save()
        before = (tmp_path / "progress.json").read_text()

        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        run(c, limit=[form_index("tener", "pres-yo")])

        assert (tmp_path / "progress.json").read_text() == before

    def test_answers_move_only_the_conjugation_schedule(self, main, tmp_path):
        learn(main, index_of("tener"))
        vocabulary = dict(main.cards[index_of("tener")].to_dict())
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        run(c, limit=[form_index("tener", "pres-yo")])

        assert main.cards[index_of("tener")].to_dict() == vocabulary
        assert form_index("tener", "pres-yo") in c.cards

    def test_the_conjugation_file_holds_no_vocabulary(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        run(c, limit=[form_index("ser", "pres-yo"),
                      form_index("tener", "pres-yo")])
        assert all(DECK[i].lemma for i in c.cards)

    def test_saves_under_the_form_id_not_the_position(self, main, tmp_path):
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        run(c, limit=[form_index("tener", "pres-yo")])
        c.save()
        stored = json.loads((tmp_path / "c.json").read_text())
        assert list(stored["cards"]) == ["tener:pres-yo"]


class TestSession:
    def test_a_correct_answer_climbs_the_same_ladder(self, main, tmp_path):
        """Nothing about grading or scheduling is special here.

        One right answer is not enough to finish a card in the ordinary drill
        either: it has to clear every rung of the session ladder first.
        """
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        index = form_index("tener", "pres-yo")
        session = run(c, limit=[index])
        # The ladder means one right answer never finishes a card: it is asked
        # again at every rung before it is let go for the day.
        assert len(session.listener.asked) > 1
        assert c.cards[index].reps >= 1
        assert c.learned(index)

    def test_a_wrong_answer_is_a_lapse_here_too(self, main, tmp_path):
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        index = form_index("tener", "pres-yo")
        listener = Answers(correct=False)
        session = ConjugationSession(c, listener, verifier=None)
        listener.bind(session)
        c.queue_override = [index]
        # A missed card requeues forever by design, so the run has to be cut
        # short rather than waited out.
        original = session._judge

        def judge(*args, **kw):
            original(*args, **kw)
            if len(listener.asked) >= 3:
                session.stop()
        session._judge = judge
        session.run()
        assert not c.learned(index)
        assert c.cards[index].reps == 0

    def test_no_verbs_learned_says_so_rather_than_queue_clear(self, conj):
        """An empty screen must not read the same as a finished one."""
        said = []
        session = ConjugationSession(conj, Answers(), verifier=None)
        session.on_status = said.append
        session.run()
        assert said and "No verbs learned" in said[0]
        assert "Queue clear." not in said

    def test_summary_counts_verbs_rather_than_forms(self, main, tmp_path):
        for lemma in verb_order(DECK)[:12]:
            learn(main, index_of(lemma))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        for i in c._chain("ser", DECK):
            learn(c, i)
        session = ConjugationSession(c, Answers(), verifier=None)
        s = session.summary()
        assert s["done"] == ["ser"]
        assert s["remaining"] == 11
        assert "ser" not in s["verbs"]


class TestTheCueIsNotReadAloud:
    """Long English cues, one-syllable answers, and the cue already on screen.

    The way *out* is untouched: a missed form still speaks the answer and its
    example, which is the part that teaches.
    """

    def test_it_is_off_here_and_on_in_the_ordinary_drill(self, main, conj):
        assert main.speak_cue is True
        assert conj.speak_cue is False

    def test_the_main_setting_does_not_switch_it_back_on(self, main, tmp_path):
        main.speak_cue = True
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        assert c.speak_cue is False

    def test_it_survives_a_save_and_reload(self, main, tmp_path):
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        c.speak_cue = True              # if you ever do want it
        c.save()
        assert ConjugationProgress.open(main, path=tmp_path / "c.json").speak_cue

    def test_no_cue_is_spoken(self, main, tmp_path, monkeypatch):
        from spanish_drill import session as sm
        spoken = []
        monkeypatch.setattr(sm, "say_english", lambda t: spoken.append(t))
        monkeypatch.setattr(sm, "say_spanish", lambda t, d: None)
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        run(c, limit=[form_index("tener", "pres-yo")])
        assert spoken == []

    def test_but_a_miss_still_speaks_the_answer(self, main, tmp_path,
                                               monkeypatch):
        from spanish_drill import session as sm
        said = []
        monkeypatch.setattr(sm, "say_english", lambda t: None)
        monkeypatch.setattr(sm, "say_spanish", lambda t, d: said.append(t))
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        index = form_index("tener", "pres-yo")
        listener = Answers(correct=False)
        session = ConjugationSession(c, listener, verifier=None)
        listener.bind(session)
        c.queue_override = [index]
        original = session._judge

        def judge(*a, **kw):
            original(*a, **kw)
            session.stop()
        session._judge = judge
        session.run()
        assert DECK[index].answers[0] in said, (
            "the answer after a miss is the part that teaches and must "
            "always be spoken")


class TestTheSecondOpinionIsSteeredForConjugations:
    def test_the_session_defaults_to_the_conjugation_verifier(self, conj):
        from spanish_drill.paradigm import _conjugation_verifier
        session = ConjugationSession(conj, Answers())
        assert session.verifier is _conjugation_verifier

    def test_an_explicit_verifier_still_wins(self, conj):
        session = ConjugationSession(conj, Answers(), verifier=None)
        assert session.verifier is None

    def test_the_local_model_is_steered_too(self, main, tmp_path):
        """A steer that only reaches the double-check leaves the first
        verdict, and every early accept, still expecting infinitives."""
        from spanish_drill.transcribe import CONJUGATION_STEER
        learn(main, index_of("tener"))
        c = ConjugationProgress.open(main, path=tmp_path / "c.json")
        seen = []

        class Recording(Answers):
            def listen(self, window, should_stop=None, accept=None,
                       fast=False, steer=None, second_pass=False):
                seen.append(steer)
                return super().listen(window, should_stop, accept, fast)


        listener = Recording()
        session = ConjugationSession(c, listener, verifier=None)
        listener.bind(session)
        c.queue_override = [form_index("tener", "pres-yo")]
        session.run()
        assert seen and set(seen) == {CONJUGATION_STEER}


class TestItIsFastEnoughToUse:
    def test_the_quick_model_decides(self):
        """The main model costs the better part of ten seconds on a long
        window and only ever changes misses, which go to the second opinion
        regardless. Placement made the same trade for the same reason."""
        assert ConjugationSession.fast_recognition is True

    def test_which_means_the_scout_answer_must_still_retry(self):
        """With `fast` on, the scout IS the graded answer, so an echo there is
        a card marked silent rather than a poll that will come round again."""
        from spanish_drill import listener as L
        calls = []

        class T:
            def transcribe(self, audio, scout=False, steer=None, retry=None):
                calls.append((scout, retry))
                return "tengo"

        class R:
            floor = 0.02
            def record(self, window, should_stop=None, on_pause=None):
                return "speech", "full", True, False
            def open(self): pass
            def close(self): pass
            def calibrate(self): return 0.02
            def set_device(self, n): pass

        L.Listener(transcriber=T(), recorder=R()).listen(5.0, fast=True)
        assert calls == [(True, True)]


class TestTheSteerNeverNamesTheAnswer:
    """The one rule that cannot be traded away for a better hit rate.

    Measured on twelve real clips this drill correctly marked wrong: with the
    expected answer written into the prompt, nine of them came back CORRECT.
    "Arco." decoded as "Hago." and "Pas, pas, pas." as "Vas, vas, vas." The
    recogniser does not check the answer against the audio, it writes down
    what it was told to expect. A drill that does this marks everything right
    and teaches nothing.
    """

    def test_no_steer_contains_a_deck_answer(self):
        from spanish_drill.transcribe import assert_steer_is_clean
        assert_steer_is_clean()

    def test_the_conjugation_steer_names_no_verb_and_no_form(self):
        from spanish_drill.transcribe import CONJUGATION_STEER
        from spanish_drill.text import normalize
        shown = set(normalize(CONJUGATION_STEER).split())
        forms = {normalize(a) for c in DECK if c.lemma for a in c.answers}
        lemmas = {normalize(c.lemma) for c in DECK if c.lemma}
        assert not (shown & forms), shown & forms
        assert not (shown & lemmas), shown & lemmas

    def test_the_steer_does_not_vary_with_the_card(self):
        """A per-card steer is how the answer gets in. It is a constant."""
        from spanish_drill.transcribe import CONJUGATION_STEER
        assert isinstance(ConjugationSession.steer, str)
        assert ConjugationSession.steer == CONJUGATION_STEER


class TestTheBlindSecondPass:
    """A steer is a prior, and on a one-syllable word the prior can swamp
    the signal. Measured on clean recordings of known forms: the steer turned
    `he` into "Hi." and `tengo` into a miss, both of which decode perfectly
    with no prompt. Steer alone 12/14, no prompt 14/14, both 14/14, and no
    wrong person or tense was accepted by any of them. On the owner's own
    recordings, 18 accepted became 22 of 42.
    """

    class T:
        def __init__(self):
            self.steers = []

        def transcribe(self, audio, scout=False, steer=None, retry=None):
            self.steers.append(steer)
            # The steered reading is wrong; the unprompted one is right.
            return "hi" if steer else "tengo"

    class R:
        floor = 0.02
        def record(self, window, should_stop=None, on_pause=None):
            return "speech", "full", True, False
        def open(self): pass
        def close(self): pass
        def calibrate(self): return 0.02
        def set_device(self, n): pass

    def listen(self, **kw):
        from spanish_drill.listener import Listener
        t = self.T()
        got = Listener(transcriber=t, recorder=self.R()).listen(
            5.0, accept=lambda text: text == "tengo", steer="S", **kw)
        return got, t.steers

    def test_it_is_on_for_conjugations(self):
        assert ConjugationSession.second_pass is True

    def test_it_is_off_for_the_ordinary_drill(self):
        """There the extra decode is the slow model, and a miss goes to the
        API second opinion anyway, which is quicker and better."""
        from spanish_drill.session import DrillSession
        assert DrillSession.second_pass is False

    def test_a_missed_reading_is_looked_at_again_with_no_prompt(self):
        got, steers = self.listen(second_pass=True)
        assert got == "tengo"
        assert steers == ["S", ""], "the second look must suggest nothing"

    def test_off_by_default(self):
        got, steers = self.listen()
        assert got == "hi" and steers == ["S"]

    def test_a_reading_that_already_matches_is_not_decoded_twice(self):
        from spanish_drill.listener import Listener
        t = self.T()
        got = Listener(transcriber=t, recorder=self.R()).listen(
            5.0, accept=lambda text: True, steer="S", second_pass=True)
        assert got == "hi" and t.steers == ["S"]

    def test_the_second_look_never_carries_the_expected_answer(self):
        """The whole safety of this. An empty prompt suggests nothing, so it
        cannot pull a wrong reading toward the answer the way naming it does.
        Measured: naming the answer scored the same 14/14 on clean audio but
        broke echo detection on real clips, turning correct answers into
        silence, because a transcript of the answer and an echo of the prompt
        become the same string."""
        _, steers = self.listen(second_pass=True)
        assert all(s in ("S", "") for s in steers)
        assert "" in steers


class TestOnlyFormsEverReachThisQueue:
    """The separation that survived the two files becoming one.

    `unlocked` refuses vocabulary, but it only decides what gets introduced.
    Reviews are drawn from whatever is already due and never pass through it,
    which was harmless while this tracker had a file of its own and wrong the
    moment both shared one. Measured on a real schedule at the time: every one
    of the 167 cards this drill offered was ordinary vocabulary, and not one
    was a conjugated form.
    """

    def test_a_due_word_is_not_offered_as_a_paradigm(self, conj):
        word = index_of("hola")
        learn(conj, word, interval=0)           # due right now
        conj.cards[word].due = today() - 1
        due, fresh = conj.queue_parts()
        assert word not in due + fresh

    def test_a_due_form_still_is(self, conj, main):
        main.cards[index_of("tener")] = Card(interval=99, reps=9,
                                             due=today() + 99)
        conj.known = known_verbs(main)
        form = form_index("tener", "pres-yo")
        learn(conj, form, interval=0)
        conj.cards[form].due = today() - 1
        due, fresh = conj.queue_parts()
        assert form in due + fresh

    def test_the_filter_is_about_forms_not_the_category_setting(self, conj):
        """`open` forces category to "all", so nothing else was filtering."""
        assert conj.category == "all"
        assert conj.in_category(form_index("ser", "pres-yo"), DECK)
        assert not conj.in_category(index_of("hola"), DECK)

    def test_the_panel_and_the_queue_cannot_disagree(self, conj):
        """Both read the same question, so a count cannot promise a card the
        drill will refuse to ask."""
        word = index_of("hola")
        learn(conj, word, interval=0)
        conj.cards[word].due = today() - 1
        due, fresh = conj.queue_parts()
        assert len(due) + len(fresh) == sum(
            1 for i in conj.cards if conj.in_category(i, DECK)
            and conj.cards[i].due <= today())
