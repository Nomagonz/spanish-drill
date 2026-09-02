"""Phase two: conjugated forms and the ladder that lets them in.

Two things are being protected here. The forms themselves have to be right,
because a wrong one is worse than a missing one: it teaches you Spanish that
is not Spanish. And the gate has to hold, because its whole purpose is to stop
nine hundred cards arriving at once.
"""
import json

import pytest

from spanish_drill.conjugation import (PERSONS, TENSES, VERBS, conjugate, cue,
                                       forms_for)
from spanish_drill.config import UNLOCK_REPS
from spanish_drill.deck import index_by_id, load_deck
from spanish_drill.progress import Progress
from spanish_drill.scheduler import MATURE_AT, Card

DECK = load_deck()
IDS = index_by_id(DECK)


def known(reps=UNLOCK_REPS, interval=1):
    """Answered right at least once, so it is in the review rotation."""
    return Card(ease=2.5, interval=interval, reps=reps, lapses=0, due=0)


def unlearned():
    """Seen and missed: reps is reset to zero by a lapse."""
    return Card(ease=2.5, interval=0, reps=0, lapses=1, due=0)


def placement_passed():
    """What a placement run writes: three weeks out, off two quick answers."""
    return Card(ease=2.5, interval=MATURE_AT, reps=2, lapses=0, due=0)


def progress_with(**cards):
    p = Progress(path=None)
    p.cards = {IDS[card_id]: state for card_id, state in cards.items()}
    return p


class TestTheFormsAreRight:
    """Spot checks on the shapes that a rule engine gets wrong."""

    @pytest.mark.parametrize("lemma,tense,expected", [
        ("tener", "pres",
         ("tengo", "tienes", "tenemos", "tiene", "tenéis", "tienen")),
        ("tener", "pret",
         ("tuve", "tuviste", "tuvimos", "tuvo", "tuvisteis", "tuvieron")),
        ("tener", "fut",
         ("tendré", "tendrás", "tendremos", "tendrá", "tendréis", "tendrán")),
        ("ser", "imp", ("era", "eras", "éramos", "era", "erais", "eran")),
        ("ir", "pret", ("fui", "fuiste", "fuimos", "fue", "fuisteis", "fueron")),
        # c -> z so the sound survives, and -eron rather than -ieron after a j
        ("hacer", "pret",
         ("hice", "hiciste", "hicimos", "hizo", "hicisteis", "hicieron")),
        ("decir", "pret",
         ("dije", "dijiste", "dijimos", "dijo", "dijisteis", "dijeron")),
        ("traer", "pret",
         ("traje", "trajiste", "trajimos", "trajo", "trajisteis", "trajeron")),
        # the stem shifts in the third person only
        ("pedir", "pret",
         ("pedí", "pediste", "pedimos", "pidió", "pedisteis", "pidieron")),
        ("dormir", "pret",
         ("dormí", "dormiste", "dormimos", "durmió", "dormisteis",
          "durmieron")),
        # the accent on the infinitive is a spelling device and does not carry
        ("oír", "fut", ("oiré", "oirás", "oiremos", "oirá", "oiréis", "oirán")),
        ("oír", "pres", ("oigo", "oyes", "oímos", "oye", "oís", "oyen")),
        ("hablar", "pret",
         ("hablé", "hablaste", "hablamos", "habló", "hablasteis",
          "hablaron")),
    ])
    def test_known_paradigms(self, lemma, tense, expected):
        assert conjugate(lemma, tense) == expected

    def test_every_verb_has_every_tense(self):
        for lemma in VERBS:
            for tense in TENSES:
                forms = conjugate(lemma, tense)
                assert len(forms) == len(PERSONS)
                assert all(forms), (lemma, tense)

    def test_the_conditional_is_the_future_stem(self):
        """Knowing tendré should get you most of tendría."""
        for lemma in VERBS:
            fut = conjugate(lemma, "fut")[0]
            cond = conjugate(lemma, "cond")[0]
            assert fut[:-1] == cond[:-2], lemma


class TestTheCuesAreAnswerable:
    def test_will_is_the_future_and_would_is_the_conditional(self):
        assert cue("tener", "fut", "yo").startswith("I will have")
        assert cue("tener", "cond", "yo").startswith("I would have")

    def test_the_two_pasts_never_share_a_cue(self):
        """"I had" alone cannot be answered: it is both tuve and tenía."""
        for lemma in VERBS:
            for person in PERSONS:
                assert cue(lemma, "pret", person) != cue(lemma, "imp", person)

    def test_no_cue_is_ambiguous_between_two_verbs(self):
        seen = {}
        for lemma in VERBS:
            for tense in TENSES:
                for _, spanish, text in forms_for(lemma, tense):
                    if text in seen and seen[text] != spanish:
                        pytest.fail(f"{text!r} wants both {seen[text]!r} "
                                    f"and {spanish!r}")
                    seen[text] = spanish

    def test_can_never_becomes_i_will_can(self):
        assert cue("poder", "fut", "yo") == "I will be able to"
        assert cue("poder", "imp", "yo") == "I used to be able to"

    def test_each_cue_has_at_most_one_aside(self):
        for lemma in VERBS:
            for tense in TENSES:
                for _, _, text in forms_for(lemma, tense):
                    assert text.count("(") <= 1, text


class TestTheDeckCarriesTheForms:
    def test_conjugations_are_tagged_as_verbs(self):
        """So that picking "verb" drills a verb and its endings together."""
        assert all(c.pos == "verb" for c in DECK if c.lemma)

    def test_every_form_points_at_a_real_infinitive(self):
        for card in DECK:
            if card.lemma:
                assert card.lemma in IDS, card.id

    def test_the_id_names_the_verb_and_the_form(self):
        card = DECK[IDS["tener:pres-yo"]]
        assert card.answers == ("tengo",) and card.lemma == "tener"


class TestTheLadder:
    def test_vocabulary_is_never_locked(self):
        p = progress_with()
        assert p.unlocked(IDS["tener"])

    def test_a_form_is_locked_until_its_infinitive_is_known(self):
        p = progress_with()
        assert not p.unlocked(IDS["tener:pres-yo"])
        p = progress_with(tener=known())
        assert p.unlocked(IDS["tener:pres-yo"])

    def test_one_correct_answer_is_the_bar(self):
        """Not maturity. Waiting three weeks per tense would put the
        conditional most of a year out."""
        p = progress_with(tener=known(reps=1, interval=1))
        assert p.unlocked(IDS["tener:pres-yo"])

    def test_a_placement_pass_counts(self):
        """Two quick correct answers is still two correct answers."""
        p = progress_with(tener=placement_passed())
        assert p.unlocked(IDS["tener:pres-yo"])

    def test_forgetting_it_closes_what_it_opened(self):
        """A lapse resets reps, so the next variation stops arriving until
        the one before it comes back."""
        p = progress_with(tener=unlearned())
        assert not p.unlocked(IDS["tener:pres-yo"])

    def test_the_present_comes_before_the_preterite(self):
        p = progress_with(tener=known())
        assert p.unlocked(IDS["tener:pres-yo"])
        assert not p.unlocked(IDS["tener:pret-yo"])

    def test_the_preterite_waits_for_every_person_of_the_present(self):
        forms = {f"tener:pres-{person}": known() for person in PERSONS}
        all_but_one = dict(forms)
        all_but_one["tener:pres-ellos"] = unlearned()
        p = progress_with(tener=known(), **all_but_one)
        assert not p.unlocked(IDS["tener:pret-yo"]), (
            "the last person of the present was still being learnt")
        p = progress_with(tener=known(), **forms)
        assert p.unlocked(IDS["tener:pret-yo"])

    def test_a_later_tense_stays_shut_until_its_turn(self):
        p = progress_with(tener=known(),
                          **{f"tener:pres-{x}": known() for x in PERSONS})
        assert p.unlocked(IDS["tener:pret-yo"])
        for tense in ("imp", "fut", "cond"):
            assert not p.unlocked(IDS[f"tener:{tense}-yo"]), tense

    def test_one_verb_unlocking_does_not_unlock_another(self):
        p = progress_with(tener=known())
        assert p.unlocked(IDS["tener:pres-yo"])
        assert not p.unlocked(IDS["hablar:pres-yo"])

    def test_only_one_new_form_of_a_verb_is_ever_waiting(self):
        """Never a whole tense at once. Five near-identical forms arriving
        together is the worst case for interference: they compete to be the
        answer to almost the same cue."""
        p = progress_with(tener=known())
        offered = [i for i in p.unseen_indexes() if DECK[i].lemma == "tener"]
        assert offered == [IDS["tener:pres-yo"]], (
            [DECK[i].id for i in offered])

    def test_the_chain_advances_one_form_at_a_time(self):
        state = {"tener": known()}
        expected = [f"tener:{t}-{x}" for t in TENSES for x in PERSONS]
        for form_id in expected:
            p = progress_with(**state)
            waiting = [DECK[i].id for i in p.unseen_indexes()
                       if DECK[i].lemma == "tener"]
            assert waiting == [form_id], f"expected only {form_id}, got {waiting}"
            state[form_id] = known()        # learn it, and the next opens

    def test_the_card_never_names_the_verb(self, tmp_path):
        """The label showed the infinitive, which is most of the answer.

        Seeing "volver" beside "I return" leaves only the ending to produce.
        Working out which verb a cue wants is part of the card.
        """
        from spanish_drill.progress import Progress
        from spanish_drill.session import DrillSession
        deck = load_deck()
        session = DrillSession(Progress(path=tmp_path / "p.json"), None,
                               verifier=None)
        for card_id in ("volver:pres-yo", "tener:pret-el", "ir:fut-nos"):
            index = next(i for i, c in enumerate(deck) if c.id == card_id)
            label = session._state_label(index)
            lemma = deck[index].lemma
            assert lemma not in label, f"{label!r} gives away {lemma!r}"

    def test_no_label_leaks_any_spanish(self, tmp_path):
        """Nothing on the card may contain the word being asked for."""
        from spanish_drill.progress import Progress
        from spanish_drill.session import DrillSession
        deck = load_deck()
        session = DrillSession(Progress(path=tmp_path / "p.json"), None,
                               verifier=None)
        for index, card in enumerate(deck):
            if not card.lemma:
                continue
            label = session._state_label(index).lower()
            assert card.answers[0].lower() not in label, card.id

    def test_the_person_order_is_the_one_that_was_asked_for(self):
        assert PERSONS == ("yo", "tu", "nos", "el", "vos", "ellos")

    def test_locked_forms_are_never_offered_as_new(self):
        p = progress_with(tener=known())
        offered = set(p.unseen_indexes())
        assert IDS["tener:pres-yo"] in offered
        assert IDS["tener:pret-yo"] not in offered
        assert IDS["hablar:pres-yo"] not in offered


class TestVosotros:
    """Everyday speech in Spain, and absent from Latin America entirely."""

    def test_it_is_taught_on_the_spain_dialect(self):
        p = progress_with(tener=known())
        p.dialect = "es-ES"
        chain = [DECK[i].id for i in p._chain("tener", DECK)]
        assert "tener:pres-vos" in chain

    def test_it_is_absent_on_latin_american_spanish(self):
        p = progress_with(tener=known())
        p.dialect = "es-MX"
        chain = [DECK[i].id for i in p._chain("tener", DECK)]
        assert not any(":pres-vos" in c for c in chain)
        assert not p.unlocked(IDS["tener:pres-vos"])

    def test_dropping_it_does_not_leave_a_hole_in_the_chain(self):
        """On es-MX the chain has to close up, or `tenéis` would become a step
        you could never pass and the preterite would never open."""
        state = {"tener": known()}
        for person in ("yo", "tu", "nos", "el"):
            state[f"tener:pres-{person}"] = known()
        p = progress_with(**state)
        p.dialect = "es-MX"
        waiting = [DECK[i].id for i in p.unseen_indexes()
                   if DECK[i].lemma == "tener"]
        assert waiting == ["tener:pres-ellos"]

    def test_it_comes_just_before_they(self):
        p = progress_with(tener=known())
        p.dialect = "es-ES"
        chain = [DECK[i].id for i in p._chain("tener", DECK)]
        assert chain.index("tener:pres-vos") == chain.index("tener:pres-ellos") - 1

    def test_you_all_is_never_confused_with_you_or_they(self):
        assert cue("tener", "pres", "vos") == "you all have (to own)"
        assert cue("tener", "pres", "tu") == "you have (to own)"
        assert cue("tener", "pres", "ellos") == "they have (to own)"


class TestProgressIsKeyedById:
    """Saving by position is what handed one word's history to another."""

    def test_it_saves_under_the_card_id(self, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        p.cards = {IDS["tener"]: known()}
        p.save()
        raw = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
        assert list(raw["cards"]) == ["tener"]

    def test_it_round_trips(self, tmp_path):
        p = Progress(path=tmp_path / "p.json")
        p.cards = {IDS["tener"]: known(), IDS["hablar:pres-yo"]: known()}
        p.save()
        back = Progress.load(tmp_path / "p.json")
        assert back.cards.keys() == p.cards.keys()

    def test_a_file_written_before_ids_still_loads(self, tmp_path):
        """Old saves are keyed by position and have to keep working."""
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"cards": {"3": known().to_dict()}}),
                        encoding="utf-8")
        assert 3 in Progress.load(path).cards

    def test_a_card_that_left_the_deck_is_dropped_not_guessed_at(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps(
            {"cards": {"tener": known().to_dict(),
                       "wordthatwentaway": known().to_dict()}}), encoding="utf-8")
        loaded = Progress.load(path)
        assert loaded.cards.keys() == {IDS["tener"]}

    def test_editing_the_deck_no_longer_moves_history(self, tmp_path):
        """The bug this replaced: history followed the slot, not the word."""
        path = tmp_path / "p.json"
        p = Progress(path=path)
        p.cards = {IDS["tener"]: known()}
        p.save()
        saved = json.loads(path.read_text(encoding="utf-8"))["cards"]
        assert "tener" in saved and not any(k.isdigit() for k in saved)


class TestEveryFormHasASentence:
    """A conjugation card with no example teaches the word and not its use.

    All 1170 of them were blank, so a missed form showed the answer alone with
    an empty space under it where the sentence goes.
    """

    @staticmethod
    def fold(s):
        import unicodedata
        d = unicodedata.normalize("NFD", (s or "").lower())
        return "".join(c for c in d if unicodedata.category(c) != "Mn")

    def test_every_conjugation_has_an_example_and_a_gloss(self):
        blank = [c.id for c in load_deck()
                 if c.lemma and not (c.example.strip() and c.gloss.strip())]
        assert not blank, f"{len(blank)} conjugations have no sentence"

    def test_the_sentence_actually_uses_the_form_it_teaches(self):
        """A sentence that does not contain the word is worse than none: it
        sits on screen as though it were showing you the form in use."""
        import re
        missing = []
        for card in load_deck():
            if not card.lemma:
                continue
            form = self.fold(card.answers[0])
            if not re.search(rf"(?<![a-zñ]){re.escape(form)}(?![a-zñ])",
                             self.fold(card.example)):
                missing.append((card.id, card.answers[0], card.example))
        assert not missing, f"{len(missing)} examples omit their own form: {missing[:3]}"

    def test_a_verb_does_not_reuse_one_sentence_for_every_person(self):
        """Otherwise the example is the cue with the ending swapped, which
        teaches the ending and nothing about using the word."""
        from collections import defaultdict
        by_lemma = defaultdict(list)
        for card in load_deck():
            if card.lemma:
                by_lemma[card.lemma].append(card.example)
        lazy = {lemma: len(set(rows)) for lemma, rows in by_lemma.items()
                if len(set(rows)) < len(rows) * 0.5}
        assert not lazy, f"these verbs reuse the same sentence: {lazy}"
