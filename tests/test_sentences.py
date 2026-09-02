"""Composing whole sentences out of words already known.

The gate is what this mode lives or dies by. A sentence built from a word you
have never seen is not a composition exercise, it is a vocabulary test with
no warning, and the tests here exist mostly to make sure that cannot happen.
"""
import tempfile
from pathlib import Path

import pytest

from spanish_drill import sentences as S
from spanish_drill.composition import SentenceDrill
from spanish_drill.deck import index_by_id, load_deck
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card, MATURE_AT


# -- helpers --------------------------------------------------------------
# Never the real progress.json. Nothing here calls save(), but a Progress
# built with the default path is one added save() away from a test writing
# over weeks of real scheduling.
_SCRATCH = Path(tempfile.mkdtemp(prefix="drill-sentences-")) / "progress.json"


def progress_with(known=(), interval=MATURE_AT, path=None, **states):
    """A Progress where exactly these card ids sit at the given interval."""
    ids = index_by_id()
    p = Progress(path=path or _SCRATCH)
    for card_id in known:
        p.cards[ids[card_id]] = Card(interval=interval, reps=2, due=0)
    for card_id, value in states.items():
        p.cards[ids[card_id]] = Card(interval=value, reps=1 if value else 0,
                                     due=0)
    return p


def sentence(es, en="cue", needs=None, also=()):
    derived, _ = S.requirements(es, needs)
    return S.Sentence(en=en, es=es, needs=derived, also=tuple(also))


def everything_for(*items):
    """Every card these sentences need, conjugated forms included.

    Knowing `tener` is no longer enough to be asked for `tengo`: the form is
    gated too, so a test granting only the infinitive is testing the old rule.
    """
    out = []
    for item in items:
        out += list(item.needs) + list(S._form_requirements(item, set()))
    return tuple(dict.fromkeys(out))


class Typer:
    """A listener that hands over scripted answers, then stops the drill."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.drill = None
        self.last_audio = None

    def clear(self):
        pass

    def listen(self, window=None, should_stop=None, accept=None,
               fast=False, steer=None, second_pass=False):
        if not self.answers:
            if self.drill:
                self.drill.stop()
            return None
        return self.answers.pop(0)


def run(drill, typer):
    typer.drill = drill
    drill.run()
    return drill


# -- the bank itself ------------------------------------------------------
class TestTheBankIsWellFormed:
    """Every property here is objective. None of it is a judgement about
    whether a sentence is a good one to learn from."""

    def test_every_word_is_accounted_for_by_the_deck(self):
        """The guard the whole gate rests on. A word the deck cannot resolve
        is a word the gate never checks, so the sentence could be offered
        while leaning on vocabulary that was never learned."""
        broken = {}
        for item in S.load_sentences():
            for wording in item.wordings:
                _, unknown = S.requirements(wording)
                if unknown:
                    broken[wording] = sorted(set(unknown))
        assert not broken, f"words no card accounts for: {broken}"

    def test_every_requirement_is_a_real_card(self):
        ids = {c.id for c in load_deck()}
        for item in S.load_sentences():
            missing = [n for n in item.needs if n not in ids]
            assert not missing, f"{item.es}: {missing}"

    def test_no_sentence_is_requirement_free(self):
        """A sentence of nothing but function words would be permanently
        available, gated on nothing at all."""
        for item in S.load_sentences():
            assert item.needs, item.es

    def test_no_sentence_appears_twice(self):
        seen = [i.es for i in S.load_sentences()]
        assert len(set(seen)) == len(seen)

    def test_no_cue_is_empty(self):
        for item in S.load_sentences():
            assert item.en.strip() and item.es.strip()

    def test_they_are_short(self):
        """Short and common was the requirement. A long sentence has more
        than one right answer however carefully it is written."""
        for item in S.load_sentences():
            assert len(S.tokens(item.es)) <= 12, item.es

    def test_every_alternate_wording_is_really_different(self):
        for item in S.load_sentences():
            assert item.es not in item.also, item.es


# -- the gate -------------------------------------------------------------
class TestNothingUsesAWordYouHaveNotLearned:
    def test_all_words_known_makes_it_available(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        assert S.is_available(item, S.known_ids(p))

    def test_a_word_never_seen_blocks_it(self):
        """The whole point. A card absent from progress.json has never been
        answered, so it can never be in the known set."""
        item = sentence("Tengo vino blanco.")
        p = progress_with([n for n in everything_for(item) if n != "blanco"])
        assert not S.is_available(item, S.known_ids(p))
        assert "blanco" in S.blocked_by(item, S.known_ids(p))

    def test_a_word_seen_but_not_yet_answered_right_blocks_it(self):
        """Interval zero is a card that is new or has just lapsed. Being in
        the file is not the same as being known."""
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        p.cards[index_by_id()["blanco"]] = Card(interval=0, reps=0, due=0)
        assert not S.is_available(item, S.known_ids(p))

    def test_a_lapsed_word_drops_out_again(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        assert S.is_available(item, S.known_ids(p))
        p.cards[index_by_id()["vino"]].interval = 0        # forgotten
        assert not S.is_available(item, S.known_ids(p))

    def test_words_still_being_learned_count(self):
        """The bar is the rotation, not maturity: answered right once and
        scheduled a day ahead is enough."""
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item), interval=1)
        assert S.is_available(item, S.known_ids(p))

    def test_raising_the_bar_to_maturity_excludes_them(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item), interval=1)
        assert not S.is_available(item, S.known_ids(p, bar=MATURE_AT))

    def test_function_words_never_gate(self):
        """`el`, `la`, `un` and `no` have no card at all. Requiring them
        would mean no sentence is ever available however much has been
        learned."""
        item = sentence("No tengo el vino.")
        assert set(item.needs) <= {"tener", "vino", "venir"}
        for word in ("no", "el", "un", "una", "de", "en", "que"):
            assert word in S.FUNCTION_WORDS

    def test_a_word_that_is_two_cards_needs_both(self):
        """`trabajo` is the noun and also "I work". Nothing in the spelling
        says which, so the strict reading is the safe one."""
        item = sentence("Trabajo en una oficina.")
        assert "trabajo" in item.needs and "trabajar" in item.needs

    def test_available_filters_the_whole_bank(self):
        item = sentence("Tengo un perro.")
        granted = set(everything_for(item))
        p = progress_with(granted)
        for offered in S.available(p, sentences=S.load_sentences()):
            assert set(offered.needs) <= granted, offered.es

    def test_nothing_known_means_nothing_offered(self):
        assert S.available(Progress(path=_SCRATCH),
                           sentences=S.load_sentences()) == []


class TestReadingTheDeckOutOfASentence:
    def test_a_conjugated_form_leads_back_to_its_verb(self):
        assert "tener" in S.resolve("tengo")
        assert "hablar" in S.resolve("hablo")

    def test_a_regular_verb_outside_the_tables_still_resolves(self):
        """Only 39 verbs have written-out paradigms, but 146 are in the deck
        and a sentence may use any of them."""
        assert "correr" in S.resolve("corre")
        assert "comprar" in S.resolve("compramos")

    def test_a_plural_noun_leads_back_to_the_singular(self):
        assert "coche" in S.resolve("coches")
        assert "luz" in S.resolve("luces")

    def test_an_adjective_agrees_and_still_resolves(self):
        assert "blanco" in S.resolve("blanca")
        assert "nuevo" in S.resolve("nuevas")

    def test_a_stem_changing_verb_outside_the_tables_resolves(self):
        """`conseguir` has no written-out paradigm, and generating it as a
        regular verb gives `consegues`. A real sentence saying `consigues`
        would have been thrown away as a word off the deck."""
        assert "conseguir" in S.resolve("consigues")
        assert "recordar" in S.resolve("recuerdo")
        assert "mostrar" in S.resolve("muestra")

    def test_the_spelling_only_yo_forms_resolve(self):
        assert "conseguir" in S.resolve("consigo")
        assert "elegir" in S.resolve("elijo")

    def test_a_regular_verb_is_not_given_a_stem_change(self):
        """The change is per verb, not a rule. `comer` must not become
        `cuemo`, or a misspelling would quietly resolve to a real word."""
        assert S.resolve("cuemo") == ()
        assert "comer" in S.resolve("come")

    def test_an_attached_pronoun_is_peeled_off(self):
        assert "ayudar" in S.resolve("ayudarme")
        assert "ver" in S.resolve("verlo")

    def test_it_does_not_find_a_verb_inside_an_ordinary_word(self):
        """`vela` is a candle, not `ver` plus `la`. Enclitics only ever come
        off an infinitive or a gerund."""
        assert S._without_enclitic("vela") is None
        assert S._without_enclitic("hola") is None

    def test_a_word_the_deck_has_never_heard_of_resolves_to_nothing(self):
        assert S.resolve("zzzqq") == ()


# -- grading --------------------------------------------------------------
class TestPerfect:
    def test_the_exact_sentence_is_perfect(self):
        item = sentence("Tengo vino blanco.")
        assert S.grade("Tengo vino blanco.", item).perfect

    def test_case_and_punctuation_are_ignored(self):
        item = sentence("Tengo vino blanco.")
        assert S.grade("tengo vino blanco", item).perfect
        assert S.grade("¡TENGO VINO BLANCO!", item).perfect

    def test_a_missing_accent_is_still_perfect(self):
        """The same decision as everywhere else in the app: reaching for the
        option key is a tax on the mode that exists for answering quickly."""
        item = sentence("Es un problema difícil.")
        assert S.grade("Es un problema dificil", item).perfect

    def test_a_missing_tilde_is_forgiven_here_too(self):
        """`ñ` survives normalising everywhere else in the app, because it is
        its own letter rather than an accented `n`. In this mode it goes with
        the accents: a whole sentence is enough work without reaching for the
        option key twice a line."""
        item = sentence("Hablo español.")
        assert S.grade("Hablo espanol", item).perfect
        assert S.grade("Hablo español", item).perfect

    def test_an_added_subject_pronoun_is_not_a_mistake(self):
        item = sentence("Hablo español.")
        assert S.grade("Yo hablo español", item).perfect

    def test_an_omitted_subject_pronoun_is_not_a_mistake(self):
        """Spanish drops it far more often than it keeps it, and both are the
        same sentence."""
        item = sentence("Yo hablo español.", needs=["hablar", "español"])
        assert S.grade("Hablo español", item).perfect

    def test_an_added_third_person_pronoun_is_forgiven(self):
        item = sentence("Tiene un perro.", needs=["tener", "perro"])
        assert S.grade("Él tiene un perro", item).perfect
        assert S.grade("El tiene un perro", item).perfect

    def test_a_dropped_leading_article_is_still_a_mistake(self):
        """The same word in the same place, and this one is wrong. Only the
        front of the sentence is forgiven, and only in the added direction
        for the two that collide with an article."""
        item = sentence("El perro corre.", needs=["perro", "correr"])
        assert not S.grade("Perro corre", item).perfect

    def test_a_pronoun_in_the_middle_is_not_forgiven(self):
        """`lo`, `la` and `le` are object pronouns there, and dropping one
        changes what the sentence says."""
        item = sentence("Tengo el vino.", needs=["tener", "vino"])
        assert not S.grade("Tengo vino", item).perfect

    def test_an_empty_answer_is_never_perfect(self):
        item = sentence("Tengo vino blanco.")
        assert not S.grade("", item).perfect
        assert not S.grade(None, item).perfect

    def test_an_alternate_wording_is_accepted(self):
        item = sentence("¿Puedes ayudarme?", also=["¿Me puedes ayudar?"])
        assert S.grade("Me puedes ayudar", item).perfect
        assert S.grade("Puedes ayudarme", item).perfect


class TestItSaysWhereYouWentWrong:
    def test_a_wrong_word_carries_what_you_wrote(self):
        item = sentence("Tengo vino blanco.")
        g = S.grade("Tengo vino blanca", item)
        assert not g.perfect
        wrong = [t for t in g.marked if t.state == "wrong"]
        assert len(wrong) == 1
        assert wrong[0].text == "blanco" and wrong[0].typed == "blanca"

    def test_a_missing_word_is_named(self):
        item = sentence("Tengo vino blanco.")
        g = S.grade("Tengo vino", item)
        assert [t.text for t in g.marked if t.state == "missing"] == ["blanco"]

    def test_an_invented_word_is_listed_separately(self):
        item = sentence("Tengo vino blanco.")
        g = S.grade("Tengo mucho vino blanco", item)
        assert g.extra == ("mucho",)

    def test_a_dropped_article_is_a_real_mistake(self):
        """`él` and `el` are the same word once accents are stripped, so
        forgiving the pronoun would forgive a stray article anywhere."""
        item = sentence("Tengo el vino.", needs=["tener", "vino"])
        assert not S.grade("Tengo vino", item).perfect
        item2 = sentence("Tengo vino blanco.")
        g = S.grade("Tengo el vino blanco", item2)
        assert not g.perfect and g.extra == ("el",)

    def test_one_missing_word_does_not_condemn_the_rest(self):
        """Aligned rather than zipped. A word left out early shifts
        everything after it, and comparing position by position would mark a
        sentence you nearly got right as entirely wrong."""
        item = sentence("El perro grande corre.",
                        needs=["perro", "grande", "correr"])
        g = S.grade("El perro corre", item)
        assert [t.state for t in g.marked] == ["right", "right", "missing",
                                               "right"]

    def test_everything_wrong_is_counted(self):
        item = sentence("Tengo vino blanco.")
        g = S.grade("Quiero agua fria hoy", item)
        assert g.mistakes >= 3

    def test_the_expected_wording_comes_back_with_its_accents(self):
        item = sentence("Es un problema difícil.")
        assert S.grade("es un problema dificil",
                       item).expected == "Es un problema difícil."


# -- the drill ------------------------------------------------------------
class TestTheSentenceDrill:
    def test_a_perfect_answer_retires_the_sentence(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer([item.es])
        d = run(SentenceDrill(progress_with(everything_for(item)), typer,
                              sentences=[item]), typer)
        assert d.summary() == {"perfect": 1, "asked": 1, "total": 1}

    def test_a_missed_sentence_comes_back(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["algo mal", item.es])
        d = run(SentenceDrill(progress_with(everything_for(item)), typer,
                              sentences=[item]), typer)
        assert d.summary() == {"perfect": 1, "asked": 2, "total": 1}

    def test_the_second_showing_says_it_is_a_repeat(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["algo mal", item.es])
        labels = []
        d = SentenceDrill(progress_with(everything_for(item)), typer, sentences=[item])
        d.on_prompt = lambda s, label: labels.append(label)
        run(d, typer)
        assert labels == ["SENTENCE", "SENTENCE · AGAIN"]

    def test_the_bar_counts_only_what_is_finished(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["algo mal", item.es])
        seen = []
        d = SentenceDrill(progress_with(everything_for(item)), typer, sentences=[item])
        d.on_progress = lambda done, total: seen.append((done, total))
        run(d, typer)
        assert seen[0] == (0, 1) and seen[-1] == (1, 1)

    def test_it_never_touches_the_schedule(self):
        """The words are already being reviewed on their own account. Grading
        them again here would count one answer twice."""
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        before = {i: c.to_dict() for i, c in p.cards.items()}
        typer = Typer(["algo mal", item.es])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        assert {i: c.to_dict() for i, c in p.cards.items()} == before

    def test_stop_ends_it(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["para"])
        d = run(SentenceDrill(progress_with(everything_for(item)), typer,
                              sentences=[item]), typer)
        assert d.summary()["perfect"] == 0

    def test_skip_puts_it_back_without_grading(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["salta", item.es])
        d = run(SentenceDrill(progress_with(everything_for(item)), typer,
                              sentences=[item]), typer)
        assert d.summary() == {"perfect": 1, "asked": 1, "total": 1}

    def test_a_miss_holds_until_it_is_released(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer(["algo mal", item.es])
        held = []
        d = SentenceDrill(progress_with(everything_for(item)), typer, sentences=[item],
                          hold_on_miss=lambda should_stop: held.append(1))
        run(d, typer)
        assert len(held) == 1        # once, on the miss, not on the pass

    def test_it_says_what_is_blocking_an_empty_bank(self):
        d = SentenceDrill(Progress(path=_SCRATCH), Typer([]))
        why = d.why_empty()
        assert "unlocked" in why and len(why) > 30

    def test_an_empty_bank_ends_without_asking_anything(self):
        """And ends straight away. Nothing is being written, so there is
        nothing to wait for."""
        typer = Typer([])
        d = SentenceDrill(Progress(path=_SCRATCH), typer, sentences=[])
        prompts = []
        d.on_prompt = lambda s, label: prompts.append(s)
        import time
        started = time.time()
        run(d, typer)
        assert prompts == [None] and d.summary()["asked"] == 0
        assert time.time() - started < 1.0

    def test_it_waits_for_a_sentence_that_is_still_being_written(self):
        """Only when something is actually writing them. The first API call
        has not come back yet and the pool is empty, which is the one case
        where sitting still beats saying there is nothing to do."""
        import threading
        item = sentence("Tengo vino blanco.")
        typer = Typer([item.es])
        d = SentenceDrill(progress_with(everything_for(item)), typer, sentences=[],
                          expecting_more=True)
        d.WAIT_SECONDS = 3
        threading.Timer(0.15, lambda: d.add([item])).start()
        run(d, typer)
        assert d.summary() == {"perfect": 1, "asked": 1, "total": 1}

    def test_it_gives_up_waiting_rather_than_hanging(self):
        typer = Typer([])
        d = SentenceDrill(Progress(path=_SCRATCH), typer, sentences=[],
                          expecting_more=True)
        d.WAIT_SECONDS = 0.3
        run(d, typer)
        assert d.summary()["asked"] == 0

    def test_a_late_arrival_is_only_queued_once(self):
        item = sentence("Tengo vino blanco.")
        other = sentence("Bebo agua.")
        d = SentenceDrill(progress_with(everything_for(item, other)), Typer([]),
                          sentences=[item])
        d.build_queue()
        assert d.add([item]) == 0        # already queued
        assert d.add([other]) == 1
        assert d.total == 2

    def test_a_sentence_delivered_before_the_run_starts_is_not_lost(self):
        """The generator can come back before the queue has been built. The
        build has to merge with what is already there, not assign over it."""
        item = sentence("Tengo vino blanco.")
        typer = Typer([item.es])
        d = SentenceDrill(progress_with(everything_for(item)), typer, sentences=[])
        d.add([item])                    # arrives before run()
        run(d, typer)
        assert d.summary() == {"perfect": 1, "asked": 1, "total": 1}


class TestTheBankIsGrammatical:
    """Checks a person can get wrong by reading and a machine cannot.

    Spanish needs prepositions English gives no hint about. "I see my father"
    is `veo a mi padre`, never `veo mi padre`, and nothing in the English says
    so. Gender is the same: `la problema` reads fine to an English speaker.
    """

    PEOPLE = {"amigo", "amiga", "hermano", "hermana", "hijo", "hija", "padre",
              "madre", "doctor", "hombre", "mujer", "bebe", "capitan",
              "agente", "persona", "presidente", "novio", "perro", "amigos",
              "hermanos", "hijos", "hombres", "mujeres", "padres", "perros"}
    ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "mi",
                "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra"}
    TAKES_PERSONAL_A = {
        "veo", "ves", "ve", "vemos", "ven", "busco", "buscas", "busca",
        "buscamos", "buscan", "conozco", "conoces", "conoce", "conocemos",
        "conocen", "llamo", "llamas", "llama", "llamamos", "llaman", "ayudo",
        "ayudas", "ayuda", "ayudamos", "ayudan", "escucho", "escuchas",
        "escucha", "miro", "miras", "mira", "invito", "invitas", "invita",
        "espero", "esperas", "espera", "pago", "pagas", "paga", "mato",
        "matas", "mata", "amo", "amas", "ama"}
    # Verbs that govern a preposition before a following infinitive.
    NEEDS_PREP = {"aprender": "a", "empezar": "a", "ayudar": "a", "ir": "a",
                  "volver": "a", "enseñar": "a", "venir": "a", "salir": "a",
                  "tratar": "de", "acabar": "de", "dejar": "de",
                  "olvidar": "de", "parar": "de"}

    @staticmethod
    def gender_of_nouns():
        """Read off the deck's own example sentences, not hand-written."""
        import re
        from collections import Counter
        from spanish_drill.text import normalize
        pattern = re.compile(r"\b(el|la|los|las|un|una|unos|unas)\s+(\w+)",
                             re.IGNORECASE)
        counts = {}
        for card in load_deck():
            for article, noun in pattern.findall(card.example or ""):
                article = article.lower()
                sex = "m" if article in ("el", "los", "un", "unos") else "f"
                counts.setdefault(normalize(noun), Counter())[sex] += 1
        return {noun: c.most_common(1)[0][0] for noun, c in counts.items()}

    def problems(self, es):
        import re
        out = []
        words = S.words_of(es)
        infinitive = re.compile(r"\w+(ar|er|ir)$")
        for at, word in enumerate(words[:-1]):
            after = words[at + 1:]
            if word in self.TAKES_PERSONAL_A:
                if after[0] in self.PEOPLE:
                    out.append(f"needs personal 'a': {word} {after[0]}")
                elif (after[0] in self.ARTICLES and len(after) > 1
                        and after[1] in self.PEOPLE):
                    out.append(f"needs personal 'a': {word} {after[0]} {after[1]}")
            if word in S.FUNCTION_WORDS:
                # `para comprar` is the preposition "to", not the verb
                # `parar`. Reading it as the verb demanded a `de` that must
                # not be there and condemned every purpose clause in the bank.
                continue
            for lemma in S.resolve(word):
                prep = self.NEEDS_PREP.get(lemma)
                if prep and infinitive.match(after[0]) and after[0] != prep:
                    out.append(f"{lemma} needs '{prep}' before {after[0]}")
        return out

    def test_the_check_catches_spanish_that_is_actually_wrong(self):
        """A checker that passes everything is worth nothing, so this pins
        that it fails the sentences it is supposed to fail."""
        assert self.problems("veo mi padre")
        assert self.problems("conozco tu hermano")
        assert self.problems("quiero aprender hablar")
        assert self.problems("voy comer")
        assert not self.problems("veo a mi padre")
        assert not self.problems("quiero aprender a hablar")
        assert not self.problems("intento entender")
        assert not self.problems("miro la pelicula")

    def test_no_sentence_is_missing_a_preposition(self):
        broken = {i.es: self.problems(i.es) for i in S.load_sentences()
                  if self.problems(i.es)}
        assert not broken, broken

    def test_articles_agree_with_their_nouns(self):
        """`la problema` reads perfectly well to an English speaker."""
        masculine = {"el", "un", "los", "unos"}
        feminine = {"la", "una", "las", "unas"}
        # `el agua` takes a masculine article and stays a feminine noun.
        exceptions = {"agua", "arma"}
        gender = self.gender_of_nouns()
        broken = []
        for item in S.load_sentences():
            words = S.words_of(item.es)
            for at, word in enumerate(words[:-1]):
                noun = words[at + 1]
                sex = gender.get(noun)
                if sex is None or noun in exceptions:
                    continue
                if (word in masculine and sex == "f") or \
                        (word in feminine and sex == "m"):
                    broken.append(f"{item.es}: '{word} {noun}'")
        assert not broken, broken

    def test_every_sentence_is_present_tense_only(self):
        """The deck has only taught the present. A preterite in the bank
        would be asking for a form that has never been drilled."""
        from spanish_drill.conjugation import TENSES, VERBS, conjugate
        past = set()
        for lemma in VERBS:
            for tense in TENSES:
                if tense == "pres":
                    continue
                for form in conjugate(lemma, tense):
                    past.add(S.normalize(form))
        present = {S.normalize(f) for lemma in VERBS
                   for f in conjugate(lemma, "pres")}
        vocabulary = {c.answers[0] for c in load_deck()
                      if not c.lemma and c.pos != "verb"}
        vocabulary = {S.normalize(w) for w in vocabulary}
        strays = []
        for item in S.load_sentences():
            for word in S.words_of(item.es):
                # `vino` is wine before it is anything venir did.
                if word in past and word not in present \
                        and word not in vocabulary:
                    strays.append((item.es, word))
        assert not strays, strays

    def test_no_sentence_needs_the_personal_a(self):
        """The deck teaches words, never the rule that a person as an object
        takes `a`. So a cue like "I know your brother" gives no way to know
        that `conozco tu hermano` is wrong, and the card is unanswerable
        rather than hard. Sentences that would need it are kept out entirely.
        """
        offenders = []
        for item in S.load_sentences():
            words = S.words_of(item.es)
            for at, word in enumerate(words[:-1]):
                if word in self.TAKES_PERSONAL_A and words[at + 1] == "a":
                    offenders.append(item.es)
                if word == "al" and at + 1 < len(words) \
                        and words[at + 1] in self.PEOPLE:
                    offenders.append(item.es)
        assert not offenders, offenders


class Speaker:
    """A listener that hands over scripted transcripts, like a microphone."""

    def __init__(self, heard, audio=b"clip"):
        self.heard = list(heard)
        self.last_audio = audio
        self.drill = None
        self.windows = []
        self.accepts = []

    def listen(self, window=None, should_stop=None, accept=None,
               fast=False, steer=None, second_pass=False):
        self.windows.append(window)
        self.accepts.append(accept)
        if not self.heard:
            if self.drill:
                self.drill.stop()
            return None
        return self.heard.pop(0)


class Log:
    """Stands in for the answer log; records what it was asked to keep."""

    def __init__(self, path="/tmp/clip.wav"):
        self.saved = []
        self.path = path

    def save_audio(self, index, audio, word=None):
        self.saved.append(word)
        return "stamp", self.path


def spoken(item, heard, verifier=None, log=None, progress=None):
    from spanish_drill.composition import SentenceDrill
    speaker = Speaker(heard)
    drill = SentenceDrill(progress or progress_with(everything_for(item)),
                          speaker, sentences=[item], typed=False,
                          verifier=verifier, answer_log=log or Log())
    speaker.drill = drill
    drill.run()
    return drill, speaker


class TestSpeakingASentence:
    """The same drill through a microphone, with the same second opinion the
    word drill uses behind it."""

    def test_a_spoken_sentence_is_graded(self):
        item = sentence("Tengo vino blanco.")
        results = []
        drill, _ = spoken(item, [item.es])
        assert drill.summary()["perfect"] == 1

    def test_punctuation_and_case_do_not_matter_from_a_recogniser(self):
        item = sentence("Tengo vino blanco.")
        drill, _ = spoken(item, ["tengo vino blanco"])
        assert drill.summary()["perfect"] == 1

    def test_the_window_grows_with_the_sentence(self):
        """The answer-wait dial is sized for one word. Ten words spoken do
        not fit in six seconds, and a window that runs out mid-clause scores
        silence."""
        from spanish_drill.composition import speech_window
        short = sentence("Tengo vino blanco.")
        long = sentence("El hombre que vende el coche vive en la ciudad.",
                        needs=["hombre", "vender", "coche", "vivir", "ciudad"])
        assert speech_window(long) > speech_window(short)

    def test_the_listener_is_given_that_window(self):
        item = sentence("Tengo vino blanco.")
        from spanish_drill.composition import speech_window
        _, speaker = spoken(item, [item.es])
        assert speaker.windows[0] == pytest.approx(speech_window(item))

    def test_a_typed_run_has_no_window_at_all(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer([item.es])
        d = SentenceDrill(progress_with(everything_for(item)), typer,
                          sentences=[item])
        run(d, typer)
        # TypedListener treats None as "wait indefinitely"
        assert d.typed is True

    def test_it_can_finish_early_when_the_answer_is_already_right(self):
        """The recogniser re-checks at each pause; saying it correctly should
        end the turn rather than waiting out the window."""
        item = sentence("Tengo vino blanco.")
        _, speaker = spoken(item, [item.es])
        accept = speaker.accepts[0]
        assert accept is not None
        assert accept("tengo vino blanco") is True
        assert accept("tengo agua") is False

    def test_the_clip_is_kept(self):
        """A verdict should be traceable to the audio behind it."""
        item = sentence("Tengo vino blanco.")
        log = Log()
        spoken(item, ["algo completamente mal", item.es], log=log)
        assert log.saved            # something was written

    def test_a_typed_run_records_no_audio(self):
        item = sentence("Tengo vino blanco.")
        typer = Typer([item.es])
        log = Log()
        d = SentenceDrill(progress_with(everything_for(item)), typer,
                          sentences=[item], answer_log=log)
        run(d, typer)
        assert log.saved == []


class TestTheSecondOpinionOnSentences:
    def test_a_misheard_sentence_is_overturned(self):
        """The whole point. The local model decides misses and is worst at
        exactly that, so a miss is never final until a stronger model has
        heard the same audio."""
        item = sentence("Tengo vino blanco.")
        results = []
        drill, _ = spoken(item, ["tengo bino blanco"],
                          verifier=lambda path: ("Tengo vino blanco.", False))
        assert drill.summary()["perfect"] == 1

    def test_the_result_says_it_was_overturned(self):
        item = sentence("Tengo vino blanco.")
        from spanish_drill.composition import SentenceDrill
        speaker = Speaker(["tengo bino blanco"])
        seen = []
        d = SentenceDrill(progress_with(everything_for(item)), speaker,
                          sentences=[item], typed=False,
                          verifier=lambda p: ("Tengo vino blanco.", False),
                          answer_log=Log())
        speaker.drill = d
        d.on_result = seen.append
        d.run()
        assert seen[0].overturned and seen[0].grade.perfect
        assert seen[0].said == "tengo bino blanco"
        assert seen[0].api_text == "Tengo vino blanco."

    def test_a_genuinely_wrong_sentence_is_kept_wrong(self):
        item = sentence("Tengo vino blanco.")
        drill, _ = spoken(item, ["quiero agua", item.es],
                          verifier=lambda path: ("quiero agua", False))
        assert drill.summary()["asked"] == 2      # it came back round

    def test_a_correct_answer_is_never_sent_for_verification(self):
        """Spending an API call to confirm something already right."""
        item = sentence("Tengo vino blanco.")
        calls = []
        spoken(item, [item.es],
               verifier=lambda path: calls.append(path) or ("x", False))
        assert calls == []

    def test_an_echo_yields_no_verdict(self):
        """The model handing our own prompt back is not evidence either way."""
        item = sentence("Tengo vino blanco.")
        drill, _ = spoken(item, ["algo mal", item.es],
                          verifier=lambda path: (None, True))
        assert drill.summary()["perfect"] == 1    # the retry, not the echo

    def test_a_typed_answer_is_never_sent_for_verification(self):
        """Nothing was recorded, and nobody misheard a keyboard."""
        item = sentence("Tengo vino blanco.")
        calls = []
        typer = Typer(["algo mal", item.es])
        d = SentenceDrill(progress_with(everything_for(item)), typer,
                          sentences=[item], typed=True,
                          verifier=lambda p: calls.append(p) or ("x", False))
        run(d, typer)
        assert calls == []

    def test_the_verifier_is_steered_for_a_sentence_not_a_word(self):
        """The word steer tells the recogniser to expect isolated words,
        which is the wrong instruction for a whole clause."""
        from spanish_drill import transcribe
        prompts = []
        def fake(path, model, prompt):
            prompts.append(prompt)
            return "Tengo vino blanco."
        transcribe.sentence_second_opinion("x", transcriber=fake)
        assert prompts[0] == transcribe.SENTENCE_STEER
        assert "sueltas" not in prompts[0]      # not the word steer


class TestSpokenSentencesAreHandsFree:
    def test_a_spoken_miss_does_not_wait_for_a_keypress(self):
        """The point of speaking is not having to touch anything."""
        item = sentence("Tengo vino blanco.")
        held = []
        from spanish_drill.composition import SentenceDrill
        speaker = Speaker(["algo mal", item.es])
        d = SentenceDrill(progress_with(everything_for(item)), speaker,
                          sentences=[item], typed=False, verifier=None,
                          answer_log=Log(), hold_on_miss=None)
        speaker.drill = d
        d.run()
        assert held == []

    def test_stop_spoken_ends_the_run(self):
        item = sentence("Tengo vino blanco.")
        drill, _ = spoken(item, ["para"])
        assert drill.summary()["perfect"] == 0

    def test_a_right_answer_is_not_read_as_a_command(self):
        """`para` is the answer to "for, to" before it is a stop."""
        item = sentence("Trabajo para ganar dinero.",
                        needs=["trabajar", "ganar", "dinero"])
        drill, _ = spoken(item, ["trabajo para ganar dinero"])
        assert drill.summary()["perfect"] == 1


class TestAMissIsReadBack:
    """The moment you have just failed to produce a sentence is the moment
    it is worth hearing. The slip shows the spelling but not the sound."""

    @staticmethod
    def _run(typed, heard, hints=True):
        from spanish_drill import composition
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        p.hints = hints
        said = []
        if typed:
            listener = Typer(heard)
            drill = SentenceDrill(p, listener, sentences=[item], typed=True)
        else:
            listener = Speaker(heard)
            drill = SentenceDrill(p, listener, sentences=[item], typed=False,
                                  verifier=None, answer_log=Log())
        listener.drill = drill
        drill.on_result = lambda r: None
        original = composition.say_spanish
        composition.say_spanish = lambda text, dialect: said.append(text)
        try:
            drill.run()
        finally:
            composition.say_spanish = original
        return item, said

    def test_a_spoken_miss_is_read_back(self):
        item, said = self._run(False, ["algo mal", "Tengo vino blanco."])
        assert item.es in said

    def test_a_typed_miss_is_read_back_too(self):
        item, said = self._run(True, ["algo mal", "Tengo vino blanco."])
        assert item.es in said

    def test_a_correct_typed_answer_stays_silent(self):
        """Speaking every right answer is what would stop the quiet mode
        being quiet."""
        item, said = self._run(True, ["Tengo vino blanco."])
        assert said == []

    def test_turning_off_say_the_answer_back_keeps_typing_silent(self):
        item, said = self._run(True, ["algo mal", "Tengo vino blanco."],
                               hints=False)
        assert said == []

    def test_a_spoken_miss_is_read_back_even_with_hints_off(self):
        """Spoken is already making noise; the read-back is the teaching."""
        item, said = self._run(False, ["algo mal", "Tengo vino blanco."],
                               hints=False)
        assert item.es in said


class TestASentenceIsFinishedWithForGood:
    """Get it right once and it never comes back, across sessions.

    Sentences have no schedule. The words they are built from are already
    being reviewed on their own account, so asking again for one you have
    produced correctly is asking twice for the same evidence.
    """

    def test_a_perfect_answer_retires_it(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        typer = Typer([item.es])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        assert item.id in p.sentences_done

    def test_a_miss_does_not_retire_it(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        typer = Typer(["algo completamente mal"])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        assert item.id not in p.sentences_done

    def test_getting_it_right_after_a_miss_still_retires_it(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        typer = Typer(["algo mal", item.es])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        assert item.id in p.sentences_done

    def test_it_survives_closing_the_app(self):
        """Written the moment it is earned, not at the end of the run: a
        session stopped half way through still got these right."""
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        typer = Typer([item.es])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        again = Progress.load(p.path)
        assert item.id in again.sentences_done

    def test_a_retired_sentence_is_not_offered_again(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        assert item.es in [s.es for s in S.unfinished(p, sentences=[item])]
        p.sentences_done.add(item.id)
        assert S.unfinished(p, sentences=[item]) == []

    def test_the_gate_is_unchanged_by_retirement(self):
        """`available` still answers "could this be asked", which is what the
        prerecorder and the blocked-word report need."""
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        p.sentences_done.add(item.id)
        assert S.available(p, sentences=[item]) != []
        assert S.unfinished(p, sentences=[item]) == []

    def test_a_second_run_asks_nothing(self):
        item = sentence("Tengo vino blanco.")
        p = progress_with(everything_for(item))
        typer = Typer([item.es])
        run(SentenceDrill(p, typer, sentences=[item]), typer)
        # the same drill again, this time with the real pool
        typer2 = Typer([])
        second = SentenceDrill(p, typer2)
        second.pool = [item]
        typer2.drill = second
        second.run()
        assert second.summary()["asked"] == 0

    def test_an_unknown_id_in_the_file_is_harmless(self):
        """The id is the sentence's own text. Editing a sentence changes it,
        and the old entry must not break anything."""
        p = progress_with()
        p.sentences_done.add("a-sentence-that-no-longer-exists")
        p.save()
        assert Progress.load(p.path).sentences_done
        assert S.unfinished(p) == S.unfinished(p)   # does not raise

    def test_it_says_when_everything_unlocked_is_done(self):
        """"Nothing to do" and "nothing unlocked" are different answers and
        a blank screen cannot tell them apart.

        Built from a sentence that really is in the bank, because that is
        what `why_empty` reads: an invented one would leave nothing unlocked
        and test the other branch by accident.
        """
        real = S.load_sentences()[0]
        p = progress_with(everything_for(real))
        unlocked = S.available(p)
        assert unlocked, "the fixture unlocked nothing, so this proves nothing"
        p.sentences_done.update(i.id for i in unlocked)
        drill = SentenceDrill(p, Typer([]))
        assert "done" in drill.why_empty().lower()
