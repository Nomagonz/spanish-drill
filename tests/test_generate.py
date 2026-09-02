"""Sentences written by the API.

The model is a proposer and nothing more. Everything here is about the part
that does not trust it: a language model told to stay inside a word list will
wander out of it, and the wandering is invisible in the output because the
sentence still reads perfectly well. So the tests that matter are the ones
proving a wandered sentence cannot reach the drill.

Nothing in this file calls the real API.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from spanish_drill import generate as G
from spanish_drill import sentences as S
from spanish_drill.config import MAX_GENERATED_SENTENCES
from spanish_drill.deck import index_by_id
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card, MATURE_AT

_SCRATCH = Path(tempfile.mkdtemp(prefix="drill-generate-")) / "progress.json"


def progress_with(known):
    ids = index_by_id()
    p = Progress(path=_SCRATCH)
    for card_id in known:
        p.cards[ids[card_id]] = Card(interval=MATURE_AT, reps=2, due=0)
    return p


class FakeClient:
    """Stands in for the OpenAI client, one scripted batch per call."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        rows = self.batches.pop(0) if self.batches else []
        # The API answers in the schema's long field names.
        body = json.dumps({"sentences": [
            {"spanish": r.get("es", ""), "english": r.get("en", "")}
            for r in rows]})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


class Exploding(FakeClient):
    def create(self, **kw):
        raise RuntimeError("no network")


# -- the judge ------------------------------------------------------------
class TestNothingUnknownGetsThrough:
    """The whole reason the mode can be trusted with a generator behind it."""

    def test_a_good_sentence_survives(self):
        known = {"tener", "vino", "venir", "blanco"}
        kept, _ = G.usable([{"en": "I have white wine.",
                             "es": "Tengo vino blanco."}], known)
        assert kept == [{"en": "I have white wine.", "es": "Tengo vino blanco."}]

    def test_a_word_that_is_not_in_the_deck_is_thrown_out(self):
        """`gato` is a perfectly good Spanish word and is not in this deck.
        The sentence reads fine, which is exactly why it has to be caught
        mechanically rather than by looking at it."""
        kept, dropped = G.usable(
            [{"en": "I have a cat.", "es": "Tengo un gato."}], {"tener"})
        assert kept == []
        assert "gato" in dropped[0][1]

    def test_a_word_in_the_deck_but_not_yet_learned_is_thrown_out(self):
        kept, dropped = G.usable(
            [{"en": "I have white wine.", "es": "Tengo vino blanco."}],
            {"tener", "vino", "venir"})           # blanco not learned
        assert kept == []
        assert "blanco" in dropped[0][1]

    def test_a_long_sentence_is_thrown_out(self):
        long = "Tengo un perro grande y un coche nuevo y una casa blanca aqui hoy ahora"
        cue = "I have a big dog and a new car and a white house here today now"
        kept, dropped = G.usable([{"en": cue, "es": long}], set())
        assert kept == [] and dropped[0][1] == "too long"

    def test_a_sentence_of_nothing_but_glue_is_thrown_out(self):
        """It would be gated on nothing and available forever."""
        kept, dropped = G.usable(
            [{"en": "The and the of in.", "es": "El y la de en."}], set())
        assert kept == [] and dropped[0][1] == "no content words"

    def test_an_empty_row_is_thrown_out(self):
        kept, _ = G.usable([{"en": "", "es": ""}, {"en": "x"}], set())
        assert kept == []

    def test_something_already_on_file_is_not_bought_twice(self):
        known = {"tener", "vino", "venir", "blanco"}
        rows = [{"en": "I have white wine.", "es": "Tengo vino blanco."}]
        kept, dropped = G.usable(rows, known, seen_es={"Tengo vino blanco."})
        assert kept == [] and dropped[0][1] == "duplicate"

    def test_a_repeat_inside_one_batch_is_only_kept_once(self):
        known = {"tener", "vino", "venir", "blanco"}
        row = {"en": "I have white wine.", "es": "Tengo vino blanco."}
        kept, _ = G.usable([row, dict(row)], known)
        assert len(kept) == 1

    def test_what_survives_would_pass_the_drill_gate_too(self):
        """The judge and the gate have to agree, or the mode would keep
        sentences it then refuses to ask."""
        p = progress_with(["tener", "vino", "venir", "blanco",
                           "tener:pres-yo"])       # the form is gated too
        known = S.known_ids(p)
        kept, _ = G.usable([{"en": "I have white wine.",
                             "es": "Tengo vino blanco."}], known)
        for item in G.as_sentences(kept):
            assert S.is_available(item, known)


class TestTheCueHasToBeEnglish:
    """Measured, not guessed at: with the fields called `en` and `es`, 58 of
    82 sentences came back with Spanish in the English field. The cue then
    *is* the answer, and the card asks you to copy what is on the screen.
    Silent, total, and invisible in the output."""

    def test_english_is_recognised(self):
        for cue in ("The old captain goes to the city.", "I have white wine.",
                    "We work together.", "She wants to open the door."):
            assert G.is_english(cue), cue

    def test_spanish_in_the_cue_is_caught(self):
        for cue in ("Te gusta la fiesta en la noche.",
                    "Este viaje es importante para mí.",
                    "La acción es importante para todos."):
            assert not G.is_english(cue), cue

    def test_a_spanish_cue_with_no_accents_at_all_is_caught(self):
        """The one that got through and reached a card on screen. Every
        letter is plain ASCII, so the accent check says nothing, and the
        single Spanish preposition `a` was enough to pass as English."""
        assert not G.is_english("Voy a tomar un segundo para pensar.")

    def test_a_cue_of_only_spanish_lookalike_words_is_caught(self):
        """Every word here is Spanish the deck teaches, so nothing is left
        for English to explain."""
        assert not G.is_english("No tengo dinero para el coche")

    def test_a_cue_made_of_words_the_deck_does_not_teach_is_still_spanish(self):
        """`eso` has no card, so nothing could account for it. That must not
        be read as evidence the sentence is English."""
        assert not G.is_english("Debo parar y pensar en eso.")
        assert not G.is_english("No quiero pensar en mi trabajo ahora.")

    def test_a_cue_whose_english_words_all_look_spanish_still_passes(self):
        """`he`, `has` and `a` are all ordinary Spanish, which is what broke
        the previous version of this check."""
        assert G.is_english("He has a dog.")

    def test_the_extras_list_never_widens_the_gate(self):
        """It exists to tell languages apart, not to let a word through. A
        sentence using one is still thrown out for having no card."""
        kept, dropped = G.usable(
            [{"en": "I must stop and think about that.",
              "es": "Debo parar y pensar en eso."}],
            {"deber", "parar", "pensar"})
        assert kept == [] and "eso" in dropped[0][1]

    def test_ordinary_english_cues_still_pass(self):
        for cue in ("The doctor has a special program.",
                    "I am going to drink water.",
                    "The general is important.",
                    "We work together."):
            assert G.is_english(cue), cue

    def test_a_spanish_cue_is_thrown_out(self):
        kept, dropped = G.usable(
            [{"en": "Tengo vino blanco.", "es": "Tengo vino blanco."}],
            {"tener", "vino", "venir", "blanco"})
        assert kept == [] and "not English" in dropped[0][1]

    def test_a_cue_identical_to_the_answer_is_thrown_out(self):
        kept, _ = G.usable([{"en": "I have wine", "es": "I have wine"}],
                           {"tener"})
        assert kept == []

    def test_the_schema_names_the_fields_in_full(self):
        """`en` and `es` were too weak a signal to switch language on."""
        props = G.SCHEMA["properties"]["sentences"]["items"]["properties"]
        assert set(props) == {"spanish", "english"}

    def test_the_long_field_names_are_read_back(self):
        client = FakeClient([[{"en": "I have white wine.",
                               "es": "Tengo vino blanco."}]])
        rows = G.call_api([], client=client)
        assert rows == [{"en": "I have white wine.", "es": "Tengo vino blanco."}]


class TestSubjectPronounsAreTakenOff:
    """The model writes them even when told not to, and keeping them would
    make the ordinary Spanish answer read as a missing word."""

    def test_a_leading_pronoun_goes(self):
        assert G._drop_subject("Él tiene un perro.") == "Tiene un perro."
        assert G._drop_subject("Nosotros queremos entrar.") == "Queremos entrar."

    def test_an_article_is_not_a_pronoun(self):
        """`El perro corre` must survive intact. The accent is the only thing
        telling `él` from `el`, which is why this runs before normalising."""
        assert G._drop_subject("El perro corre rápido.") == "El perro corre rápido."
        assert G._drop_subject("Tu hermano trabaja aquí.") == "Tu hermano trabaja aquí."

    def test_a_question_keeps_its_opening_mark(self):
        assert G._drop_subject("¿Tú hablas español?") == "¿Hablas español?"

    def test_a_two_word_sentence_is_left_alone(self):
        """Taking the subject off "Yo soy" leaves a sentence, not a cue."""
        assert G._drop_subject("Yo soy.") == "Yo soy."

    def test_the_stored_sentence_is_the_stripped_one(self):
        kept, _ = G.usable([{"en": "He has a dog.", "es": "Él tiene un perro."}],
                           {"tener", "perro"})
        assert kept == [{"en": "He has a dog.", "es": "Tiene un perro."}]

    def test_the_model_is_told_to_leave_them_out(self):
        assert "subject pronoun" in G.SYSTEM


# -- what gets asked for --------------------------------------------------
class TestTheRequest:
    def test_the_menu_is_only_words_that_are_known(self):
        p = progress_with(["tener", "vino", "blanco"])
        menu = G.word_menu(p)
        assert {es for es, _, _ in menu} <= {"tener", "vino", "blanco"}

    def test_the_menu_leaves_out_conjugation_cards(self):
        """A verb covers its own forms: the gate reads a conjugated form back
        to its infinitive, so listing forms separately would say nothing new."""
        p = progress_with(["tener", "vino", "blanco"])
        assert all(":" not in es for es, _, _ in G.word_menu(p))

    def test_nothing_known_asks_for_nothing(self):
        kept, dropped = G.generate_batch(Progress(path=_SCRATCH), 5,
                                         client=FakeClient([]))
        assert kept == [] and dropped == []

    def test_the_words_are_put_in_front_of_the_model(self):
        messages = G.build_messages([("vino", "wine", "noun")], 3)
        assert "vino = wine" in messages[1]["content"]
        assert "Write 3" in messages[1]["content"]

    def test_it_is_told_what_it_has_already_written(self):
        messages = G.build_messages([("vino", "wine", "noun")], 3,
                                    already=["Tengo vino."])
        assert "Tengo vino." in messages[1]["content"]

    def test_the_rules_name_the_allowed_glue(self):
        for word in ("el", "la", "un", "de", "en", "no"):
            assert word in G.SYSTEM


# -- money ----------------------------------------------------------------
class TestTheCeiling:
    def test_the_allowance_counts_down_from_the_cap(self):
        assert G.remaining_allowance([]) == MAX_GENERATED_SENTENCES
        assert G.remaining_allowance([{"en": "a", "es": "b"}]) == \
            MAX_GENERATED_SENTENCES - 1

    def test_a_full_store_never_calls_the_api_again(self):
        full = [{"en": str(i), "es": str(i)} for i in range(MAX_GENERATED_SENTENCES)]
        client = FakeClient([[{"en": "x", "es": "Tengo vino blanco."}]])
        kept, _ = G.generate_batch(progress_with(["tener", "vino", "venir",
                                                  "blanco"]), 20,
                                   client=client, stored=full)
        assert kept == [] and client.calls == []

    def test_a_batch_cannot_overshoot_the_ceiling(self):
        nearly = [{"en": str(i), "es": str(i)}
                  for i in range(MAX_GENERATED_SENTENCES - 1)]
        rows = [{"en": "a", "es": "Tengo vino blanco."},
                {"en": "b", "es": "Bebo agua."}]
        kept, _ = G.generate_batch(
            progress_with(["tener", "vino", "venir", "blanco", "beber",
                           "agua"]), 20,
            client=FakeClient([rows]), stored=nearly)
        assert len(kept) == 1        # only one slot was left

    def test_it_never_asks_for_more_than_is_left(self):
        nearly = [{"en": str(i), "es": str(i)}
                  for i in range(MAX_GENERATED_SENTENCES - 2)]
        client = FakeClient([[]])
        G.generate_batch(progress_with(["tener", "vino"]), 20,
                         client=client, stored=nearly)
        assert "Write 2" in client.calls[0]["messages"][1]["content"]


# -- failure --------------------------------------------------------------
class TestItFailsQuietly:
    def test_an_api_error_is_an_empty_batch_not_a_crash(self):
        kept, dropped = G.generate_batch(
            progress_with(["tener", "vino"]), 5, client=Exploding([]))
        assert kept == [] and dropped == []

    def test_rubbish_json_is_an_empty_batch(self):
        class Rubbish(FakeClient):
            def create(self, **kw):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="not json"))])
        assert G.call_api([], client=Rubbish([])) == []

    def test_a_missing_store_reads_as_empty(self, tmp_path):
        assert G.load_generated(tmp_path / "nope.json") == []

    def test_a_corrupt_store_reads_as_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{{{", encoding="utf-8")
        assert G.load_generated(path) == []

    def test_the_store_round_trips(self, tmp_path):
        path = tmp_path / "s.json"
        rows = [{"en": "I have white wine.", "es": "Tengo vino blanco."}]
        G.save_generated(rows, path)
        assert G.load_generated(path) == rows

    def test_half_written_rows_are_ignored(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps([{"en": "a"}, {"es": "b"}, "junk",
                                    {"en": "a", "es": "b"}]), encoding="utf-8")
        assert G.load_generated(path) == [{"en": "a", "es": "b"}]
