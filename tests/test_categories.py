"""Restricting the drill to one part of speech."""
import pytest

from spanish_drill.deck import categories, load_deck
from spanish_drill.progress import Progress

DECK = load_deck()

# The deck holds two kinds of card. Vocabulary carries an example sentence and
# a unique answer; a conjugated form carries neither, on purpose. These checks
# describe the vocabulary, so that is what they are measured against.
VOCABULARY = tuple(c for c in DECK if not c.lemma)


class TestDeckMetadata:
    def test_every_card_has_a_category(self):
        assert all(c.pos for c in DECK)

    def test_the_categories_are_the_ones_we_expect(self):
        assert {"verb", "noun", "adjective", "adverb"} <= set(categories(DECK))

    def test_categories_are_ordered_by_size(self):
        found = categories(DECK)
        sizes = [sum(1 for c in DECK if c.pos == p) for p in found]
        assert sizes == sorted(sizes, reverse=True)


class TestFiltering:
    def test_all_lets_everything_through(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", category="all")
        assert all(p.in_category(i) for i in range(len(DECK)))

    def test_a_category_admits_only_its_own(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", category="verb")
        allowed = [i for i in range(len(DECK)) if p.in_category(i)]
        assert allowed, "there should be verbs in the deck"
        assert all(DECK[i].pos == "verb" for i in allowed)

    def test_unseen_respects_the_filter(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", category="verb", new_per=500)
        assert all(DECK[i].pos == "verb" for i in p.unseen_indexes())

    def test_the_queue_respects_the_filter(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", category="adjective", new_per=25)
        queue = p.build_queue()
        assert queue
        assert all(DECK[i].pos == "adjective" for i in queue)

    def test_switching_category_changes_the_queue(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", new_per=10)
        p.category = "verb"
        verbs = p.build_queue()
        p.category = "noun"
        nouns = p.build_queue()
        assert not (set(verbs) & set(nouns))

    def test_the_setting_survives_a_save(self, tmp_path):
        p = Progress(path=tmp_path / "p.json", category="verb")
        p.save()
        assert Progress.load(tmp_path / "p.json").category == "verb"


class TestDeckIntegrity:
    """The deck was rebuilt from an external source; check it is well formed."""

    def test_it_is_the_expected_size(self):
        assert len(VOCABULARY) == 500

    def test_no_card_is_missing_a_field(self):
        for c in VOCABULARY:
            assert c.prompt and c.answers and c.example and c.gloss

    def test_no_duplicate_answers(self):
        """Vocabulary only. Conjugations repeat by design: `hablamos` is both
        "we speak" and "we spoke", and `fui` belongs to both ser and ir."""
        first = [c.answers[0] for c in VOCABULARY]
        assert len(first) == len(set(first))

    def test_examples_are_not_shared_between_cards(self):
        examples = [c.example for c in VOCABULARY]
        assert len(examples) == len(set(examples))

    def test_prompts_are_short_enough_to_speak(self):
        long = [(c.answers[0], c.prompt) for c in DECK if len(c.prompt) > 60]
        assert not long, f"these prompts are a mouthful: {long[:5]}"

    def test_frequency_order_is_preserved(self):
        """The most common words should still come first."""
        assert [c.answers[0] for c in DECK[:3]] == ["de", "que", "ser"]


class TestSpokenPrompt:
    """The parenthetical carries the disambiguation and must be read aloud.

    "to know (a fact)" and "to know (a person or place)" are saber and
    conocer. Strip the brackets and both prompts sound identical, and the
    answer becomes a coin flip.
    """

    def test_the_parenthetical_is_kept(self):
        card = next(c for c in DECK if c.answers[0] == "saber")
        assert "fact" in card.spoken_prompt

    def test_brackets_become_a_pause(self):
        card = next(c for c in DECK if c.answers[0] == "saber")
        assert "(" not in card.spoken_prompt and ")" not in card.spoken_prompt
        assert "," in card.spoken_prompt

    def test_the_two_to_knows_are_distinguishable_aloud(self):
        saber = next(c for c in DECK if c.answers[0] == "saber")
        conocer = next(c for c in DECK if c.answers[0] == "conocer")
        assert saber.spoken_prompt != conocer.spoken_prompt

    def test_a_prompt_without_brackets_is_untouched(self):
        card = next(c for c in DECK if c.answers[0] == "hablar")
        assert card.spoken_prompt == card.prompt

    def test_no_prompt_speaks_as_empty_or_ragged(self):
        for c in DECK:
            spoken = c.spoken_prompt
            assert spoken and not spoken.startswith(",") and "  " not in spoken


class TestPromptsAreClean:
    """The prompts come from Wiktionary, whose markup does not always close.

    Only objective damage is checked here. An earlier version of this test
    flagged anything ending in "or", "and" or "to", which called y, con, o and
    "according to" broken. A check that fires on correct data is worse than no
    check, because it trains you to ignore it.
    """

    def test_no_leftover_markup(self):
        import re
        bad = [(c.answers[0], c.prompt) for c in DECK
               if re.search(r"[\[\]{}|]", c.prompt)]
        assert not bad, f"wiki markup survived cleaning: {bad}"

    def test_no_grammatical_complement_markers(self):
        import re
        bad = [(c.answers[0], c.prompt) for c in DECK
               if re.search(r"(?:^|\s)\+\S", c.prompt)]
        assert not bad, f"complement markers left in: {bad}"

    def test_no_wiktionary_boilerplate(self):
        bad = [(c.answers[0], c.prompt) for c in DECK
               if c.prompt.lower().startswith(("used ", "forms ", "indicating "))]
        assert not bad, f"grammatical commentary, not a prompt: {bad}"

    def test_no_ragged_whitespace_or_punctuation(self):
        bad = [(c.answers[0], c.prompt) for c in DECK
               if "  " in c.prompt or c.prompt != c.prompt.strip()
               or c.prompt[:1] in ",;:" or c.prompt[-1:] in ",;:"]
        assert not bad, f"ragged prompts: {bad}"


class TestSpainIsTheDefault:
    """Castilian, not Latin American: the c/z "th" and vosotros are what you
    hear in Spain, and drilling a Mexican accent teaches the wrong sounds."""

    def test_a_fresh_install_starts_in_spain(self):
        from spanish_drill.progress import Progress
        assert Progress().dialect == "es-ES"

    def test_the_spanish_voice_is_the_peninsular_one(self):
        from spanish_drill.speech import spanish_voice
        assert spanish_voice("es-ES") == "Mónica"
        assert spanish_voice("es-MX") == "Paulina"

    def test_an_unknown_dialect_falls_back_to_spain(self):
        """A stored setting from an older save, or a typo, must not quietly
        land on the accent the default was moved away from."""
        from spanish_drill.speech import spanish_voice
        from spanish_drill import voice
        from spanish_drill.deck import load_deck
        assert spanish_voice("es-XX") == "Mónica"
        voices = {v for _, v, _ in voice.phrases_for(load_deck(), "es-XX", [0]) if v}
        assert voices == {"Mónica"}

    def test_mexico_is_still_available(self):
        """Switching the default is not the same as removing the choice."""
        from spanish_drill.config import VOICES
        assert set(VOICES) == {"es-MX", "es-ES"}
