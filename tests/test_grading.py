"""What counts as a correct answer.

These are deliberately strict about the near-miss rules. A false accept is far
worse than a false reject here: a rejected answer costs one extra review, an
accepted wrong answer banks a mistake and hides the word for weeks.
"""
import pytest

from spanish_drill.deck import load_deck
from spanish_drill.grading import check, command_of, quality
from spanish_drill.text import lev, normalize, tolerance

DECK = load_deck()


def card(word):
    for c in DECK:
        if c.answers[0] == word:
            return c
    raise AssertionError(f"{word!r} is not in the deck")


class TestExactAndAlternates:
    def test_exact(self):
        assert check("volver", card("volver"))

    def test_second_alternate_is_accepted(self):
        assert check("regresar", card("volver"))

    def test_case_and_punctuation_ignored(self):
        assert check("VOLVER.", card("volver"))
        assert check("¡Volver!", card("volver"))

    def test_accents_are_optional(self):
        assert check("dia", card("día"))
        assert check("día", card("día"))

    def test_wrong_word_rejected(self):
        assert not check("comer", card("volver"))

    def test_empty_rejected(self):
        assert not check("", card("volver"))
        assert not check(None, card("volver"))


class TestEmbeddedAndRepeated:
    """The answer may arrive buried in filler, or repeated several times."""

    @pytest.mark.parametrize("said", [
        "llevar",
        "llevar llevar llevar",
        "llevar. llevar. llevar.",
        "um llevar uh llevar",
        "no se... llevar",
        "y el bar llevar",
        "LLEVAR LLEVAR",
    ])
    def test_accepted(self, said):
        assert check(said, card("llevar"))

    @pytest.mark.parametrize("said", [
        "comer comer comer",
        "y el bar",
        "yeh bar",
        "gracias por ver el video",
    ])
    def test_rejected(self, said):
        assert not check(said, card("llevar"))

    def test_leading_article_is_stripped(self):
        assert check("la casa", card("casa"))


class TestNearMiss:
    def test_typo_within_tolerance_is_accepted(self):
        v = check("entiender", card("entender"))
        assert v and v.close

    def test_exact_match_is_not_flagged_close(self):
        v = check("entender", card("entender"))
        assert v and not v.close

    def test_another_real_deck_word_is_never_a_typo(self):
        """llevar and llegar differ by one character, inside fuzzy tolerance.

        Accepting one for the other would silently mark a wrong answer correct
        on the single hardest pair in the deck.
        """
        assert lev(normalize("llevar"), normalize("llegar")) <= tolerance("llegar")
        assert not check("llevar", card("llegar"))
        assert not check("llegar", card("llevar"))

    def test_conjugations_are_not_accepted_for_infinitives(self):
        assert not check("quieres", card("querer"))
        assert not check("quiere", card("querer"))


class TestCommands:
    @pytest.mark.parametrize("said,cmd", [
        ("para", "stop"), ("stop", "stop"),
        ("skip", "skip"), ("salta", "skip"),
        ("repite", "repeat"), ("again", "repeat"),
        ("no se", "reveal"), ("no sé", "reveal"),
    ])
    def test_recognised(self, said, cmd):
        assert command_of(said) == cmd

    def test_a_plain_answer_is_not_a_command(self):
        assert command_of("volver") is None


class TestQuality:
    """Answer quality on SM-2's 0-5 scale, from what is observable."""

    def test_silence_scores_below_a_wrong_guess(self):
        assert quality(ok=False, close=False, silent=True, elapsed=1, window=6) == 0
        assert quality(ok=False, close=False, silent=False, elapsed=1, window=6) == 1

    def test_mangled_but_right_scores_below_clean(self):
        assert quality(ok=True, close=True, silent=False, elapsed=1, window=6) == 3

    def test_hesitation_is_penalised(self):
        fast = quality(ok=True, close=False, silent=False, elapsed=1.0, window=6)
        slow = quality(ok=True, close=False, silent=False, elapsed=5.0, window=6)
        assert fast == 5 and slow == 4


class TestSilentH:
    """Spanish `h` is never pronounced, so a recogniser rightly drops it.

    Marking that wrong told you off for saying the word correctly, and on a
    short answer the edit-distance pass cannot rescue it: the tolerance for a
    four-letter word is zero on purpose. These are real transcripts from a
    real session.
    """

    def card(self, card_id):
        from spanish_drill.deck import load_deck
        return next(c for c in load_deck() if c.id == card_id)

    @pytest.mark.parametrize("said", ["E.", "Eh", "¡Eh!", "e"])
    def test_a_bare_vowel_is_the_answer_for_he(self, said):
        assert check(said, self.card("haber:pres-yo"))

    @pytest.mark.parametrize("said", ["Ago. Ago.", "ago", "hago"])
    def test_ago_is_hago(self, said):
        assert check(said, self.card("hacer:pres-yo"))

    def test_it_is_a_full_match_not_a_close_one(self):
        """Nothing was fumbled: the letter was never spoken in the first place."""
        match = check("ago", self.card("hacer:pres-yo"))
        assert match and not match.close

    def test_a_genuinely_different_word_is_still_wrong(self):
        assert check("Agua, agua, agua.", self.card("hacer:pres-yo")) is None
        assert check("Algo.", self.card("hacer:pres-yo")) is None

    @pytest.mark.parametrize("said,card_id", [
        ("boy", "ir:pres-yo"),              # b and v are one phoneme in Spanish
        ("Boy, boy, boy.", "ir:pres-yo"),
        ("voy", "ir:pres-yo"),
        ("bas", "ir:pres-tu"),
    ])
    def test_b_and_v_are_the_same_sound(self, said, card_id):
        """Not a dialect: no Spanish speaker anywhere separates them, so a
        recogniser writing either letter heard the same thing."""
        assert check(said, self.card(card_id))

    def test_p_and_b_stay_separate(self):
        """A voicing contrast Spanish really does make. `paz` is not `vas`."""
        assert check("Pas, pas, pas.", self.card("ir:pres-tu")) is None

    def test_yeismo(self):
        from spanish_drill.text import normalize, sounds_as
        assert sounds_as(normalize("llamar")) == sounds_as(normalize("yamar"))

    def test_the_ch_digraph_is_not_touched(self):
        """`h` is silent alone, not inside `ch`. `coche` must not become `coce`."""
        from spanish_drill.text import sounds_as, normalize
        assert sounds_as(normalize("coche")) == "coche"
        assert sounds_as(normalize("chico")) == "chico"

    def test_the_deck_is_never_merged_by_any_of_it(self):
        """The test every sound rule has to keep passing.

        A rule may fix a spelling. It may never merge two words the deck
        actually distinguishes. Measured across all 1579 answers, silent h
        plus b/v plus ll/y together merge exactly one pair.
        """
        import collections
        from spanish_drill.deck import load_deck
        from spanish_drill.text import normalize, sounds_as
        groups = collections.defaultdict(set)
        for c in load_deck():
            for a in c.answers:
                if (n := normalize(a)):
                    groups[sounds_as(n)].add(n)
        clashing = {k: v for k, v in groups.items() if len(v) > 1}
        assert clashing == {"o": {"o", "oh"}}, clashing
