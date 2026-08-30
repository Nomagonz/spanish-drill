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
