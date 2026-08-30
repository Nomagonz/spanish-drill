"""Steering the recogniser, and refusing to grade its echo of our own prompt."""
import pytest

from spanish_drill.deck import load_deck
from spanish_drill.text import normalize
from spanish_drill.transcribe import (STEER, STEER_WORDS, assert_steer_is_clean,
                                      is_steer_echo, second_opinion)


class TestSteerCannotLeakAnswers:
    """The prompt biases decoding. If it names a deck answer, the recogniser
    agrees with whatever was said and the grade stops meaning anything."""

    def test_no_example_word_is_a_deck_answer(self):
        deck_answers = {normalize(a) for c in load_deck() for a in c.answers}
        assert not (STEER_WORDS & deck_answers)

    def test_the_assertion_guards_it(self):
        assert_steer_is_clean()

    def test_the_assertion_actually_fires(self, monkeypatch):
        monkeypatch.setattr("spanish_drill.transcribe.STEER_WORDS", {"casa"})
        with pytest.raises(AssertionError):
            assert_steer_is_clean()


class TestEchoDetection:
    @pytest.mark.parametrize("text", [
        "Palabras sueltas en español, a veces repetidas varias veces.",
        "pintar, nadar, saltar, bailar, fresa, morado, tijeras",
        "Palabras sueltas en español",
        "por ejemplo pintar nadar saltar",
        "pintar",            # a lone example word is still the prompt talking
        "Nadar.",
        "TIJERAS",
    ])
    def test_flagged(self, text):
        assert is_steer_echo(text)

    @pytest.mark.parametrize("text", [
        "llevar", "llevar llevar llevar", "Deber, deber.", "Ir, ir.",
        "Seguir  Seguir", "conocer", "hoy en día", "sin embargo", "a veces",
    ])
    def test_not_flagged(self, text):
        assert not is_steer_echo(text)

    def test_empty_is_not_an_echo(self):
        assert not is_steer_echo("")
        assert not is_steer_echo(None)

    def test_no_real_answer_is_ever_swallowed(self):
        swallowed = [a for c in load_deck() for a in c.answers if is_steer_echo(a)]
        assert swallowed == []


class TestSecondOpinion:
    """On an echo we retry WITHOUT the prompt, because the prompt caused it.

    The retry must never be triggered by 'it didn't say the answer' — asking
    again until the answer appears is just a slower way of leaking it.
    """

    def test_a_clean_answer_is_returned_untouched(self):
        calls = []

        def fake(path, model, prompt):
            calls.append(prompt)
            return "llevar"

        text, echoed = second_opinion("x.wav", transcriber=fake)
        assert (text, echoed) == ("llevar", False)
        assert calls == [STEER], "should not retry when the first answer is usable"

    def test_an_echo_is_retried_without_the_prompt(self):
        calls = []

        def fake(path, model, prompt):
            calls.append(prompt)
            return "pintar" if prompt else "llevar"

        text, echoed = second_opinion("x.wav", transcriber=fake)
        assert (text, echoed) == ("llevar", False)
        assert calls == [STEER, None]

    def test_a_persistent_echo_yields_no_verdict(self):
        text, echoed = second_opinion(
            "x.wav", transcriber=lambda path, model, prompt: "pintar")
        assert text is None and echoed is True

    def test_a_wrong_answer_is_not_retried(self):
        calls = []

        def fake(path, model, prompt):
            calls.append(prompt)
            return "comer"

        text, echoed = second_opinion("x.wav", transcriber=fake)
        assert text == "comer" and not echoed
        assert calls == [STEER], "retrying a wrong answer biases toward accepting"

    def test_api_failure_is_not_an_echo(self):
        text, echoed = second_opinion(
            "x.wav", transcriber=lambda path, model, prompt: None)
        assert text is None and echoed is True
