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


class TestTheConjugationSteer:
    """The word steer's examples are all infinitives, and the models follow.

    Measured in a real session: "seguir" for `sigo`, "poner" for `pongo`,
    "salir" for `sales`, "oir" for `oigo`. Four verbs, all pulled the same way.
    """

    def test_it_names_no_deck_answer(self):
        from spanish_drill.transcribe import assert_steer_is_clean
        assert_steer_is_clean()             # raises if any example is a card

    def test_the_prompt_itself_carries_no_example_words(self):
        """Examples are the part a model hands back instead of a transcript.

        Measured on nine real clips: a four-example version echoed on two of
        them, the instruction alone on none. The instruction is what fixes
        the infinitive bias; the examples only cost echoes. The example words
        stay known to the echo guard, which is free, but they are not in the
        text the recogniser is shown.
        """
        from spanish_drill.transcribe import (CONJUGATION_STEER,
                                              CONJUGATION_STEER_WORDS)
        from spanish_drill.text import normalize
        shown = set(normalize(CONJUGATION_STEER).split())
        assert not (shown & CONJUGATION_STEER_WORDS)
        assert CONJUGATION_STEER_WORDS, "the echo guard still needs them"

    def test_echoing_it_back_is_not_a_transcript(self):
        from spanish_drill.transcribe import CONJUGATION_STEER, is_steer_echo
        assert is_steer_echo(CONJUGATION_STEER, CONJUGATION_STEER)
        assert is_steer_echo("Un verbo español conjugado", CONJUGATION_STEER)

    def test_an_echo_is_caught_without_the_tilde(self):
        """A recogniser that writes `espanol` is still echoing."""
        from spanish_drill.transcribe import CONJUGATION_STEER, is_steer_echo
        assert is_steer_echo("un verbo espanol conjugado una sola",
                             CONJUGATION_STEER)

    def test_it_is_measured_against_the_steer_actually_used(self):
        """The fallback used to compare with the word steer whatever was passed,
        so a conjugation prompt coming back got through as a transcript."""
        from spanish_drill.transcribe import CONJUGATION_STEER, is_steer_echo
        assert is_steer_echo("Un verbo español conjugado, una sola palabra.",
                             CONJUGATION_STEER)

    def test_a_real_answer_is_never_an_echo(self):
        from spanish_drill.transcribe import CONJUGATION_STEER, is_steer_echo
        for said in ("tengo", "hago", "voy", "hablabamos"):
            assert not is_steer_echo(said, CONJUGATION_STEER)

    def test_the_second_opinion_uses_it(self):
        from spanish_drill.transcribe import (CONJUGATION_STEER,
                                              conjugation_second_opinion)
        seen = []

        def fake(path, model, prompt):
            seen.append(prompt)
            return "tengo"

        conjugation_second_opinion("clip.wav", transcriber=fake)
        assert seen == [CONJUGATION_STEER]

    def test_the_drill_hands_it_to_the_local_model(self):
        from spanish_drill.paradigm import ConjugationSession
        from spanish_drill.transcribe import CONJUGATION_STEER
        assert ConjugationSession.steer == CONJUGATION_STEER


class TestTheLocalModelRetriesWithoutTheSteer:
    """An echo used to be recorded as silence: shouted at, marked unattempted.

    Measured on nine real clips, the steer came back instead of a transcript
    on five of them.
    """

    class Model:
        """Hands back the steer when given one, and the truth when not."""

        def __init__(self, truth="hago"):
            self.truth = truth
            self.prompts = []

        def transcribe(self, audio, **kw):
            self.prompts.append(kw.get("initial_prompt"))
            said = STEER if kw.get("initial_prompt") else self.truth
            return [type("S", (), {"text": said})()], None

    def build(self, truth="hago"):
        from spanish_drill.transcribe import LocalTranscriber
        t = LocalTranscriber.__new__(LocalTranscriber)
        t.model = self.Model(truth)
        t.scout = self.Model(truth)
        return t

    def test_an_echo_is_retried_with_no_prompt(self):
        t = self.build()
        assert t.transcribe(None) == "hago"
        assert t.model.prompts == [STEER, None]

    def test_the_poll_does_not_retry(self):
        """The early accept runs at every pause and must stay cheap.

        `retry` is about whether this decode is the final answer, not which
        model runs it: when the scout IS the answer, as it is in the
        conjugation drill, it keeps the retry.
        """
        t = self.build()
        assert t.transcribe(None, scout=True, retry=False) is None
        assert t.scout.prompts == [STEER]

    def test_the_scout_retries_when_it_is_the_answer(self):
        """A fast mode grades on the scout, so its echo is a mark, not a poll."""
        t = self.build()
        assert t.transcribe(None, scout=True, retry=True) == "hago"
        assert t.scout.prompts == [STEER, None]

    def test_a_caller_supplied_steer_is_the_one_used(self):
        from spanish_drill.transcribe import CONJUGATION_STEER
        t = self.build()
        t.transcribe(None, steer=CONJUGATION_STEER)
        assert t.model.prompts[0] == CONJUGATION_STEER


class TestTheGradedDecodeAlwaysRetries:
    """The transcriber defaults the retry off for the scout, which is right
    for the early-accept poll and wrong for the answer: with `fast` on the
    scout IS the answer, and an echo there marks a card you did answer as
    silent. So the listener asks for it explicitly on the decode it grades.
    """

    class Recorder:
        floor = 0.02

        def __init__(self):
            self.calls = []

        def record(self, window, should_stop=None, on_pause=None):
            if on_pause:
                on_pause("partial")         # one early-accept poll
            return "speech", "full", True, False

        def open(self): pass
        def close(self): pass
        def calibrate(self): return 0.02
        def set_device(self, name): pass

    class Transcriber:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, scout=False, steer=None, retry=None):
            self.calls.append({"audio": audio, "scout": scout, "retry": retry})
            return None if audio == "partial" else "tengo"

    def listen(self, **kw):
        from spanish_drill.listener import Listener
        t = self.Transcriber()
        listener = Listener(transcriber=t, recorder=self.Recorder())
        listener.listen(5.0, accept=lambda text: False, **kw)
        return t.calls

    def test_the_poll_does_not_retry(self):
        poll = [c for c in self.listen() if c["audio"] == "partial"]
        assert poll and all(c["retry"] is False for c in poll)

    def test_the_main_model_answer_retries(self):
        graded = [c for c in self.listen() if c["audio"] == "speech"]
        assert graded == [{"audio": "speech", "scout": False, "retry": True}]

    def test_the_fast_scout_answer_retries_too(self):
        graded = [c for c in self.listen(fast=True) if c["audio"] == "speech"]
        assert graded == [{"audio": "speech", "scout": True, "retry": True}], (
            "a fast mode grades on the scout, so its echo must be retried "
            "rather than recorded as silence")

    def test_the_early_accept_poll_turns_it_off(self):
        """Pinned: the listener must not pay for a second decode per pause."""
        import inspect
        from spanish_drill import listener
        source = inspect.getsource(listener.Listener.listen)
        assert "retry=False" in source

    def test_a_good_first_answer_is_not_decoded_twice(self):
        from spanish_drill.transcribe import LocalTranscriber
        t = LocalTranscriber.__new__(LocalTranscriber)

        class Plain:
            def __init__(self): self.prompts = []
            def transcribe(self, audio, **kw):
                self.prompts.append(kw.get("initial_prompt"))
                return [type("S", (), {"text": "tengo"})()], None
        t.model = t.scout = Plain()
        assert t.transcribe(None) == "tengo"
        assert t.model.prompts == [STEER]

