"""Test-wide safety net.

The drill speaks out loud, records audio and writes files next to the app.
None of that belongs in a test run, so all of it is redirected before any test
touches it.
"""
import os

# Qt needs a platform before it is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from spanish_drill import session as session_module



@pytest.fixture(autouse=True)
def silent(monkeypatch):
    """No test should make the machine talk or beep.

    Every module that speaks has to be named here, not just the drill loop.
    `composition` imports `say_english` into its own namespace, so patching
    it on `session` alone left the spoken sentence tests calling the real
    text-to-speech API and playing the result out loud.
    """
    from spanish_drill import composition, cues, placement, speech
    for module in (session_module, composition):
        monkeypatch.setattr(module, "say_english", lambda *a, **k: None)
        monkeypatch.setattr(module, "say_spanish", lambda *a, **k: None)
    monkeypatch.setattr(placement, "say_spanish", lambda *a, **k: None)
    monkeypatch.setattr(cues, "play", lambda *a, **k: None)
    # The catch-all goes here rather than on `voice.speak`, one level down.
    # Everything that speaks funnels through `speech.say` whatever way it
    # imported its own wrapper, and stubbing it leaves voice.py's own tests
    # free to test the thing they exist to test.
    monkeypatch.setattr(speech, "say", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_stray_writes(monkeypatch, tmp_path):
    """Answer audio and logs go to a scratch directory, never the real one."""
    from spanish_drill import answers
    monkeypatch.setattr(answers, "ANSWERS_DIR", tmp_path / "answers")
    monkeypatch.setattr(answers, "ANSWER_LOG", tmp_path / "answers" / "answers.jsonl")


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """Drop the pauses that only exist to pace spoken feedback.

    Deliberately narrow. Patching time.sleep globally would also silence the
    pacing that audio capture depends on, which quietly turned the recorder's
    timing tests into tests that could never fail.
    """
    monkeypatch.setattr(session_module, "pace", lambda *_: None)
