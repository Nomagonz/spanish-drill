"""Pre-recorded prompts and the long-lived player.

The timings that justify this, on a one-second prompt:

    live `say` every time          1.96s
    spawn afplay on a cached file  2.05s   (no better: the cost is the spawn)
    long-lived player              1.02s   (exactly the audio, nothing more)
"""
import subprocess

import pytest

from spanish_drill import voice
from spanish_drill.deck import load_deck


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "PROMPT_CACHE", tmp_path / "prompts")
    return tmp_path / "prompts"


@pytest.fixture
def fake_say(monkeypatch):
    """Stand in for the `say` binary; record what it was asked to render."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        target = command[command.index("-o") + 1]
        import wave
        with wave.open(target, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"\0\0" * 100)
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr(voice.subprocess, "run", run)
    return calls


class TestCaching:
    def test_a_phrase_is_recorded_once(self, fake_say):
        voice.render("to happen")
        voice.render("to happen")
        assert len(fake_say) == 1, "it re-recorded a phrase it already had"

    def test_different_phrases_get_different_files(self, fake_say):
        a = voice.render("to happen")
        b = voice.render("to carry")
        assert a != b and a.exists() and b.exists()

    def test_voice_and_rate_are_part_of_the_identity(self, fake_say):
        plain = voice.path_for("hola", None, 180)
        spanish = voice.path_for("hola", "Paulina", 180)
        faster = voice.path_for("hola", "Paulina", 240)
        assert len({plain, spanish, faster}) == 3

    def test_a_half_written_file_is_never_used(self, monkeypatch, cache):
        """A crash mid-render must not leave a truncated prompt behind."""
        def die(command, **kwargs):
            target = command[command.index("-o") + 1]
            open(target, "wb").write(b"junk")       # started but not finished
            raise OSError("interrupted")
        monkeypatch.setattr(voice.subprocess, "run", die)
        assert voice.render("to happen") is None
        assert not voice.path_for("to happen").exists()
        assert not list(cache.glob("*.partial.wav"))


class TestSpeaking:
    def test_it_plays_the_recording(self, fake_say, monkeypatch):
        played = []
        monkeypatch.setattr(voice, "_play", lambda p: played.append(p) or True)
        voice.speak("to happen")
        assert played == [voice.path_for("to happen")]

    def test_it_falls_back_to_live_speech_if_playback_fails(self, fake_say,
                                                            monkeypatch):
        monkeypatch.setattr(voice, "_play", lambda p: False)
        voice.speak("to happen")
        assert any(c[0] == "say" and "-o" not in c for c in fake_say)

    def test_silence_is_not_recorded(self, fake_say):
        voice.speak("")
        assert fake_say == []


class TestPrerecording:
    def test_it_records_everything_asked_for(self, fake_say):
        made = voice.prerecord([("one", None, 180), ("two", None, 180)])
        assert made == 2

    def test_it_skips_what_it_already_has(self, fake_say):
        voice.render("one")
        made = voice.prerecord([("one", None, 180), ("two", None, 180)])
        assert made == 1

    def test_it_can_be_stopped_part_way(self, fake_say):
        phrases = [(f"phrase {i}", None, 180) for i in range(20)]
        voice.prerecord(phrases, should_stop=lambda: len(fake_say) >= 3)
        assert len(fake_say) <= 4

    def test_it_reports_progress(self, fake_say):
        seen = []
        voice.prerecord([("a", None, 180), ("b", None, 180)],
                        on_progress=lambda d, t: seen.append((d, t)))
        assert seen == [(1, 2), (2, 2)]

    def test_a_card_needs_both_its_prompt_and_its_answer(self):
        deck = load_deck()
        phrases = voice.phrases_for(deck, "es-MX", [0])
        spoken = {text for text, _, _ in phrases}
        assert deck[0].spoken_prompt in spoken
        assert deck[0].answers[0] in spoken

    def test_the_spanish_voice_follows_the_dialect(self):
        deck = load_deck()
        mx = {v for _, v, _ in voice.phrases_for(deck, "es-MX", [0]) if v}
        es = {v for _, v, _ in voice.phrases_for(deck, "es-ES", [0]) if v}
        assert mx != es
