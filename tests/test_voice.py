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

# Captured at import, before any fixture can replace it. The autouse `silent`
# fixture in conftest stubs `voice.speak` so that no test anywhere can make
# the machine talk, which is right everywhere except the handful of tests
# below whose whole subject is `speak` itself.
_REAL_SPEAK = voice.speak


def spoken_audio(seconds=0.5, rate=22050, level=0.3):
    """Bytes that look like something was actually said.

    Deliberately not silence. A rendered prompt is only accepted if it carries
    signal, because the API sometimes returns a well-formed silent clip, and a
    fixture that writes zeroes would make every test agree that silence is
    fine.
    """
    import array, io, math, wave
    frames = array.array("h", (
        int(level * 32767 * math.sin(2 * math.pi * 220 * i / rate))
        for i in range(int(seconds * rate))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(frames.tobytes())
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    """Isolate the prompt cache, and never let a test reach the network.

    The engine defaults to the API. A test that fakes `say` but leaves the
    default alone quietly renders for real, which is slow, costs money, and
    passes or fails for reasons that have nothing to do with the test. Every
    test opts in to an engine; the API ones fake the client.
    """
    monkeypatch.setattr(voice, "PROMPT_CACHE", tmp_path / "prompts")
    monkeypatch.setattr(voice, "TTS_ENGINE", "say")
    return tmp_path / "prompts"


@pytest.fixture
def fake_say(monkeypatch):
    """Stand in for the `say` binary; record what it was asked to render."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        open(command[command.index("-o") + 1], "wb").write(spoken_audio())
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
    """The one place `voice.speak` is called for real.

    Safe because every route out of it is closed here anyway: `fake_say`
    takes the subprocess, and each test takes `_play`. Nothing reaches an
    API or an output device.
    """

    @pytest.fixture(autouse=True)
    def real_speak(self, monkeypatch):
        monkeypatch.setattr(voice, "speak", _REAL_SPEAK)

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
        from spanish_drill.config import ENGLISH_RATE
        voice.render("one")
        made = voice.prerecord([("one", None, ENGLISH_RATE),
                                ("two", None, ENGLISH_RATE)])
        assert made == 1

    def test_it_can_be_stopped_part_way(self, fake_say):
        """Recording runs in parallel, so stopping is prompt, not instant:
        whatever is already in flight finishes."""
        phrases = [(f"phrase {i}", None, 180) for i in range(400)]
        voice.prerecord(phrases, should_stop=lambda: len(fake_say) >= 3,
                        workers=4)
        assert len(fake_say) < 400, "stop was ignored"

    def test_it_records_in_parallel(self, monkeypatch, cache):
        """Each phrase is an independent `say` that mostly waits. Serially the
        deck takes over twenty minutes."""
        import threading, time
        peak, live, lock = [0], [0], threading.Lock()

        def slow(command, **kwargs):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            open(command[command.index("-o") + 1], "wb").write(spoken_audio(0.05))
            with lock:
                live[0] -= 1
            class Result: returncode = 0
            return Result()

        monkeypatch.setattr(voice.subprocess, "run", slow)
        voice.prerecord([(f"p {i}", None, 180) for i in range(24)], workers=8)
        assert peak[0] > 1, "recordings ran one at a time"

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


class TestRecordingsAreShared:
    """A recording made in one mode must be the file another mode plays.

    That only holds if phrases are stored under their own text. Speaking an
    answer and its example as one joined string would make a phrase unique to
    a single card, recorded once and never reused.
    """

    def test_the_same_phrase_is_one_file_whatever_asked_for_it(self):
        assert (voice.path_for("llevar", "Paulina")
                == voice.path_for("llevar", "Paulina"))

    def test_placement_and_the_drill_want_the_same_files(self):
        deck = load_deck()
        placement_needs = set(voice.phrases_for(deck, "es-MX", [12],
                                                examples=False))
        drill_needs = set(voice.phrases_for(deck, "es-MX", [12]))
        assert placement_needs < drill_needs, (
            "placement should need a subset, so its recordings are reused")

    def test_a_card_covers_prompt_answer_and_example(self):
        deck = load_deck()
        texts = {t for t, _, _ in voice.phrases_for(deck, "es-MX", [12])}
        card = deck[12]
        assert {card.spoken_prompt, card.answers[0], card.example} <= texts

    def test_no_phrase_is_requested_twice(self):
        deck = load_deck()
        phrases = voice.phrases_for(deck, "es-MX")
        assert len(phrases) == len(set(phrases))

    def test_the_answer_is_never_glued_to_its_example(self):
        """A joined string is unique to one card and can never be shared."""
        deck = load_deck()
        card = deck[12]
        joined = f"{card.answers[0]}. {card.example}"
        texts = {t for t, _, _ in voice.phrases_for(deck, "es-MX", [12])}
        assert joined not in texts

    def test_shared_words_are_recorded_once_across_cards(self):
        """Two cards asking for the same phrase share one recording.

        This used to assert the whole deck came to fewer than three phrases a
        card, which held only because the deck contained duplicate cards. Once
        those were merged the count hit exactly three a card and the test
        failed, having been a measure of the duplication rather than of the
        sharing. The mechanism is now built rather than hoped for.
        """
        deck = load_deck()
        twins = tuple(deck[0] for _ in range(2))
        assert len(voice.phrases_for(twins, "es-MX")) == 3, (
            "two cards wanting the same phrases asked for six recordings")

    def test_an_example_reused_by_two_cards_is_one_recording(self):
        from dataclasses import replace
        deck = load_deck()
        pair = (deck[0], replace(deck[1], example=deck[0].example,
                                 gloss=deck[0].gloss))
        examples = [t for t, v, _ in voice.phrases_for(pair, "es-MX") if v]
        assert examples.count(deck[0].example) == 1


class TestTheApiBackend:
    """The API has no speed control of its own, so rate is applied afterwards."""

    @pytest.fixture
    def fake_api(self, monkeypatch):
        """Stand in for the OpenAI client. Records what it was asked for."""
        calls = []

        class Speech:
            def create(self, **kwargs):
                calls.append(kwargs)
                return type("Reply", (), {"content": spoken_audio()})()

        class Client:
            audio = type("Audio", (), {"speech": Speech()})()

        module = type("M", (), {"OpenAI": lambda *a, **k: Client()})
        monkeypatch.setitem(__import__("sys").modules, "openai", module)
        monkeypatch.setattr(voice, "TTS_ENGINE", "openai")
        return calls

    def test_it_renders_through_the_api(self, fake_api):
        path = voice.render("to happen")
        assert path is not None and path.exists()
        assert len(fake_api) == 1

    def test_the_spanish_voice_differs_from_the_english_one(self, fake_api):
        voice.render("to happen", None)
        voice.render("pasar", "Paulina")
        assert fake_api[0]["voice"] != fake_api[1]["voice"]

    def test_a_failed_call_leaves_nothing_behind(self, monkeypatch, cache):
        """A half-written prompt is worse than a missing one: it plays."""
        monkeypatch.setattr(voice, "TTS_ENGINE", "openai")
        monkeypatch.setattr(voice, "_render_openai",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
        assert voice.render("to happen") is None
        assert not voice.path_for("to happen").exists()
        assert not list(cache.rglob("*.partial*"))


class TestSpeed:
    """The English cue is read fast; the Spanish being learned is not.

    The API renders at one fixed pace whatever it is asked for, so the rate has
    to be applied to the audio afterwards. Resampling would be exact but raises
    the pitch; ffmpeg's atempo stretches time and leaves the pitch alone.
    """

    def duration(self, path):
        import wave
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()

    @pytest.fixture
    def one_second(self, tmp_path):
        import wave
        path = tmp_path / "one.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"\0\0" * 22050)
        return path

    def test_it_shortens_the_audio(self, one_second, tmp_path):
        out = tmp_path / "fast.wav"
        assert voice.retempo(one_second, out, 1.5)
        assert self.duration(out) == pytest.approx(1 / 1.5, abs=0.05)

    def test_it_keeps_the_pitch(self, one_second, tmp_path):
        """Resampling would shorten it too, and sound like a chipmunk. The
        sample rate surviving unchanged is what separates the two."""
        import wave
        out = tmp_path / "fast.wav"
        voice.retempo(one_second, out, 1.5)
        with wave.open(str(out)) as w:
            assert w.getframerate() == 22050

    def test_it_declines_to_touch_audio_at_the_right_speed(self, one_second,
                                                            tmp_path):
        assert voice.retempo(one_second, tmp_path / "same.wav", 1.0) is False

    def test_it_handles_a_speed_past_what_one_pass_allows(self, one_second,
                                                          tmp_path):
        """atempo takes 0.5 to 2.0, so anything beyond has to be chained."""
        out = tmp_path / "very-fast.wav"
        assert voice.retempo(one_second, out, 3.0)
        assert self.duration(out) == pytest.approx(1 / 3.0, abs=0.05)

    def test_the_english_cue_is_faster_than_the_spanish(self):
        from spanish_drill.config import ENGLISH_RATE, SPANISH_RATE
        assert ENGLISH_RATE > SPANISH_RATE

    def test_each_side_is_recorded_at_its_own_speed(self):
        from spanish_drill.config import ENGLISH_RATE, SPANISH_RATE
        deck = load_deck()
        spoken = voice.phrases_for(deck, "es-MX", [12])
        assert [r for _, v, r in spoken if v is None] == [ENGLISH_RATE], (
            "the English cue is not read fast")
        assert {r for _, v, r in spoken if v} == {SPANISH_RATE}, (
            "the Spanish should stay at a natural pace")

    def test_changing_the_english_speed_does_not_orphan_the_spanish(self):
        """The rate is part of a recording's identity. If both sides shared one
        constant, reading the English faster would silently discard two
        thousand Spanish recordings."""
        from spanish_drill.config import SPANISH_RATE
        assert (voice.path_for("tener", "Paulina", SPANISH_RATE)
                != voice.path_for("tener", "Paulina", SPANISH_RATE + 90))


class TestSilenceIsNeverAccepted:
    """The API sometimes returns a valid, correctly-headed, silent clip.

    Nothing downstream can tell: it plays, it takes the right amount of time,
    and the card goes by without a cue. Ten of five hundred English prompts
    came back this way on the first run.
    """

    def write(self, path, **kwargs):
        path.write_bytes(spoken_audio(**kwargs))
        return path

    def test_speech_is_audible(self, tmp_path):
        assert voice.is_audible(self.write(tmp_path / "a.wav"))

    def test_digital_silence_is_not(self, tmp_path):
        assert not voice.is_audible(self.write(tmp_path / "a.wav", level=0))

    def test_a_barely_there_hiss_is_not(self, tmp_path):
        """The duds were not perfectly zero, just far below any real prompt:
        0.0012 against a quietest-real of 0.027."""
        assert not voice.is_audible(self.write(tmp_path / "a.wav", level=0.001))

    def test_an_empty_or_broken_file_is_not(self, tmp_path):
        (tmp_path / "empty.wav").write_bytes(b"")
        (tmp_path / "junk.wav").write_bytes(b"not a wav at all")
        assert not voice.is_audible(tmp_path / "empty.wav")
        assert not voice.is_audible(tmp_path / "junk.wav")
        assert not voice.is_audible(tmp_path / "absent.wav")

    def test_a_silent_render_is_not_kept(self, monkeypatch, cache):
        """It must not land in the cache, or it is silent forever after."""
        def silent(command, **kwargs):
            open(command[command.index("-o") + 1], "wb").write(
                spoken_audio(level=0))
            class Result: returncode = 0
            return Result()
        monkeypatch.setattr(voice.subprocess, "run", silent)
        assert voice.render("to happen") is None
        assert not voice.path_for("to happen").exists()

    def test_the_api_is_asked_again_rather_than_kept(self, monkeypatch, cache):
        """Silence is a bad roll of the dice, not a permanent failure: the
        retry that is already there for rate limits fixes it."""
        monkeypatch.setattr(voice, "TTS_ENGINE", "openai")
        monkeypatch.setattr(voice.time, "sleep", lambda _: None)
        tries = []

        class Speech:
            def create(self, **kwargs):
                tries.append(kwargs)
                return type("Reply", (), {
                    "content": spoken_audio(level=0 if len(tries) < 3 else 0.3)})()

        module = type("M", (), {"OpenAI": lambda *a, **k: type(
            "C", (), {"audio": type("A", (), {"speech": Speech()})()})()})
        monkeypatch.setitem(__import__("sys").modules, "openai", module)

        path = voice.render("to happen")
        assert len(tries) == 3, "it settled for silence instead of asking again"
        assert path is not None and voice.is_audible(path)


class TestEveryPromptTheDrillWillSpeakExists:
    """Nothing may be recorded mid-drill: the first card would stall on a
    network call, and offline it would be silent.

    This one looks at the real cache rather than a temporary one, so it is a
    check on this installation, not on the code. The recordings are rebuilt on
    demand and not in the repository, so a fresh clone has nothing to check.
    """

    @pytest.fixture(autouse=True)
    def real_cache(self, monkeypatch):
        from spanish_drill.config import PROMPT_CACHE
        monkeypatch.setattr(voice, "PROMPT_CACHE", PROMPT_CACHE)
        if not any(PROMPT_CACHE.glob("*/*.wav")):
            pytest.skip("no prompts recorded yet; run the app to build them")

    def test_every_recording_is_on_disk_and_audible(self):
        """Vocabulary only. A locked conjugation cannot be asked yet, and
        recording all of them up front would be thousands of clips for words
        the drill will not reach for weeks."""
        deck = load_deck()
        spoken = [i for i, c in enumerate(deck) if not c.lemma]
        phrases = list(dict.fromkeys(voice.phrases_for(deck, "es-MX", spoken)
                                     + voice.phrases_for(deck, "es-ES", spoken)))
        missing = [p for p in phrases if not voice.path_for(*p).exists()]
        assert not missing, f"{len(missing)} prompts were never recorded: {missing[:5]}"
        silent = [p for p in phrases if not voice.is_audible(voice.path_for(*p))]
        assert not silent, f"{len(silent)} prompts are silent: {silent[:5]}"
