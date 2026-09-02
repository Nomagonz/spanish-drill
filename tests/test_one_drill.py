"""Not one schedule: one session.

The window used to build a session of its own and the phone built another, so
the two agreed about what you knew and sat on different cards, each waiting
for an answer the other could not give. Sharing a database fixes the first
half of that and none of the second.

Driven in a process of its own. See `one_drill_probe.py` for why.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parent / "one_drill_probe.py"


@pytest.fixture(scope="module")
def seen():
    done = subprocess.run([sys.executable, str(PROBE)],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_there_is_only_ever_one_session(seen):
    assert seen["one_session"], (
        "the window built a session of its own instead of running the hub's")


def test_both_screens_are_shown_the_same_card(seen):
    assert seen["window_card"], "the window never showed a card"
    assert seen["browser_card"] == seen["window_card"], (
        f"window is on {seen['window_card']!r} while the browser is on "
        f"{seen['browser_card']!r}")


def test_answering_in_the_browser_moves_the_window(seen):
    after = seen["after_browser_answer"]
    assert after["window_moved"], (
        "the window ignored an answer given on another screen")
    assert after["browser"] == after["window"], (
        "the two screens parted company after the browser answered")


def test_answering_in_the_window_moves_the_browser(seen):
    after = seen["after_window_answer"]
    assert after["window_moved"], "the window did not move on"
    assert after["browser"] == after["window"], (
        "the browser is still showing the card the window has left")
