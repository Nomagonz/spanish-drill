"""The worker in front of the shared database.

It was the one piece with nothing standing behind it, and it shipped a fault
that every request from a browser hit and no request from curl did: the CORS
preflight threw, so the page could not make a single keyed call while every
check run by hand passed. A browser reports that as "failed to fetch" and says
nothing about the preflight, so from the drill's side it looks like the network.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "tests" / "worker_bridge.mjs"
WORKER = ROOT / "worker" / "src" / "index.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the worker")


def run(*jobs):
    done = subprocess.run(["node", str(BRIDGE), str(WORKER)],
                          input=json.dumps(list(jobs)),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


class TestTheBrowserCanReachItAtAll:
    """A page on another origin sends a preflight before anything else."""

    def test_the_preflight_does_not_throw(self):
        got = run({"path": "/state", "method": "OPTIONS",
                   "origin": "https://nomagonz.github.io",
                   "request_method": "PUT",
                   "request_headers": "authorization,content-type"})[0]
        assert "threw" not in got, (
            f"the preflight raised {got.get('threw')!r}, which is a 500 in "
            "production and reads as a dead network in the browser")
        assert got["status"] == 204

    def test_the_preflight_carries_no_body(self):
        """A 204 is defined as having none, and building one with a body is
        exactly what threw."""
        got = run({"path": "/state", "method": "OPTIONS"})[0]
        assert got["body"] is None

    def test_the_preflight_allows_what_the_page_actually_sends(self):
        got = run({"path": "/state", "method": "OPTIONS"})[0]
        assert got["allow_origin"] == "*"
        assert "Authorization" in got["allow_headers"]
        assert "Content-Type" in got["allow_headers"]
        assert "PUT" in got["allow_methods"]

    def test_a_preflight_is_answered_without_a_key(self):
        """A browser sends it before it has been told to send anything, so
        turning it away on authorisation blocks every keyed request there is."""
        got = run({"path": "/state", "method": "OPTIONS"})[0]
        assert got["status"] == 204

    def test_ordinary_replies_carry_the_headers_too(self):
        got = run({"path": "/version", "token": "right-key"})[0]
        assert got["allow_origin"] == "*"


class TestItRefusesWhatItShould:
    def test_no_key_is_turned_away(self):
        assert run({"path": "/version"})[0]["status"] == 401

    def test_a_wrong_key_is_turned_away(self):
        assert run({"path": "/version", "token": "wrong"})[0]["status"] == 401

    def test_a_key_of_the_same_length_is_still_wrong(self):
        got = run({"path": "/version", "token": "right-kex"})[0]
        assert got["status"] == 401

    def test_an_unset_secret_refuses_everything(self):
        """Failing closed. A worker deployed before its secret was set must
        not be an open database."""
        got = run({"path": "/version", "token": "", "env_token": ""})[0]
        assert got["status"] == 401

    def test_an_unknown_path_is_a_404(self):
        assert run({"path": "/nope", "token": "right-key"})[0]["status"] == 404


class TestReadingAndWriting:
    def test_the_version_comes_back(self):
        got = run({"path": "/version", "token": "right-key",
                   "db": {"state": {}, "version": 7}})[0]
        assert got["body"] == {"version": 7}

    def test_the_state_comes_back(self):
        got = run({"path": "/state", "token": "right-key",
                   "db": {"state": {"cards": {"de": 1}}, "version": 2}})[0]
        assert got["body"]["state"] == {"cards": {"de": 1}}
        assert got["body"]["version"] == 2

    def test_a_write_on_the_current_version_lands(self):
        got = run({"path": "/state", "method": "PUT", "token": "right-key",
                   "db": {"state": {}, "version": 3},
                   "body": {"state": {"cards": {"de": 9}}, "base": 3}})[0]
        assert got["status"] == 200
        assert got["db_state"] == {"cards": {"de": 9}}
        assert got["db_version"] == 4

    def test_a_write_on_a_stale_version_is_refused_with_the_current_one(self):
        """What stops the phone and the desktop overwriting each other."""
        got = run({"path": "/state", "method": "PUT", "token": "right-key",
                   "db": {"state": {"cards": {"de": 1}}, "version": 5},
                   "body": {"state": {"cards": {"de": 9}}, "base": 2}})[0]
        assert got["status"] == 409
        assert got["body"]["version"] == 5
        assert got["body"]["state"] == {"cards": {"de": 1}}
        assert got["db_state"] == {"cards": {"de": 1}}, "the refused write landed anyway"

    def test_rubbish_is_not_taken_as_a_state(self):
        for body in ({"state": "not an object"}, {"state": None}, {}):
            got = run({"path": "/state", "method": "PUT", "token": "right-key",
                       "body": body})[0]
            assert got["status"] == 400, body
