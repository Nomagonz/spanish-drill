"""Where progress is kept, and what happens when it cannot be reached.

The interesting tests here are the failure ones. A drill saves after every
answered card, so a store that loses a write quietly, or that lets a stale
local copy overwrite a shared one, throws away real work in a way nobody
notices until a week of scheduling has gone wrong.
"""
import json
import time
import urllib.error

import pytest

from spanish_drill.deck import load_deck
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card
from spanish_drill.store import CONFLICT, FileStore, RemoteStore


class FakeDatabase:
    """One shared database, in memory, speaking the worker's API.

    Including the refusal. A PUT carries the version it was built on and is
    turned away with 409 if that is not the current one, which is the whole
    of what stops two clients overwriting each other.
    """

    def __init__(self, state=None, version=0):
        self.state = state if state is not None else {}
        self.version = version
        self.calls = []
        self.slow = 0.0             # seconds a call takes, for timing tests

    def __call__(self, path, method="GET", body=None):
        self.calls.append((path, method))
        if self.slow:
            time.sleep(self.slow)
        if path == "/version":
            return 200, {"version": self.version}
        if path == "/state" and method == "GET":
            return 200, {"version": self.version, "state": self.state}
        if path == "/state" and method == "PUT":
            base = body.get("base")
            if base is not None and base != self.version:
                return 409, {"version": self.version, "state": self.state}
            self.state = body["state"]
            self.version += 1
            return 200, {"version": self.version}
        raise AssertionError(f"unexpected call: {method} {path}")


class Link:
    """One client's connection to the database, cuttable on its own.

    Per client rather than per database on purpose. The case worth testing is
    one machine losing the network while another carries on drilling, and a
    single switch on the database cannot express that.
    """

    def __init__(self, database):
        self.database = database
        self.down = False

    def __call__(self, path, method="GET", body=None):
        if self.down:
            raise urllib.error.URLError("no route to host")
        return self.database(path, method, body)


class RacingLink(Link):
    """A connection where somebody else lands in the gap.

    The version check at the top of a save and the write at the bottom are
    two round trips, and another client can get in between them. That gap is
    the only way to reach the refusal path, and it cannot be reached by
    ordering two ordinary saves: the second one sees the moved version on the
    way in and merges before it ever writes.
    """

    def __init__(self, database, interloper):
        super().__init__(database)
        self.interloper = interloper        # slipped in once, on our first PUT

    def __call__(self, path, method="GET", body=None):
        if path == "/state" and method == "PUT" and self.interloper is not None:
            slipped, self.interloper = self.interloper, None
            self.database("/state", "PUT", {"state": slipped, "base": None})
        return super().__call__(path, method, body)


def connect(database, cache):
    """A client of `database`, caching to `cache`."""
    store = RemoteStore("https://example.invalid", "token", cache=cache)
    store.link = Link(database)
    store._call = store.link
    return store


@pytest.fixture
def remote(tmp_path):
    """A store pointed at a fake database, with its cache in a scratch file."""
    database = FakeDatabase()
    return connect(database, tmp_path / "cache.json"), database


def _flush(progress):
    """Wait for a save to reach the database.

    A save puts it on disk and hands the database to a thread, because the
    drill shows the next card straight after and a round trip in front of
    every word is what that costs. Anything asserting on the database has to
    wait for it to get there.
    """
    store = progress._store()
    if hasattr(store, "flush"):
        # Best effort. Several of these cut the link on purpose, where not
        # landing is the thing being tested.
        store.flush(timeout=6)


def a_card(due=0):
    return Card(ease=2.5, interval=1, reps=1, lapses=0, due=due)


class TestTheFileStore:
    """The original backend, unchanged in behaviour by having an interface."""

    def test_it_round_trips(self, tmp_path):
        store = FileStore(tmp_path / "p.json")
        store.write({"cards": {"de": 1}})
        assert store.read() == {"cards": {"de": 1}}

    def test_a_missing_file_reads_as_empty(self, tmp_path):
        assert FileStore(tmp_path / "nothing.json").read() == {}

    def test_a_corrupt_file_reads_as_empty(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{ this is not json")
        assert FileStore(path).read() == {}

    def test_no_path_means_no_file(self, tmp_path):
        """Tests build throwaway progress this way. It must touch nothing."""
        store = FileStore(None)
        assert store.write({"cards": {}}) is True
        assert store.read() == {}
        assert store.stamp() is None
        assert list(tmp_path.iterdir()) == []

    def test_the_stamp_moves_when_the_file_does(self, tmp_path):
        store = FileStore(tmp_path / "p.json")
        store.write({"a": 1})
        before = store.stamp()
        store.write({"a": 1, "b": 2})
        assert store.stamp() != before


class TestTheRemoteStore:
    def test_it_round_trips_through_the_database(self, remote):
        store, database = remote
        store.write({"cards": {"de": 1}})
        assert store.flush(), "the push never landed"
        assert database.state == {"cards": {"de": 1}}
        assert store.read() == {"cards": {"de": 1}}

    def test_a_write_lands_on_disk_before_the_network(self, remote, tmp_path):
        """The cushion. A drill must not lose a card to a slow database."""
        store, database = remote
        store.link.down = True
        store.write({"cards": {"de": 1}})
        assert json.loads((tmp_path / "cache.json").read_text()) == {
            "cards": {"de": 1}}

    def test_a_write_does_not_wait_for_the_database(self, remote):
        """A third of a second in front of every word is what this is for."""
        store, database = remote
        database.slow = 0.4
        started = time.time()
        store.write({"cards": {"de": 1}})
        assert time.time() - started < 0.1, (
            "the drill is waiting for the database before the next card")
        assert store.flush(timeout=5)
        assert database.state == {"cards": {"de": 1}}

    def test_a_write_that_could_not_get_out_is_kept_and_retried(self, remote):
        store, database = remote
        store.link.down = True
        store.write({"cards": {"de": 7}})
        assert not store.flush(timeout=0.5), "it cannot have landed"
        store.link.down = False
        assert store.flush(timeout=6), "it never went out once the link was up"
        assert database.state == {"cards": {"de": 7}}

    def test_only_the_newest_state_is_sent(self, remote):
        """Each push carries the whole state, so an older one still queued
        has nothing in it the newer one does not."""
        store, database = remote
        store.link.down = True
        for n in range(5):
            store.write({"cards": {"de": n}})
        store.link.down = False
        assert store.flush(timeout=6)
        assert database.state == {"cards": {"de": 4}}

    def test_an_unreachable_read_falls_back_to_the_cache(self, remote):
        store, database = remote
        store.write({"cards": {"de": 1}})
        store.link.down = True
        assert store.read() == {"cards": {"de": 1}}
        assert store.stale() is True

    def test_an_unreachable_stamp_keeps_the_last_version(self, remote):
        """Not `None`. Claiming the store is empty invites a bad merge."""
        store, database = remote
        store.write({"cards": {}})
        seen = store.stamp()
        store.link.down = True
        assert store.stamp() == seen

    def test_the_version_moves_on_every_write(self, remote):
        store, database = remote
        store.write({"a": 1}); store.flush()
        first = store.settled()
        store.write({"a": 2}); store.flush()
        assert store.settled() != first

    def test_settled_never_calls_out(self, remote):
        store, database = remote
        store.write({"a": 1}); store.flush()
        before = len(database.calls)
        store.settled()
        assert len(database.calls) == before

    def test_the_version_is_not_asked_for_on_every_save(self, remote):
        """Once per card is a round trip in front of the next word, for a
        question whose answer only changes when somebody else writes."""
        store, database = remote
        store.write({"a": 1}); store.flush()
        before = len(database.calls)
        for _ in range(10):
            store.moved(store.settled())
        assert len(database.calls) == before, (
            "the drill is asking the database its version once per card")

    def test_a_write_built_on_an_old_version_is_refused(self, remote):
        """And the refusal is remembered, so the next save reads and merges
        rather than pushing over the top of somebody else."""
        store, database = remote
        store.write({"a": 1}); store.flush()
        stale = store.settled()
        database.state = {"a": "somebody else"}     # they got in first
        database.version += 1
        store.write({"a": 2}, base=stale)
        store.flush(timeout=2)
        assert database.state == {"a": "somebody else"}, "it overwrote them"
        assert store.moved(stale), "the next save will not know to merge"

    def test_a_write_built_on_the_current_version_is_taken(self, remote):
        store, database = remote
        store.write({"a": 1}); store.flush()
        store.write({"a": 2}, base=store.settled())
        assert store.flush()
        assert database.state == {"a": 2}

    def test_our_own_writes_do_not_look_like_somebody_else(self, remote):
        """`moved` is what a save asks before deciding to merge. Answering yes
        to our own push sent it off to merge with a copy of what it had just
        sent, once per answered card."""
        store, database = remote
        store.write({"a": 1}); store.flush()
        seen = store.settled()
        store.write({"a": 2}); store.flush()
        assert store.moved(seen) is False

    def test_somebody_else_writing_does_look_like_it(self, remote):
        """Noticed shortly after, rather than the instant it happens.

        Asking the database on every save is a round trip in front of every
        word, so the version is refreshed on the background thread instead.
        A second writer is therefore seen a few seconds late, which is the
        right trade for a drill that normally has exactly one.
        """
        store, database = remote
        store.FRESH_SECONDS = 0     # look every chance it gets
        store.write({"a": 1}); store.flush()
        seen = store.settled()
        database.state = {"a": "theirs"}
        database.version += 1
        for _ in range(200):
            if store.moved(seen):
                break
            time.sleep(0.02)
        assert store.moved(seen) is True

    def test_asking_it_to_go_and_look_sees_them_at_once(self, remote):
        store, database = remote
        store.write({"a": 1}); store.flush()
        seen = store.settled()
        database.state = {"a": "theirs"}
        database.version += 1
        assert store.stamp(fresh=True) != seen

    def test_a_conflict_is_not_mistaken_for_being_offline(self, remote):
        """A refusal means the database is up and has something to say."""
        store, database = remote
        store.write({"a": 1}); store.flush()
        stale = store.settled()
        database.version += 1
        store.write({"a": 2}, base=stale)
        store.flush(timeout=2)
        assert store.stale() is False


class TestProgressOnASharedDatabase:
    """The point of the exercise: two clients, one schedule."""

    def test_it_round_trips(self, remote, tmp_path):
        store, database = remote
        p = Progress.load(path=tmp_path / "p.json", store=store)
        p.cards[3] = a_card(due=99)
        p.new_done = 7
        p.save()
        _flush(p)

        back = Progress.load(path=tmp_path / "p.json", store=store)
        assert back.new_done == 7
        assert back.cards[3].due == 99

    def test_what_one_client_writes_the_other_sees(self, remote, tmp_path):
        store, database = remote
        desk = Progress.load(path=tmp_path / "a.json", store=store)
        desk.cards[3] = a_card(due=99)
        desk.save()
        _flush(desk)

        phone = Progress.load(path=tmp_path / "b.json",
                              store=connect(database, tmp_path / "b.json"))
        assert phone.cards[3].due == 99

    def test_two_clients_both_keep_their_work(self, remote, tmp_path):
        """The merge that already existed for two processes on one file."""
        store, database = remote
        desk = Progress.load(path=tmp_path / "a.json", store=store)

        phone = Progress.load(path=tmp_path / "b.json",
                              store=connect(database, tmp_path / "b.json"))

        desk.cards[3] = a_card(due=11)
        desk.save()
        _flush(desk)
        phone.cards[5] = a_card(due=22)
        phone.save()
        _flush(phone)

        after = Progress.load(path=tmp_path / "c.json",
                              store=connect(database, tmp_path / "c.json"))
        assert after.cards[3].due == 11
        assert after.cards[5].due == 22

    def test_a_session_that_loses_the_network_is_not_thrown_away(
            self, remote, tmp_path):
        """The bug this design exists to avoid.

        Work done while the database was unreachable must still win the merge
        once it comes back. If a failed push were recorded as agreed, the
        reconnect would treat these cards as already shared and let the
        database's older copy overwrite them.
        """
        store, database = remote
        p = Progress.load(path=tmp_path / "p.json", store=store)
        p.cards[3] = a_card(due=11)
        p.save()
        _flush(p)

        store.link.down = True
        p.cards[5] = a_card(due=22)         # a whole session, offline
        p.cards[7] = a_card(due=33)
        p.save()
        _flush(p)

        store.link.down = False
        p.save()
        _flush(p)                            # the network comes back

        after = Progress.load(path=tmp_path / "q.json",
                              store=connect(database, tmp_path / "q.json"))
        assert after.cards[3].due == 11
        assert after.cards[5].due == 22
        assert after.cards[7].due == 33

    def test_a_save_onto_a_moved_version_merges_first(self, remote, tmp_path):
        """The ordinary case: the mismatch is caught on the way in."""
        store, database = remote
        desk = Progress.load(path=tmp_path / "a.json", store=store)
        phone = Progress.load(path=tmp_path / "b.json",
                              store=connect(database, tmp_path / "b.json"))

        desk.cards[3] = a_card(due=11)
        phone.cards[5] = a_card(due=22)
        desk.save()
        _flush(desk)
        phone.save()
        _flush(phone)

        after = Progress.load(path=tmp_path / "c.json",
                              store=connect(database, tmp_path / "c.json"))
        assert after.cards[3].due == 11
        assert after.cards[5].due == 22

    def test_a_save_beaten_to_it_mid_flight_is_not_lost(self, tmp_path):
        """The race the compare-and-swap exists for.

        Somebody else's write lands after this save has checked the version
        and before its own arrives. The database refuses ours, and neither
        card may be dropped: theirs is already in, and ours has to reach the
        next save that goes out.

        That merge used to happen in a retry loop inside the save itself.
        It cannot now, because a save hands the database to a thread and
        returns: the refusal comes back after the drill has moved on. The
        store remembers it instead, and the next save reads and merges before
        it writes, which is the same guarantee one card later.
        """
        database = FakeDatabase()
        store = RemoteStore("https://example.invalid", "token",
                            cache=tmp_path / "cache.json")
        p = Progress.load(path=tmp_path / "p.json", store=store)

        deck = load_deck()
        theirs = {"cards": {deck[9].id: a_card(due=44).to_dict()}}
        store.link = RacingLink(database, theirs)
        store._call = store.link

        p.cards[5] = a_card(due=22)
        p.save()
        _flush(p)
        assert any(call == ("/state", "PUT") for call in database.calls)

        # The next card answered is the one that carries it, and by then the
        # store knows it was refused.
        p.cards[7] = a_card(due=55)
        p.save()
        _flush(p)

        after = Progress.load(path=tmp_path / "q.json",
                              store=connect(database, tmp_path / "q.json"))
        assert after.cards[5].due == 22     # ours, refused once then carried
        assert after.cards[7].due == 55
        assert after.cards[9].due == 44     # theirs, landed in the gap

    def test_the_offline_work_survives_the_other_client_moving(
            self, remote, tmp_path):
        """The same, with the database genuinely changing underneath."""
        store, database = remote
        p = Progress.load(path=tmp_path / "p.json", store=store)
        p.cards[3] = a_card(due=11)
        p.save()
        _flush(p)

        store.link.down = True              # only this client is cut off
        p.cards[5] = a_card(due=22)
        p.save()
        _flush(p)

        # Somebody else drills a different card while we cannot see it.
        other = Progress.load(path=tmp_path / "o.json",
                              store=connect(database, tmp_path / "o.json"))
        other.cards[9] = a_card(due=44)
        other.save()
        _flush(other)

        store.link.down = False
        p.save()
        _flush(p)
        # One more, for the same reason as the race above: the refusal comes
        # back after the save has returned, so the merge is the next one out.
        p.save()
        _flush(p)

        after = Progress.load(path=tmp_path / "q.json",
                              store=connect(database, tmp_path / "q.json"))
        assert after.cards[5].due == 22     # ours, done offline
        assert after.cards[9].due == 44     # theirs, done meanwhile

    def test_starting_up_offline_keeps_every_card(self, remote, tmp_path):
        """A restart with no network reads the cache, and must not later
        hand that session back to whatever the database still held."""
        store, database = remote
        first = Progress.load(path=tmp_path / "p.json", store=store)
        first.cards[3] = a_card(due=11)
        first.save()
        _flush(first)

        # A fresh process, same machine, same cache file, no network.
        cold_store = connect(database, tmp_path / "cache.json")
        cold_store.link.down = True
        cold = Progress.load(path=tmp_path / "p.json", store=cold_store)
        assert cold.cards[3].due == 11      # served from the cache
        cold.cards[5] = a_card(due=22)
        cold.save()
        _flush(cold)

        cold_store.link.down = False
        cold.save()
        _flush(cold)

        after = Progress.load(path=tmp_path / "q.json",
                              store=connect(database, tmp_path / "q.json"))
        assert after.cards[3].due == 11
        assert after.cards[5].due == 22


class TestNothingChangesUntilItIsSwitchedOn:
    def test_the_default_is_still_a_plain_file(self, tmp_path, monkeypatch):
        """No configuration means the drill behaves exactly as it always did."""
        monkeypatch.setattr("spanish_drill.store.SYNC_URL", "")
        monkeypatch.setattr("spanish_drill.store.SYNC_TOKEN", "")
        from spanish_drill.store import default_store
        assert isinstance(default_store(tmp_path / "p.json"), FileStore)

    def test_a_configured_url_gets_the_database(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spanish_drill.store.SYNC_URL", "https://x.invalid")
        monkeypatch.setattr("spanish_drill.store.SYNC_TOKEN", "t")
        from spanish_drill.store import default_store
        assert isinstance(default_store(tmp_path / "p.json"), RemoteStore)
