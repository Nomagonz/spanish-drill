"""Where progress is kept.

Two backends behind one small interface, so nothing above this file knows
whether the schedule lives on this disk or in a database every client shares.

The wire format is the file format, field for field. That is the whole trick:
the merge in `progress.py` takes a parsed dict and does not care where it came
from, so one shared database costs a backend rather than a rewrite. Whatever
`save()` used to write into progress.json is exactly what goes over the wire.

Three operations, because that is all `Progress` ever needed of a file:

    stamp()     what it looks like from outside, cheaply. Used to answer
                "did somebody else write while we were thinking" without
                paying to read the whole thing.
    moved(seen) whether it has moved since `seen` for a reason that is not
                this client's own last write. Asked instead of comparing
                stamps by hand, because a store that pushes in the background
                moves on its own and answering "yes" to that would send the
                drill off to merge with itself.
    settled()   the same, but as of the last call that already succeeded,
                and never over the network. A save has just been told the
                new version by the write itself, and asking again would put
                a second round trip in the path of every answered card.
    stale()     whether the last operation missed the database and was served
                locally instead. What came back is then a guess, not the
                truth, and the merge above has to be told so.
    read()      the saved state, as a dict. `{}` when there is none.
    write(data, base)
                commit what we built from version `base`, reporting whether
                it landed: True, False for a store we could not reach, or
                CONFLICT for one that has moved on since `base` and will not
                take a write built on it.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import SYNC_TIMEOUT, SYNC_TOKEN, SYNC_URL

# The store has moved since the version this write was built on. Somebody else
# saved first, and what they saved has to be folded in before trying again.
CONFLICT = object()


class FileStore:
    """One file on this disk, written atomically.

    `path=None` means hold nothing: reads come back empty and writes go
    nowhere. Tests build throwaway `Progress` objects that way to work on
    cards in memory, and they must not touch the filesystem at all.
    """

    def __init__(self, path):
        self.path = Path(path) if path is not None else None

    def stamp(self):
        if self.path is None:
            return None
        try:
            info = self.path.stat()
            return (info.st_mtime_ns, info.st_size)
        except OSError:
            return None

    def settled(self):
        return self.stamp()         # a stat is already as cheap as it gets

    def moved(self, seen):
        return self.stamp() != seen

    def pending(self):
        return False                # a write here has landed by the time it returns

    def stale(self):
        return False                # the disk is either there or it raised

    def read(self):
        if self.path is None:
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def write(self, data, base=None):
        if self.path is None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)  # atomic: a crash cannot shred progress
        return True


class RemoteStore:
    """One database, shared by every client that drills this deck.

    The local file stays underneath as a write-through cache, for one reason:
    a drill answers a card every few seconds and saves after each one, and
    stalling a spoken word on a round trip that may not come back is a worse
    failure than a late write. So a write lands on disk first and is pushed
    after. Losing the network degrades the app to exactly what it was before
    this file existed, and the next save that gets through carries everything,
    because every save sends the whole state rather than a delta.

    The same reasoning covers reads. If the database cannot be reached we
    report the version we last saw rather than `None`: claiming "nothing
    changed" is honest when we cannot look, where claiming the store is empty
    would invite the merge to treat a live schedule as a fresh one.
    """

    # How long an answer about the version is worth before asking again. The
    # drill saves after every card, and asking the database each time put a
    # round trip in front of the next word for a question whose answer only
    # changes when somebody else writes.
    FRESH_SECONDS = 5.0
    # How long to let writes gather before sending. Each push carries the
    # whole schedule, so one covers every card answered while it waited and
    # sending them separately buys nothing. It costs real time: encoding
    # sixty kilobytes of JSON is work the drill is competing with for the
    # interpreter, and doing it once a second instead of once a card is the
    # difference between the next word arriving promptly and not.
    SETTLE_SECONDS = 0.75

    def __init__(self, url, token, cache=None, timeout=SYNC_TIMEOUT):
        self.url = url.rstrip("/")
        self.token = token
        self.cache = FileStore(cache)
        self.timeout = timeout
        self._last = None           # newest version the database admitted to
        self._lock = threading.RLock()
        self.offline = False        # last call failed; surfaced in the UI
        self._checked = 0.0         # when the version was last asked for
        self._ours = set()          # versions this client's own writes made
        self._pending = None        # the newest state waiting to go out
        self._pending_base = None
        self._wake = threading.Event()
        self._pusher = None
        self._inflight = False      # a push is out there right now
        self._suspect = False       # somebody else wrote; re-read before trusting
        # Given by whoever owns the schedule: takes the other side's state
        # and hands back the two folded together, ready to send. Set because
        # a refusal now arrives after the save that caused it has returned,
        # so there is nobody left to ask by then.
        self.merge = None
        # Called with the state that has just been accepted. Whoever owns the
        # schedule uses it to move its idea of what the database holds, which
        # cannot be done when the write is queued: it has not been taken by
        # anything yet, and saying otherwise is how a card gets dropped.
        self.landed = None

    # -- plumbing ---------------------------------------------------------
    def _call(self, path, method="GET", body=None):
        """(status, body). A refusal is an answer, not a failure.

        The distinction matters: a 409 means the database is up and has
        something to say, and treating it as a dead connection would send the
        drill into its offline path over a conflict it could have resolved.
        """
        request = urllib.request.Request(
            f"{self.url}{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json",
                     # Named, because Cloudflare turns away urllib's default
                     # signature with a 403 before the worker is ever reached.
                     # Measured against the live worker: curl got a 200 and
                     # the drill got "error code: 1010" on the same token.
                     "User-Agent": "spanish-drill"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                return answer.status, json.loads(answer.read().decode() or "{}")
        except urllib.error.HTTPError as refusal:
            try:
                return refusal.code, json.loads(refusal.read().decode() or "{}")
            except (ValueError, OSError):
                return refusal.code, {}

    def _try(self, *args, **kwargs):
        """The same call, but a dead network is an outcome and not a crash."""
        try:
            status, out = self._call(*args, **kwargs)
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError):
            self.offline = True
            return None, None
        self.offline = False
        return status, out

    # -- the interface ----------------------------------------------------
    def stamp(self, fresh=False):
        """The version, from what we already know unless asked to go and look.

        Never over the network by default. This is asked once per answered
        card, and a round trip in front of the next word is exactly what this
        store exists to stop paying. The background thread keeps the answer
        current instead, so another client's write is noticed a few seconds
        later rather than immediately, which is the right trade for a drill
        that only ever has one writer.
        """
        with self._lock:
            known = self._last
            stale_enough = (known is None or self._suspect
                            or time.monotonic() - self._checked >= self.FRESH_SECONDS)
            if not (fresh or stale_enough):
                return known
            if known is not None and not fresh:
                # Let the thread go and look; answer with what we have.
                self._wake.set()
                self._ensure_pusher()
                return known
        status, out = self._try("/version")
        with self._lock:
            if status != 200:
                return self._last       # cannot see; do not guess
            self._last = ("remote", out.get("version"))
            self._checked = time.monotonic()
            self._suspect = False
            return self._last

    def moved(self, seen):
        """Has somebody else written since `seen`?

        Our own pushes move the version too, and treating that as news sent
        the drill to re-read and merge with a copy of what it had just sent,
        once per answered card. Only a version this client did not produce
        counts.
        """
        now = self.stamp()
        if now == seen:
            return False
        with self._lock:
            return not (isinstance(now, tuple) and now[1] in self._ours)

    def settled(self):
        return self._last

    def pending(self):
        """Is anything of ours still on its way?

        What the caller does with this is decide whether the database can be
        said to hold what it was just handed. Saying yes too early is the one
        fault here that quietly loses a card: the next merge would treat that
        card as already shared and let the database's copy, which never got
        it, win.
        """
        with self._lock:
            return self._pending is not None or self._inflight

    def stale(self):
        return self.offline

    def read(self):
        with self._lock:
            status, out = self._try("/state")
            if status != 200:
                return self.cache.read()        # carry on from what we have
            self._last = ("remote", out.get("version"))
            state = out.get("state") or {}
            self.cache.write(state)             # keep the fallback current
            return state

    def write(self, data, base=None):
        """Disk first, then the database, and the database on its own thread.

        The drill saves after every answered card and then shows the next one.
        Pushing from here put a third of a second of network in front of every
        word, which is most of what made answering from a phone feel slow. The
        write that matters for not losing anything is the local one, and that
        is three milliseconds.

        Only the newest state is ever waiting: each push carries the whole of
        it, so an older one still queued has nothing in it the newer one does
        not, and dropping it loses nothing.

        `base` is the version this write was built from. The database refuses
        it if that is no longer the current one, which is what keeps two
        writers from overwriting each other.
        """
        with self._lock:
            # What the drill just earned is safe on disk before anything slow
            # is attempted with it.
            self.cache.write(data)
            self._pending = data
            self._pending_base = base
        self._ensure_pusher()
        self._wake.set()
        return True

    # -- the thread that does the waiting -----------------------------------
    def _ensure_pusher(self):
        with self._lock:
            if self._pusher is not None and self._pusher.is_alive():
                return
            self._pusher = threading.Thread(target=self._push_forever,
                                            daemon=True)
            self._pusher.start()

    def _push_forever(self):
        while True:
            self._wake.wait(1.0)
            self._wake.clear()
            # Let a burst finish arriving. A drill answering a card every
            # couple of seconds would otherwise push every one of them.
            deadline = time.monotonic() + self.SETTLE_SECONDS
            while time.monotonic() < deadline:
                time.sleep(0.05)
                with self._lock:
                    if self._pending is None:
                        break
            self._push_once()
            self._catch_up()

    def _catch_up(self):
        """Ask what version the database is on, off the drill's path."""
        with self._lock:
            if self._pending is not None:
                return              # a push will bring back a version anyway
            if time.monotonic() - self._checked < self.FRESH_SECONDS:
                return
        status, out = self._try("/version")
        if status != 200:
            return
        with self._lock:
            self._last = ("remote", out.get("version"))
            self._checked = time.monotonic()
            self._suspect = False

    def _push_once(self):
        with self._lock:
            data, base = self._pending, self._pending_base
            self._pending = None
            self._inflight = data is not None
        if data is None:
            return
        try:
            status, out = self._try(
                "/state", method="PUT",
                body={"state": data,
                      "base": base[1] if isinstance(base, tuple) else None})

            if status == 200:
                with self._lock:
                    version = out.get("version")
                    self._last = ("remote", version)
                    self._ours.add(version)
                    self._checked = time.monotonic()
                    self._suspect = False
                    tell = self.landed
                if tell is not None:
                    tell(data)      # outside the lock: it takes another one
                return

            if status == 409:
                # Somebody else got in first and their copy is the current
                # one, so ours cannot go as it stands. Folded together and
                # offered again rather than dropped: nothing else is going to
                # do it, because the save that queued this returned long
                # before the refusal came back.
                version = out.get("version")
                theirs = out.get("state") or {}
                with self._lock:
                    self._last = ("remote", version)
                    self._checked = time.monotonic()
                    self._suspect = self.merge is None
                    merge = self.merge
                if merge is None:
                    return          # the next save will read and merge
                # Outside the lock. The merge takes the schedule's own lock,
                # and a save holds that while calling in here to write, so
                # holding both in that order is a deadlock waiting for a
                # slow enough network.
                folded = merge(theirs)
                with self._lock:
                    if self._pending is None:
                        self._pending = folded
                        self._pending_base = ("remote", version)
                self._wake.set()
                return

            # Unreachable, or refused for a reason worth retrying. Put it back
            # unless something newer has already taken its place.
            with self._lock:
                if self._pending is None:
                    self._pending = data
                    self._pending_base = base
            time.sleep(1.0)
            self._wake.set()
        finally:
            with self._lock:
                self._inflight = False

    def flush(self, timeout=10.0):
        """Wait for what is queued to land. For shutting down, and for tests.

        Waits on the push in flight as well as the one still queued: a state
        the pusher has picked up has not arrived anywhere yet, and returning
        on that alone said a write had landed while it was still on the wire.
        """
        deadline = time.monotonic() + timeout
        self._ensure_pusher()
        self._wake.set()
        while time.monotonic() < deadline:
            with self._lock:
                if self._pending is None and not self._inflight:
                    return True
            time.sleep(0.02)
        return False


def default_store(path):
    """The store this machine is configured for.

    Unset `SYNC_URL` means the original behaviour in full: this file, this
    machine, nobody else. That is the default on purpose, so nothing about
    the drill changes until the database is deliberately switched on.
    """
    if SYNC_URL and SYNC_TOKEN:
        return RemoteStore(SYNC_URL, SYNC_TOKEN, cache=path)
    return FileStore(path)
