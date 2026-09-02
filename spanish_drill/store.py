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

    def __init__(self, url, token, cache=None, timeout=SYNC_TIMEOUT):
        self.url = url.rstrip("/")
        self.token = token
        self.cache = FileStore(cache)
        self.timeout = timeout
        self._last = None           # newest version the database admitted to
        self._lock = threading.RLock()
        self.offline = False        # last call failed; surfaced in the UI

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
    def stamp(self):
        with self._lock:
            status, out = self._try("/version")
            if status != 200:
                return self._last       # cannot see; do not guess
            self._last = ("remote", out.get("version"))
            return self._last

    def settled(self):
        return self._last

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
        """Disk first, then the database. True only if the database took it.

        The caller uses that answer to decide whether to move its idea of
        what has been agreed with the database. Saying yes to a push that
        never landed is the one bug here that silently eats a day's work:
        the next reconnect would see this session as already synced and let
        the database's older copy win the merge.

        `base` is the version this write was built from. The database refuses
        it if that is no longer the current one, which is what keeps the phone
        and the desktop from overwriting each other when both save inside the
        same round trip.
        """
        with self._lock:
            # What the drill just earned is safe on disk before anything slow
            # is attempted with it.
            self.cache.write(data)
            status, out = self._try(
                "/state", method="PUT",
                body={"state": data,
                      "base": base[1] if isinstance(base, tuple) else None})
            if status == 409:
                self._last = ("remote", out.get("version"))
                return CONFLICT
            if status != 200:
                return False
            self._last = ("remote", out.get("version"))
            return True


def default_store(path):
    """The store this machine is configured for.

    Unset `SYNC_URL` means the original behaviour in full: this file, this
    machine, nobody else. That is the default on purpose, so nothing about
    the drill changes until the database is deliberately switched on.
    """
    if SYNC_URL and SYNC_TOKEN:
        return RemoteStore(SYNC_URL, SYNC_TOKEN, cache=path)
    return FileStore(path)
