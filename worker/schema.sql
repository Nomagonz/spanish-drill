-- The one database, in full.
--
-- One row. That is deliberate and worth the paragraph, because the obvious
-- shape is a row per card and it is the wrong one here.
--
-- The drill saves after every answered card, a few seconds apart. A card per
-- row means rewriting all 1670 of them on each of those saves, which is on
-- the order of 167,000 row writes in a single sitting and past D1's daily
-- allowance before the first session is over. As one row it is one write.
--
-- Nothing is lost by it. The merging that matters is per card and happens on
-- the client, in `progress.py`, where it already lived when two processes
-- shared one file: whoever touched a card keeps it, and counters are folded
-- in as deltas. The database's job is to hold the agreed copy and to say what
-- version it is on, so a client that has fallen behind is told rather than
-- allowed to overwrite.
CREATE TABLE IF NOT EXISTS progress (
  id      INTEGER PRIMARY KEY CHECK (id = 1),
  -- The whole of progress.json, unchanged. The wire format is the file
  -- format, so nothing here needs to know what a card is.
  state   TEXT    NOT NULL DEFAULT '{}',
  -- Bumped on every accepted write. A client sends the version it based its
  -- write on and is refused if this has moved since, which is what stops the
  -- phone and the desktop from quietly overwriting each other.
  version INTEGER NOT NULL DEFAULT 0,
  updated TEXT    NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO progress (id, state, version) VALUES (1, '{}', 0);
