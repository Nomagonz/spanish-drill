/**
 * The one database, over HTTP.
 *
 * Small on purpose. It holds the agreed copy of the schedule and says what
 * version it is on; it does not know what a card is, when one is due, or how
 * to grade an answer. All of that stays in the drill, which is the only place
 * it has ever lived and the only place it can stay correct.
 *
 *   GET  /version  -> {version}
 *   GET  /state    -> {version, state}
 *   PUT  /state    -> {version}            body {state, base}
 *
 * A PUT carries the version it was based on. If the stored version has moved
 * since, the write is refused with 409 and the current state comes back with
 * it, so the client can fold the two together and try again. Without that,
 * the phone and the desktop saving within one round trip of each other would
 * mean whichever landed second silently threw the other's cards away.
 */

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      // The drill page is served from somewhere else, and a browser will not
      // call this without being told that is allowed.
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
      "Access-Control-Allow-Methods": "GET, PUT, OPTIONS",
      "Cache-Control": "no-store",
    },
  });

/**
 * Constant time, so a wrong token cannot be narrowed down one character at a
 * time by how quickly it is rejected.
 */
function tokenMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string") return false;
  if (given.length !== expected.length) return false;
  let differences = 0;
  for (let i = 0; i < given.length; i++) {
    differences |= given.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return differences === 0;
}

function authorised(request, env) {
  const header = request.headers.get("Authorization") || "";
  const given = header.startsWith("Bearer ") ? header.slice(7) : "";
  return Boolean(env.SYNC_TOKEN) && tokenMatches(given, env.SYNC_TOKEN);
}

async function current(env) {
  const row = await env.DB.prepare(
    "SELECT state, version FROM progress WHERE id = 1"
  ).first();
  // A database that has had its schema applied but never been written to.
  return row || { state: "{}", version: 0 };
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);

    if (request.method === "OPTIONS") return json({}, 204);

    if (!authorised(request, env)) {
      return json({ error: "unauthorised" }, 401);
    }

    if (pathname === "/version" && request.method === "GET") {
      const row = await current(env);
      return json({ version: row.version });
    }

    if (pathname === "/state" && request.method === "GET") {
      const row = await current(env);
      return json({ version: row.version, state: JSON.parse(row.state) });
    }

    if (pathname === "/state" && request.method === "PUT") {
      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "expected json" }, 400);
      }
      if (!body || typeof body.state !== "object" || body.state === null) {
        return json({ error: "expected a state object" }, 400);
      }

      const text = JSON.stringify(body.state);

      // The compare-and-swap. `base` is the version the client merged
      // against; the UPDATE only fires while the stored version still
      // matches it, so two writes racing cannot both win. A client that
      // sends no base is taking the old last-writer-wins behaviour
      // knowingly, which is what a first upload from an empty database is.
      const guard =
        body.base === undefined || body.base === null
          ? env.DB.prepare(
              "UPDATE progress SET state = ?1, version = version + 1," +
                " updated = datetime('now') WHERE id = 1"
            ).bind(text)
          : env.DB.prepare(
              "UPDATE progress SET state = ?1, version = version + 1," +
                " updated = datetime('now') WHERE id = 1 AND version = ?2"
            ).bind(text, body.base);

      const done = await guard.run();
      if (done.meta.changes === 0) {
        const row = await current(env);
        return json(
          {
            error: "version moved",
            version: row.version,
            state: JSON.parse(row.state),
          },
          409
        );
      }

      const row = await current(env);
      return json({ version: row.version });
    }

    return json({ error: "not found" }, 404);
  },
};
