/**
 * Runs the worker's request handler outside Cloudflare and reports what it did.
 *
 * The worker was the one piece with nothing standing behind it, and it shipped
 * a fault that every request from a browser hit and no request from curl did:
 * the CORS preflight threw, so the page could not make a single keyed call
 * while every hand-run check passed. This is what makes that testable without
 * deploying anything.
 *
 * The database is a stand-in holding one row, which is all the real schema is.
 */
import { writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { readFileSync } from "node:fs";

const source = process.argv[2];
const dir = mkdtempSync(join(tmpdir(), "worker-"));
const copy = join(dir, "worker.mjs");
writeFileSync(copy, readFileSync(source));
const worker = (await import("file://" + copy)).default;

/** Just enough D1: one row, and an UPDATE that may or may not match. */
function database(initial) {
  const row = { state: JSON.stringify(initial.state), version: initial.version };
  return {
    prepare(sql) {
      return {
        bind(...args) {
          return {
            async run() {
              const guarded = sql.includes("AND version = ?2");
              if (guarded && row.version !== args[1]) {
                return { meta: { changes: 0 } };
              }
              row.state = args[0];
              row.version += 1;
              return { meta: { changes: 1 } };
            },
            async first() { return { ...row }; },
          };
        },
        async first() { return { ...row }; },
        async run() { return { meta: { changes: 0 } }; },
      };
    },
    _row: row,
  };
}

const jobs = JSON.parse(readFileSync(0, "utf8"));
const out = [];

for (const job of jobs) {
  const db = database(job.db || { state: {}, version: 0 });
  const env = { DB: db, SYNC_TOKEN: job.env_token ?? "right-key" };
  const headers = {};
  if (job.token !== undefined) headers.Authorization = "Bearer " + job.token;
  if (job.body !== undefined) headers["Content-Type"] = "application/json";
  if (job.origin) headers.Origin = job.origin;
  if (job.request_headers) headers["Access-Control-Request-Headers"] = job.request_headers;
  if (job.request_method) headers["Access-Control-Request-Method"] = job.request_method;

  const request = new Request("https://example.invalid" + job.path, {
    method: job.method || "GET",
    headers,
    body: job.body === undefined ? undefined : JSON.stringify(job.body),
  });

  let record;
  try {
    const answer = await worker.fetch(request, env);
    const text = await answer.text();
    record = {
      status: answer.status,
      body: text ? JSON.parse(text) : null,
      allow_origin: answer.headers.get("Access-Control-Allow-Origin"),
      allow_headers: answer.headers.get("Access-Control-Allow-Headers"),
      allow_methods: answer.headers.get("Access-Control-Allow-Methods"),
      db_version: db._row.version,
      db_state: JSON.parse(db._row.state),
    };
  } catch (e) {
    // The failure being guarded against: a throw here is a 500 in production,
    // and from the browser it is indistinguishable from the network being out.
    record = { threw: String(e && e.message || e) };
  }
  out.push(record);
}

process.stdout.write(JSON.stringify(out));
