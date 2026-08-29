# spanish-drill

Static single-page Spanish vocabulary voice drill. 253 cards, spaced repetition,
spoken prompts and spoken answers. No build step, no dependencies, no server side.
The whole app is `index.html`.

**Live:** https://nomagonz.github.io/spanish-drill/

## Status

| Item | State |
|---|---|
| `index.html` (the app) | Present. 253 cards, unchanged logic/styling/word list. |
| `localStorage` patch | Applied |
| GitHub repo `Nomagonz/spanish-drill` | Public |
| GitHub Pages | Enabled, `main` / root |
| Mic on real hardware | **Not verified — yours to run.** See below. |

## The one code change

The app persisted progress through `window.storage`, which exists only inside
Claude's artifact host. Anywhere else that object is missing and the code fell back
to a plain in-memory object, so it ran but lost all progress on reload.

The backing of the `store` wrapper is now `localStorage`. Keys were already
namespaced under `esdrill:v2`, so no key changes were needed.

Before:

    const store = {
      async get(k){
        try{ if(window.storage){ const r = await window.storage.get(k,false); return r? r.value : null; } }
        catch(e){ return mem[k] ?? null; }
        return mem[k] ?? null;
      },
      async set(k,v){
        mem[k]=v;
        try{ if(window.storage) await window.storage.set(k,v,false); }catch(e){}
      }
    };

After:

    const store = {
      async get(k){
        try{ const r = localStorage.getItem(k); if(r !== null) return r; }
        catch(e){ return mem[k] ?? null; }
        return mem[k] ?? null;
      },
      async set(k,v){
        mem[k]=v;
        try{ localStorage.setItem(k,v); }catch(e){}
      }
    };

Semantics preserved on purpose:

* `get` still returns `null` on a miss, never `undefined`.
* Values are still stored raw, not JSON-wrapped. `save()` already hands the wrapper
  a `JSON.stringify`'d string and `load()` already does the `JSON.parse`.
* The `async` signatures stay. `localStorage` is synchronous, but an `async` method
  returning a plain value still returns a promise, so every existing `await` works
  untouched.
* The `try/catch` and the `mem` mirror stay. Safari Private Mode and quota-exceeded
  both throw; that is swallowed and the app degrades to the same in-memory
  no-persistence behavior it had before.

Nothing else was touched. The only other edit was deleting the handoff comment block
at the top of the file, which that block explicitly said could be removed.

## Verifying the mic

Open the live URL **on the iPhone** and press "Test the mic" on the start screen.

This cannot be verified headlessly. `webkitSpeechRecognition` needs real microphone
hardware plus Google's speech backend, and `getUserMedia` needs a user gesture on a
secure origin. An automated check can confirm the page loads over https and that the
origin is secure. It cannot confirm speech is actually captured.

If the mic reports blocked: confirm the URL is `https://` (not `file://`), and that
Safari has microphone permission for the site under Settings -> Safari.

## Unrelated repo — do not touch

`Nomagonz/boerne-site` is a live production business site serving
`boernephotoboothco.com` via Netlify continuous deployment. Every push to its `main`
auto-deploys, and its `/index.html` is the real homepage. This project is deliberately
kept in a separate repo.
