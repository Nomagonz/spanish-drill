"""Drives a real window and a real browser against one hub, and reports.

Run as a script, in a process of its own, because that is the only way this
is worth anything. Qt aborts rather than fails when a window or a thread is
left in a state an earlier test put it in, and an abort takes the whole run
with it: the suite stops saying anything about anything. A separate process
starts from nothing every time and cannot be poisoned by what ran before.

Prints one JSON document describing what the two screens saw.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Never the shared database: this is about which card is on screen.
os.environ.pop("SPANISH_DRILL_SYNC_URL", None)
os.environ.pop("SPANISH_DRILL_SYNC_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication          # noqa: E402
from spanish_drill import ui as ui_module         # noqa: E402
from spanish_drill.progress import Progress       # noqa: E402
from spanish_drill.scheduler import Card, today   # noqa: E402
from spanish_drill.serve import Hub, start_server  # noqa: E402

ui_module.Window._start_preload = lambda self: None
ui_module.Window._start_prerecording = lambda self: None
plain = lambda text: re.sub(r"<[^>]+>", "", text)


def main():
    tmp = Path(tempfile.mkdtemp())
    progress = Progress(path=tmp / "p.json")
    progress.day = today()
    for i in range(12):                 # enough due to have something to ask
        progress.cards[i] = Card(interval=1, reps=1, due=0)

    hub = Hub(progress)
    app = QApplication(sys.argv[:1])
    window = ui_module.Window(hub=hub)
    window.go.setEnabled(True)          # the model preload normally does this
    server, _ = start_server(hub, port=0, host="127.0.0.1", token="probe")
    base = "http://127.0.0.1:%d" % server.server_address[1]

    def pump(seconds=1.0):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def post(path, body=None):
        r = urllib.request.Request(f"{base}{path}?t=probe", method="POST",
                                   data=json.dumps(body or {}).encode(),
                                   headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(r, timeout=30).read()

    def browser(since):
        with urllib.request.urlopen(f"{base}/poll?t=probe&since={since}",
                                    timeout=30) as a:
            d = json.loads(a.read())
        shown = [json.loads(e) for e in d["events"]
                 if json.loads(e)["event"] == "prompt"]
        return (shown[-1]["text"] if shown else None), d["next"]

    out = {}
    try:
        window.typing.setChecked(True)
        pump(0.2)
        window.toggle()
        pump(3.0)

        theirs, since = browser(0)
        out["window_card"] = plain(window.prompt_label.text())
        out["browser_card"] = theirs
        out["one_session"] = window.worker.session is hub.session

        was = plain(window.prompt_label.text())
        card = hub.deck[hub.session.current]
        post("/answer", {"text": card.answers[0]})
        pump(4.0)
        theirs, since = browser(since)
        out["after_browser_answer"] = {
            "window": plain(window.prompt_label.text()),
            "browser": theirs,
            "window_moved": plain(window.prompt_label.text()) != was,
        }

        was = plain(window.prompt_label.text())
        card = hub.deck[hub.session.current]
        window.answer_box.setText(card.answers[0])
        window._submit_typed()
        pump(4.0)
        theirs, since = browser(since)
        out["after_window_answer"] = {
            "window": plain(window.prompt_label.text()),
            "browser": theirs,
            "window_moved": plain(window.prompt_label.text()) != was,
        }
    finally:
        hub.stop()
        for _ in range(200):
            if not hub.state()["running"]:
                break
            pump(0.05)
        if window.thread is not None:
            window.worker.stop()
            window.thread.quit()
            window.thread.wait(5000)
        server.shutdown()
        server.server_close()
        window.close()
        pump(0.2)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
