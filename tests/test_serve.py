"""Drilling from a phone.

The browser is a keyboard and a screen. Everything tested here is about that
being all it is: the deck, the schedule and the grading stay on this machine,
and what crosses the wire is a cue out and an answer back.

Nothing here binds a public interface or touches the real progress file.
"""
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from spanish_drill import sentences as S
from spanish_drill.deck import index_by_id
from spanish_drill.progress import Progress
from spanish_drill.scheduler import Card, MATURE_AT
from spanish_drill.serve import Handler, Hub

TOKEN = "test-token"


def scratch_progress(card_ids=(), interval=MATURE_AT):
    from spanish_drill.scheduler import today
    path = Path(tempfile.mkdtemp(prefix="drill-serve-")) / "progress.json"
    p = Progress(path=path)
    p.day = today()             # or every load rolls the tallies over
    ids = index_by_id()
    for cid in card_ids:
        p.cards[ids[cid]] = Card(interval=interval, reps=2, due=0)
    return p


def everything_for(item):
    """Every card a sentence needs, conjugated forms included."""
    return tuple(item.needs) + tuple(S._form_requirements(item, set()))


@pytest.fixture
def hub():
    """A hub with exactly one sentence unlocked, so a drill has something
    to ask. Granting nothing would test the empty case by accident."""
    item = S.load_sentences()[0]
    return Hub(scratch_progress(everything_for(item)))


@pytest.fixture
def server(hub):
    """A real server on a loopback port, torn down after the test."""
    Handler.hub, Handler.token = hub, TOKEN
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    hub.stop()
    srv.shutdown()
    srv.server_close()


def post(base, path, body=None, token=TOKEN):
    request = urllib.request.Request(
        f"{base}{path}?t={token}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=5)


def collect(base, into, token=TOKEN):
    """Drive the poll loop the browser runs, collecting what it is told."""
    def run():
        mark = 0
        while True:
            try:
                reply = urllib.request.urlopen(
                    f"{base}/poll?t={token}&since={mark}", timeout=30)
                payload = json.loads(reply.read())
            except Exception:
                return
            mark = payload["next"]
            for event in payload["events"]:
                into.append(json.loads(event))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def wait_for(events, name, timeout=4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hits = [e for e in events if e["event"] == name]
        if hits:
            return hits[-1]
        time.sleep(0.02)
    raise AssertionError(f"no {name!r} event; saw {[e['event'] for e in events]}")


# -- the token ------------------------------------------------------------
class TestNothingIsOpenToTheWorld:
    """The point of --serve is to put this behind a tunnel, which means the
    URL will be reachable by anyone who finds it."""

    def test_the_page_needs_the_token(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{server}/", timeout=5)
        assert e.value.code == 403

    def test_a_wrong_token_is_refused(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{server}/?t=nope", timeout=5)
        assert e.value.code == 403

    def test_polling_without_the_token_is_refused(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{server}/poll?t=nope", timeout=5)
        assert e.value.code == 403

    def test_posting_without_the_token_is_refused(self, server, hub):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(server, "/start", {"mode": "sentences"}, token="nope")
        assert e.value.code == 403
        assert hub.session is None      # and nothing was started

    def test_the_right_token_gets_the_page(self, server):
        reply = urllib.request.urlopen(f"{server}/?t={TOKEN}", timeout=5)
        assert reply.status == 200
        assert b"type the Spanish" in reply.read()


# -- fan-out --------------------------------------------------------------
class TestEveryScreenSeesTheSameDrill:
    """One drill, however many browsers. Opening the page on a second device
    shows the same card rather than starting a competing session."""

    def test_two_browsers_are_told_the_same_thing(self, hub):
        hub.publish("status", text="hello")
        one = hub.since(0, timeout=0.1)
        two = hub.since(0, timeout=0.1)
        assert one == two

    def test_each_browser_keeps_its_own_place(self, hub):
        hub.publish("status", text="one")
        mark, _ = hub.since(0, timeout=0.1)
        hub.publish("status", text="two")
        caught_up = hub.since(mark, timeout=0.1)[1]
        from_scratch = hub.since(0, timeout=0.1)[1]
        assert len(caught_up) == 1 and len(from_scratch) == 2

    def test_the_log_is_ordered(self, hub):
        for word in ("one", "two", "three"):
            hub.publish("status", text=word)
        _, events = hub.since(0, timeout=0.1)
        assert [json.loads(e)["text"] for e in events] == ["one", "two", "three"]


# -- running a drill ------------------------------------------------------
class TestTheSentenceDrillOverTheWire:
    def test_a_cue_arrives_and_an_answer_is_graded(self, server, hub):
        events = []
        collect(server, events)
        wait_for(events, "ready")
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")

        current = hub.session.current
        post(server, "/answer", {"text": current.es})
        result = wait_for(events, "result")
        assert result["perfect"] is True
        assert result["expected"] == current.es

    def test_a_wrong_answer_comes_back_marked(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/answer", {"text": "algo completamente mal"})
        result = wait_for(events, "result")
        assert result["perfect"] is False
        assert result["hold"] is True           # held until Enter
        assert any(t["state"] != "right" for t in result["marked"])

    def test_enter_releases_a_held_card(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        first = wait_for(events, "prompt")
        post(server, "/answer", {"text": "algo completamente mal"})
        wait_for(events, "result")
        assert hub.gate.waiting
        post(server, "/answer", {"text": ""})   # the release
        deadline = time.time() + 3
        while hub.gate.waiting and time.time() < deadline:
            time.sleep(0.02)
        assert not hub.gate.waiting

    def test_stop_ends_it(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/stop")
        wait_for(events, "finished")

    def test_only_one_drill_runs_at_a_time(self, server, hub):
        """Starting again replaces the drill rather than refusing, but the
        one it replaces has to actually stop: two loops sharing a keyboard
        would take turns eating each other's answers."""
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        first = hub.session
        post(server, "/start", {"mode": "sentences"})
        assert hub.session is not first
        assert not first.running and hub.session.running

    def test_progress_is_reported(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        assert wait_for(events, "progress")["total"] > 0


class TestTheBaseUrlIsEnoughAfterTheFirstVisit:
    """The address worth bookmarking is the bare hostname.

    The token still guards the drill; it just stops having to be typed. Hand
    it over once and the browser keeps it, so the plain URL works from then
    on and the address bar is not carrying the key in the clear.
    """

    @staticmethod
    def _get(url, cookie=None):
        req = urllib.request.Request(url)
        if cookie:
            req.add_header("Cookie", cookie)
        return urllib.request.urlopen(req, timeout=10)

    def test_the_long_link_hands_the_token_over(self, server):
        r = self._get(f"{server}/?t={TOKEN}")
        assert r.status == 200
        setc = r.headers.get("Set-Cookie") or ""
        assert f"drill={TOKEN}" in setc
        assert "HttpOnly" in setc and "SameSite=Lax" in setc

    def test_the_bare_url_then_works(self, server):
        r = self._get(f"{server}/", cookie=f"drill={TOKEN}")
        assert r.status == 200
        assert b"DRILL" in r.read()

    def test_polling_works_on_the_cookie_alone(self, server):
        r = self._get(f"{server}/poll?since=0", cookie=f"drill={TOKEN}")
        assert r.status == 200
        assert "next" in json.loads(r.read())

    def test_the_bare_url_is_still_shut_without_it(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            self._get(f"{server}/")
        assert e.value.code == 403

    def test_a_wrong_cookie_is_no_better(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            self._get(f"{server}/", cookie="drill=not-the-token")
        assert e.value.code == 403

    def test_it_is_not_handed_out_again_every_load(self, server):
        r = self._get(f"{server}/", cookie=f"drill={TOKEN}")
        assert not r.headers.get("Set-Cookie")


class TestBothBarsCountToTheSameTotal:
    """74 / 295 on the desk has to be 74 / 295 on the phone.

    Each side holds its own copy of progress.json and only the panel ever
    re-read it. The queue did not, so a drill started on the phone was sized
    from whatever that process booted with: two screens, one deck, and two
    different totals on the bar.
    """

    def test_the_queue_is_sized_from_the_file_not_the_snapshot(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        before = wait_for(events, "progress")["total"]
        post(server, "/stop")

        # The window, drilling the same file while the phone sat idle.
        desk = Progress.load(hub.progress.path)
        due, fresh = desk.queue_parts()
        for index in list(due)[:3] or list(fresh)[:3]:
            desk.cards[index] = Card(interval=MATURE_AT * 4, reps=9, due=99)
        desk.save()

        events.clear()
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        after = wait_for(events, "progress")["total"]
        assert after != before, (
            "the phone sized its queue from a stale copy: the window's work "
            f"never reached it (both runs said {before})")

    def test_starting_a_run_rereads_the_file(self, server, hub):
        seen = []
        real = hub.progress.refresh
        hub.progress.refresh = lambda: (seen.append(1), real())[1]
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        assert seen, "start() built the queue without re-reading progress.json"


class TestTheBarSaysTheSameThingAsTheWindow:
    """The desk app prints "12 / 340" beside its bar. The phone drew the
    same fill and no numbers, so the two did not read alike."""

    def test_progress_carries_the_numbers_the_window_prints(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        m = wait_for(events, "progress")
        assert m["done"] is not None and m["total"] > 0
        assert "skipped" in m           # placement counts them; the rest send 0

    def test_the_page_shows_them(self):
        from spanish_drill.serve import PAGE
        assert 'id="count"' in PAGE
        assert "m.done + ' / ' + m.total" in PAGE
        assert "skipped)" in PAGE


class TestTheWordDrillOverTheWire:
    def test_a_card_is_asked_and_graded(self, server, hub):
        """The real SM-2 drill, typed, with the schedule still on this
        machine."""
        hub.progress = scratch_progress(["perro"], interval=0)
        hub.progress.cards[index_by_id()["perro"]].due = 0
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        card = hub.deck[hub.session.current]
        post(server, "/answer", {"text": card.answers[0]})
        result = wait_for(events, "result")
        assert result["perfect"] is True
        assert result["expected"] == card.answers[0]
        assert result["next_review"]        # the schedule moved

    def test_it_never_writes_the_real_progress_file(self, hub):
        from spanish_drill.config import PROGRESS_PATH
        assert hub.progress.path != PROGRESS_PATH


class TestWhatTheServerKnows:
    def test_state_counts_what_is_actually_unlocked(self, hub):
        state = hub.state()
        assert state["sentences"] == len(S.available(hub.progress, hub.deck))
        assert state["bank"] == len(S.load_sentences())
        assert state["running"] is False

    def test_an_empty_sentence_run_says_why(self, server, hub):
        """A blank screen is indistinguishable from a broken page."""
        hub.progress = scratch_progress()        # nothing learned at all
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        prompt = wait_for(events, "prompt")
        assert prompt["note"]           # nothing is unlocked for a fresh save
        assert "unlocked" in prompt["note"]


class TestEveryAnswerIsBounded:
    """Why this is polled rather than streamed.

    Server-sent events worked on the machine and died behind a Cloudflare
    quick tunnel: a 200, the right headers, and no body however long you
    waited. A proxy cannot forward a response whose end it cannot see. Every
    answer here carries a Content-Length, and those go through anything.
    """

    def test_the_poll_reply_has_a_content_length(self, server):
        reply = urllib.request.urlopen(f"{server}/poll?t={TOKEN}&since=0",
                                       timeout=30)
        assert reply.headers.get("Content-Length")
        assert reply.headers.get("Transfer-Encoding") is None

    def test_the_page_has_a_content_length(self, server):
        reply = urllib.request.urlopen(f"{server}/?t={TOKEN}", timeout=5)
        assert reply.headers.get("Content-Length")

    def test_the_page_does_not_use_a_stream(self):
        from spanish_drill.serve import PAGE
        assert "EventSource" not in PAGE
        assert "/poll?" in PAGE


class TestThePollCatchesUp:
    def test_a_fresh_browser_is_told_the_state(self, server, hub):
        got = []
        collect(server, got)
        assert wait_for(got, "ready")["sentences"] >= 0

    def test_it_returns_at_once_when_something_is_waiting(self, hub):
        hub.publish("status", text="hello")
        mark, events = hub.since(0, timeout=5)
        assert mark == 1 and len(events) == 1

    def test_it_waits_when_nothing_has_happened(self, hub):
        started = time.time()
        mark, events = hub.since(0, timeout=0.3)
        assert events == [] and time.time() - started >= 0.25

    def test_a_mark_only_gets_what_came_after_it(self, hub):
        hub.publish("status", text="one")
        hub.publish("status", text="two")
        mark, events = hub.since(1, timeout=0.1)
        assert len(events) == 1 and "two" in events[0]

    def test_a_browser_joining_late_is_caught_up(self, hub):
        hub.publish("prompt", text="I have a dog.", label="SENTENCE")
        _, events = hub.since(0, timeout=0.1)
        assert any("I have a dog." in e for e in events)

    def test_the_backlog_does_not_grow_without_bound(self, hub):
        from spanish_drill.serve import BACKLOG
        for i in range(BACKLOG + 50):
            hub.publish("status", text=str(i))
        assert len(hub.events) <= BACKLOG


class TestThePhoneHasTheWholePanel:
    """Everything the window shows, minus the parts that need a microphone.

    Read from the same `Progress` methods the window uses, so the two cannot
    end up disagreeing about the same deck.
    """

    def test_it_reports_todays_work(self, hub):
        s = hub.state()
        for key in ("learned", "new_left", "new_per", "reviews", "missed",
                    "learning", "mature", "words_due", "new_in_queue"):
            assert key in s, key

    def test_the_ladder_comes_from_progress(self, hub):
        assert hub.state()["ladder"] == hub.progress.ladder_steps()

    def test_in_queue_matches_what_the_drill_would_ask(self, hub):
        due, fresh = hub.progress.queue_parts()
        assert hub.state()["words_due"] == len(due) + len(fresh)

    def test_it_offers_every_category(self, hub):
        names = [c[0] for c in hub.state()["categories"]]
        assert names[0] == "all" and "verb" in names

    def test_it_reports_the_placement_scope(self, hub):
        assert hub.state()["unsorted"], hub.state()["in_scope"]

    def test_it_says_why_nothing_is_due(self, hub):
        assert hub.state()["why_empty"].endswith(".")

    def test_the_second_opinion_tally_is_there(self, hub):
        s = hub.state()
        assert s["kept"] == hub.progress.kept
        assert s["overturned"] == hub.progress.overturned


class TestTheDialsWorkFromThePhone:
    def test_the_category_can_be_changed(self, hub):
        hub.configure(category="noun")
        assert hub.progress.category == "noun"
        assert hub.state()["category"] == "noun"

    def test_the_new_word_allowance_can_be_changed(self, hub):
        hub.configure(new_per=7)
        assert hub.progress.new_per == 7

    def test_the_allowance_is_bounded(self, hub):
        hub.configure(new_per=9999)
        assert hub.progress.new_per == 100
        hub.configure(new_per=-5)
        assert hub.progress.new_per == 0

    def test_the_answer_window_is_bounded(self, hub):
        hub.configure(window=999)
        assert hub.progress.window == 20
        hub.configure(window=1)
        assert hub.progress.window == 3

    def test_a_nonsense_dialect_is_ignored(self, hub):
        before = hub.progress.dialect
        hub.configure(dialect="klingon")
        assert hub.progress.dialect == before

    def test_settings_are_saved_not_just_set(self, hub):
        hub.configure(category="noun", new_per=11)
        again = Progress.load(hub.progress.path)
        assert again.category == "noun" and again.new_per == 11

    def test_changing_a_dial_republishes_the_panel(self, hub):
        hub.configure(new_per=13)
        _, events = hub.since(0, timeout=0.1)
        assert any('"new_per": 13' in e for e in events)


class TestPlacementFromThePhone:
    def test_it_runs_a_placement_session(self, server, hub):
        from spanish_drill.placement import PlacementSession
        events = []
        collect(server, events)
        wait_for(events, "ready")
        post(server, "/start", {"mode": "placement"})
        wait_for(events, "prompt")
        assert isinstance(hub.session, PlacementSession)

    def test_the_scope_reaches_the_session(self, server, hub):
        hub.configure(scope="all")
        post(server, "/start", {"mode": "placement"})
        deadline = time.time() + 4
        while hub.session is None and time.time() < deadline:
            time.sleep(0.02)
        assert hub.session.retest is True

    def test_it_is_typed_and_never_calls_the_api(self, server, hub):
        post(server, "/start", {"mode": "placement"})
        deadline = time.time() + 4
        while hub.session is None and time.time() < deadline:
            time.sleep(0.02)
        assert hub.session.typed is True
        assert hub.session.verifier is None


class TestTheWordSlipCarriesWhatTheWindowShows:
    def test_a_word_result_has_the_schedule_and_the_sentence(self, server, hub):
        hub.progress = scratch_progress(["perro"], interval=0)
        hub.progress.cards[index_by_id()["perro"]].due = 0
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        card = hub.deck[hub.session.current]
        post(server, "/answer", {"text": card.answers[0]})
        r = wait_for(events, "result")
        assert r["next_review"] and r["example"] and r["gloss"]
        assert "ease" in r["detail"]


class TestTheScopeCountIsHonest:
    """The panel promised nothing was left to sort while the button started a
    run of eleven hundred cards. A count that does not match what the button
    does is worse than no count."""

    def test_it_matches_what_placement_would_offer(self, hub):
        from spanish_drill.placement import PlacementSession
        unsorted, in_scope = hub.progress.placement_scope(hub.deck)
        session = PlacementSession(hub.progress, None, typed=True,
                                   verifier=None, deck=hub.deck)
        offered = len(session.next_queue())
        assert offered == unsorted, (offered, unsorted)
        assert session.skipped == in_scope - unsorted

    def test_retest_offers_everything_in_scope(self, hub):
        from spanish_drill.placement import PlacementSession
        _, in_scope = hub.progress.placement_scope(hub.deck)
        session = PlacementSession(hub.progress, None, typed=True,
                                   verifier=None, deck=hub.deck, retest=True)
        assert len(session.next_queue()) == in_scope


class TestBothSidesShareOneDeck:
    """The app and the phone each hold their own copy of progress.json.

    A plain write from one used to wipe whatever the other had done:
    measured, the desktop erased a card, five reviews and a finished
    sentence. Whichever screen you pick up, the day's work has to be there.
    """

    @staticmethod
    def two_copies():
        path = Path(tempfile.mkdtemp(prefix="drill-shared-")) / "progress.json"
        Progress(path=path).save()
        return Progress.load(path), Progress.load(path)

    def test_neither_side_wipes_the_other(self):
        desktop, phone = self.two_copies()
        phone.cards[10] = Card(interval=6, reps=2, due=0)
        phone.save()
        desktop.cards[20] = Card(interval=1, reps=1, due=0)
        desktop.save()
        assert sorted(Progress.load(desktop.path).cards) == [10, 20]

    def test_the_days_work_adds_up(self):
        """Five reviews on the phone and one at the desk is six, not one."""
        desktop, phone = self.two_copies()
        phone.reviews_done = 5
        phone.save()
        desktop.reviews_done += 1
        desktop.save()
        assert Progress.load(desktop.path).reviews_done == 6

    def test_finished_sentences_are_pooled(self):
        desktop, phone = self.two_copies()
        phone.sentences_done.add("one")
        phone.save()
        desktop.sentences_done.add("two")
        desktop.save()
        assert Progress.load(desktop.path).sentences_done == {"one", "two"}

    def test_the_side_that_answered_a_card_wins_it(self):
        """Not last-write-wins: whoever actually touched the card."""
        desktop, phone = self.two_copies()
        desktop.cards[5] = Card(interval=1, reps=1, due=0)
        desktop.save()
        phone.refresh()
        phone.cards[5] = Card(interval=6, reps=2, due=0)   # answered again
        phone.save()
        assert Progress.load(desktop.path).cards[5].interval == 6

    def test_a_card_removed_here_stays_removed(self):
        desktop, phone = self.two_copies()
        desktop.cards[7] = Card(interval=1, reps=1, due=0)
        desktop.save()
        phone.refresh()
        del phone.cards[7]
        phone.save()
        assert 7 not in Progress.load(desktop.path).cards

    def test_refresh_keeps_unsaved_work(self):
        desktop, phone = self.two_copies()
        phone.cards[1] = Card(interval=6, reps=2, due=0)   # not saved yet
        desktop.cards[2] = Card(interval=1, reps=1, due=0)
        desktop.save()
        phone.refresh()
        assert sorted(phone.cards) == [1, 2]

    def test_refresh_says_when_nothing_moved(self):
        desktop, _ = self.two_copies()
        assert desktop.refresh() is False

    def test_a_tally_from_another_day_is_not_added_in(self):
        """Yesterday's reviews are not part of today's bar."""
        desktop, phone = self.two_copies()
        phone.reviews_done = 5
        phone.day = desktop.day - 1          # the phone is a day behind
        phone.save()
        desktop.reviews_done = 2
        desktop.save()
        assert Progress.load(desktop.path).reviews_done == 2

    def test_the_lifetime_tally_still_adds_across_days(self):
        desktop, phone = self.two_copies()
        phone.kept = 4
        phone.day = desktop.day - 1
        phone.save()
        desktop.kept += 3
        desktop.save()
        assert Progress.load(desktop.path).kept == 7

    def test_the_phone_panel_picks_up_the_desk(self, hub):
        """Answer at the desk, and the phone's panel shows it without a
        restart. The day has to be today's, or loading rolls the tally over
        and resets it, which is right and would prove nothing here."""
        from spanish_drill.scheduler import today
        hub.progress.day = today()
        hub.progress.save()
        other = Progress.load(hub.progress.path)
        other.reviews_done += 9
        other.save()
        assert hub.state()["reviews"] >= 9


class TestPauseFromThePhone:
    def test_it_pauses_and_resumes(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/pause", {})
        assert wait_for(events, "paused")["paused"] is True
        assert hub.paused is True
        post(server, "/pause", {})
        assert hub.state()["paused"] is False

    def test_the_card_stays_put(self, server, hub):
        """Not a stop: the session, its queue and its place survive."""
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        first = wait_for(events, "prompt")["text"]
        session = hub.session
        post(server, "/pause", {})
        time.sleep(0.3)
        post(server, "/pause", {})
        assert hub.session is session
        assert hub.session.current.en == first

    def test_a_paused_drill_can_still_be_stopped(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/pause", {})
        post(server, "/stop")
        wait_for(events, "finished")

    def test_pausing_nothing_is_harmless(self, hub):
        assert hub.pause() is False

    def test_the_panel_reports_it(self, server, hub):
        post(server, "/start", {"mode": "sentences"})
        deadline = time.time() + 4
        while hub.typed is None and time.time() < deadline:
            time.sleep(0.02)
        hub.pause(True)
        assert hub.state()["paused"] is True

    def test_the_button_is_in_the_page(self):
        from spanish_drill.serve import PAGE
        assert 'id="pause"' in PAGE and "/pause" in PAGE


class TestSwitchingModes:
    """Pressing a mode is an instruction to be in that mode.

    Refusing while something else ran meant the only way out of the sentence
    drill was to find STOP first, and a session that stuck for any reason
    left every button on the page dead.
    """

    def test_a_mode_can_be_started_over_a_running_one(self, server, hub):
        from spanish_drill.composition import SentenceDrill
        from spanish_drill.session import DrillSession
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        assert isinstance(hub.session, SentenceDrill)

        hub.progress = scratch_progress(["perro"], interval=0)
        hub.progress.cards[index_by_id()["perro"]].due = 0
        post(server, "/start", {"mode": "words"})
        deadline = time.time() + 5
        while not isinstance(hub.session, DrillSession) and time.time() < deadline:
            time.sleep(0.02)
        assert isinstance(hub.session, DrillSession)

    def test_the_old_session_really_stops(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        first = hub.session
        post(server, "/start", {"mode": "sentences"})
        assert hub.session is not first
        assert not first.running

    def test_switching_out_of_a_paused_drill_works(self, server, hub):
        """The worst version of the trap: paused, so running never clears."""
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/pause", {})
        assert hub.paused
        post(server, "/start", {"mode": "sentences"})
        assert hub.paused is False      # the new session is not paused
        assert hub.session.running

    def test_the_page_does_not_grey_out_the_modes(self):
        from spanish_drill.serve import PAGE
        assert "['go','sentences','placement'].forEach(b => $(b).disabled" not in PAGE


class TestTheAddressDoesNotChange:
    """A link that changes every launch is not an address.

    The whole point of running this on a phone is opening the same URL
    tomorrow, so the key has to outlive the process.
    """

    def test_the_token_is_kept(self, tmp_path):
        from spanish_drill.serve import stored_token
        path = tmp_path / ".serve-token"
        first = stored_token(path)
        assert first and stored_token(path) == first

    def test_it_is_read_back_from_the_file(self, tmp_path):
        from spanish_drill.serve import stored_token
        path = tmp_path / ".serve-token"
        path.write_text("a-known-key\n", encoding="utf-8")
        assert stored_token(path) == "a-known-key"

    def test_it_is_not_world_readable(self, tmp_path):
        from spanish_drill.serve import stored_token
        path = tmp_path / ".serve-token"
        stored_token(path)
        assert path.stat().st_mode & 0o077 == 0, "anyone could read the key"

    def test_an_unwritable_home_still_serves(self, tmp_path):
        """It should degrade to a one-off key, not refuse to start."""
        from spanish_drill.serve import stored_token
        assert stored_token(tmp_path / "no" / "such" / "dir" / "t")

    def test_it_is_not_committed(self):
        from spanish_drill.config import ROOT
        assert ".serve-token" in (ROOT / ".gitignore").read_text(encoding="utf-8")


class TestEnterMovesToTheNextCard:
    """One Enter answers; only a miss costs a second one.

    The phone drill briefly held every card, right ones included, so the
    verdict could be read before the next cue wiped it. In practice that put
    a keypress between every card of a run you are getting right, which is
    most of them, and the drill on this machine never did it. A correct
    answer goes straight on; a miss is still held, by the drill itself.
    """

    def test_a_correct_answer_moves_on_at_once(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        before = sum(1 for e in events if e["event"] == "prompt")
        post(server, "/answer", {"text": hub.session.current.es})
        result = wait_for(events, "result")
        assert result["perfect"] is True and result["hold"] is False
        assert self._moved_on(events, before), \
            "a right answer sat there waiting for Enter"

    def test_a_right_answer_does_not_swallow_the_next_one(self, server, hub):
        """The Enter after a correct card is an answer, not a release.

        Nothing is being held, so if that keypress were still treated as a
        release the next card would eat it and sit unanswered.
        """
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "words"})
        wait_for(events, "prompt")
        card = hub.deck[hub.session.current]
        post(server, "/answer", {"text": card.answers[0]})
        assert wait_for(events, "result")["perfect"] is True
        assert hub._hold_pending is False
        assert not hub.gate.waiting

    @staticmethod
    def _moved_on(events, prompts_before, timeout=4):
        """Either the next card, or the run ending. Counting prompts rather
        than comparing text: with one sentence unlocked the next card is the
        same sentence, and a right answer ends the run instead."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            prompts = sum(1 for e in events if e["event"] == "prompt")
            if prompts > prompts_before or any(e["event"] == "finished"
                                               for e in events):
                return True
            time.sleep(0.02)
        return False

    def test_a_miss_still_takes_only_one_enter(self, server, hub):
        """The drill holds a miss itself; this must not stack a second hold
        on top and ask for two."""
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/answer", {"text": "algo completamente mal"})
        assert wait_for(events, "result")["hold"] is True
        before = sum(1 for e in events if e["event"] == "prompt")
        post(server, "/answer", {"text": ""})          # one Enter only
        assert self._moved_on(events, before), "a miss needed more than one Enter"

    def test_stopping_releases_a_held_miss(self, server, hub):
        events = []
        collect(server, events)
        post(server, "/start", {"mode": "sentences"})
        wait_for(events, "prompt")
        post(server, "/answer", {"text": "algo completamente mal"})
        wait_for(events, "result")
        post(server, "/stop")
        wait_for(events, "finished")

    def test_the_page_asks_on_every_hold(self):
        from spanish_drill.serve import PAGE
        assert "if (m.hold) d +=" in PAGE


class TestThePanelAgreesWithTheDrill:
    """`running` drives PAUSE and STOP, so a stale answer disables the only
    way out of a session."""

    def test_starting_says_so_at_once(self, hub):
        """Nothing published a panel after the thread started, so the browser
        kept the last one it had, taken before the drill began. PAUSE and
        STOP read `running` off that and stayed disabled for the whole
        session, leaving no way to stop one."""
        assert hub.state()["running"] is False
        hub.start("sentences")
        try:
            assert hub.state()["running"] is True
            said = [json.loads(e)["event"] for e in hub.since(0)[1]]
            assert "ready" in said
        finally:
            hub.stop()

    def test_finishing_says_so_before_the_panel_that_follows(self, hub):
        """`on_finished` is raised from inside the session thread, so asking
        whether that thread is alive answers yes at exactly the moment the
        drill has ended. The panel published a beat later then contradicted
        the finish it had just sent, and the controls came back."""
        hub.start("sentences")
        for _ in range(100):
            if hub.state()["running"]:
                break
            time.sleep(0.02)
        hub.live = True                 # as a running drill leaves it
        hub._finished()                 # raised on the session's own thread
        assert hub.state()["running"] is False

    def test_a_stop_settles(self, hub):
        hub.start("sentences")
        for _ in range(100):
            if hub.state()["running"]:
                break
            time.sleep(0.02)
        hub.stop()
        for _ in range(200):
            if not hub.state()["running"]:
                break
            time.sleep(0.02)
        assert hub.state()["running"] is False
