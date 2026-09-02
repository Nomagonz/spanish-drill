"""What the published page must keep being.

It was its own program: 253 cards written into the HTML, Leitner boxes keyed
by deck position, and a schedule in localStorage that nothing else could read.
Every number it showed disagreed with the desktop's, and none of that was
visible from either side. These are the properties that stop it drifting back.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "index.html").read_text(encoding="utf-8")
PORT = (ROOT / "web" / "drill.js").read_text(encoding="utf-8")


class TestItUsesTheOneDeck:
    def test_the_deck_is_fetched_not_written_in(self):
        assert 'fetch("deck.json")' in PAGE

    def test_no_deck_is_embedded(self):
        """The old page carried 253 of its own cards as object literals."""
        assert len(re.findall(r'\{en:"', PAGE)) == 0
        assert PAGE.count('{en:') == 0

    def test_the_page_is_no_longer_carrying_a_deck_sized_payload(self):
        assert len(PAGE) < 40_000, (
            "the page grew by the size of a deck; it should be fetching one")


class TestItUsesTheOneScheduler:
    def test_the_leitner_ladder_is_gone(self):
        """`IVL` was the old eight-rung ladder every card walked."""
        assert "IVL" not in PAGE
        assert "[0,1,3,7,16,35,90,180]" not in PAGE.replace(" ", "")

    def test_the_ported_rules_are_loaded(self):
        assert 'src="web/drill.js"' in PAGE

    def test_it_grades_through_the_port(self):
        assert "D.check(" in PAGE
        assert "D.schedule(" in PAGE
        assert "D.typedQuality(" in PAGE

    def test_the_port_still_migrates_a_leitner_save(self):
        """Years of the old format sit in people's browsers."""
        assert "LEITNER_LADDER" in PORT
        assert "esdrill:v2" in PAGE


class TestItUsesTheOneDatabase:
    def test_it_talks_to_the_worker(self):
        assert "spanish-drill-sync" in PAGE
        assert "/state" in PAGE

    def test_a_write_carries_the_version_it_was_built_on(self):
        """Without it the page would overwrite whatever the desktop just did."""
        assert "base: version" in PAGE

    def test_a_refused_write_is_retried_rather_than_dropped(self):
        assert "409" in PAGE
        assert "touched" in PAGE

    def test_the_key_is_never_written_into_the_page(self):
        """It is typed in and kept on the device, not published with the
        page, which anyone can read."""
        for line in PAGE.splitlines():
            if "Bearer" in line:
                assert '" + key' in line or "+ key" in line, line

    def test_state_is_keyed_by_card_id(self):
        """The old page keyed by deck position, which hands one word's history
        to whatever word lands in its slot when the deck is edited."""
        assert "self.deck[i].id" in PORT or "deck[i].id" in PORT
