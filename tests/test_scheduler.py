"""SM-2 scheduling, ease adaptation, and repairing a miss that never happened."""
import pytest

from spanish_drill import scheduler as S
from spanish_drill.scheduler import (EASE_MIN, EASE_START, MATURE_AT, Card,
                                     describe_interval, ease_delta, is_leech,
                                     migrate, schedule)


def answered(times, q, card=None):
    c = card or Card.new(today=0)
    for _ in range(times):
        schedule(c, q, today=0)
    return c


class TestLadder:
    def test_first_two_intervals_are_fixed(self):
        c = Card.new(today=0)
        schedule(c, 4, today=0)
        assert c.interval == 1
        schedule(c, 4, today=0)
        assert c.interval == 6

    def test_intervals_grow_after_that(self):
        c = Card.new(today=0)
        seen = []
        for _ in range(5):
            schedule(c, 4, today=0)
            seen.append(c.interval)
        assert seen == sorted(seen) and len(set(seen)) == len(seen)

    def test_due_date_follows_the_interval(self):
        c = Card.new(today=0)
        schedule(c, 5, today=100)
        assert c.due == 100 + c.interval


class TestEase:
    def test_perfect_recall_raises_ease(self):
        assert answered(6, 5).ease > EASE_START

    def test_struggling_lowers_ease(self):
        assert answered(6, 3).ease < EASE_START

    def test_ease_has_a_floor(self):
        assert answered(40, 0).ease == pytest.approx(EASE_MIN)

    def test_an_easy_card_outpaces_a_hard_one(self):
        assert answered(6, 5).interval > answered(6, 3).interval * 3

    def test_delta_is_symmetric_with_schedule(self):
        """repair relies on being able to undo exactly what schedule applied."""
        c = Card.new(today=0)
        before = c.ease
        schedule(c, 1, today=0)
        assert c.ease == pytest.approx(max(EASE_MIN, before + ease_delta(1)))


class TestFailure:
    def test_a_miss_resets_to_relearning(self):
        c = answered(4, 5)
        assert c.interval > 6
        schedule(c, 1, today=0)
        assert c.interval == 0 and c.reps == 0

    def test_a_miss_counts_a_lapse_and_lowers_ease(self):
        c = answered(4, 5)
        before = c.ease
        schedule(c, 1, today=0)
        assert c.lapses == 1 and c.ease < before

    def test_a_miss_is_due_immediately(self):
        c = answered(2, 5)
        schedule(c, 0, today=57)
        assert c.due == 57


class TestLeech:
    def test_repeated_lapses_flag_a_leech(self):
        assert is_leech(answered(8, 0))

    def test_a_healthy_card_is_not_a_leech(self):
        assert not is_leech(Card.new(today=0))


class TestMaturity:
    def test_mature_needs_a_long_interval(self):
        c = Card.new(today=0)
        while c.interval < MATURE_AT:
            schedule(c, 5, today=0)
        assert c.interval >= MATURE_AT

    def test_a_fresh_correct_card_is_not_mature(self):
        assert answered(1, 5).interval < MATURE_AT


class TestMigration:
    """Saves written by the original Leitner scheduler must survive."""

    def test_box_becomes_an_interval(self):
        c = migrate({"b": 4, "d": 116, "r": 9, "l": 3}, today=100)
        assert c.interval == 16

    def test_history_is_kept(self):
        c = migrate({"b": 4, "d": 116, "r": 9, "l": 3}, today=100)
        assert c.lapses == 3 and c.ease == EASE_START

    def test_migrating_twice_is_harmless(self):
        old = {"b": 2, "d": 103, "r": 4, "l": 1}
        once = migrate(old, today=100)
        assert migrate(once.to_dict(), today=100).to_dict() == once.to_dict()

    def test_current_format_round_trips(self):
        c = answered(3, 5)
        assert Card.from_dict(c.to_dict()).to_dict() == c.to_dict()


class TestDescribeInterval:
    @pytest.mark.parametrize("ivl,text", [
        (0, "again this session"),
        (1, "again tomorrow"),
        (6, "again in 6 days"),
        (30, "again in 1 month"),
        (90, "again in 3 months"),
    ])
    def test_wording(self, ivl, text):
        assert describe_interval(ivl) == text
