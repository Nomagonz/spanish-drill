"""Re-judge past answers against a stronger recogniser.

Most misses are now caught during the session. This exists for anything
recorded while the live check was off, or before it existed.
"""
from .answers import AnswerLog, resolve_index
from .config import VERIFY_MODEL
from .deck import load_deck
from .grading import check, quality
from .progress import Progress
from .scheduler import (Card, PASSING_QUALITY, migrate, schedule,
                        unschedule_penalty)
from .transcribe import assert_steer_is_clean, second_opinion


def repair(progress, record, new_quality, index=None):
    """Undo a miss the recogniser invented, without trampling what came after.

    Only the interval is restored, and only when the card is still sitting in
    the state that bad miss left it in. If it has been answered correctly since,
    that repetition is real and its scheduling is already right.

    The index is resolved by the caller, never read off the record: the stored
    position belongs to whatever deck was loaded the day it was written.
    """
    if index is None:
        index = resolve_index(record)
    if index is None:
        # Silently filing this under None would create a phantom card and
        # leave the real one unrepaired. Refusing is the safe failure.
        raise ValueError("cannot tell which card this answer was about")
    card = progress.card_or_new(index)
    unschedule_penalty(card, record["quality"], new_quality)
    restored = card.reps == 0 and card.interval == 0
    if restored:
        before = migrate(record["before"])
        before.ease, before.lapses = card.ease, card.lapses
        schedule(before, new_quality)
        card = before
    progress.cards[index] = card
    progress.missed_today = max(0, progress.missed_today - 1)
    return card, restored


def review(model=VERIFY_MODEL, log=None, progress=None, verifier=None):
    """Re-check every answer that has not been judged yet."""
    assert_steer_is_clean()
    log = log or AnswerLog()
    progress = progress or Progress.load()
    verifier = verifier or (lambda path: second_opinion(str(path), model))
    deck = load_deck()

    records = log.all()
    pending = [r for r in records if r.get("verdict") is None]
    if not pending:
        print(f"{len(records)} answer(s) on file, none awaiting review.")
        return {"overturned": [], "confirmed": [], "unusable": [], "accepted": []}

    print(f"{len(pending)} answer(s) to re-check with '{model}'.\n")
    buckets = {"overturned": [], "confirmed": [], "unusable": [], "accepted": []}

    for record in pending:
        index = resolve_index(record, deck)
        if index is None:
            # The card it was about is not in this deck any more, or the
            # record is too old to say which one it was. Repairing a guess
            # would move a schedule that belongs to a different word.
            record["verdict"] = "unknown-card"
            buckets["unusable"].append(record)
            continue
        card = deck[index]
        path = log.audio_path(record)
        if path is None:
            record["verdict"] = "no-audio"
            buckets["unusable"].append(record)
            continue

        text, echoed = verifier(path)
        record["second_opinion"] = text
        if text is None:
            # An echoed prompt is not evidence either way. Leaving it pending
            # would re-bill it forever, so mark it and move on.
            record["verdict"] = "no-signal" if echoed else "unchecked"
            buckets["unusable"].append(record)
            continue

        was_miss = record["quality"] < PASSING_QUALITY
        match = check(text, card)
        if match and was_miss:
            new_q = quality(True, match.close, False,
                            record["elapsed"], progress.window)
            _, restored = repair(progress, record, new_q, index)
            record.update(verdict="false-miss", corrected_quality=new_q,
                          interval_restored=restored)
            buckets["overturned"].append(record)
        elif was_miss:
            record["verdict"] = "confirmed"
            buckets["confirmed"].append(record)
        else:
            record["verdict"] = "accepted"
            buckets["accepted"].append(record)

    progress.save()
    log.rewrite(records)
    _report(buckets, len(pending))
    return buckets


def _prompt(record):
    """Log lines written by the previous version used different keys."""
    return record.get("prompt") or record.get("en") or "?"


def _report(buckets, total):
    line = "-" * 72
    print(line)
    if buckets["accepted"]:
        print(f"MARKED CORRECT live ({len(buckets['accepted'])})")
    if buckets["overturned"]:
        print(f"\nOVERTURNED — you were right ({len(buckets['overturned'])}):")
        for r in buckets["overturned"]:
            note = "" if r.get("interval_restored") else "  [ease only]"
            print(f"  {_prompt(r)[:30]:<32} expected {r['expected'][0]!r}")
            print(f"     live {r['heard']!r:<22} -> recheck "
                  f"{r['second_opinion']!r}{note}")
    if buckets["confirmed"]:
        print(f"\nCONFIRMED misses ({len(buckets['confirmed'])}):")
        for r in buckets["confirmed"]:
            print(f"  {_prompt(r)[:30]:<32} expected {r['expected'][0]!r:<16} "
                  f"heard {r['second_opinion']!r}")
    if buckets["unusable"]:
        print(f"\nNo usable speech, left alone ({len(buckets['unusable'])}).")
    print(line)
    wrong = len(buckets["overturned"])
    print(f"Local model got {total - wrong} of {total} right.")
    if wrong:
        print(f"{wrong} were the recogniser's fault; those cards were repaired.")
