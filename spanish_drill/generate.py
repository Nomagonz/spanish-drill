"""Sentences written by the API, judged locally before any of them count.

The model proposes and this module disposes. Every sentence that comes back
is put through exactly the same resolver the curated bank is held to, and
anything leaning on a word that is not in the known set is dropped on the
floor. That check is not a formality: a language model asked to stay inside a
word list will drift out of it, reliably, and the drift is invisible in the
output because the sentence still reads perfectly well. Prompting is how the
hit rate is kept high. The gate is why a miss cannot cost anything.

Nothing here is required for the mode to work. The curated bank in
sentences.json needs no key, no network and no spend, and this only ever adds
to it.
"""
import json
import os
import random

from .config import (GENERATED_SENTENCES_PATH, MAX_GENERATED_SENTENCES,
                     SENTENCE_MODEL, WORDS_PER_CALL)
from .deck import load_deck
from .sentences import (FUNCTION_WORDS, Sentence, known_ids, requirements,
                        resolve, tokens)
from .text import normalize

MAX_WORDS = 12          # the same ceiling the curated bank is tested against

# Characters that only Spanish writes. A cue carrying one is not English.
SPANISH_ONLY = set("ñáéíóúü¿¡")

# Ordinary Spanish that this deck happens not to teach. Used ONLY to tell
# which language a cue is written in — never to let a word past the gate. A
# sentence containing one of these is still thrown out for using a word with
# no card; this list just stops it being mistaken for English on the way.
SPANISH_EXTRAS = frozenset("""
eso esto esa ese esta este aquello aquel aquella muy todavia contigo conmigo
ti sino cual cuyo cada otro otra otros otras algo alguien nada nadie
tan tanto mucho mucha muchos muchas poco poca todo toda todos todas
siempre nunca aqui alli ahi ahora luego entonces tambien tampoco
pues asi bien mal mismo propio hay hacia sobre entre hasta desde
""".split())

def is_english(text):
    """Is this actually an English cue?

    The model returns Spanish in the English field often enough to matter,
    and the failure is silent and total: the cue becomes the answer, and the
    card asks you to copy out what is already on the screen.

    The test is which language can account for the whole sentence. Spanish
    built from this deck is, by construction, entirely deck words and glue —
    that is what the gate guarantees. So a cue with even one word Spanish
    cannot explain is English, and a cue Spanish explains completely is not.

    Two earlier attempts were worse and both shipped a bad card. Looking for
    accents missed "Voy a tomar un segundo para pensar", which has none.
    Looking for English marker words then failed the other way, because the
    obvious markers are Spanish too: `a`, `no`, `he`, `has` and `me` are all
    ordinary Spanish, and dropping them left "He has a dog." with no marker
    at all.
    """
    if set((text or "").lower()) & SPANISH_ONLY:
        return False
    words = normalize(text).split()
    if not words:
        return False
    return any(word not in FUNCTION_WORDS and word not in SPANISH_EXTRAS
               and not resolve(word) for word in words)


SYSTEM = (
    "You write single, short Spanish sentences for a beginner's drill, with "
    "an English translation. You are given the only Spanish words the learner "
    "knows. Obey these rules exactly.\n"
    "1. Use ONLY words from the supplied list, plus articles, prepositions, "
    "pronouns and conjunctions (el, la, un, una, de, en, con, que, no, y, a, "
    "por, para, mi, tu, su, me, te, se, lo).\n"
    "2. Conjugate the listed verbs in the PRESENT tense only.\n"
    "3. Keep every sentence under twelve words, and plain enough that there "
    "is only one natural way to translate the English back into Spanish. No "
    "idioms, no wordplay, no names, no numbers.\n"
    "4. The English must be an unambiguous cue for exactly that Spanish. If "
    "two Spanish sentences would both be fair translations, do not use it. "
    "In particular never write a dummy 'it': 'it is easy to learn' can be "
    "'es facil aprender' or 'es facil de aprender' depending on what 'it' "
    "points at, so both are right and the learner is marked wrong for "
    "picking the other. Name the subject, or drop the sentence.\n"
    "5. Translate every word with the meaning given for it in the list, and "
    "no other. The list is what the learner was taught. If it says "
    "'deber = to owe; should, ought to' then deber is 'should', never "
    "'must'; 'must' is a different word they have not been taught. A cue "
    "that glosses a word differently from the list teaches the wrong "
    "meaning, which is worse than no sentence at all.\n"
    "6. Do not write the subject pronoun (yo, tu, el, ella, nosotros, ellos) "
    "unless the sentence genuinely needs it. Spanish normally drops it, and "
    "'Tiene un perro' is the natural form of 'He has a dog'.\n"
    "7. Never repeat a sentence you have been shown as already written.\n"
    "8. For each item write the Spanish sentence first, then translate it. "
    "The 'spanish' field is Spanish and the 'english' field is English. An "
    "'english' field containing Spanish is the one mistake that makes the "
    "whole item useless."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            # Named in full and ordered Spanish first, both on purpose.
            # Called "en" and "es" the model filled both with Spanish, three
            # times out of four: the abbreviations are too weak a signal to
            # switch language on. Writing the Spanish first then makes the
            # English an actual translation of something, rather than two
            # independent attempts at the same sentence.
            "items": {
                "type": "object",
                "properties": {
                    "spanish": {
                        "type": "string",
                        "description": "The Spanish sentence the learner has "
                                       "to produce."},
                    "english": {
                        "type": "string",
                        "description": "The English translation of the "
                                       "spanish field. Written in English, "
                                       "never in Spanish. This is the cue "
                                       "shown on screen."}},
                "required": ["spanish", "english"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


# -- the store ------------------------------------------------------------
def load_generated(path=None):
    """What has been paid for already. Missing or corrupt reads as empty."""
    path = path or GENERATED_SENTENCES_PATH
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    return [r for r in raw if isinstance(r, dict) and r.get("en") and r.get("es")]


def save_generated(rows, path=None):
    """Written atomically, like progress: a crash must not shred the store."""
    path = path or GENERATED_SENTENCES_PATH
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def as_sentences(rows):
    out = []
    for row in rows:
        needs, _ = requirements(row["es"])
        out.append(Sentence(en=row["en"], es=row["es"], needs=needs))
    return out


def remaining_allowance(stored=None):
    """How many more may ever be bought."""
    stored = load_generated() if stored is None else stored
    return max(0, MAX_GENERATED_SENTENCES - len(stored))


# -- what to ask for ------------------------------------------------------
def word_menu(progress, deck=None, rng=None, limit=WORDS_PER_CALL):
    """A sample of the known words, as `spanish = english` lines.

    Verbs go in as infinitives with a note that they may be conjugated: the
    gate reads a conjugated form back to its infinitive, so a verb the
    learner knows is usable in any person without widening what is allowed.
    """
    deck = deck or load_deck()
    rng = rng or random
    known = known_ids(progress, deck)
    rows = []
    for card in deck:
        if card.lemma or card.id not in known:
            continue            # conjugation cards are covered by their verb
        if card.pos not in ("noun", "verb", "adjective", "adverb"):
            continue            # function words are always allowed anyway
        rows.append((card.answers[0], card.prompt, card.pos))
    rng.shuffle(rows)
    return rows[:limit]


def build_messages(menu, count, already=()):
    listing = "\n".join(f"{es} = {en} ({pos})" for es, en, pos in menu)
    seen = "\n".join(already)
    ask = [f"Words available:\n{listing}\n",
           f"Write {count} different sentences."]
    if seen:
        ask.append(f"\nAlready written, do not repeat these:\n{seen}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(ask)}]


def call_api(messages, model=SENTENCE_MODEL, client=None):
    """One request. Returns rows, or [] on any failure.

    A generator that raises takes the mode down with it, and the mode has a
    perfectly good curated bank to fall back on, so every failure here is
    just an empty batch.
    """
    try:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        reply = client.chat.completions.create(
            model=model, messages=messages, temperature=1.0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "sentences",
                                             "schema": SCHEMA,
                                             "strict": True}})
        payload = json.loads(reply.choices[0].message.content or "{}")
        rows = payload.get("sentences") or []
        # Back to the short keys the rest of the app stores and reads.
        return [{"en": r.get("english", ""), "es": r.get("spanish", "")}
                for r in rows if isinstance(r, dict)]
    except Exception as exc:
        print(f"    [sentence api: {type(exc).__name__}: {str(exc)[:70]}]")
        return []


# Matched with their accents on, before anything is normalised. That accent
# is the only thing separating the pronoun `él` from the article `el` and
# `tú` from the possessive `tu`, and stripping the article off "El perro
# corre" would turn a good sentence into a broken one.
LEADING_SUBJECTS = frozenset(
    "yo tú él ella usted ustedes nosotros nosotras vosotros vosotras "
    "ellos ellas".split())


def _drop_subject(spanish):
    """Take a leading subject pronoun off, so the stored form is the natural one.

    The model writes them even when told not to. Keeping "Él tiene un perro"
    as the answer would make the ordinary "Tiene un perro" read as a missing
    word, which is the grader inventing a mistake.
    """
    text = (spanish or "").strip()
    words = text.split()
    if len(words) > 2 and words[0].strip("¿¡").lower() in LEADING_SUBJECTS:
        rest = " ".join(words[1:])
        lead = "".join(c for c in words[0] if c in "¿¡")
        return lead + rest[0].upper() + rest[1:] if rest else text
    return text


# -- the judge ------------------------------------------------------------
def usable(rows, known, seen_es=()):
    """Keep only what the gate would have allowed anyway.

    Four ways to be thrown out and they are all cheap to check: a word no
    card accounts for, a word that is not in the known set, too long, or one
    already on file. Returns (kept, why_dropped) so the caller can say how
    the model is doing rather than silently binning most of a batch.
    """
    kept, dropped = [], []
    seen = set(seen_es)
    for row in rows:
        en, es = (row.get("en") or "").strip(), _drop_subject(row.get("es"))
        if not en or not es:
            dropped.append((es, "empty"))
            continue
        if not is_english(en) or normalize(en) == normalize(es):
            dropped.append((es, f"the cue is not English: {en!r}"))
            continue
        if es in seen:
            dropped.append((es, "duplicate"))
            continue
        if len(tokens(es)) > MAX_WORDS:
            dropped.append((es, "too long"))
            continue
        needs, unknown = requirements(es)
        if unknown:
            dropped.append((es, f"not in the deck: {', '.join(unknown)}"))
            continue
        if not needs:
            dropped.append((es, "no content words"))
            continue
        outside = [n for n in needs if n not in known]
        if outside:
            dropped.append((es, f"not learned yet: {', '.join(outside)}"))
            continue
        seen.add(es)
        kept.append({"en": en, "es": es})
    return kept, dropped


def generate_batch(progress, count, deck=None, rng=None, client=None,
                   model=SENTENCE_MODEL, stored=None):
    """Ask for `count` sentences and return only the ones that survive."""
    stored = load_generated() if stored is None else stored
    allowance = remaining_allowance(stored)
    if allowance <= 0:
        return [], []
    deck = deck or load_deck()
    known = known_ids(progress, deck)
    menu = word_menu(progress, deck, rng)
    if not menu:
        return [], []
    rng = rng or random
    sample = [r["es"] for r in stored]
    rng.shuffle(sample)
    messages = build_messages(menu, min(count, allowance), sample[:25])
    rows = call_api(messages, model, client)
    kept, dropped = usable(rows, known, {r["es"] for r in stored})
    return kept[:allowance], dropped
