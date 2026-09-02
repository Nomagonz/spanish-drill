"""Whole sentences, composed only from words already in the rotation.

The drill teaches one word at a time. This asks you to put them together: an
English sentence on screen, the Spanish typed back, graded word by word with
the mistakes marked in place.

What makes it worth having is the gate. A sentence is offered only once every
content word in it is a word you have got right and are being reviewed on, so
composing is never also a vocabulary test. See `requirements` for what counts
as a content word and `available` for the gate itself.

Requirements are read off the Spanish, never authored beside it. A list kept
by hand drifts from the sentence it describes, and a drifted list is
invisible: the sentence still reads fine while the gate has quietly stopped
checking a word. Deriving them means the two cannot disagree, and a word the
deck cannot account for is a broken sentence the tests refuse to ship.
"""
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from .config import (SENTENCE_KNOWN_INTERVAL, SENTENCE_REQUIRES_KNOWN_FORM,
                     SENTENCES_PATH)
from .conjugation import TENSES, VERBS, conjugate
from .deck import load_deck
from .text import normalize

# Glue, and nothing but glue. Never gated, because these are structure
# rather than vocabulary and most of them have no card at all.
#
# Kept deliberately short. An earlier version also exempted `eso`, `este`,
# `muy`, `sobre`, `hasta`, `ya`, `solo` and a dozen more, which put words the
# learner had never seen into sentences and broke the one promise this mode
# makes. Everything with a card in the deck belongs on the other side of the
# gate, however small a word it is: `este`, `entre` and `porque` are
# vocabulary, and they are gated like any other word.
FUNCTION_WORDS = frozenset("""
el la los las un una unos unas lo
de del a al en con por para sin
y e o u ni que si no pero
yo tu el ella usted nosotros nosotras vosotros vosotras ellos ellas ustedes
me te se nos os le les
mi mis tu tus su sus nuestro nuestra nuestros nuestras vuestro vuestra
""".split())

# Spanish drops the subject pronoun far more often than it keeps it, and both
# readings are correct: "hablo español" and "yo hablo español" are the same
# sentence. Marking one of them wrong would be the mode inventing a mistake,
# which is the failure this whole app is built to avoid. So these are matched
# when present and never counted against you either way.
# Written normalised, because that is the form they are compared in: `él`
# arrives as `el` and `tú` as `tu`.
SUBJECT_PRONOUNS = frozenset(
    "yo tu el ella usted nosotros nosotras vosotros vosotras "
    "ellos ellas ustedes".split())

# Which of them may be left out. `el` and `tu` are missing from this set on
# purpose: once accents are gone they are indistinguishable from the article
# `el` and the possessive `tu`, and forgiving a missing one would forgive a
# dropped article, which is a real mistake. Added at the front they are
# unambiguous — nobody opens a sentence with a bare article and no noun — so
# `SUBJECT_PRONOUNS` is what gets forgiven in that direction.
OMITTABLE = SUBJECT_PRONOUNS - {"el", "tu"}

# Forms no table produces. `hay` is the impersonal present of `haber` and
# looks like nothing else in its own paradigm.
ALIASES = {"hay": "haber",
           # Apocopes. The adjective loses its ending before a masculine
           # noun, and the shortened form is not what the deck stores.
           "primer": "primero", "gran": "grande", "buen": "bueno",
           "tercer": "tercero", "algun": "alguno", "ningun": "ninguno",
           # Spelling, not conjugation: the yo form keeps the sound and so
           # has to change the letter. Three verbs in this deck do it.
           "consigo": "conseguir", "elijo": "elegir", "cojo": "coger"}


@dataclass(frozen=True)
class Sentence:
    en: str                 # the cue, read off the screen
    es: str                 # the answer, accents and all
    needs: tuple = ()       # deck card ids every content word maps to
    also: tuple = ()        # equally correct wordings, graded against too

    @property
    def id(self):
        return normalize(self.es).replace(" ", "-")

    @property
    def wordings(self):
        return (self.es,) + tuple(self.also)


# -- reading the deck back out of a sentence ------------------------------
def _variants(word):
    """The surface forms one dictionary word appears as in a sentence.

    Number and gender only. A noun is plural about as often as not and an
    adjective agrees with whatever it describes, so indexing only the citation
    form would leave half the words in the bank resolving to nothing.
    """
    out = {word}
    if word.endswith("z"):
        out.add(word[:-1] + "ces")          # luz -> luces
    elif word[-1:] in "aeiou":
        out.add(word + "s")
    else:
        out.add(word + "es")                # papel -> papeles
    if word.endswith("o"):
        stem = word[:-1]
        out.update({stem + "a", stem + "os", stem + "as"})
    return out


# Stem changers the conjugation tables do not cover. Those tables hold the
# verbs worth teaching; these are the rest of the deck, listed only so their
# forms can be recognised. Without it `consigues` leads nowhere and a
# perfectly good sentence is thrown away as if it used a word off the deck.
#
# Written out rather than derived for the reason conjugation.py gives: which
# verbs change and which do not is a fact about each verb, not a rule.
STEM_CHANGES = {
    "conseguir": ("e", "i"), "elegir": ("e", "i"), "repetir": ("e", "i"),
    "vestir": ("e", "i"), "medir": ("e", "i"), "impedir": ("e", "i"),
    "mentir": ("e", "ie"), "referir": ("e", "ie"), "convertir": ("e", "ie"),
    "preferir": ("e", "ie"), "despertar": ("e", "ie"), "sentar": ("e", "ie"),
    "divertir": ("e", "ie"), "defender": ("e", "ie"), "encender": ("e", "ie"),
    "detener": ("e", "ie"), "mantener": ("e", "ie"), "obtener": ("e", "ie"),
    "recordar": ("o", "ue"), "mostrar": ("o", "ue"), "sonar": ("o", "ue"),
    "acostar": ("o", "ue"), "probar": ("o", "ue"), "volar": ("o", "ue"),
    "costar": ("o", "ue"), "doler": ("o", "ue"), "resolver": ("o", "ue"),
    "devolver": ("o", "ue"), "soler": ("o", "ue"), "colgar": ("o", "ue"),
}


def _change_stem(stem, change):
    """Swap the last instance of the changing vowel: recib -> ... , cont -> cuent."""
    old, new = change
    at = stem.rfind(old)
    return stem[:at] + new + stem[at + len(old):] if at >= 0 else stem


def _regular_forms(infinitive):
    """A regular paradigm for a verb the conjugation tables do not cover.

    Those tables are deliberately small: they hold the irregulars and stem
    changers, which is what has to be memorised. But the deck has 146 verbs
    in the rotation and a sentence may use any of them, so `corre` has to
    lead back to `correr` or the gate cannot see the word at all.

    Only ever used to recognise a word, never to teach one. A stem changer
    the tables have not got would generate a wrong form here, and a wrong
    form is inert: no sentence contains it, so nothing ever looks it up.
    """
    from .conjugation import (CONDITIONAL_ENDINGS, FUTURE_ENDINGS, REGULAR,
                              _split)
    stem, ending = _split(infinitive)
    if ending not in ("ar", "er", "ir"):
        return ()
    out = []
    for tense in ("pres", "pret", "imp"):
        out.extend(stem + e for e in REGULAR[tense][ending])
    out.extend(infinitive + e for e in FUTURE_ENDINGS)
    out.extend(infinitive + e for e in CONDITIONAL_ENDINGS)
    change = STEM_CHANGES.get(normalize(infinitive))
    if change:
        # The boot: every present form but nosotros. The regular ones stay
        # indexed alongside, because an extra spelling nobody ever writes
        # costs nothing and a missing one costs a whole sentence.
        changed = _change_stem(stem, change)
        endings = REGULAR["pres"][ending]
        out.extend(changed + endings[i] for i in (0, 1, 3, 4))
    return tuple(out)


@lru_cache(maxsize=1)
def _forms():
    """Every surface form the deck accounts for -> the card id behind it.

    A form can belong to more than one card: `vino` is wine and also what
    `venir` did yesterday, `trabajo` is the noun and also "I work". Every
    reading is kept and the gate demands all of them, because it cannot tell
    from the spelling which one a sentence meant, and the safe direction is
    the strict one: a sentence is never offered while it might be leaning on
    a word that has not been learned. A sentence that would rather be judged
    on one reading says so with an explicit `needs`.
    """
    index, exact = {}, {}

    def add(form, card_id, is_citation=False):
        for store in ((index, exact) if is_citation else (index,)):
            seen = store.setdefault(form, [])
            if card_id not in seen:
                seen.append(card_id)

    for card in load_deck():
        if card.lemma:
            continue
        for answer in card.answers:
            n = normalize(answer)
            if not n or " " in n:
                continue                    # multi-word entries are phrases
            add(n, card.id, is_citation=True)     # the word as the deck writes it
            for form in _variants(n):
                add(form, card.id)
    for lemma in VERBS:
        for tense in TENSES:
            for spanish in conjugate(lemma, tense):
                add(normalize(spanish), lemma)
    # Regular verbs last, so a curated paradigm always wins over a generated
    # one: `puedo` stays poder's rather than being overwritten by `podo`.
    curated = {normalize(v) for v in VERBS}
    for card in load_deck():
        if card.lemma or card.pos != "verb":
            continue
        infinitive = card.answers[0]
        if " " in infinitive or normalize(infinitive) in curated:
            continue
        for form in _regular_forms(infinitive):
            add(normalize(form), card.id)
    for surface, lemma in ALIASES.items():
        add(surface, lemma)
    verbs = {c.id for c in load_deck() if not c.lemma and c.pos == "verb"}
    return ({form: tuple(ids) for form, ids in index.items()},
            {form: tuple(ids) for form, ids in exact.items()}, verbs)


# Object pronouns ride on the back of an infinitive or a gerund in Spanish:
# "ayudarme", "verlo", "decírtelo". Longest first, so "melo" is taken off
# whole rather than leaving a stray "me".
ENCLITICS = ("noslo", "nosla", "melo", "mela", "telo", "tela", "selo", "sela",
             "selos", "selas", "nos", "les", "los", "las", "me", "te", "se",
             "lo", "la", "le", "os")


def _without_enclitic(word):
    """`verlo` -> `ver`, or None.

    Only ever peeled off an infinitive or a gerund, which is where Spanish
    actually attaches them. Stripping from anything else would start finding
    verbs inside ordinary nouns: `vela` is a candle, not `ver` plus `la`.
    """
    for tail in ENCLITICS:
        if word.endswith(tail) and len(word) > len(tail) + 2:
            base = word[: -len(tail)]
            if base.endswith("r") or base.endswith("ndo"):
                return base
    return None


def resolve(word):
    """Every deck card a sentence word could belong to. Empty if none does.

    Empty means one of two things and the caller has to tell them apart: a
    function word, which is fine and ungated, or a word the deck cannot
    account for, which is a broken sentence.
    """
    n = normalize(word)
    index, exact, verbs = _forms()

    def readings(form):
        """Every card the form could be, with dead-weight inflections cut.

        A word spelled exactly as the deck writes it is that word, and only
        incidentally the inflection of some other noun or adjective: `lista`
        is the noun, not the feminine of `listo`. Those inflected readings
        are dropped, because carrying them meant ordinary vocabulary dragged
        in adjectives the sentence never used and the gate held back a
        sentence you could read perfectly well.

        Verb readings are never dropped. An unlearned conjugated form is the
        one thing that must not slip through, and `vino` really can be what
        venir did yesterday however plainly it is also wine.
        """
        found = index.get(form, ())
        if form not in exact:
            return found
        keep = set(exact[form])
        return tuple(r for r in found if r in keep or r in verbs)

    if n in index:
        return readings(n)
    bare = _without_enclitic(n)
    return readings(bare) if bare else ()


def words_of(spanish):
    return [w for w in normalize(spanish).split() if w]


def requirements(spanish, override=None):
    """(card ids that must be known, words nothing can account for)."""
    if override:
        return tuple(override), ()
    needs, unknown = [], []
    for word in words_of(spanish):
        if word in FUNCTION_WORDS:
            continue
        found = resolve(word)
        if not found:
            unknown.append(word)
        for card_id in found:
            if card_id not in needs:
                needs.append(card_id)
    return tuple(needs), tuple(unknown)


# Cached on the file's timestamp rather than for the life of the process.
# Held outright, editing the bank did nothing until the app was restarted,
# and a sentence deleted an hour ago kept coming up as though it were still
# there. Reading a stat() per call is nothing next to that confusion.
_loaded = {"key": None, "value": ()}


def load_sentences(path=None):
    """The bank, reloaded whenever the file on disk has changed."""
    from pathlib import Path
    path = Path(path or SENTENCES_PATH)
    try:
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        key = (str(path), None, None)
    if _loaded["key"] == key:
        return _loaded["value"]
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []
    out = []
    for row in raw:
        needs, _ = requirements(row["es"], row.get("needs"))
        out.append(Sentence(en=row["en"], es=row["es"], needs=needs,
                            also=tuple(row.get("also", ()))))
    _loaded["key"], _loaded["value"] = key, tuple(out)
    return _loaded["value"]


# -- the gate -------------------------------------------------------------
def known_ids(progress, deck=None, bar=None):
    """Card ids at or past the bar, as ids rather than positions.

    The bar is an interval in days. One means answered right and scheduled
    ahead — the panel's LEARNING — which takes in both the words still being
    learned and the ones that have matured past them.
    """
    bar = SENTENCE_KNOWN_INTERVAL if bar is None else bar
    deck = deck or load_deck()
    return {deck[i].id for i, card in progress.cards.items()
            if 0 <= i < len(deck) and card.interval >= bar}


def _form_requirements(sentence, known):
    """The conjugated forms a strict gate demands on top of the infinitives.

    Off by default. Requiring the exact form is right in principle and much
    narrower in practice: only the present tense has any forms in the
    rotation at all, and a thoroughly known verb like `hablar` has none.
    """
    if not SENTENCE_REQUIRES_KNOWN_FORM:
        return ()
    wanted = []
    for word in words_of(sentence.es):
        if word in FUNCTION_WORDS:
            continue
        readings = resolve(word)
        # A word that is ordinary vocabulary you already know needs no verb
        # form behind it. `vino` is wine; demanding venir's preterite for it
        # would hold back a sentence that never used the verb at all.
        if any(r in known and r not in VERBS for r in readings):
            continue
        for lemma in readings:
            if lemma not in VERBS:
                continue
            for form_id, spanish in _form_ids(lemma):
                if normalize(spanish) == word and form_id not in wanted:
                    wanted.append(form_id)
    return tuple(wanted)


@lru_cache(maxsize=1)
def _all_form_ids():
    out = {}
    for lemma in VERBS:
        rows = []
        for tense in TENSES:
            from .conjugation import PERSONS
            for person, spanish in zip(PERSONS, conjugate(lemma, tense)):
                rows.append((f"{lemma}:{tense}-{person}", spanish))
        out[lemma] = tuple(rows)
    return out


def _form_ids(lemma):
    return _all_form_ids().get(lemma, ())


def _form_groups(sentence, known):
    """The conjugated forms a sentence needs, one group per word.

    A group holds every card a written form could be, and any one of them
    satisfies it. Spanish spells the nosotros present and preterite of an
    -ir verb identically — `salimos`, `vivimos`, `escribimos` — so demanding
    both would hold a sentence back until a tense it never uses is learned.
    You produce one form; learning one reading of it is the bar.
    """
    if not SENTENCE_REQUIRES_KNOWN_FORM:
        return ()
    groups = []
    for word in words_of(sentence.es):
        if word in FUNCTION_WORDS:
            continue
        readings = resolve(word)
        if any(r in known and r not in VERBS for r in readings):
            continue
        found = []
        for lemma in readings:
            if lemma not in VERBS:
                continue
            for form_id, spanish in _form_ids(lemma):
                if normalize(spanish) == word and form_id not in found:
                    found.append(form_id)
        if found:
            groups.append(tuple(found))
    return tuple(groups)


def is_available(sentence, known):
    """Every content word known. That is the whole rule."""
    if not sentence.needs:
        return False
    if not all(n in known for n in sentence.needs):
        return False
    return all(any(f in known for f in group)
               for group in _form_groups(sentence, known))


def all_sentences():
    """The curated bank plus everything the API has written so far.

    The bank ships with the repo and needs no key, no network and no spend.
    Generated ones only ever add to it, and are held to exactly the same
    gate, so where a sentence came from changes nothing about whether it may
    be asked.
    """
    from .config import SENTENCE_GENERATION
    if not SENTENCE_GENERATION:
        return tuple(load_sentences())      # the verified bank, on its own
    from .generate import as_sentences, load_generated
    return tuple(load_sentences()) + tuple(as_sentences(load_generated()))


def available(progress, deck=None, sentences=None, bar=None):
    known = known_ids(progress, deck, bar)
    pool = all_sentences() if sentences is None else sentences
    return [s for s in pool if is_available(s, known)]


def unfinished(progress, deck=None, sentences=None, bar=None):
    """Unlocked, and not already answered correctly.

    Sentences retire for good rather than coming round again. There is no
    schedule behind them: the words they are built from are already being
    reviewed on their own account, and asking for a sentence you have
    produced correctly is asking twice for the same evidence.
    """
    done = set(getattr(progress, "sentences_done", None) or ())
    return [s for s in available(progress, deck, sentences, bar)
            if s.id not in done]


def blocked_by(sentence, known):
    """Which words are holding a sentence back, for saying so on screen."""
    missing = [n for n in sentence.needs if n not in known]
    for group in _form_groups(sentence, known):
        if not any(f in known for f in group):
            missing.append(group[0])    # name one; they are the same word
    return tuple(missing)


# -- grading --------------------------------------------------------------
# Accents are not graded, exactly as they are not on a single typed word, and
# for the reason written down in grading.py: reaching for the option key on
# every other word is a tax on the mode that exists for answering quickly and
# quietly. The correct sentence is shown back with its accents intact, so a
# missing one is still visible. It just is not marked wrong.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def tokens(text):
    """The sentence as it is compared: words in order, punctuation gone."""
    return [w for w in _PUNCTUATION.sub(" ", text or "").split() if w]


def compare_as(word):
    """The key two words are matched on.

    Accents go, and so does the tilde on `ñ`. Everywhere else in the app the
    tilde survives, because `ñ` is a distinct letter rather than an accented
    `n` and `año` is not `ano`. Here it goes with the rest: writing a whole
    sentence is enough work without reaching for the option key twice a line,
    and the right spelling is shown back afterwards either way.
    """
    return normalize(word).replace("ñ", "n")


@dataclass(frozen=True)
class Token:
    """One word of the marked-up answer.

    `state` is right, wrong, missing or extra. `text` is what to show, which
    for a wrong or missing word is the expected spelling with its accents,
    because that is the part worth reading.
    """
    text: str
    state: str
    typed: str = ""         # what was written instead, on a wrong word


@dataclass(frozen=True)
class Grade:
    perfect: bool
    marked: tuple           # Tokens, in the order the sentence should read
    extra: tuple            # words written that belong nowhere
    typed: str
    expected: str           # the wording this was graded against

    @property
    def mistakes(self):
        return sum(1 for t in self.marked if t.state != "right") + len(self.extra)


def _grade_against(typed, expected):
    """Word-by-word comparison, aligned rather than zipped.

    difflib rather than position-by-position: one missing word early shifts
    everything after it, and a zip would mark the whole rest of the sentence
    wrong when only the first word was actually missed.
    """
    want, got = tokens(expected), tokens(typed)
    want_key = [compare_as(w) for w in want]
    got_key = [compare_as(w) for w in got]

    marked, extra = [], []
    for op, i1, i2, j1, j2 in SequenceMatcher(
            a=want_key, b=got_key, autojunk=False).get_opcodes():
        if op == "equal":
            marked.extend(Token(want[i], "right") for i in range(i1, i2))
        elif op == "replace":
            for at in range(max(i2 - i1, j2 - j1)):
                i, j = i1 + at, j1 + at
                if i < i2 and j < j2:
                    marked.append(Token(want[i], "wrong", got[j]))
                elif i < i2:
                    marked.append(Token(want[i], "missing"))
                else:
                    extra.append((j, got[j]))
        elif op == "delete":
            marked.extend(Token(want[i], "missing") for i in range(i1, i2))
        else:
            extra.extend((j, got[j]) for j in range(j1, j2))

    # Spanish drops the subject pronoun far more often than it keeps it, and
    # both readings are correct: "hablo español" and "yo hablo español" are
    # the same sentence. Forgiven only at the very front, which is the only
    # place a subject pronoun can be: anywhere else the same word is an
    # article or an object pronoun and dropping it is a real mistake.
    if marked and marked[0].state == "missing" \
            and normalize(marked[0].text) in OMITTABLE:
        marked = marked[1:]
    extra = tuple(word for at, word in extra
                  if not (at == 0 and normalize(word) in SUBJECT_PRONOUNS))

    marked = tuple(marked)
    perfect = bool(got) and not extra and all(t.state == "right" for t in marked)
    return Grade(perfect=perfect, marked=marked, extra=extra,
                 typed=typed or "", expected=expected)


def grade(typed, sentence):
    """Grade against every accepted wording and keep the kindest verdict.

    A sentence with an `also` is one where two wordings are genuinely equal
    Spanish. Marking the second one wrong because the first was written down
    first would be the mode inventing a mistake.
    """
    best = None
    for wording in sentence.wordings:
        result = _grade_against(typed, wording)
        if result.perfect:
            return result
        if best is None or result.mistakes < best.mistakes:
            best = result
    return best
