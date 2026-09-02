"""The deck itself has to be answerable.

Everything here is an objective property of the data, not a guess at what a
word ought to mean. The one that matters most: the drill speaks a cue and
grades what you say against one card's answers. If two cards say the same
thing out loud but accept different words, one of them will mark a correct
answer wrong every time it comes up, and there is nothing you could have said
to avoid it.
"""
import re
from collections import defaultdict

from spanish_drill.deck import load_deck
from spanish_drill.grading import check


def heard_as(cue):
    """A cue reduced to what a listener actually receives: words, in order.

    Punctuation is inaudible, so it goes. Nothing else does. An earlier version
    also stripped a leading "to", which collapsed "to change" onto "change" and
    flagged every honest verb/noun pair in the deck.
    """
    return " ".join(re.sub(r"[^a-z ]", " ", cue.lower()).split())


def senses_offered(card):
    """Every meaning a cue offers, as a listener would take them.

    The bracket is the qualifier, so the sense is what is left of the cue
    without it. And a list of infinitives shares one "to": "to pardon, excuse"
    offers *to excuse*, not the bare noun, so the leading "to" is carried onto
    the senses that follow it. Comparing them without it let `perdonar` offer
    "excuse" while `disculpar` offered "to excuse", and the two never met.
    """
    bare = re.sub(r"\([^)]*\)", "", card.prompt)
    parts = [heard_as(p) for p in re.split(r"[,;]", bare)]
    parts = [p for p in parts if p]
    if parts and parts[0].startswith("to "):
        parts = [parts[0]] + [p if p.startswith("to ") else "to " + p
                              for p in parts[1:]]
    return parts


def leading_sense(card):
    """The first meaning a cue offers, which is the one you answer from.

    "to return, to come back" and "to return, go back" are different strings
    that both open with "to return" and both want the same thing. A card
    carrying a parenthetical is exempt: the bracket is the deck saying this cue
    has been deliberately separated from its neighbour.
    """
    if "(" in card.prompt:
        return heard_as(card.prompt)
    return heard_as(re.split(r"[,;]", card.prompt)[0])


def groups_that_sound_alike():
    """Cards a listener could not tell apart, grouped.

    Two passes, because neither catches the other's cases: the whole cue, and
    the leading sense within one part of speech. Leading sense is compared only
    among cards of the same kind, since "to change" and "change, shift" are
    told apart by the "to" and are not a collision.

    Not exhaustive and not meant to be. It pins the collisions this deck
    actually had; a pair overlapping only in a later sense still needs a person
    to notice.
    """
    deck = load_deck()
    groups = defaultdict(set)
    for index, card in enumerate(deck):
        groups[("cue", heard_as(card.spoken_prompt))].add(index)
        groups[("sense", card.pos, leading_sense(card))].add(index)
    return {key: sorted(rows) for key, rows in groups.items() if len(rows) > 1}


class TestEveryCueIsAnswerable:
    def test_cards_that_sound_alike_accept_the_same_words(self):
        """The bug this exists for: "have" wanted haber, "to have" wanted
        tener, and each rejected the other. Whichever came up, half the time
        the only sensible answer was marked wrong."""
        deck = load_deck()
        broken = [
            [(i, deck[i].prompt, deck[i].answers) for i in rows]
            for rows in groups_that_sound_alike().values()
            if len({frozenset(deck[i].answers) for i in rows}) > 1
        ]
        assert not broken, (
            "these cues sound the same but want different answers, so there is "
            f"no way to answer them right: {broken}")

    def test_a_synonym_is_not_correct_on_one_card_and_wrong_on_another(self):
        """"to return, to come back" took volver or regresar; "to return, go
        back" took only regresar. Same cue by ear, and volver was a wrong
        answer on one of them."""
        deck = load_deck()
        for key, rows in groups_that_sound_alike().items():
            accepted = {frozenset(deck[i].answers) for i in rows}
            assert len(accepted) == 1, (
                f"{key}: {[(i, deck[i].prompt, deck[i].answers) for i in rows]}")

    def test_no_english_meaning_is_offered_by_two_cards(self):
        """One meaning, one card, unless every claimant says which it is.

        Stricter than the two checks above, which only compare whole cues and
        leading senses. This catches an overlap in any later sense: `tomar`
        used to offer "to drink" alongside `beber`, and `decir` offered "to
        tell" alongside `contar`. Hearing either cue both answers are right
        and only one is accepted, so the card cannot be answered.

        A meaning may be shared, but only when EVERY card offering it carries
        a parenthetical saying which one it is: `ser` and `estar` both mean
        "to be" and both say so. One bare claim is enough to break it, because
        that is the cue with nothing to tell it apart from the others.
        """
        owners = defaultdict(list)
        for card in load_deck():
            if card.lemma:
                continue
            for sense in senses_offered(card):
                owners[sense].append(card)

        broken = {}
        for sense, cards in owners.items():
            if len(cards) < 2:
                continue
            unqualified = [c.answers[0] for c in cards if "(" not in c.prompt]
            if unqualified:
                broken[sense] = {"claimed bare by": unqualified,
                                 "also offered by": [c.answers[0] for c in cards]}
        assert not broken, (
            "these meanings are offered by more than one card with nothing to "
            f"tell them apart: {broken}")

    def test_every_verb_is_cued_as_an_infinitive(self):
        """This is what keeps a verb from colliding with its own noun. The bug
        that started all of this was a card cueing haber as "have": the missing
        "to" is the only thing separating it from tener, and it was missing."""
        odd = [(i, c.prompt) for i, c in enumerate(load_deck())
               if c.pos == "verb" and not c.lemma
               and not c.spoken_prompt.lower().startswith("to ")]
        assert not odd, f"verbs cued as if they were nouns: {odd}"

    def test_no_cue_is_silent(self):
        assert all(card.spoken_prompt for card in load_deck())

    def test_the_parenthetical_survives_being_spoken(self):
        """Dropping it is what made "to know (a fact)" unanswerable."""
        for card in load_deck():
            if "(" in card.prompt:
                inside = card.prompt.split("(", 1)[1].split(")", 1)[0]
                assert inside.lower() in card.spoken_prompt.lower(), card.prompt

    def test_no_cue_reads_out_a_bracket(self):
        """A cue is read aloud verbatim, so any bracket left in it is either
        spoken or silently swallowed, and the disambiguating aside goes with
        it. Only round brackets are converted, so only round brackets belong."""
        stray = [c.prompt for c in load_deck()
                 if set(c.spoken_prompt) & set("()[]{}<>")]
        assert not stray, f"brackets survive into the spoken cue: {stray}"


class TestTheSubjectIsFindableInACue:
    """The screen marks it, so it has to come back for every form and for
    nothing else. A form whose subject came back empty would read as plain
    vocabulary, which is exactly the card it is easiest to answer in the
    wrong person."""

    def test_every_conjugated_cue_names_who_it_is_about(self):
        missing = [c.prompt for c in load_deck() if c.form and not c.subject]
        assert not missing, f"conjugated cues with no subject: {missing}"

    def test_the_subject_opens_the_cue_it_came_from(self):
        for card in load_deck():
            if card.subject:
                assert card.prompt.startswith(card.subject), card.prompt

    def test_you_all_is_never_read_as_a_bare_you(self):
        """Both are subjects and one is a prefix of the other, so the shorter
        match would leave "all" sitting outside the mark."""
        plural = [c for c in load_deck() if c.form and c.form.endswith("-vos")]
        assert plural
        assert all(c.subject == "you all" for c in plural)

    def test_a_pronoun_card_is_not_treated_as_a_paradigm(self):
        """"you (informal)" is the word being taught, not a conjugation."""
        assert not any(c.subject for c in load_deck() if not c.form)


class TestAnswersAreGradeable:
    def test_every_card_accepts_its_own_answers(self):
        for card in load_deck():
            for answer in card.answers:
                assert check(answer, card), f"{card.prompt}: {answer} rejected"

    def test_no_card_is_missing_its_pieces(self):
        for index, card in enumerate(load_deck()):
            assert card.answers and all(a.strip() for a in card.answers), index
            # Conjugated forms are checked by TestEveryFormHasASentence,
            # which also pins that the sentence uses the form it teaches.
            if not card.lemma:
                assert card.example.strip() and card.gloss.strip(), index

    def test_an_answer_is_never_listed_twice_on_one_card(self):
        for card in load_deck():
            assert len(set(card.answers)) == len(card.answers), card.prompt


class TestTheDeckMatchesStoredProgress:
    """Progress used to be keyed by position, which made the deck order
    load-bearing: inserting a card repointed every card after it at someone
    else's history, and it happened. It is keyed by card id now, so what has
    to hold is that ids are unique and that every saved key still resolves."""

    def test_every_card_has_a_unique_id(self):
        deck = load_deck()
        ids = [c.id for c in deck]
        assert len(set(ids)) == len(ids)
        assert all(ids)

    def test_a_conjugation_id_can_never_collide_with_a_word(self):
        """`como`, `vino` and `paso` are vocabulary cards and also conjugated
        forms. The namespace separator is what keeps them apart."""
        deck = load_deck()
        vocabulary = {c.id for c in deck if not c.lemma}
        assert not any(":" in i for i in vocabulary)
        assert all(":" in c.id for c in deck if c.lemma)

    def test_every_saved_key_resolves_to_a_card(self):
        from spanish_drill.config import PROGRESS_PATH
        import json
        if not PROGRESS_PATH.exists():
            return
        saved = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        ids = {c.id for c in load_deck()}
        unknown = [k for k in saved.get("cards", {}) if k not in ids]
        assert not unknown, f"progress refers to cards that do not exist: {unknown}"


class TestBracketsDoNotSwallowAnotherCard:
    """A qualifier may reuse a word, but not one that is a whole other card.

    `obra` was cued "work (a play, a piece of art)" while `juego` was cued
    "play" outright, so within the nouns the word "play" identified two cards
    and neither cue said which. Comparing only the text outside the brackets
    could never see it, because the collision was inside one.

    Restricted to a single part of speech on purpose. Across two, the cue
    carries the difference on its own: "to ask (a question)" and "question"
    are never mistaken for each other.
    """

    def test_no_parenthetical_names_another_cue_of_the_same_kind(self):
        deck = [c for c in load_deck() if not c.lemma]
        whole_cue = {}
        for card in deck:
            bare = heard_as(re.sub(r"\([^)]*\)", "", card.prompt))
            if bare:
                whole_cue.setdefault((card.pos, bare), card)

        swallowed = []
        for card in deck:
            for inside in re.findall(r"\(([^)]*)\)", card.prompt):
                for part in re.split(r"[,;]", inside):
                    phrase = re.sub(r"^(a|an|the) ", "", heard_as(part))
                    other = whole_cue.get((card.pos, phrase))
                    if other is not None and other is not card:
                        swallowed.append(
                            (card.answers[0], card.prompt,
                             other.answers[0], other.prompt))
        assert not swallowed, (
            "a bracket names another card of the same kind outright, so the "
            f"word points at two cards: {swallowed}")
