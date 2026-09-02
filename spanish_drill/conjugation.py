"""Conjugated forms, for the drill's second phase.

A verb earns its conjugations only after the infinitive itself has survived
real spaced review, and then one tense at a time. See `unlocked_forms`.

Why only some verbs. Regular conjugation is a pattern, not vocabulary: once
you can do `hablar` you can do the other ninety regular -ar verbs in the deck,
and drilling them one at a time is ninety cards teaching one suffix. What
actually has to be memorised is the irregulars and the stem changes, so those
get the full treatment and exactly one regular verb per ending comes along to
keep the pattern warm.

The present tense is written out per verb rather than derived. It is where
almost all the irregularity lives, and a rule engine subtle enough to get
`tengo`, `quepo`, `conozco` and `juego` right from a stem is harder to check
than the five words themselves. Everything after the present is systematic
enough to generate: the imperfect has three irregular verbs in the whole
language, the future and conditional share one stem list, and the preterite
irregulars are a closed set of strong stems.
"""
from .config import PERSON_ORDER, TENSE_ORDER

# The five distinct forms, in the order they are taught. Spanish collapses
# more than English does: he, she and formal usted are one form, and they and
# ustedes are another, so drilling seven pronouns would mean two pairs of
# cards with identical answers.
PERSONS = PERSON_ORDER                  # yo, tu, nos, el, vos, ellos
TENSES = TENSE_ORDER                    # ("pres", "pret", "imp", "fut", "cond")

PRONOUN = {"yo": "I", "tu": "you", "nos": "we", "el": "he",
           "vos": "you all", "ellos": "they"}

# vosotros. Written apart from the tables above rather than folded into them,
# because it is a dialect choice rather than a sixth fact about the verb: on
# Latin American Spanish these forms are never taught at all.
VOSOTROS_PRESENT_ENDING = {"ar": "áis", "er": "éis", "ir": "ís"}
VOSOTROS_PRESENT = {"ser": "sois", "ir": "vais", "ver": "veis", "dar": "dais"}
VOSOTROS_PRETERITE = {"ser": "fuisteis", "ir": "fuisteis", "dar": "disteis",
                      "ver": "visteis", "oír": "oísteis"}
VOSOTROS_IMPERFECT = {"ser": "erais", "ir": "ibais", "ver": "veíais"}

# -- regular endings ------------------------------------------------------
REGULAR = {
    "pres": {"ar": ("o", "as", "amos", "a", "an"),
             "er": ("o", "es", "emos", "e", "en"),
             "ir": ("o", "es", "imos", "e", "en")},
    "pret": {"ar": ("é", "aste", "amos", "ó", "aron"),
             "er": ("í", "iste", "imos", "ió", "ieron"),
             "ir": ("í", "iste", "imos", "ió", "ieron")},
    "imp":  {"ar": ("aba", "abas", "ábamos", "aba", "aban"),
             "er": ("ía", "ías", "íamos", "ía", "ían"),
             "ir": ("ía", "ías", "íamos", "ía", "ían")},
}
# Future and conditional hang their endings off the whole infinitive, which is
# why they share an irregular-stem list: know `tendré` and you nearly know
# `tendría`.
FUTURE_ENDINGS = ("é", "ás", "emos", "á", "án")
CONDITIONAL_ENDINGS = ("ía", "ías", "íamos", "ía", "ían")

IRREGULAR_STEM = {          # future and conditional both build on these
    "decir": "dir", "haber": "habr", "hacer": "har", "poder": "podr",
    "poner": "pondr", "querer": "querr", "saber": "sabr", "salir": "saldr",
    "tener": "tendr", "valer": "valdr", "venir": "vendr", "caber": "cabr",
    "obtener": "obtendr", "mantener": "mantendr", "detener": "detendr",
    "suponer": "supondr",
    # The accent in `oír` only exists to break the diphthong in the
    # infinitive. Adding a syllable does that job instead, so it goes.
    "oír": "oir",
}

# -- present, written out -------------------------------------------------
# Order is always yo, tú, nosotros, él, ellos.
PRESENT = {
    "ser":       ("soy", "eres", "somos", "es", "son"),
    "estar":     ("estoy", "estás", "estamos", "está", "están"),
    "haber":     ("he", "has", "hemos", "ha", "han"),
    "tener":     ("tengo", "tienes", "tenemos", "tiene", "tienen"),
    "ir":        ("voy", "vas", "vamos", "va", "van"),
    "hacer":     ("hago", "haces", "hacemos", "hace", "hacen"),
    "poder":     ("puedo", "puedes", "podemos", "puede", "pueden"),
    "decir":     ("digo", "dices", "decimos", "dice", "dicen"),
    "querer":    ("quiero", "quieres", "queremos", "quiere", "quieren"),
    "ver":       ("veo", "ves", "vemos", "ve", "ven"),
    "saber":     ("sé", "sabes", "sabemos", "sabe", "saben"),
    "dar":       ("doy", "das", "damos", "da", "dan"),
    "venir":     ("vengo", "vienes", "venimos", "viene", "vienen"),
    "poner":     ("pongo", "pones", "ponemos", "pone", "ponen"),
    "salir":     ("salgo", "sales", "salimos", "sale", "salen"),
    "seguir":    ("sigo", "sigues", "seguimos", "sigue", "siguen"),
    "conocer":   ("conozco", "conoces", "conocemos", "conoce", "conocen"),
    "traer":     ("traigo", "traes", "traemos", "trae", "traen"),
    "oír":       ("oigo", "oyes", "oímos", "oye", "oyen"),
    "caer":      ("caigo", "caes", "caemos", "cae", "caen"),
    "volver":    ("vuelvo", "vuelves", "volvemos", "vuelve", "vuelven"),
    "pensar":    ("pienso", "piensas", "pensamos", "piensa", "piensan"),
    "sentir":    ("siento", "sientes", "sentimos", "siente", "sienten"),
    "encontrar": ("encuentro", "encuentras", "encontramos", "encuentra",
                  "encuentran"),
    "empezar":   ("empiezo", "empiezas", "empezamos", "empieza", "empiezan"),
    "pedir":     ("pido", "pides", "pedimos", "pide", "piden"),
    "contar":    ("cuento", "cuentas", "contamos", "cuenta", "cuentan"),
    "jugar":     ("juego", "juegas", "jugamos", "juega", "juegan"),
    "dormir":    ("duermo", "duermes", "dormimos", "duerme", "duermen"),
    "morir":     ("muero", "mueres", "morimos", "muere", "mueren"),
    "perder":    ("pierdo", "pierdes", "perdemos", "pierde", "pierden"),
    "entender":  ("entiendo", "entiendes", "entendemos", "entiende",
                  "entienden"),
    "cerrar":    ("cierro", "cierras", "cerramos", "cierra", "cierran"),
    "mover":     ("muevo", "mueves", "movemos", "mueve", "mueven"),
    "servir":    ("sirvo", "sirves", "servimos", "sirve", "sirven"),
    "andar":     ("ando", "andas", "andamos", "anda", "andan"),
    "hablar":    ("hablo", "hablas", "hablamos", "habla", "hablan"),
    "comer":     ("como", "comes", "comemos", "come", "comen"),
    "vivir":     ("vivo", "vives", "vivimos", "vive", "viven"),
}

# -- preterite ------------------------------------------------------------
# Strong preterites: an irregular stem plus a set of endings that carry no
# accent, which is what makes `hablo`/`habló` a regular-verb problem only.
STRONG_PRETERITE = {
    "tener": "tuv", "estar": "estuv", "poder": "pud", "poner": "pus",
    "saber": "sup", "querer": "quis", "venir": "vin", "andar": "anduv",
    "haber": "hub", "caber": "cup", "hacer": "hic", "decir": "dij",
    "traer": "traj",
}
STRONG_ENDINGS = ("e", "iste", "imos", "o", "ieron")

PRETERITE = {           # the ones that follow no pattern at all
    "ser":    ("fui", "fuiste", "fuimos", "fue", "fueron"),
    "ir":     ("fui", "fuiste", "fuimos", "fue", "fueron"),
    "dar":    ("di", "diste", "dimos", "dio", "dieron"),
    "ver":    ("vi", "viste", "vimos", "vio", "vieron"),
    "oír":    ("oí", "oíste", "oímos", "oyó", "oyeron"),
}
# -ir verbs whose stem shifts in the third person only, and -er/-ir verbs
# whose ending vowel turns to a y between vowels.
PRETERITE_THIRD = {
    "pedir": ("pidió", "pidieron"), "dormir": ("durmió", "durmieron"),
    "morir": ("murió", "murieron"), "sentir": ("sintió", "sintieron"),
    "seguir": ("siguió", "siguieron"), "servir": ("sirvió", "sirvieron"),
    "oír": ("oyó", "oyeron"), "caer": ("cayó", "cayeron"),
}

IMPERFECT = {           # the only three irregular imperfects in the language
    "ser": ("era", "eras", "éramos", "era", "eran"),
    "ir":  ("iba", "ibas", "íbamos", "iba", "iban"),
    "ver": ("veía", "veías", "veíamos", "veía", "veían"),
}

# -- English side ---------------------------------------------------------
# (base, third person, simple past, form used after "will"/"used to").
# The fourth entry exists for "can", which has no infinitive: "I will can" is
# not English, so the future and conditional borrow "be able to". Everything
# else repeats its base and the field is filled in by _english().
ENGLISH = {
    "ser": ("be", "is", None), "estar": ("be", "is", None),
    "haber": ("have", "has", "had"), "tener": ("have", "has", "had"),
    "ir": ("go", "goes", "went"), "hacer": ("do", "does", "did"),
    "poder": ("can", "can", "could", "be able to"),
    "decir": ("say", "says", "said"),
    "querer": ("want", "wants", "wanted"), "ver": ("see", "sees", "saw"),
    "saber": ("know", "knows", "knew"), "dar": ("give", "gives", "gave"),
    "venir": ("come", "comes", "came"), "poner": ("put", "puts", "put"),
    "salir": ("leave", "leaves", "left"), "seguir": ("follow", "follows",
                                                     "followed"),
    "conocer": ("know", "knows", "knew"), "traer": ("bring", "brings",
                                                    "brought"),
    "oír": ("hear", "hears", "heard"), "caer": ("fall", "falls", "fell"),
    "volver": ("return", "returns", "returned"),
    "pensar": ("think", "thinks", "thought"),
    "sentir": ("feel", "feels", "felt"),
    "encontrar": ("find", "finds", "found"),
    "empezar": ("start", "starts", "started"), "pedir": ("request", "requests",
                                                         "requested"),
    "contar": ("count", "counts", "counted"), "jugar": ("play", "plays",
                                                        "played"),
    "dormir": ("sleep", "sleeps", "slept"), "morir": ("die", "dies", "died"),
    "perder": ("lose", "loses", "lost"),
    "entender": ("understand", "understands", "understood"),
    "cerrar": ("close", "closes", "closed"), "mover": ("move", "moves",
                                                       "moved"),
    "servir": ("serve", "serves", "served"), "andar": ("walk", "walks",
                                                       "walked"),
    "hablar": ("speak", "speaks", "spoke"), "comer": ("eat", "eats", "ate"),
    "vivir": ("live", "lives", "lived"),
}

# Two verbs both mean "to have" and two both mean "to be". Without this the
# cue is the same sentence for both and there is no way to tell which word is
# wanted, which is the one deck error the collision test refuses to allow.
DISAMBIGUATE = {
    "ser": "(identity)", "estar": "(state, location)",
    "haber": "(helping verb)", "tener": "(to own)",
    "conocer": "(a person)", "saber": "(a fact)",
    "salir": "(to go out)", "andar": "(to go about)",
    "contar": "(to count, to tell)",
}

_BE = {"pres": ("am", "are", "are", "is", "are", "are"),
       "pret": ("was", "were", "were", "was", "were", "were")}


def _split(lemma):
    """Stem and conjugation class.

    The class is read with the accent removed: `oír` conjugates as an -ir
    verb, and its accent is a spelling device for the infinitive alone.
    """
    ending = lemma[-2:].replace("í", "i").replace("é", "e").replace("á", "a")
    return lemma[:-2], ending


def vosotros(lemma, tense):
    """The Spain-only plural "you" form.

    Mostly regular even where the rest of the verb is not: a stem change
    never reaches nosotros or vosotros, so `tener` gives `tenéis` while it
    gives `tienes` and `tienen`.
    """
    stem, ending = _split(lemma)
    if tense == "pres":
        return VOSOTROS_PRESENT.get(
            lemma, stem + VOSOTROS_PRESENT_ENDING[ending])
    if tense == "imp":
        return VOSOTROS_IMPERFECT.get(
            lemma, stem + ("abais" if ending == "ar" else "íais"))
    if tense == "fut":
        return IRREGULAR_STEM.get(lemma, lemma) + "éis"
    if tense == "cond":
        return IRREGULAR_STEM.get(lemma, lemma) + "íais"
    if lemma in VOSOTROS_PRETERITE:
        return VOSOTROS_PRETERITE[lemma]
    if lemma in STRONG_PRETERITE:
        return STRONG_PRETERITE[lemma] + "isteis"
    return stem + ("asteis" if ending == "ar" else "isteis")


def conjugate(lemma, tense):
    """Every form of one verb in one tense, in PERSONS order.

    vosotros is spliced in ahead of ellos rather than living in the tables,
    so the five universal forms stay written exactly as they are spoken
    everywhere and the Spain-only one is derived beside them.
    """
    base = _without_vosotros(lemma, tense)
    at = PERSONS.index("vos")
    return base[:at] + (vosotros(lemma, tense),) + base[at:]


def _without_vosotros(lemma, tense):
    stem, ending = _split(lemma)
    if tense == "pres":
        return PRESENT[lemma]
    if tense == "imp":
        if lemma in IMPERFECT:
            return IMPERFECT[lemma]
        return tuple(stem + e for e in REGULAR["imp"][ending])
    if tense in ("fut", "cond"):
        base = IRREGULAR_STEM.get(lemma, lemma)
        endings = FUTURE_ENDINGS if tense == "fut" else CONDITIONAL_ENDINGS
        return tuple(base + e for e in endings)
    if tense == "pret":
        if lemma in PRETERITE:
            return PRETERITE[lemma]
        if lemma in STRONG_PRETERITE:
            forms = [STRONG_PRETERITE[lemma] + e for e in STRONG_ENDINGS]
            if lemma == "hacer":
                forms[3] = "hizo"           # c -> z to keep the sound
            if lemma in ("decir", "traer"):
                forms[4] = STRONG_PRETERITE[lemma] + "eron"     # not -ieron
            return tuple(forms)
        forms = [stem + e for e in REGULAR["pret"][ending]]
        if lemma in PRETERITE_THIRD:
            forms[3], forms[4] = PRETERITE_THIRD[lemma]
        return tuple(forms)
    raise KeyError(tense)


def _english(lemma):
    entry = ENGLISH[lemma]
    base, third, past = entry[0], entry[1], entry[2]
    return base, third, past, (entry[3] if len(entry) > 3 else base)


def cue(lemma, tense, person):
    """The English prompt for one form.

    Every tense reads differently on purpose. "I had" alone is ambiguous
    between the finished past and the habitual one, and two cards with the
    same spoken cue and different answers is the one deck error the collision
    test refuses to allow. So the finished past says "yesterday" and the
    habitual one says "used to".

    Any disambiguating note goes last, always, so each cue has exactly one
    parenthetical and it reads as a single aside rather than a list.
    """
    base, third, past, nonfinite = _english(lemma)
    who = PRONOUN[person]
    if tense == "pres":
        word = (_BE["pres"][PERSONS.index(person)] if base == "be"
                else third if person == "el" else base)
        text = f"{who} {word}"
    elif tense == "pret":
        word = _BE["pret"][PERSONS.index(person)] if base == "be" else past
        text = f"{who} {word} yesterday"
    elif tense == "imp":
        text = f"{who} used to {nonfinite}"
    elif tense == "fut":
        text = f"{who} will {nonfinite}"
    else:
        text = f"{who} would {nonfinite}"
    note = DISAMBIGUATE.get(lemma, "")
    return f"{text} {note}".strip()


VERBS = tuple(PRESENT)          # every verb that has a written-out present


def forms_for(lemma, tense):
    """[(form_key, spanish, english_cue)] for one verb in one tense."""
    out = []
    for person, spanish in zip(PERSONS, conjugate(lemma, tense)):
        out.append((f"{tense}-{person}", spanish, cue(lemma, tense, person)))
    return out
