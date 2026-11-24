"""
Different ways to write the same words:
- join_acronyms ("S.N.C.F." -> "SNCF"), expand_titles ("Dr. Smith" -> "Doctor Smith"), expand_units ("10 km" -> "10 kilomètres")
- normalize_spelling: single spelling of the words whose spelling depends on the country, or on the spelling reform
- remove_disfluencies ("euh", "uh", "hmm", ...)
The case is kept (used by format_text, which then lowers the case if asked to).
"""

import re
from functools import cache

# ------------------------------------------------------------------------------------
# Acronyms


def join_acronyms(text):
    """ "S.N.C.F." -> "SNCF", "U.S." -> "US" (a dot at the end of the text also ends the sentence, and is kept)"""

    def join(match):
        acronym = match.group()
        end = "." if acronym.endswith(".") and not text[match.end() :].strip() else ""
        return acronym.replace(".", "") + end

    return re.sub(r"(?<![\w.])[^\W\d_a-zà-öø-ÿ]\.(?:[^\W\d_a-zà-öø-ÿ]\.)*(?:[^\W\d_a-zà-öø-ÿ]\.|[^\W\d_a-zà-öø-ÿ](?![\w.]))", join, text)


# ------------------------------------------------------------------------------------
# Titles and other abbreviations

_capitalized = r"(?=\s+[^\W\d_a-zà-öø-ÿ])"
_TITLES = {
    "fr": [
        (r"\bMM\." + _capitalized, "messieurs"),
        (r"\bM\." + _capitalized, "monsieur"),
        (r"\bMr\.?" + _capitalized, "monsieur"),
        (r"\bDr\.?" + _capitalized, "docteur"),
        (r"\bPr\.?" + _capitalized, "professeur"),
        (r"\bMme\.?" + _capitalized, "madame"),
        (r"\bMlle\.?" + _capitalized, "mademoiselle"),
    ],
    "en": [
        (r"\bMr\.?" + _capitalized, "Mister"),
        (r"\bMrs\.?" + _capitalized, "Missus"),
        (r"\bMs\.?" + _capitalized, "Miss"),
        (r"\bDr\.?" + _capitalized, "Doctor"),
        (r"\bProf\.?" + _capitalized, "Professor"),
        (r"\b[Vv]s\b\.?", "versus"),
    ],
}


def expand_titles(text, lang):
    for pattern, replacement in _TITLES.get(lang, []):
        text = re.sub(pattern, replacement, text)
    return text


# ------------------------------------------------------------------------------------
# Units after a number

# {unit: (singular, plural)}
_UNITS = {
    "fr": {
        "km/h": ("kilomètre par heure", "kilomètres par heure"),
        "m/s": ("mètre par seconde", "mètres par seconde"),
        "km": ("kilomètre", "kilomètres"),
        "m": ("mètre", "mètres"),
        "cl": ("centilitre", "centilitres"),
        "ha": ("hectare", "hectares"),
        "min": ("minute", "minutes"),
    },
    "en": {
        "km/h": ("kilometer per hour", "kilometers per hour"),
        "mph": ("mile per hour", "miles per hour"),
        "km": ("kilometer", "kilometers"),
        "cm": ("centimeter", "centimeters"),
        "mm": ("millimeter", "millimeters"),
        "kg": ("kilogram", "kilograms"),
        "mg": ("milligram", "milligrams"),
        "ml": ("milliliter", "milliliters"),
        "lbs": ("pound", "pounds"),
        "lb": ("pound", "pounds"),
        "ft": ("foot", "feet"),
    },
}


def expand_units(text, lang):
    """ "10 km" -> "10 kilomètres", "1 km" -> "1 kilomètre" """
    units = _UNITS.get(lang)
    if not units:
        return text

    def expand(match):
        number, unit = match.groups()
        value = float(re.sub(r"[   ]", "", number).replace(",", "." if lang == "fr" else ""))
        singular = value < 2 if lang == "fr" else value == 1
        return f"{number} {units[unit][0 if singular else 1]}"

    names = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    return re.sub(r"(?<![\w.,])(\d+(?:[   ]\d{3})*(?:[.,]\d+)?)\s?(" + names + r")(?![\w²³/'’-])", expand, text)


# ------------------------------------------------------------------------------------
# Spelling


def _keep_case(original, replacement):
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# French: spelling reform of 1990 (beginnings of words, followed by any ending)
_STEMS_FR = {
    # Circumflex on i and u
    "maître": "maitre",
    "maîtri": "maitri",
    "contremaître": "contremaitre",
    "boît": "boit",
    "déboît": "déboit",
    "emboît": "emboit",
    "chaîn": "chain",
    "enchaîn": "enchain",
    "déchaîn": "déchain",
    "traîn": "train",
    "entraîn": "entrain",
    "fraîch": "fraich",
    "rafraîch": "rafraich",
    "défraîch": "défraich",
    "aîné": "ainé",
    "abîm": "abim",
    "cloîtr": "cloitr",
    "épître": "épitre",
    "gîte": "gite",
    "huître": "huitre",
    "traîtr": "traitr",
    "dîn": "din",
    "goût": "gout",
    "dégoût": "dégout",
    "ragoût": "ragout",
    "coût": "cout",
    "surcoût": "surcout",
    "voût": "vout",
    "envoût": "envout",
    "croût": "crout",
    "brûl": "brul",
    "bûch": "buch",
    "embûche": "embuche",
    "flût": "flut",
    "affût": "affut",
    "piqûre": "piqure",
    "dûment": "dument",
    "assidûment": "assidument",
    "îlot": "ilot",
    # Accents
    "événement": "évènement",
    "réglement": "règlement",
    "allégement": "allègement",
    "crémerie": "crèmerie",
    "céleri": "cèleri",
    "sécheresse": "sècheresse",
    # Others
    "ambiguï": "ambigüi",
    "imbécillité": "imbécilité",
}
# French: whole words
_WORDS_FR = {
    "île": "ile",
    "îles": "iles",
    "presqu'île": "presqu'ile",
    "août": "aout",
    "surcroît": "surcroit",
    "aiguë": "aigüe",
    "aiguës": "aigües",
    "ambiguë": "ambigüe",
    "ambiguës": "ambigües",
    "exiguë": "exigüe",
    "exiguës": "exigües",
    "week-end": "weekend",
    "week-ends": "weekends",
    "e-mail": "email",
    "e-mails": "emails",
    "ok": "okay",
    "charriot": "chariot",
    "charriots": "chariots",
    "serpillère": "serpillière",
    "serpillères": "serpillières",
}
# French: "connaître", "connaît" (naître, paraître, ... and their compounds), "plaît"
_VERBS_FR = r"\b((?:re|mé)?(?:con)?na|(?:ap|com|dis|trans)?para|repara|réappara|(?:dé|com)?pla)î(t(?:re|rai|ras|ra|rons|rez|ront|rais|rait|rions|riez|raient)?)\b"

# English: American spelling
_OUR_EN = ("arm ard behavi cand clam col demean endeav fav ferv flav harb hon hum lab neighb od parl ranc rig rum sav savi splend succ tum val vap vig").split()
_ISE_EN = (
    "amort anonym apolog author capital categor central character civil colon commercial computer critic crystall custom digit econom "
    "emphas energ equal familiar fertil final formal general global harmon hospital hypothes ideal immun incentiv industrial initial "
    "item jeopard legal liberal local material maxim mechan memor minim mobil modern monet national natural neutral normal optim "
    "organ patron penal personal polar popular pressur priorit privat public rational real recogn reorgan revolution scrutin sensit "
    "social special stabil standard stigmat subsid summar symbol sympath synchron terror theor tranquill trivial urban util vandal victim visual"
).split()
_DOUBLE_L_EN = "cancel channel counsel dial duel enamel equal fuel grovel initial jewel label level libel marvel model panel pedal quarrel refuel rival shovel signal spiral total travel tunnel unravel yodel".split()
_WORDS_EN = {
    "grey": "gray",
    "greys": "grays",
    "programme": "program",
    "programmes": "programs",
    "cheque": "check",
    "cheques": "checks",
    "tyre": "tire",
    "tyres": "tires",
    "aluminium": "aluminum",
    "plough": "plow",
    "ploughs": "plows",
    "pyjamas": "pajamas",
    "ageing": "aging",
    "judgement": "judgment",
    "judgements": "judgments",
    "acknowledgement": "acknowledgment",
    "acknowledgements": "acknowledgments",
    "cosy": "cozy",
    "moustache": "mustache",
    "mould": "mold",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "analogue": "analog",
    "defence": "defense",
    "offence": "offense",
    "offences": "offenses",
    "licence": "license",
    "licences": "licenses",
    "pretence": "pretense",
    "practise": "practice",
    "practised": "practiced",
    "practising": "practicing",
    "jewellery": "jewelry",
    "enrolment": "enrollment",
    "fulfil": "fulfill",
    "fulfils": "fulfills",
    "skilful": "skillful",
    "wilful": "willful",
    "instalment": "installment",
    "instalments": "installments",
    "woollen": "woolen",
    "marvellous": "marvelous",
    "manoeuvre": "maneuver",
    "manoeuvres": "maneuvers",
    "foetus": "fetus",
    "diarrhoea": "diarrhea",
    "oestrogen": "estrogen",
    "encyclopaedia": "encyclopedia",
    "mediaeval": "medieval",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "analyser": "analyzer",
    "paralyse": "paralyze",
    "paralysed": "paralyzed",
    "sceptic": "skeptic",
    "sceptics": "skeptics",
    "sceptical": "skeptical",
    "scepticism": "skepticism",
    "centre": "center",
    "centres": "centers",
    "centred": "centered",
    "theatre": "theater",
    "theatres": "theaters",
    "fibre": "fiber",
    "fibres": "fibers",
    "calibre": "caliber",
    "spectre": "specter",
    "sombre": "somber",
    "e-mail": "email",
    "e-mails": "emails",
    "ok": "okay",
    "weekend": "weekend",
    "week-end": "weekend",
}
_PATTERNS_EN = [
    (r"\b((?:dis|un|mis)?(?:" + "|".join(_OUR_EN) + r"))our((?:s|ed|ing|ings|able|ably|ful|less|ite|ites|er|ers|ly|hood|hoods|al|ist|ists|y|some)?)\b", r"\1or\2"),
    (r"\b((?:re|de|un|over)?(?:" + "|".join(_ISE_EN) + r"))is(e|es|ed|er|ers|ing|ation|ations|ational|able)\b", r"\1iz\2"),
    (r"\b(" + "|".join(_DOUBLE_L_EN) + r")l(ed|ing|er|ers|or|ors|ings)\b", r"\1\2"),
    (r"\b((?:kilo|centi|milli|deci|micro)?met|(?:milli|centi|deci)?lit)re(s?)\b", r"\1er\2"),
    (r"\b(p)aed(iatric\w*|ophil\w*)\b", r"\1ed\2"),
    (r"\b(orthop)aed(ic\w*)\b", r"\1ed\2"),
    (r"\b(an)aesth(\w+)\b", r"\1esth\2"),
    (r"\b(h)aemo(\w+)\b", r"\1emo\2"),
    (r"\b(gyn)aeco(\w+)\b", r"\1eco\2"),
    (r"\b(an|leuk)aemi(a|as|c)\b", r"\1emi\2"),
]


@cache
def _spelling_rules(lang):
    if lang == "fr":
        stems, words = _STEMS_FR, _WORDS_FR
    elif lang == "en":
        stems, words = {}, _WORDS_EN
    else:
        return None, None, {}
    boundary, end = r"(?<![\w-])", r"(?![\w-]|'(?=\w))" if lang == "en" else r"(?![\w-])"
    stems_pattern = re.compile(r"\b(" + "|".join(sorted(stems, key=len, reverse=True)) + ")", re.IGNORECASE) if stems else None
    # The words with a dash are looked for first
    words_pattern = re.compile(boundary + "(" + "|".join(sorted(words, key=len, reverse=True)) + ")" + end, re.IGNORECASE)
    return stems_pattern, words_pattern, stems | words


def normalize_spelling(text, lang):
    """
    French: spelling reform ("événement" -> "évènement", "maître" -> "maitre", "week-end" -> "weekend")
    English: American spelling ("colour" -> "color", "organise" -> "organize", "travelled" -> "traveled")
    """
    stems_pattern, words_pattern, replacements = _spelling_rules(lang)
    for pattern in stems_pattern, words_pattern:
        if pattern is not None:
            text = pattern.sub(lambda m: _keep_case(m.group(), replacements[m.group().lower()]), text)
    if lang == "fr":
        text = re.sub(_VERBS_FR, r"\1i\2", text, flags=re.IGNORECASE)
    if lang == "en":
        for pattern, replacement in _PATTERNS_EN:
            text = re.sub(pattern, lambda m: _keep_case(m.group(), m.expand(replacement).lower()), text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------------------------
# Disfluencies

# Only the sounds that are not words of the language ("um" is a word in German and in Portuguese),
# and that do not mean anything (not "mhm" or "uh-huh", that mean yes)
_DISFLUENCIES = {
    "fr": r"e+u+h+m*|h+e+u+h*|h+u+m+|h+m+|m+h+|m{2,}",
    "en": r"u+h+|u+h+m+|u+m+|e+r+m+|h+m+|h+u+m{2,}|m{2,}",
    "es": r"e+h+m+|h+m+|m{2,}",
    "de": r"ä+h+m*|ö+h+m*|h+m+|m{2,}",
    "it": r"e+h+m+|h+m+|m{2,}",
    "pt": r"h+m+|m{2,}|ã+h+",
    "nl": r"e+h+m+|u+h+m*|h+m+|m{2,}",
}


@cache
def _disfluencies_pattern(lang):
    if lang not in _DISFLUENCIES:
        return None
    # With the spaces and the comma that go with it
    return re.compile(rf"(?P<before>,?\s*)(?<![\w'’-])(?:{_DISFLUENCIES[lang]})(?![\w-])(?P<after>\s*(?:,|\.{{3}}|…)?)", re.IGNORECASE)


def remove_disfluencies(text, lang):
    """
    "Euh, le matin" -> "Le matin", "le matin, euh, et le soir" -> "le matin, et le soir", "So uh we" -> "So we"
    """
    pattern = _disfluencies_pattern(lang)
    if pattern is None or not pattern.search(text):
        return text
    capital = re.match(r"\s*\W*[^\W\d_]", text)
    capital = bool(capital) and capital.group()[-1].isupper()

    def remove(match):
        # One comma when it was between two
        return ", " if "," in match.group("before") and "," in match.group("after") else " "

    text = " " + text + " "
    previous = None
    while previous != text:
        previous, text = text, pattern.sub(remove, text)
    text = re.sub(r"\s+([.,])", r"\1", text)  # "le matin, euh." -> "le matin."
    text = re.sub(r"(^|[.?!])\s*[.,]+", r"\1", text)
    text = re.sub(r" {2,}", " ", text).strip()
    if capital and text:
        text = re.sub(r"^(\W*)([^\W\d_])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text
