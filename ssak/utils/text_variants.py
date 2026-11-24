"""
Different ways to write the same words:
- join_acronyms ("S.N.C.F." -> "SNCF"), expand_titles ("Dr. Smith" -> "Doctor Smith"), expand_units ("10 km" -> "10 kilomètres")
- normalize_spelling: single spelling of the words whose spelling depends on the country, or on the spelling reform
- remove_disfluencies ("euh", "uh", "hmm", ...)
The case is kept (used by format_text, which then lowers the case if asked to).
"""

import re
from functools import cache

from ssak.utils.text_numbers import ONE_MARK, is_singular_number

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

# {symbols: (singular, plural)}
_UNITS = {
    "fr": {
        "km": ("kilomètre", "kilomètres"),
        "m": ("mètre", "mètres"),
        "cm": ("centimètre", "centimètres"),
        "mm": ("millimètre", "millimètres"),
        "kg": ("kilogramme", "kilogrammes"),
        "g": ("gramme", "grammes"),
        "mg": ("milligramme", "milligrammes"),
        "µg|μg|mcg": ("microgramme", "microgrammes"),
        "dl|dL": ("décilitre", "décilitres"),
        "cl|cL": ("centilitre", "centilitres"),
        "ml|mL": ("millilitre", "millilitres"),
        "ha": ("hectare", "hectares"),
        "min": ("minute", "minutes"),
        "°C": ("degré Celsius", "degrés Celsius"),
        "°F": ("degré Fahrenheit", "degrés Fahrenheit"),
        "°": ("degré", "degrés"),
    },
    "en": {
        "km": ("kilometer", "kilometers"),
        "cm": ("centimeter", "centimeters"),
        "mm": ("millimeter", "millimeters"),
        "kg": ("kilogram", "kilograms"),
        "mg": ("milligram", "milligrams"),
        "µg|μg|mcg": ("microgram", "micrograms"),
        "dl|dL": ("deciliter", "deciliters"),
        "cl|cL": ("centiliter", "centiliters"),
        "ml|mL": ("milliliter", "milliliters"),
        "mph": ("mile per hour", "miles per hour"),
        "lbs|lb": ("pound", "pounds"),
        "ft": ("foot", "feet"),
        "min": ("minute", "minutes"),
        "°C": ("degree Celsius", "degrees Celsius"),
        "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
        "°": ("degree", "degrees"),
    },
    "es": {
        "km": ("kilómetro", "kilómetros"),
        "m": ("metro", "metros"),
        "cm": ("centímetro", "centímetros"),
        "mm": ("milímetro", "milímetros"),
        "kg": ("kilogramo", "kilogramos"),
        "g": ("gramo", "gramos"),
        "mg": ("miligramo", "miligramos"),
        "µg|μg|mcg": ("microgramo", "microgramos"),
        "l|L": ("litro", "litros"),
        "dl|dL": ("decilitro", "decilitros"),
        "cl|cL": ("centilitro", "centilitros"),
        "ml|mL": ("mililitro", "mililitros"),
        "ha": ("hectárea", "hectáreas"),
        "min": ("minuto", "minutos"),
        "°C": ("grado Celsius", "grados Celsius"),
    },
    "de": {
        "km": ("Kilometer", "Kilometer"),
        "m": ("Meter", "Meter"),
        "cm": ("Zentimeter", "Zentimeter"),
        "mm": ("Millimeter", "Millimeter"),
        "kg": ("Kilogramm", "Kilogramm"),
        "g": ("Gramm", "Gramm"),
        "mg": ("Milligramm", "Milligramm"),
        "µg|μg|mcg": ("Mikrogramm", "Mikrogramm"),
        "l|L": ("Liter", "Liter"),
        "dl|dL": ("Deziliter", "Deziliter"),
        "cl|cL": ("Zentiliter", "Zentiliter"),
        "ml|mL": ("Milliliter", "Milliliter"),
        "ha": ("Hektar", "Hektar"),
        "min": ("Minute", "Minuten"),
        "°C": ("Grad Celsius", "Grad Celsius"),
    },
    "it": {
        "km": ("chilometro", "chilometri"),
        "m": ("metro", "metri"),
        "cm": ("centimetro", "centimetri"),
        "mm": ("millimetro", "millimetri"),
        "kg": ("chilogrammo", "chilogrammi"),
        "g": ("grammo", "grammi"),
        "mg": ("milligrammo", "milligrammi"),
        "µg|μg|mcg": ("microgrammo", "microgrammi"),
        "l|L": ("litro", "litri"),
        "dl|dL": ("decilitro", "decilitri"),
        "cl|cL": ("centilitro", "centilitri"),
        "ml|mL": ("millilitro", "millilitri"),
        "ha": ("ettaro", "ettari"),
        "min": ("minuto", "minuti"),
        "°C": ("grado Celsius", "gradi Celsius"),
    },
    "pt": {
        "km": ("quilómetro", "quilómetros"),
        "m": ("metro", "metros"),
        "cm": ("centímetro", "centímetros"),
        "mm": ("milímetro", "milímetros"),
        "kg": ("quilograma", "quilogramas"),
        "g": ("grama", "gramas"),
        "mg": ("miligrama", "miligramas"),
        "µg|μg|mcg": ("micrograma", "microgramas"),
        "l|L": ("litro", "litros"),
        "dl|dL": ("decilitro", "decilitros"),
        "cl|cL": ("centilitro", "centilitros"),
        "ml|mL": ("mililitro", "mililitros"),
        "ha": ("hectare", "hectares"),
        "min": ("minuto", "minutos"),
        "°C": ("grau Celsius", "graus Celsius"),
    },
    "nl": {
        "km": ("kilometer", "kilometer"),
        "m": ("meter", "meter"),
        "cm": ("centimeter", "centimeter"),
        "mm": ("millimeter", "millimeter"),
        "kg": ("kilogram", "kilogram"),
        "g": ("gram", "gram"),
        "mg": ("milligram", "milligram"),
        "µg|μg|mcg": ("microgram", "microgram"),
        "l|L": ("liter", "liter"),
        "dl|dL": ("deciliter", "deciliter"),
        "cl|cL": ("centiliter", "centiliter"),
        "ml|mL": ("milliliter", "milliliter"),
        "ha": ("hectare", "hectare"),
        "min": ("minuut", "minuten"),
        "°C": ("graad Celsius", "graden Celsius"),
    },
}

# Squares and cubes of the lengths
_POWERS = {
    "fr": ("{} carré", "{} carrés", "{} cube", "{} cubes"),
    "en": ("square {}", "square {}", "cubic {}", "cubic {}"),
    "es": ("{} cuadrado", "{} cuadrados", "{} cúbico", "{} cúbicos"),
    "it": ("{} quadrato", "{} quadrati", "{} cubo", "{} cubi"),
    "pt": ("{} quadrado", "{} quadrados", "{} cúbico", "{} cúbicos"),
    "de": ("Quadrat{}", "Quadrat{}", "Kubik{}", "Kubik{}"),
    "nl": ("vierkante {}", "vierkante {}", "kubieke {}", "kubieke {}"),
}
_METER = {"fr": ("mètre", "mètres"), "en": ("meter", "meters")}

# Units that are too ambiguous to be converted, except before "/" and another unit ("3x/j", "1 cp/j", "2 h/jour")
_UNITS_BEFORE_SLASH = {
    "fr": {"l|L": ("litre", "litres"), "h": ("heure", "heures"), "j": ("jour", "jours"), "cp|cps|cpr": ("comprimé", "comprimés"), "x": ("fois", "fois")},
    "en": {
        "m": ("meter", "meters"),
        "g": ("gram", "grams"),
        "l|L": ("liter", "liters"),
        "h|hr|hrs": ("hour", "hours"),
        "mi": ("mile", "miles"),
        "oz": ("ounce", "ounces"),
        "x": ("time", "times"),
    },
    "es": {"h": ("hora", "horas"), "x": ("vez", "veces")},
    "de": {"h": ("Stunde", "Stunden"), "x": ("Mal", "Mal")},
    "it": {"h": ("ora", "ore"), "x": ("volta", "volte")},
    "pt": {"h": ("hora", "horas"), "x": ("vez", "vezes")},
    "nl": {"h|u": ("uur", "uur"), "x": ("keer", "keer")},
}
# Words that can be before "/" and a unit ("2 comprimés/jour"), besides the names of the units
_WORDS_BEFORE_SLASH = {
    "fr": "fois|comprimés?|gélules?|capsules?|sachets?|gouttes?|doses?|ampoules?|prises?|injections?|bouffées?",
    "en": "times?|tablets?|pills?|capsules?|drops?|doses?|puffs?|injections?",
    "es": "ve(?:z|ces)|comprimidos?|cápsulas?|gotas?|dosis|sobres?|inyección|inyecciones",
    "de": "[Mm]al|Tabletten?|Kapseln?|Tropfen|Dosis|Dosen",
    "it": "volt[ae]|compress[ae]|capsul[ae]|gocc(?:ia|e)|dos[ei]",
    "pt": "ve(?:z|zes)|comprimidos?|cápsulas?|gotas?|doses?",
    "nl": "keer|tablet(?:ten)?|capsules?|druppels?|dosis|doses",
}
# After "/", besides the units above
_AFTER_SLASH = {
    "fr": {
        "j|jours?": "par jour",
        "h|heures?": "par heure",
        "s|secondes?": "par seconde",
        "minutes?": "par minute",
        "sem|semaines?": "par semaine",
        "mois": "par mois",
        "ans?|année": "par an",
        "l|L|litre": "par litre",
        "personne": "par personne",
        "kilo": "par kilo",
    },
    "en": {
        "d|days?": "per day",
        "h|hr|hours?": "per hour",
        "s|sec|seconds?": "per second",
        "minutes?": "per minute",
        "wk|weeks?": "per week",
        "months?": "per month",
        "yr|years?": "per year",
        "g": "per gram",
        "l|L|liter|litre": "per liter",
        "m": "per meter",
        "night": "per night",
        "share": "per share",
        "person": "per person",
        "unit": "per unit",
        "bbl|barrel": "per barrel",
        "ton": "per ton",
        "oz|ounce": "per ounce",
        "mile": "per mile",
    },
    "es": {
        "d|días?|dias?": "por día",
        "h|horas?": "por hora",
        "s|segundos?": "por segundo",
        "minutos?": "por minuto",
        "sem|semanas?": "por semana",
        "mes": "por mes",
        "años?": "por año",
        "persona": "por persona",
    },
    "de": {
        "d|Tag": "pro Tag",
        "h|Std|Stunde": "pro Stunde",
        "s|Sekunde": "pro Sekunde",
        "Minute": "pro Minute",
        "Woche": "pro Woche",
        "Monat": "pro Monat",
        "Jahr": "pro Jahr",
        "Person": "pro Person",
    },
    "it": {
        "die|dì|giorno": "al giorno",
        "h|ora": "all'ora",
        "s|secondo": "al secondo",
        "min|minuto": "al minuto",
        "settimana": "alla settimana",
        "mese": "al mese",
        "anno": "all'anno",
        "persona": "a persona",
    },
    "pt": {
        "d|dias?": "por dia",
        "h|horas?": "por hora",
        "s|segundos?": "por segundo",
        "minutos?": "por minuto",
        "sem|semanas?": "por semana",
        "mês": "por mês",
        "anos?": "por ano",
        "pessoa": "por pessoa",
    },
    "nl": {
        "d|dag": "per dag",
        "h|u|uur": "per uur",
        "s|seconde": "per seconde",
        "minuut": "per minuut",
        "week": "per week",
        "maand": "per maand",
        "jaar": "per jaar",
        "persoon": "per persoon",
    },
}
_PER = {"fr": "par", "en": "per", "es": "por", "de": "pro", "it": "per", "pt": "por", "nl": "per"}

# "24h/24", "7j/7"
_AROUND_THE_CLOCK = {
    "fr": [(r"h|heures?", "24", "heures sur"), (r"j|jours?", "7", "jours sur")],
    "it": [(r"h|ore", "24", "ore su"), (r"g|gg|giorni", "7", "giorni su")],
}

# Forms of "one" before a unit
_ONE_BEFORE_MASCULINE = {"es": ("uno", "un"), "it": ("uno", "un"), "de": ("eins", "ein")}
_ONE_BEFORE_FEMININE = {"fr": ("un", "une"), "es": ("uno", "una"), "it": ("uno", "una"), "de": ("eins", "eine"), "pt": ("um", "uma")}
_FEMININE_UNITS = {
    "fr": "livre|heure|minute|gélule|capsule|goutte|dose|ampoule|prise|injection|bouffée",
    "es": "libra|hora|hectárea|vez|cápsula|gota|dosis",
    "it": "sterlina|ora|volta|compressa|capsula|goccia|dose",
    "de": "Stunde|Minute|Tablette|Kapsel|Dosis",
    "pt": "libra|hora|vez|cápsula|gota|dose",
}


def _expand_symbols(units):
    return {symbol: names for symbols, names in units.items() for symbol in symbols.split("|")}


@cache
def _unit_tables(lang):
    """units after a number, units before a slash, {unit after a slash: words}"""
    units = _expand_symbols(_UNITS[lang])
    for prefix in "", "c", "m", "k":
        names = units.get(prefix + "m") if prefix else units.get("m", _METER.get(lang))
        square, squares, cube, cubes = _POWERS[lang]
        for power, singular, plural in ("2", square, squares), ("3", cube, cubes):
            for symbol in power, {"2": "²", "3": "³"}[power]:
                units[f"{prefix}m{symbol}"] = (singular.format(names[0].lower()), plural.format(names[1].lower())) if lang == "de" else (singular.format(names[0]), plural.format(names[1]))
    before = units | _expand_symbols(_UNITS_BEFORE_SLASH[lang])
    after = {re.escape(symbol): f"{_PER[lang]} {names[0]}" for symbol, names in units.items() if "ph" not in symbol}
    after |= {re.escape(names[0]): f"{_PER[lang]} {names[0]}" for symbol, names in units.items() if "ph" not in symbol}
    for patterns, words in _AFTER_SLASH[lang].items():
        after |= dict.fromkeys(patterns.split("|"), words)
    return units, before, after


def _alternatives(patterns):
    return "|".join(sorted(patterns, key=len, reverse=True))


_NUMBER = r"(?<![\w.,])(\d+/\d+|\d+(?:[    ]\d{3})*(?:[.,]\d+)*)"
_NOT_IN_WORD = r"(?![\w²³'’-])"


@cache
def _unit_patterns(lang):
    units, before, after = _unit_tables(lang)
    after = _alternatives(after)
    words = _alternatives({re.escape(n) for names in before.values() for n in names} | {_WORDS_BEFORE_SLASH[lang]})
    slashes = rf"((?:\s?/\s?(?:{after}){_NOT_IN_WORD})+)"
    single = [u for u in units if len(u) == 1]
    return {
        "ratio": re.compile(rf"{_NUMBER}(\s?(?:{_alternatives(map(re.escape, before))})|\s(?:{words})){slashes}"),
        "price": re.compile(rf"([$€£]\s?\d[\d.,]*(?:\s?(?:[kKMB]|bn|million|billion)\b)?|\d[\d.,]*\s?[$€£]){slashes}"),
        "unit": re.compile(rf"{_NUMBER}\s?({_alternatives(map(re.escape, units))}){_NOT_IN_WORD}"),
        "single": single,
        "after": re.compile(rf"/\s?({after}){_NOT_IN_WORD}"),
        "after_words": {re.compile(p): w for p, w in _unit_tables(lang)[2].items()},
    }


def expand_units(text, lang):
    """
    "10 km" -> "10 kilomètres", "1 km" -> "1 kilomètre"
    "3 mg/j" -> "3 milligrammes par jour", "5 €/m²" -> "5 € par mètre carré" (only between a number with a unit, and a unit)
    "24h/24" -> "24 heures sur 24"
    """
    if lang not in _UNITS:
        return text
    units, before, _ = _unit_tables(lang)
    patterns = _unit_patterns(lang)

    def number_and_name(number, names):
        # Singular after one (French: below two), or a fraction ("1/2 km")
        if "/" in number:
            numerator, denominator = map(int, number.split("/"))
            is_singular = numerator <= denominator
        else:
            is_singular = is_singular_number(number, lang)
        return f"{number}{ONE_MARK if number == '1' else ''} {names[0 if is_singular else 1]}"

    def per(slashes):
        def words(match):
            return " " + next(w for p, w in patterns["after_words"].items() if p.fullmatch(match.group(1)))

        return re.sub(r"\s+", " ", patterns["after"].sub(words, slashes))

    def ratio(match):
        number, unit, slashes = match.groups()
        unit = unit.strip()
        # ("une" for "1 goutte/jour")
        names = before.get(unit, (unit, unit))
        return number_and_name(number, names) + per(slashes)

    def simple(match):
        number, unit = match.groups()
        # "m/..." with something that is not a unit
        if unit in patterns["single"] and re.match(r"\s?/\s?[^\W\d]", text[match.end() :]):
            return match.group()
        return number_and_name(number, units[unit])

    for unit, total, words in _AROUND_THE_CLOCK.get(lang, []):
        text = re.sub(rf"(?<![\w.,/])(\d+)\s?(?:{unit})\s?/\s?{total}(?:\s?(?:{unit}))?(?![\w/])", rf"\1 {words} {total}", text)
    text = patterns["ratio"].sub(ratio, text)
    text = patterns["price"].sub(lambda m: m.group(1) + per(m.group(2)), text)
    return patterns["unit"].sub(simple, text)


def agree_one(text, lang):
    """ "uno kilómetro" -> "un kilómetro", "un heure" -> "une heure" (only for the numbers one with the mark)"""
    if ONE_MARK not in text:
        return text

    def agree(match):
        one, noun = match.groups()
        feminine = re.fullmatch(_FEMININE_UNITS.get(lang, "(?!)"), noun)
        word, replacement = (_ONE_BEFORE_FEMININE if feminine else _ONE_BEFORE_MASCULINE).get(lang, (None, None))
        return f"{_keep_case(one, replacement) if one.lower() == word else one} {noun}"

    text = re.sub(rf"(\w+)\s*{ONE_MARK}\s*(\w+)", agree, text)
    return re.sub(rf"\s*{ONE_MARK}", "", text)


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

# Portuguese: European spelling
_STEMS_PT = {
    "recepç": "receç",
    "concepç": "conceç",
    "percepç": "perceç",
    "decepç": "deceç",
    "contracepç": "contraceç",
    "registr": "regist",
    "umid": "humid",
    "indeniz": "indemniz",
    "anisti": "amnisti",
}
_WORDS_PT = {
    "dezesseis": "dezasseis",
    "dezessete": "dezassete",
    "dezenove": "dezanove",
    "quatorze": "catorze",
    "contato": "contacto",
    "contatos": "contactos",
    "cotidiano": "quotidiano",
    "cotidiana": "quotidiana",
    "úmido": "húmido",
    "úmida": "húmida",
    "úmidos": "húmidos",
    "úmidas": "húmidas",
    "súdito": "súbdito",
    "sutil": "subtil",
    "sutis": "subtis",
    "bebê": "bebé",
    "bebês": "bebés",
    "comitê": "comité",
    "comitês": "comités",
    "purê": "puré",
    "judô": "judo",
}
# "econômico" -> "económico", "gênero" -> "género" (but not "fêmea", "estômago", that are the same in Portugal)
_PATTERN_PT = r"\b(?!(?:fême[ao]s?|sêmeas?|estômagos?)\b)(\w*)([ôê])(?=[mn][aeiouáéíóúâêôãõ])"


@cache
def _spelling_rules(lang):
    if lang == "fr":
        stems, words = _STEMS_FR, _WORDS_FR
    elif lang == "en":
        stems, words = {}, _WORDS_EN
    elif lang == "pt":
        stems, words = _STEMS_PT, _WORDS_PT
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
    Portuguese: European spelling ("econômico" -> "económico", "dezesseis" -> "dezasseis", "contato" -> "contacto")
    English: American spelling ("colour" -> "color", "organise" -> "organize", "travelled" -> "traveled")
    """
    stems_pattern, words_pattern, replacements = _spelling_rules(lang)
    for pattern in stems_pattern, words_pattern:
        if pattern is not None:
            text = pattern.sub(lambda m: _keep_case(m.group(), replacements[m.group().lower()]), text)
    if lang == "fr":
        text = re.sub(_VERBS_FR, r"\1i\2", text, flags=re.IGNORECASE)
    if lang == "pt":
        text = re.sub(_PATTERN_PT, lambda m: m.group(1) + {"ô": "ó", "ê": "é", "Ô": "Ó", "Ê": "É"}[m.group(2)], text, flags=re.IGNORECASE)
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
