"""
Numbers in texts:
- clean_number_formats, currencies_after_numbers: make the ways of writing numbers in digits non ambiguous
  before converting them into letters (used by numbers_and_symbols_to_letters)
- normalize_spoken_numbers: single way to write numbers in letters, to compare texts (used by compute_wer)
"""

import re
from functools import lru_cache

# Languages where the decimal mark is the point (and the comma separates the thousands)
_DECIMAL_POINT_LANGUAGES = ["en"]
# Languages where a single group of 3 digits after a point is for thousands ("1.500" = 1500).
# Not in French, where the usual separator is a space, and where "1.500" can be 1,500.
_THOUSANDS_POINT_LANGUAGES = ["es", "de", "it", "pt", "nl"]


# Space added around the words of a conversion, only kept where it is needed (see clean_padding)
PADDING = "\ue0f0"


def clean_padding(text):
    """Replace the padding by a space, except next to another space, at the ends of the text, before a closing punctuation and after an opening one"""
    if PADDING not in text:
        return text
    p = PADDING + "+"
    text = re.sub(rf"^{p}|{p}$|(?<=[\s(\[{{'’]){p}|{p}(?=[\s.,;:!?…)\]}}])", "", text)
    return re.sub(p, " ", text)


def base_language(lang):
    """ "fr-FR" -> "fr" (but "ar_tn" -> "ar_tn")"""
    return lang.split("-")[0]


# ------------------------------------------------------------------------------------
# Before the conversion of digits into letters


def clean_number_formats(text, lang):
    """
    Numbers, hours, ... written in a non ambiguous way.
    "COVID-19" -> "COVID 19" (not "minus 19"), "n°5" -> "numéro 5", "14:30" -> "14 heures 30", "1.234,56" -> "1234,56",
    "1990s" -> "nineteen nineties", ...
    """
    # A dash after a letter is not a minus
    text = re.sub(r"(?<=[^\W\d_])-(?=\d)", " ", text)

    # Both separators: the last one is the decimal mark
    mark = "." if lang in _DECIMAL_POINT_LANGUAGES else ","
    text = re.sub(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+),(\d+)(?![.,]?\d)", lambda m: m.group(1).replace(".", "") + mark + m.group(2), text)
    text = re.sub(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)\.(\d+)(?![.,]?\d)", lambda m: m.group(1).replace(",", "") + mark + m.group(2), text)

    # Thousands
    if lang in _DECIMAL_POINT_LANGUAGES:
        text = re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!,?\d)", lambda m: m.group().replace(",", ""), text)
    elif lang in _THOUSANDS_POINT_LANGUAGES:
        text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?![.,]?\d)", lambda m: m.group().replace(".", ""), text)
    else:
        text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3}){2}(?![.,]?\d)", lambda m: m.group().replace(".", ""), text)

    if lang == "fr":
        text = re.sub(r"\b[nN][°º]\s*(?=\d)", "numéro ", text)
        # Hours
        text = re.sub(r"(?<![\d:,.])([01]?\d|2[0-4]):([0-5]\d)(?![\d:])", r"\1h\2", text)
        text = re.sub(r"(?<![\d,.])([01]?\d|2[0-4])\s?h\s?([0-5]\d)?(?![\w])", _french_hour, text)
    elif lang == "en":
        text = re.sub(r"\bNo\.\s*(?=\d)", "number ", text)
        text = re.sub(r"(?<!\w)#(?=\d)", "number ", text)
        text = re.sub(r"(?<![\d/])24/7(?![\d/])", "24 7", text)
        # Hours
        text = re.sub(r"(?<![\d:,.])([01]?\d|2[0-4]):([0-5]\d)(?![\d:])", _english_hour, text)
        # Decades
        text = re.sub(r"(?<![\w.,])(1[1-9]\d0|20[1-9]0|[2-9]0)s\b", _english_decade, text)
    return text


def _french_hour(match):
    hour, minutes = int(match.group(1)), match.group(2)
    # "une heure": converted from digits, it would be "un"
    text = "une heure" if hour == 1 else f"{hour} heure{'s' if hour > 1 else ''}"
    return text + (f" {int(minutes)}" if minutes and int(minutes) else "") + (" " if match.group().endswith(" ") else "")


def _english_hour(match):
    hour, minutes = match.groups()
    return f"{int(hour)}" + ("" if minutes == "00" else f" oh {int(minutes)}" if minutes[0] == "0" else f" {minutes}")


_DECADES_EN = ["twenties", "thirties", "forties", "fifties", "sixties", "seventies", "eighties", "nineties"]


def _english_decade(match):
    decade = match.group(1)
    if len(decade) == 2:
        return _DECADES_EN[int(decade[0]) - 2]
    if decade[2] == "1":
        return f"{decade[:2]} tens"
    return f"{decade[:2]} {_DECADES_EN[int(decade[2]) - 2]}" if decade[2] != "0" else f"{decade[:2]} hundreds"


_SCALES = {
    "en": r"hundreds?|thousands?|millions?|billions?|trillions?",
    "fr": r"mille|millions?|milliards?",
}
_SCALE_SUFFIXES = {
    "en": {"k": "thousand", "m": "million", "b": "billion", "bn": "billion"},
    "fr": {"k": "mille", "m": "millions", "md": "milliards", "mds": "milliards"},
}


def currencies_after_numbers(text, lang, symbol_to_word):
    """
    The currency is said after the number, and after the scale word
    "$1.5 million" -> "1.5 million dollars", "$5bn" -> "5 billion dollars", "$1" -> "1 dollar", "$4.99" -> "4 dollars 99"
    In French: "$5" -> "5 dollars", "1,5 M€" -> "1,5 millions d'euros"

    :param symbol_to_word: words of the symbols in the language (with the currencies, in plural)
    """
    symbols = [c for c in "$€£¥₹" if c in symbol_to_word]
    if not symbols or not any(c in text for c in symbols):
        return text
    scales, suffixes = _SCALES.get(lang, "(?!)"), _SCALE_SUFFIXES.get(lang, {})
    decimal = r"\." if lang in _DECIMAL_POINT_LANGUAGES else "[.,]"
    symbol = "([" + "".join(symbols) + "])"
    number = r"(\d+(?:[   ,.]\d{3})*(?:" + decimal + r"\d+)?)"
    suffix = r"(" + "|".join(sorted(suffixes, key=len, reverse=True)) + ")" if suffixes else "((?!))"
    scale = r"(?:\s*(" + scales + r")\b|\s?" + suffix + r"(?!\w))?"

    def said(number, scale, suffix, word):
        scale = scale or (suffixes[suffix.lower()] if suffix else None)
        if scale:
            if lang == "fr" and not scale.startswith("mille"):
                return f"{number} {scale} " + ("d'" if word[0] in "aeiouy" else "de ") + word
            return f"{number} {scale} {word}"
        if number == "1":
            word = re.sub("s$", "", word)
        cents = re.fullmatch(r"([\d,. ]+)" + decimal + r"(\d\d)", number)
        if cents:
            return f"{cents.group(1)} {re.sub('s$', '', word) if cents.group(1) == '1' else word} {cents.group(2)}"
        return f"{number} {word}"

    # Before the number
    text = re.sub(
        r"(?<!\d)(?<!\d\s)" + symbol + r"\s*" + number + scale + r"(?!\d)",
        lambda m: PADDING + said(m.group(2), m.group(3), m.group(4), symbol_to_word[m.group(1)]),
        text,
        flags=re.IGNORECASE,
    )
    # After the number, with a scale (French: "1,5 M€", "3 millions €")
    if lang == "fr":
        text = re.sub(
            r"(?<![\d.,])" + number + r"(?:\s*(" + scales + r")|\s?" + suffix + r")\s*" + symbol,
            lambda m: said(m.group(1), m.group(2), m.group(3), symbol_to_word[m.group(4)]),
            text,
            flags=re.IGNORECASE,
        )
    return clean_padding(text)


# ------------------------------------------------------------------------------------
# Numbers in letters, to compare texts

_NUMBERS_EN = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    "nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion"
)
_NUMBERS_FR = (
    "zéro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|"
    "cinquante|soixante|cents?|milles?|millions?|milliards?"
)


def normalize_spoken_numbers(text, lang):
    """
    Single way to write numbers in letters, for comparing texts, once the digits are converted into letters.
    The case of the other words is kept.

    English:
    - "two hundred and fifty", "two hundred fifty" -> "two hundred fifty"
    - "twenty twenty-one", "two thousand and twenty one" -> "two thousand twenty one" (1910 to 2026)
    - "nineteen oh five" -> "one thousand nine hundred five", "twelve hundred" -> "one thousand two hundred"
    - "a hundred" -> "one hundred", "per cent" -> "percent"
    French:
    - "une" -> "un"
    - "un point cinq" -> "un virgule cinq"
    - "deux cents", "quatre-vingts" -> "deux cent", "quatre vingt"
    - "dix-neuf cent" -> "mille neuf cent"
    - "pourcent" -> "pour cent"
    Spanish, Italian, Portuguese: gender of one (and two)
    """
    lang = base_language(lang)
    if lang == "en":
        return _normalize_english(text)
    if lang == "fr":
        return _normalize_french(text)
    if lang == "es":
        return re.sub(r"\bun[ao]\b", "un", re.sub(r"\bporciento\b", "por ciento", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    if lang == "it":
        return re.sub(r"\bun[ao]\b", "un", re.sub(r"\bpercento\b", "per cento", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    if lang == "pt":
        text = re.sub(r"\bporcento\b", "por cento", text, flags=re.IGNORECASE)
        return re.sub(r"\bduas\b", "dois", re.sub(r"\buma\b", "um", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    return text


def _split_compounds(text, numbers):
    """ "vingt-cinq" -> "vingt cinq" """
    return re.sub(rf"\b({numbers})-(?=(?:{numbers})\b)", r"\1 ", text, flags=re.IGNORECASE)


def _normalize_english(text):
    years, pattern = _english_years()
    text = _split_compounds(text, _NUMBERS_EN)
    text = re.sub(r"\ba\s+(?=(?:hundred|thousand|million|billion|trillion)\b)", "one ", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(hundred|thousand|million|billion|trillion),?\s+and\s+(?=(?:{_NUMBERS_EN})\b)", r"\1 ", text, flags=re.IGNORECASE)
    text = pattern.sub(lambda m: years[" ".join(m.group().lower().split())], text)
    text = re.sub(
        rf"(?<!\w )\b({'|'.join(_HUNDREDS_EN)})\s+hundred\b", lambda m: _HUNDREDS_EN[m.group(1).lower()], text, flags=re.IGNORECASE
    )
    return re.sub(r"\bper\s+cent\b", "percent", text, flags=re.IGNORECASE)


_UNITS_EN = "one two three four five six seven eight nine".split()
_HUNDREDS_EN = {
    "eleven": "one thousand one hundred",
    "twelve": "one thousand two hundred",
    "thirteen": "one thousand three hundred",
    "fourteen": "one thousand four hundred",
    "fifteen": "one thousand five hundred",
    "sixteen": "one thousand six hundred",
    "seventeen": "one thousand seven hundred",
    "eighteen": "one thousand eight hundred",
    "nineteen": "one thousand nine hundred",
}


@lru_cache(maxsize=1)
def _english_years():
    """{"twenty twenty one": "two thousand twenty one", "nineteen oh five": ...} and the pattern to find them"""
    from .text_utils import undigit

    def words(n):
        return re.sub(r"\band\b", " ", undigit(str(n), lang="en").replace("-", " ").replace(",", " "))

    years = {}
    for year in list(range(1910, 2000)) + list(range(2010, 2027)):
        if year % 100 >= 10:
            years[f"{words(year // 100)} {words(year % 100)}"] = " ".join(words(year).split())
    for century in (1900, 2000):
        for unit, name in enumerate(_UNITS_EN, 1):
            years[f"{words(century // 100)} oh {name}"] = " ".join(words(century + unit).split())
    pattern = "|".join(y.replace(" ", r"\s+") for y in sorted(years, key=len, reverse=True))
    return years, re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])", flags=re.IGNORECASE)


_HUNDREDS_FR = dict(
    zip(
        "onze douze treize quatorze quinze seize dix-sept dix-huit dix-neuf".split(),
        ["mille cent"] + [f"mille {n} cent" for n in "deux trois quatre cinq six sept huit neuf".split()],
    )
)


def _normalize_french(text):
    text = re.sub(r"\bpourcents?\b", "pour cent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(u)ne\b", r"\1n", text, flags=re.IGNORECASE)
    text = _split_compounds(text, _NUMBERS_FR)
    text = re.sub(rf"\b({_NUMBERS_FR})\s+point\s+(?=(?:{_NUMBERS_FR})\b)", r"\1 virgule ", text, flags=re.IGNORECASE)
    # "dix-neuf cent" -> "mille neuf cent" (not in "quatre-vingt-dix-neuf cents", after another number)
    hundreds = "|".join(h.replace("-", r"\s+") for h in _HUNDREDS_FR)
    text = re.sub(
        rf"\b((?:{_NUMBERS_FR})\s+)?({hundreds})\s+cents?\b",
        lambda m: m.group() if m.group(1) else _HUNDREDS_FR["-".join(m.group(2).lower().split())],
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(rf"\b((?:{_NUMBERS_FR})\s+)(cent|vingt|mille)s\b", r"\1\2", text, flags=re.IGNORECASE)
