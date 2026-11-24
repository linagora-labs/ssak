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


# After a number one that has to agree with the next word
ONE_MARK = "\ue0f2"

# Beginning of a sentence with a number, to get a capital letter
_SENTENCE_START = "\ue0f1"
# A dot after these words does not end a sentence
_ABBREVIATIONS = "no|vs|n|p|pp|art|fig|vol|cf|approx|env|st|dr|mr|mrs|ms|m|mme|mlle|etc|max|min"


def mark_sentence_starts(text):
    """Mark the numbers at the beginning of a sentence, in a text with case or punctuation (not in "3 comprimés", nor in lower case without punctuation)"""
    if not re.search(r"[^\W\d_a-zà-öø-ÿ][a-zà-öø-ÿ]|[.?!…]\s*$", text):
        return text

    def mark(match):
        before = re.search(r"(\w+)$", text[: match.start()])
        if match.group().startswith(".") and before and re.fullmatch(_ABBREVIATIONS, before.group(1), flags=re.IGNORECASE):
            return match.group()
        return match.group() + _SENTENCE_START

    return re.sub(r"(?:^\s*|(?:(?<!\.)\.|[?!])\s+)(?=[\d$€£¥₹]|[Nn]o\.\s*\d|[IVX]+(?:er|e)?\b)", mark, text)


def capitalize_sentence_starts(text):
    if _SENTENCE_START not in text:
        return text
    return re.sub(_SENTENCE_START + r"\s*(.?)", lambda m: m.group(1).upper(), text)


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
    # Non-breaking and thin spaces between thousands
    text = re.sub(r"(?<=\d)[\u00a0\u2009\u202f](?=\d{3}(?!\d))", " ", text)

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
        text = re.sub(r"\bno\.\s*(?=\d)", "number ", text)
        # Decades
        text = re.sub(r"(?<![\w.,])(1[1-9]\d0|20\d0|[2-9]0)s\b", _english_decade, text)
    elif lang == "de":
        text = _german_ordinals(text)
    return text


_MONTHS_DE = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"


def _german_ordinals(text):
    """ "am 2. Mai" -> "am zweiten Mai", "der 2. Mai" -> "der zweite Mai", "1960er" -> "neunzehnhundertsechziger" """
    from .text_utils import undigit

    def ordinal(match):
        article, number = match.groups()
        word = undigit(number, lang="de", to="ordinal")
        if article and article.strip().lower() not in ["der", "die", "das"]:
            word += "n"
        elif not article:
            word += "r"
        return (article or "") + word

    def decade(match):
        year = undigit(match.group(1), lang="de")
        return _german_hundreds(year) + "er"

    text = re.sub(r"(?<![\w.,])(\d{2}|\d{3}0)er\b", decade, text)
    # The number before a dot is an ordinal before a month, or after an article and before a noun
    text = re.sub(r"\b((?:am|dem|den|des|vom|zum|im|beim|der|die|das)\s+)(\d{1,2})\.(?=\s+[A-ZÄÖÜ])", ordinal, text, flags=0)
    return re.sub(rf"(?<![\w.,])()(\d{{1,2}})\.(?=\s+(?:{_MONTHS_DE})\b)", ordinal, text)


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
    if decade == "2000":
        return "two thousands"
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


_SINGULAR_CURRENCIES = {
    "es": {"dólares": "dólar", "yenes": "yen"},
    "pt": {"dólares": "dólar"},
    "it": {"dollari": "dollaro", "sterline": "sterlina"},
}


def _singular_currency(word, lang):
    return _SINGULAR_CURRENCIES.get(lang, {}).get(word) or re.sub("s$", "", word)


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
            number, word = "1" + ONE_MARK, _singular_currency(word, lang)
        cents = re.fullmatch(r"([\d,. ]+)" + decimal + r"(\d\d)", number)
        if cents:
            return f"{cents.group(1)} {_singular_currency(word, lang) if cents.group(1) == '1' else word} {cents.group(2)}"
        return f"{number} {word}"

    # Before the number
    text = re.sub(
        r"(?<!\d)(?<!\d\s)" + symbol + r"\s*" + number + scale + r"(?!\d)",
        lambda m: PADDING + said(m.group(2), m.group(3), m.group(4), symbol_to_word[m.group(1)]),
        text,
        flags=re.IGNORECASE,
    )
    # After the number one
    text = re.sub(
        r"(?<![\d.,/])(?<!\d[ \u00a0\u2009\u202f])1(\s?)" + symbol + r"(?!\s?\d)",
        lambda m: f"1{ONE_MARK} " + _singular_currency(symbol_to_word[m.group(2)], lang) + PADDING,
        text,
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

_NUMBERS_EN = "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion"
_NUMBERS_FR = "zéro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|millions?|milliards?"


def normalize_spoken_numbers(text, lang):
    """
    Single way to write numbers in letters, for comparing texts, once the digits are converted into letters.
    The case of the other words is kept.

    English:
    - "two hundred and fifty", "two hundred fifty" -> "two hundred fifty"
    - "twenty twenty-one", "two thousand and twenty one" -> "two thousand twenty one" (1010 to 2099)
    - "nineteen oh five" -> "one thousand nine hundred five", "twelve hundred" -> "one thousand two hundred"
    - "a hundred" -> "one hundred", "per cent" -> "percent"
    French:
    - "une" -> "un"
    - "un point cinq" -> "un virgule cinq"
    - "deux cents", "quatre-vingts" -> "deux cent", "quatre vingt"
    - "dix-neuf cent" -> "mille neuf cent"
    - "pourcent" -> "pour cent"
    - "milligrammes", "micro grammes", "mètres carrés" -> "milligramme", "microgramme", "mètre carré"
    - "zéro virgule zéro vingt-cinq" -> "zéro virgule zéro deux cinq"
    - "october seventh", "october the seventh" -> "october seven", "seventh october", "the seventh of october" -> "seven october"
    - "three point twenty-five" -> "three point two five"
    Spanish, Italian, Portuguese, German: gender of the numbers
    Portuguese: "dezesseis" -> "dezasseis"
    German, Dutch: years ("eintausendneunhundert..." -> "neunzehnhundert...")
    """
    lang = base_language(lang)
    if lang == "en":
        return _normalize_english(text)
    if lang == "fr":
        return _normalize_french(text)
    if lang == "es":
        text = re.sub(r"\bporciento\b", "por ciento", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(veinti)(?:un[ao]|ún)\b", r"\1un", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w*cient|quinient)as\b", r"\1os", text, flags=re.IGNORECASE)
        return re.sub(r"\bun[ao]\b", "un", text, flags=re.IGNORECASE)
    if lang == "it":
        return re.sub(r"\bun[ao]\b", "un", re.sub(r"\bpercento\b", "per cento", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    if lang == "pt":
        text = re.sub(r"\bporcento\b", "por cento", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(duz|trez|quatroc|quinh|seisc|setec|oitoc|novec)entas\b", r"\1entos", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdeze(sseis|ssete|nove)\b", r"deza\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\bquatorze\b", "catorze", text, flags=re.IGNORECASE)
        return re.sub(r"\bduas\b", "dois", re.sub(r"\buma\b", "um", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    if lang == "de":
        text = re.sub(r"\bein(?:e|en|em|er|es|s)\b", "ein", text, flags=re.IGNORECASE)
        return _german_hundreds(text)
    if lang == "nl":
        return _normalize_dutch_years(text)
    return text


_HUNDREDS_DE = {
    "ein": "elf",
    "zwei": "zwölf",
    "drei": "dreizehn",
    "vier": "vierzehn",
    "fünf": "fünfzehn",
    "sechs": "sechzehn",
    "sieben": "siebzehn",
    "acht": "achtzehn",
    "neun": "neunzehn",
}


def _german_hundreds(text):
    """ "eintausendneunhundert..." -> "neunzehnhundert..." """
    return re.sub(rf"\b(?:ein)?tausend({'|'.join(_HUNDREDS_DE)})hundert", lambda m: _HUNDREDS_DE[m.group(1).lower()] + "hundert", text, flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def _dutch_years():
    from .text_utils import undigit

    words = {n: undigit(str(n), lang="nl").replace(" ", "") for n in range(1, 100)}
    numbers = {words[n]: n for n in range(1, 100)} | {"een": 1}
    return {words[c]: c for c in range(11, 20)}, numbers


def _normalize_dutch_years(text):
    """ "negentienachtennegentig", "negentienhonderd achtennegentig", "negentienhonderdvijf" -> "duizendnegenhonderd..." """
    from .text_utils import undigit

    centuries, numbers = _dutch_years()

    def by_length(words):
        return "|".join(sorted(words, key=len, reverse=True))

    def year(match):
        century, number = match.group(1), match.group(2) or match.group(3)
        value = 100 * centuries[century.lower()] + (numbers[number.lower()] if number else 0)
        return undigit(str(value), lang="nl").replace(" ", "")

    two_digits = by_length(n for n, value in numbers.items() if value >= 10)
    return re.sub(rf"\b({by_length(centuries)})(?:\s?honderd(?:\s?({by_length(numbers)}))?|\s?({two_digits}))\b", year, text, flags=re.IGNORECASE)


def _split_compounds(text, numbers, next_numbers=None):
    """ "vingt-cinq" -> "vingt cinq" """
    return re.sub(rf"\b({numbers})-(?=(?:{next_numbers or numbers})\b)", r"\1 ", text, flags=re.IGNORECASE)


_UNITS_EN = "one two three four five six seven eight nine".split()
_TEENS_EN = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS_EN = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS_EN = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
_MONTHS_EN = "january|february|march|april|may|june|july|august|september|october|november|december"

# "nineteen ninety eight", "nineteen oh five", "nineteen hundred"
_YEAR_EN = re.compile(
    rf"(?<![\w-])({'|'.join(_TEENS_EN)}|twenty)\s+"
    rf"(?:(oh\s+)({'|'.join(_UNITS_EN)})|({'|'.join(_TEENS_EN)})|({'|'.join(_TENS_EN)})(?:\s+({'|'.join(_UNITS_EN)}))?|(hundred))"
    rf"(?![\w-])(?!\s+(?:hundred|thousand|million|billion|trillion)\b)",
    flags=re.IGNORECASE,
)
_DAY_EN = re.compile(rf"\b({_MONTHS_EN})\s+(?:the\s+)?(?:(twenty|thirty)\s+({'|'.join(_ORDINALS_EN[:9])})|({'|'.join(_ORDINALS_EN)}|thirtieth))\b", flags=re.IGNORECASE)


_DAY_FIRST_EN = re.compile(
    rf"\b(the\s+)?((?:twenty|thirty)\s+(?:{'|'.join(_ORDINALS_EN[:9])})|{'|'.join(_ORDINALS_EN)}|thirtieth)(\s+of)?(?=\s+(?:{_MONTHS_EN})\b)",
    flags=re.IGNORECASE,
)


def _cardinal_en(n):
    """1998 -> "one thousand nine hundred ninety eight" """
    if n >= 1000:
        return (_cardinal_en(n // 1000) + " thousand " + _cardinal_en(n % 1000)).strip()
    if n >= 100:
        return (_UNITS_EN[n // 100 - 1] + " hundred " + _cardinal_en(n % 100)).strip()
    if n >= 20:
        return (_TENS_EN[n // 10 - 2] + " " + _cardinal_en(n % 10)).strip()
    return ([""] + _UNITS_EN + _TEENS_EN)[n]


def _normalize_english(text):
    text = _split_compounds(text, _NUMBERS_EN)
    text = _split_compounds(text, "twenty|thirty", "|".join(_ORDINALS_EN[:9]))
    text = re.sub(r"\boh-(?=(?:" + "|".join(_UNITS_EN) + r")\b)", "oh ", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s+(?=(?:hundred|thousand|million|billion|trillion)\b)", "one ", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(hundred|thousand|million|billion|trillion),?\s+and\s+(?=(?:{_NUMBERS_EN})\b)", r"\1 ", text, flags=re.IGNORECASE)

    def year(match):
        # Not in a longer number ("one hundred twenty twenty")
        before = text[: match.start()].split()
        if before and re.fullmatch(_NUMBERS_EN, before[-1].lower()):
            return match.group()
        century, _, unit, teen, tens, tens_unit, hundred = [g.lower() if g else g for g in match.groups()]
        if hundred and century in ["ten", "twenty"]:
            return match.group()
        value = 100 * (10 + _TEENS_EN.index(century) if century != "twenty" else 20)
        if unit:
            value += 1 + _UNITS_EN.index(unit)
        elif teen:
            value += 10 + _TEENS_EN.index(teen)
        elif tens:
            value += 20 + 10 * _TENS_EN.index(tens) + (1 + _UNITS_EN.index(tens_unit) if tens_unit else 0)
        return _cardinal_en(value)

    text = _YEAR_EN.sub(year, text)

    def day(match):
        month, tens, unit, ordinal = match.groups()
        if tens:
            return f"{month} {tens} {_UNITS_EN[_ORDINALS_EN.index(unit.lower())]}"
        return f"{month} " + ("thirty" if ordinal.lower() == "thirtieth" else _cardinal_en(1 + _ORDINALS_EN.index(ordinal.lower())))

    def day_first(match):
        the, ordinal, of = match.groups()
        cardinal = day(_DAY_EN.match("january " + ordinal))[len("january ") :]
        return cardinal if of or not the else the + cardinal

    text = _DAY_EN.sub(day, text)
    text = _DAY_FIRST_EN.sub(day_first, text)

    def decimals(match):
        zeros, teen, tens, unit = match.group(2, 3, 4, 5)
        if teen:
            digits = f"one {(['zero'] + _UNITS_EN)[_TEENS_EN.index(teen.lower())]}"
        else:
            digits = f"{_UNITS_EN[1 + _TENS_EN.index(tens.lower())]} {unit or 'zero'}"
        return f"{match.group(1)} point {zeros}{digits}"

    text = re.sub(
        rf"\b({_NUMBERS_EN}) point ((?:(?:zero|oh) )*)(?:({'|'.join(_TEENS_EN)})|({'|'.join(_TENS_EN)})(?: ({'|'.join(_UNITS_EN)}))?)\b",
        decimals,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<=\bpoint )oh\b", "zero", text, flags=re.IGNORECASE)
    return re.sub(r"\bper\s+cent\b", "percent", text, flags=re.IGNORECASE)


_HUNDREDS_FR = dict(
    zip(
        "onze douze treize quatorze quinze seize dix-sept dix-huit dix-neuf".split(),
        ["mille cent"] + [f"mille {n} cent" for n in "deux trois quatre cinq six sept huit neuf".split()],
    )
)


@lru_cache(maxsize=1)
def _french_decimals():
    """{"vingt cinq": "deux cinq", ...} (10 to 999) and the pattern to find them after "virgule" """
    from .text_utils import undigit

    def words(n):
        return re.sub(r"(cent|vingt)s\b", r"\1", undigit(str(n), lang="fr").replace("-", " "))

    decimals = {words(n): " ".join(undigit(d, lang="fr") for d in str(n)) for n in range(10, 1000)}
    alternatives = "|".join(w.replace(" ", r"\s+") for w in sorted(decimals, key=len, reverse=True))
    return decimals, re.compile(rf"\b((?:virgule)\s+(?:zéro\s+)*)({alternatives})\b", flags=re.IGNORECASE)


def _normalize_french(text):
    text = re.sub(r"\bpourcents?\b", "pour cent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(u)ne\b", r"\1n", text, flags=re.IGNORECASE)
    # The singular and the plural of the units sound the same
    text = re.sub(r"\b(micro|milli|centi|déci|kilo)\s+(?=(?:gramme|litre|mètre)s?\b)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b((?:micro|milli|centi|déci|kilo)?(?:gramme|litre|mètre))s\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=mètre )(carré|cube)s\b", r"\1", text, flags=re.IGNORECASE)
    text = _split_compounds(text, _NUMBERS_FR)
    text = re.sub(rf"\b({_NUMBERS_FR})\s+point\s+(?=(?:{_NUMBERS_FR})\b)", r"\1 virgule ", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b((?:{_NUMBERS_FR})\s+)(cent|vingt|mille)s\b", r"\1\2", text, flags=re.IGNORECASE)
    decimals, pattern = _french_decimals()
    text = pattern.sub(lambda m: m.group(1) + decimals[" ".join(m.group(2).lower().split())], text)
    # "dix-neuf cent" -> "mille neuf cent" (not in "quatre-vingt-dix-neuf cents", after another number)
    hundreds = "|".join(h.replace("-", r"\s+") for h in _HUNDREDS_FR)
    text = re.sub(
        rf"\b((?:{_NUMBERS_FR})\s+)?({hundreds})\s+cents?\b",
        lambda m: m.group() if m.group(1) else _HUNDREDS_FR["-".join(m.group(2).lower().split())],
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(rf"\b((?:{_NUMBERS_FR})\s+)(cent|vingt|mille)s\b", r"\1\2", text, flags=re.IGNORECASE)
