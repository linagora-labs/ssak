import re
import warnings

from num2words import CONVERTER_CLASSES, num2words

from ssak.utils.misc import flatten
from ssak.utils.text_basic import (  # noqa: F401 (some functions are only there to be imported from this module)
    collapse_whitespace,
    format_special_characters,
    regex_escape,
    remove_parenthesis,
    remove_punctuations,
    remove_quotes,
    remove_special_words,
    replace_ligatures,
    split_around,
    split_around_apostrophe,
    split_around_space_and_apostrophe,
    transliterate,
)
from ssak.utils.text_numbers import (
    PADDING,
    base_language,
    capitalize_sentence_starts,
    clean_number_formats,
    clean_padding,
    currencies_after_numbers,
    mark_sentence_starts,
)
from ssak.utils.text_variants import agree_one


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning

# Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
_emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    flags=re.UNICODE,
)
#                            u"\U000024C2-\U0001F251"  # chinese char
# _emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)

_currencies = ["€", "$", "£", "¥", "₽"]

_symbol_to_word = {
    "fr": {
        "%": "pour cent",
        "٪": "pour cent",
        "‰": "pour mille",
        "~": "environ",
        "÷": "divisé par",
        # "*": "fois",  # ?
        "×": "fois",
        "±": "plus ou moins",
        "+": "plus",
        "⁺": "plus",
        "⁻": "moins",
        "=": "égale",
        "&": "et",
        "@": "arobase",
        "µ": "micro",
        "mm²": "millimètres carrés",
        "mm³": "millimètres cubes",
        "cm²": "centimètres carrés",
        "cm³": "centimètres cubes",
        "m²": "mètres carrés",
        "m³": "mètres cubes",
        "²": "au carré",
        "³": "au cube",
        "⁵": "à la puissance cinq",
        "⁷": "à la puissance sept",
        "½": "un demi",
        "⅓": "un tiers",
        "⅔": "deux tiers",
        "¼": "un quart",
        "¾": "trois quarts",
        "§": "paragraphe",
        "°C": "degrés Celsius",
        "°F": "degrés Fahrenheit",
        "°K": "kelvins",
        "°": "degrés",
        "€": "euros",
        "¢": "cents",
        "$": "dollars",
        "£": "livres",
        "¥": "yens",
        "₹": "roupies",
        # Below: not in Whisper tokens
        # "₩": "wons",
        # "₽": "roubles",
        # "₺": "liras",
        # "₪": "shekels",
        # "₴": "hryvnias",
        # "₮": "tugriks",
        # "℃": "degrés Celsius",
        # "℉": "degrés Fahrenheit",
        # "Ω": "ohms",
        # "Ω": "ohms",
        # "K": "kelvins",
        # "ℓ": "litres",
    },
    "en": {
        "%": "percent",
        "٪": "percent",
        "‰": "per mille",
        "~": "about",
        "÷": "divided by",
        # "*": "times",  # ?
        "×": "times",
        "±": "plus or minus",
        "+": "plus",
        "⁺": "plus",
        "⁻": "minus",
        "&": "and",
        "@": "at",
        "µ": "micro",
        "mm²": "square millimeters",
        "mm³": "cubic millimeters",
        "cm²": "square centimeters",
        "cm³": "cubic centimeters",
        "m²": "square meters",
        "m³": "cubic meters",
        "²": "squared",
        "³": "cubed",
        "⁵": "to the fifth power",
        "⁷": "to the seventh power",
        "½": "one half",
        "⅓": "one third",
        "⅔": "two thirds",
        "¼": "one quarter",
        "¾": "three quarters",
        "§": "section",
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
        "°K": "kelvins",
        "°": "degrees",
        "€": "euros",
        "¢": "cents",
        "$": "dollars",
        "£": "pounds",
        "¥": "yens",
        "₹": "rupees",
    },
    "ar": {
        "%": "في المئة",
        "٪": "في المئة",
        "‰": "بالألف",
        "~": "حوالي",
        "=": "يساوي",
        "÷": "مقسوما على",
        # "*": "مضروبا بـ",  # ?
        "×": "مضروبا بـ",
        "±": "بالإضافة أو الطرح",
        "+": "بالإضافة",
        "⁺": "بالإضافة",
        "⁻": "بالطرح",
        "&": "و",
        "@": "على",
        "µ": "ميكرو",
        "mm²": "مم مربع",
        "مم²": "مم مربع",
        "mm³": "مم مكعب",
        "مم³": "مم مكعب",
        "هـ": "هجري",
        "ق.م": "قبل الميلاد",
        "cm²": "سم مربع",
        "cm³": "سم مكعب",
        "سم²": "سم مربع",
        "سم³": "سم مكعب",
        "m²": "م مربع",
        "m³": "م مكعب",
        "م²": "م مربع",
        "م³": "م مكعب",
        "²": "مربع",
        "³": "مكعب",
        "⁵": "الخامسة",
        "⁷": "السابعة",
        "½": "نصف",
        "⅓": "ثلث",
        "⅔": "ثلثين",
        "¼": "ربع",
        "¾": "ربعين",
        "§": "فقرة",
        "°C": "درجة مئوية",
        "°F": "درجة فهرنهايت",
        "°K": "كيلفن",
        "°": "درجة",
        # "/":"أو",
        "€": "يورو",
        "¢": "سنت",
        "$": "دولار",
        "£": "جنيه",
        "¥": "ين",
        "₹": "روبية هندية",
        "₽": "روبل روسي",
        "C$": "دولار كندي",
    },
    "ru": {
        "№": "номер",
        "%": "процентов",
        "٪": "процентов",
        "=": "равно",
        "‰": "промилле",
        "~": "примерно",  # can I put several translations?
        "÷": "разделить на",
        "*": "умножить на",
        "×": "умножить на",
        "±": "плюс минус",
        "+": "плюс",
        "⁺": "плюс",
        "⁻": "минус",
        "&": "энд",
        "@": "собака",
        "кг": "килограмм",
        # "г": "грамм",
        "мм²": "квадратный миллиметр",
        "мл": "миллилитр",
        "мм³": "миллиметр в кубе",
        "см²": "квадратный сантиметр",
        "см³": "сантиметр в кубе",
        "м²": "квадратный метр",
        "м³": "метр в кубе",
        "²": "в квадрате",
        "³": "в кубе",
        "⁵": "в пятой степени",
        "⁷": "в седьмой степени",
        "½": "одна вторая",
        "⅓": "одна треть",
        "⅔": "две трети",
        "¼": "одна четверть",
        "¾": "три четверти",
        "§": "параграф",
        "°C": "градус цельсия",
        "°F": "градус по фаренгейту",
        "°K": "градус кельвина",
        "°": "градус",
        "€": "евро",
        "¢": "цент",
        "$": "доллар",
        "£": "фунт",
        "¥": "йен",
        "₹": "рупий",
        "₽": "рубль",
    },
    "es": {
        "%": "por ciento",
        "‰": "por mil",
        "+": "más",
        "&": "y",
        "@": "arroba",
        "°C": "grados Celsius",
        "°": "grados",
        "€": "euros",
        "$": "dólares",
        "£": "libras",
        "¥": "yenes",
    },
    "de": {
        "%": "Prozent",
        "‰": "Promille",
        "+": "plus",
        "&": "und",
        "@": "at",
        "°C": "Grad Celsius",
        "°": "Grad",
        "€": "Euro",
        "$": "Dollar",
        "£": "Pfund",
        "¥": "Yen",
    },
    "it": {
        "%": "per cento",
        "‰": "per mille",
        "+": "più",
        "&": "e",
        "@": "chiocciola",
        "°C": "gradi Celsius",
        "°": "gradi",
        "€": "euro",
        "$": "dollari",
        "£": "sterline",
        "¥": "yen",
    },
    "pt": {
        "%": "por cento",
        "‰": "por mil",
        "+": "mais",
        "&": "e",
        "@": "arroba",
        "°C": "graus Celsius",
        "°": "graus",
        "€": "euros",
        "$": "dólares",
        "£": "libras",
        "¥": "ienes",
    },
    "nl": {
        "%": "procent",
        "‰": "promille",
        "+": "plus",
        "&": "en",
        "@": "apenstaartje",
        "°C": "graden Celsius",
        "°": "graden",
        "€": "euro",
        "$": "dollar",
        "£": "pond",
        "¥": "yen",
    },
}

_ar_currencies = {
    "ar": {
        "EGP": "جنيه مصري",
        "ج.م": "جنيه مصري",
        "IQD": "دينار عراقي",
        "د.ع": "دينار عراقي",
        "SYP": "ليرة سورية",
        "ل.س": "ليرة سورية",
        "ل.ل": "ليرة لبنانية",
        "LBP": "ليرة لبنانية",
        "JOD": "دينار أردني",
        "د.ا": "دينار أردني",
        "SAR": "ريال سعودي",
        "ر.س": "ريال سعودي",
        "YER": "ريال يمني",
        "ر.ي": "ريال يمني",
        "LYD": "دينار ليبي",
        "د.ل": "دينار ليبي",
        "SDG": "جنيه سوداني",
        "ج.س": "جنيه سوداني",
        "MAD": "درهم مغربي",
        "د.م": "درهم مغربي",
        "TND": "دينار تونسي",
        "د.ت": "دينار تونسي",
        "KWD": "دينار كويتي",
        "د.ك": "دينار كويتي",
        "DZD": "دينار جزائري",
        "د.ج": "دينار جزائري",
        "MRO": "أوقية موريتانية",
        "أ.م": "أوقية موريتانية",
        "BHD": "دينار بحريني",
        "د.ب": "دينار بحريني",
        "QAR": "ريال قطري",
        "ر.ق": "ريال قطري",
        "AED": "درهم إماراتي",
        "د.إ": "درهم إماراتي",
        "OMR": "ريال عماني",
        "ر.ع": "ريال عماني",
        "SOS": "شلن صومالي",
        "ش.ص": "شلن صومالي",
        "FDJ": "فرنك جيبوتي",
        "ف.ج": "فرنك جيبوتي",
        "KMF": "فرنك قمري",
        "EUR": "يورو",
        "USD": "دولار أمريكي",
    }
}


def replace_keeping_word_boundaries(orig, dest, text):
    if orig in text:
        _orig = regex_escape(orig)
        text = re.sub(r"(\W)" + _orig + r"(\W)", r"\1" + dest + r"\2", text)
        text = re.sub(_orig + r"(\W)", PADDING + dest + r"\1", text)
        text = re.sub(r"(\W)" + _orig, r"\1" + dest + PADDING, text)
        text = re.sub(_orig, PADDING + dest + PADDING, text)
        text = clean_padding(text)
    return text


def normalize_arabic_currencies(text, lang="ar"):
    symbol_table = _ar_currencies.get(lang, {})
    for k, v in symbol_table.items():
        text = replace_keeping_word_boundaries(k, v, text)
    return text


def symbols_to_letters(text, lang, lower_case=False):
    symbol_table = _symbol_to_word.get(lang, {})
    for k, v in symbol_table.items():
        if lower_case:
            k = k.lower()
            v = v.lower()
        text = replace_keeping_word_boundaries(k, v, text)
    return text


_not_latin_characters_pattern = re.compile("[^a-zA-Z0-9\u00C0-\u00FF\\-'\\.?!,;: ]")

_ALL_SPECIAL_CHARACTERS = []


def remove_special_characters(
    string,
    replace_by="",
    latin_characters_only=False,
    fid=None,
):
    """
    Remove emojis from string
    """
    if latin_characters_only:
        output = _not_latin_characters_pattern.sub(replace_by, string)
    else:
        output = _emoji_pattern.sub(replace_by, string)
    if fid is not None:
        global _ALL_SPECIAL_CHARACTERS
        removed = [c for c in string if c not in output]
        for char in removed:
            if char not in _ALL_SPECIAL_CHARACTERS:
                print(f"{ord(char):06d} {char}", file=fid)
                fid.flush()
                _ALL_SPECIAL_CHARACTERS.append(char)
    return output


def is_number_language(lang):
    """Whether the numbers can be converted into letters in this language"""
    return lang in CONVERTER_CLASSES or lang[:2] in CONVERTER_CLASSES


def cardinal_numbers_to_letters(text, lang, verbose=False):
    """
    Convert numbers to letters
    """
    if not is_number_language(lang):
        return text
    punct_to_word = _punct_to_word.get(lang, _punct_to_word["en"])
    if lang == "en":
        # Decimals are read digit by digit ("3.25" -> "3 point two five"), unlike versions ("1.2.3")
        text = re.sub(
            r"(?<![\d.])(\d+)\.(\d+)(?!\.?\d)",
            lambda m: m.group(1) + " point " + " ".join(undigit(d, lang=lang) for d in m.group(2)) + PADDING,
            text,
        )
    elif not lang.startswith("ar"):
        # Long decimal parts are read digit by digit ("3,14159"), and not as a number
        text = re.sub(
            r"(?<![\d.,])(\d+)([.,])(0*[1-9]\d{3,})(?![.,]?\d)",
            lambda m: f"{m.group(1)} {punct_to_word[m.group(2)]} " + " ".join(undigit(d, lang=lang) for d in m.group(3)) + PADDING,
            text,
        )
    # Floating point numbers
    text = re.sub(r"(?<=\d)[,،](?=\d)", f" {punct_to_word[',']} ", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", f" {punct_to_word['.']} ", text)
    # wav2vec -> wav to vec
    text = re.sub(r"([a-z])2([a-z])", r"\1 to \2", text)
    # space after digits
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    # For things like 40-MFCC
    text = re.sub(r"(\d)-(?=\s|$)", r"\1", text)
    text = re.sub(r"(\d)-", r"\1 ", text)

    digits = re.findall(r"(?:\-?\b[\d/]*\d+(?: \d\d\d)+\b)|(?:\-?\d[/\d]*)", text)
    digits = list(map(lambda s: s.strip(r"[/ ]"), digits))
    digits = list(set(digits))
    digits = digits + flatten([c.split() for c in digits if " " in c])
    digits = digits + flatten([c.split("/") for c in digits if "/" in c])
    digits = sorted(digits, reverse=True, key=lambda x: (len(x), x))

    # Format some dates for Russian
    if lang == "ru":
        text = ru_convert_dates(text)

    for digit in digits:
        digitf = re.sub("/+", "/", digit)
        if not digitf:
            continue
        numslash = len(re.findall("/", digitf))
        if numslash == 0:
            word = undigit(digitf, lang=lang)
        elif numslash == 1:  # Fraction or date
            i = digitf.index("/")
            is_date = False
            if len(digitf[i + 1 :]) in [1, 2]:
                try:
                    first = int(digitf[:i])
                    second = int(digitf[i + 1 :])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13 and len(digitf[i + 1 :]) == 2
                except:
                    pass
            if is_date:
                first = digitf[:i].lstrip("0")
                use_ordinal = (lang == "ru") or (lang == "fr" and first == "1") or (lang not in ["fr", "ar", "ar_tn"] and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang, to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get(lang, {}).get(second, digitf[i + 1 :])
            else:
                first = undigit(digitf[:i], lang=lang)
                second = undigit(digitf[i + 1 :], to="denominator", lang=lang)
                if float(digitf[:i]) > 2.0 and second[-1] != "s":
                    second += "s"
            word = first + " " + second
        elif numslash == 2:  # Maybe a date
            i1 = digitf.index("/")
            i2 = digitf.index("/", i1 + 1)
            is_date = False
            is_islamic_date = False
            sfirst = digitf[:i1]
            ssecond = digitf[i1 + 1 : i2]
            sthird = digitf[i2 + 1 :]
            if len(ssecond) in [1, 2]:
                try:
                    first = int(sfirst)
                    second = int(ssecond)
                    third = int(sthird)
                    if len(sthird) == 4:  # 1/1/2019
                        is_date = first > 0 and first < 32 and second > 0 and second < 13 and third > 1000
                    elif len(sfirst) == 4:  # 2019/1/1
                        is_date = third > 0 and third < 32 and second > 0 and second < 13 and first > 1000
                        if is_date:
                            if lang == "ar" or lang == "ar_tn":
                                is_islamic_date = is_date and first < 1600
                            first, third = third, first
                            sfirst, sthird = sthird, sfirst
                except ValueError:
                    pass
            if is_date:
                first = sfirst.lstrip("0")
                use_ordinal = (lang == "ru") or (lang == "fr" and first == "1") or (lang not in ["fr", "ar", "ar_tn"] and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang, to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get("ar_islamic" if is_islamic_date else lang, {}).get(int(ssecond), ssecond)
                use_ordinal = lang == "ru"
                third = undigit(sthird, lang=lang, to="ordinal" if use_ordinal else "cardinal")
                if is_islamic_date:
                    word = " ".join([third, second, first])
                else:
                    word = " ".join([first, second, third])
            else:
                word = " / ".join([undigit(s, lang=lang) for s in digitf.split("/")])
        else:
            word = " / ".join([undigit(s, lang=lang) for s in digitf.split("/")])
        if verbose:
            print(digit, "->", word)
        # text = replace_keeping_word_boundaries(digit, word, text)
        if " " in digit:
            text = re.sub(r"\b" + str(digit) + r"\b", PADDING + word + PADDING, text)
        else:
            text = re.sub(str(digit), PADDING + word + PADDING, text)

    text = clean_padding(text)

    if lang == "ru":
        text = ru_fix_ordinals(text)

    return text


WARNING_NOTIMPLEMENTED_ORDINAL = {}


def ordinal_numbers_to_letters(text, lang):
    global WARNING_NOTIMPLEMENTED_ORDINAL

    if not is_number_language(lang):
        return text

    if lang == "en":
        digits = re.findall(r"\b(?:\d*1st|\d*2nd|\d*3rd|\d+(?:º|th))\b", text)
    elif lang == "fr":
        digits = re.findall(r"\b(?:1(?:ères?|eres?|ers?|res?|r)|2(?:ndes?|nds?)|\d+(?:º|[èe]mes?|es?))\b", text)
    elif lang in ["es", "it", "pt"]:
        text = _degree_signs_to_ordinals(text, lang)
        digits = re.findall(r"\b\d+\.?[ºª]", text)
    elif lang == "nl":
        digits = re.findall(r"\b\d+(?:ste|de|e)\b", text)
    elif lang == "de":
        digits = []  # see clean_number_formats
    else:
        if not WARNING_NOTIMPLEMENTED_ORDINAL.get(lang):
            warnings.warn(f"WARNING: Normalization of ordinal numbers not supported for language '{lang}'")
            WARNING_NOTIMPLEMENTED_ORDINAL[lang] = True
        digits = []

    if digits:
        digits = sorted(list(set(digits)), reverse=True, key=lambda x: (len(x), x))
        for digit in digits:
            word = undigit(re.findall(r"\d+", digit)[0], to="ordinal", lang=lang)
            if lang == "fr":
                suffix = re.sub(r"\d+", "", digit)
                if re.match(r"1[^\d]", digit) and suffix.rstrip("s") in ["ère", "ere", "re"]:
                    word = "première"
                elif re.match(r"2[^\d]", digit) and suffix.startswith("nd"):
                    word = "seconde" if suffix.startswith("nde") else "second"
                if suffix.endswith("s"):
                    word += "s"
            elif digit.endswith("ª"):
                word = re.sub(r"o\b", "a", word)
            text = re.sub(r"\b" + re.escape(digit) + (r"\b" if digit[-1].isalpha() and digit[-1] not in "ºª" else ""), word, text)

    return text


# "2°" is read as an ordinal ("2º", "2ª") after a singular article, before a month, or before these nouns
_MASCULINE_ARTICLES = {"es": "el|del|al", "it": "il|del|al|dal|nel|sul", "pt": "o|do|ao|no|pelo"}
_FEMININE_ARTICLES = {"es": "la", "it": "la|della|alla|dalla|nella|sulla", "pt": "da|na|pela"}
_NOUNS_AFTER_ORDINALS = {
    "es": "piso|grado|lugar|puesto|clasificado|premio|capítulo|artículo|siglo|aniversario|curso|año|día|mes|trimestre|semestre|equipo|hijo|nivel|"
    "de (?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)",
    "it": "piano|grado|posto|classificato|premio|capitolo|articolo|secolo|anniversario|corso|anno|giorno|mese|trimestre|semestre|figlio|livello|"
    "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre",
    "pt": "andar|grau|lugar|colocado|classificado|prémio|prêmio|capítulo|artigo|século|aniversário|curso|ano|dia|mês|trimestre|semestre|filho|nível|"
    "de (?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)",
}


def _degree_signs_to_ordinals(text, lang):
    """
    Spanish, Italian, Portuguese: "el 2° piso" -> "el 2º piso", "1° maggio" -> "1º maggio", "la 2° vez" -> "la 2ª vez"
    (but not "fa 10°", "a 10° bajo cero", "20 °C", "45°"). Only the small numbers.
    """

    def ordinal(match):
        article, number = match.groups()
        article = article or ""
        if re.fullmatch(_FEMININE_ARTICLES[lang], article.strip(), flags=re.IGNORECASE):
            return f"{article}{number}ª"
        is_ordinal = article or re.match(rf"\s+(?:{_NOUNS_AFTER_ORDINALS[lang]})\b", text[match.end() :], flags=re.IGNORECASE)
        return f"{article}{number}" + ("º" if is_ordinal else "°")

    articles = f"{_MASCULINE_ARTICLES[lang]}|{_FEMININE_ARTICLES[lang]}"
    return re.sub(rf"(?<![\w.,])(\b(?:{articles})\s+)?(?<![\w.,])(\d|1[0-2])°(?!\s?[CFK]\b|\w)", ordinal, text, flags=re.IGNORECASE)


_ROMAN = r"(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})"
# (not "Vers", "Ver", "Ie")
_ROMAN_ORDINALS = {
    "fr": rf"(?<![\w-])(?:(I)(ers?|ères?|res?)|(?!I(?![IVX])){_ROMAN}(?:èmes?|emes?|es?|º))(?!\w)",
    "en": rf"(?<![\w-])(?!I(?![IVX])){_ROMAN}(?:st|nd|rd|th|º)(?!\w)",
}
# After these words, or a name, the short roman numerals ("II", "IV", ...), that could be acronyms, are numbers
_ROMAN_CONTEXT = {
    "fr": "chapitre|tome|acte|scène|article|titre|volume|partie|livre|section|annexe|phase|type|grade|stade|niveau|classe|guerre|mondiale|pape|roi|empereur",
    "en": "chapter|part|act|scene|article|title|volume|book|section|appendix|phase|type|grade|stage|level|class|war|king|queen|pope|emperor",
}
# Words before "IV" for "intravenous"
_NOT_ROMAN_CONTEXT = "give|start|stop|via|by|with|in|and|the|en|par|avec|et|prendre|injecter|donner|injections?|perfusions?|voie|administration|doses?|traitement|antibiotiques?|drugs?|fluids?|therapy|lines?|access|antibiotics?|infusions?"
# English: "Henry VIII" is "Henry the eighth", but "Rocky II" is "Rocky two"
_RULERS_EN = (
    "Henry|Louis|Charles|Elizabeth|George|James|Edward|Richard|William|Mary|Paul|Pius|Benedict|Leo|Gregory|Innocent|Clement|Philip|"
    "Frederick|Francis|Napoleon|Nicholas|Alexander|Peter|Catherine|Otto|Ramses|Ramesses|Urban|Boniface|Sixtus|Ferdinand|Carlos"
)


def roman_numbers_to_letters(text, lang):
    """
    "XIXe siècle", "Ve République", "François Ier", "Louis XIV", "Henri IV", "World War II"
    The numerals of a single letter are not converted without an ordinal suffix ("X-Prep", "vitamine C"),
    and the ones of two letters that could be acronyms ("IV", "II") only after some words, or a name.
    """
    if not is_number_language(lang) or not re.search(r"[IVX]", text):
        return text

    def words(numeral, to="cardinal"):
        return undigit(str(roman_to_decimal(numeral)), lang=lang, to=to)

    if lang in _ROMAN_ORDINALS:

        def ordinal(match):
            numeral = re.match(r"[IVX]+", match.group()).group()
            suffix = match.group()[len(numeral) :]
            if lang == "fr" and numeral == "I":
                word = "première" if suffix.rstrip("s") in ["ère", "re"] else "premier"
            else:
                word = words(numeral, "ordinal")
            return word + ("s" if suffix.endswith("s") and lang == "fr" else "")

        text = re.sub(_ROMAN_ORDINALS[lang], ordinal, text)

    if lang == "en":
        text = re.sub(rf"\b({_RULERS_EN}) ({_ROMAN})(?![\w-])", lambda m: m.group() if len(m.group(2)) == 1 else f"{m.group(1)} the {words(m.group(2), 'ordinal')}", text)

    def cardinal(match):
        numeral = match.group()
        if len(numeral) == 1:
            return numeral
        if not numeral.startswith("X") and len(numeral) == 2:
            before = re.search(r"([^\W\d_]+)\s+$", text[: match.start()])
            before = before.group(1) if before else ""
            is_context = re.fullmatch(_ROMAN_CONTEXT.get(lang, ""), before, flags=re.IGNORECASE)
            is_name = re.fullmatch(r"[^\W\d_a-zà-öø-ÿ][^\W\d_A-ZÀ-ÖØ-Þ]+", before) and not re.fullmatch(_NOT_ROMAN_CONTEXT, before, flags=re.IGNORECASE)
            if lang not in _ROMAN_CONTEXT or not (is_context or is_name):
                return numeral
        return words(numeral)

    return re.sub(rf"(?<!\w)(?<![a-zà-öø-ÿ]-){_ROMAN}(?!\w)(?!-[^\W\d_]*[a-zà-öø-ÿ])", cardinal, text)


def roman_to_decimal(str):
    def value(r):
        if r == "I":
            return 1
        if r == "V":
            return 5
        if r == "X":
            return 10
        if r == "L":
            return 50
        if r == "C":
            return 100
        if r == "D":
            return 500
        if r == "M":
            return 1000
        return -1

    res = 0
    i = 0
    while i < len(str):
        s1 = value(str[i])
        if i + 1 < len(str):
            s2 = value(str[i + 1])
            if s1 >= s2:
                # Value of current symbol is greater or equal to the next symbol
                res = res + s1
                i = i + 1
            else:
                # Value of current symbol is greater or equal to the next symbol
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res


def numbers_and_symbols_to_letters(text, lang, verbose=False):
    """
    Convert numbers and symbols to letters
    A number at the beginning of a sentence gets a capital letter, when the text has case or punctuation ("3 comprimés." -> "Trois comprimés.")
    In a language where the numbers cannot be converted, only the symbols known in this language are.
    """
    lang = base_language(lang)
    # "10 000"/"10,000"/"10.000" -> "10000"
    numbers = re.findall(r"\b[1-9]\d{0,2}(?:[\.,]000)+\b", text)
    for n in numbers:
        text = re.sub(r"\b" + n + r"\b", re.sub(r"[,.]", "", n), text)

    text = mark_sentence_starts(text)
    text = clean_number_formats(text, lang=lang)
    text = currencies_after_numbers(text, lang=lang, symbol_to_word=_symbol_to_word.get(lang, {}))

    # Reorder currencies, with cents (1,20€ -> 1 € 20)
    coma = r"\." if lang in ["en", "ar", "ar_tn", "ru"] else ","
    for c in _currencies:
        if c in text:
            c = regex_escape(c)
            text = re.sub(r"\b(\d+)" + coma + r"(\d\d)\s*" + c, r"\1 " + c + r" \2", text)

    text = roman_numbers_to_letters(text, lang=lang)
    text = ordinal_numbers_to_letters(text, lang=lang)
    text = cardinal_numbers_to_letters(text, lang=lang, verbose=verbose)
    text = symbols_to_letters(text, lang, lower_case=False)

    text = agree_one(text, lang)
    return capitalize_sentence_starts(text)


def undigit(s, lang, to="cardinal", type="masc_gen", ignore_first_zeros=False):
    s = re.sub(" ", "", s)
    if "." in s:
        n = float(s)
    else:
        n = int(s)
    if to == "denominator":
        if lang == "fr":
            if s == "2":
                return "demi"
            if s == "3":
                return "tiers"
            if s == "4":
                return "quart"
        elif lang == "en":
            if s == "2":
                return "half"
            if s == "4":
                return "quarter"
        elif lang == "ar_tn" or lang == "ar":
            if s == "2":
                return "نصف"
            if s == "1/3":
                return "ثلث"
            if s == "1/4":
                return "ربع"
        elif lang == "es":
            if s == "2":
                return "mitad"
            if s == "3":
                return "tercio"
        elif lang == "ru":
            if s == "2":
                return "вторых"
            if s == "3":
                return "третьих"
            if s == "4":
                return "четверти"
        to = "ordinal"
    if lang == "ru" and to == "ordinal" and type == "masc_gen":
        return ru_card_to_ord_masc_gen(undigit(s, lang, to="cardinal", ignore_first_zeros=True))
    if not ignore_first_zeros and s.startswith("0") and to == "cardinal":
        numZeros = len(re.findall(r"0+", s)[0])
        zeros = numZeros * (robust_num2words(0, lang=lang, orig=s) + " ")
        if numZeros < len(s):
            return zeros + robust_num2words(n, lang=lang, to=to, orig=s)
        else:
            assert float(s) == 0
            return zeros.strip()
    return robust_num2words(n, lang=lang, to=to, orig=s)


def robust_num2words(x, lang, to="cardinal", orig=""):
    """
    Bugfixes for num2words
    - 20th in French was wrong
    - comma in Arabic
    - avoid overflow error on big numbers
    """
    if lang == "ar" or lang == "ar_tn":
        to = "cardinal"  # See https://github.com/savoirfairelinux/num2words/issues/403
    try:
        res = num2words(x, lang=lang, to=to)
    except Exception as err:
        # Here we should expect a OverflowError, but...
        # * TypeError can occur: https://github.com/savoirfairelinux/num2words/issues/509
        # * IndexError can occur: https://github.com/savoirfairelinux/num2words/issues/511
        # * decimal.InvalidOperation can occur: https://github.com/savoirfairelinux/num2words/issues/511
        # (who knows what else can occur...)
        warnings.warn(f"Got error of type {type(err)} on {x}")
        if x > 0:  # !
            res = " ".join(robust_num2words(int(xi), lang=lang, to=to, orig=xi) for xi in orig)
        else:
            res = _minus.get(lang, _minus["en"]) + " " + robust_num2words(-x, lang=lang, to=to, orig=orig.replace("-", ""))
    if lang == "en":
        res = res.replace(",", "")  # "one thousand, two hundred"
    if lang == "fr" and to == "ordinal":
        res = res.replace("vingtsième", "vingtième")
    elif lang == "ar":
        res = res.replace(",", "فاصيله")
    elif lang == "ar_tn":
        res = res.replace(",", "فاصل")
    return res


def ru_card_to_ord_masc_gen(d):
    separate = d.split(" ")
    last = separate.pop(-1)

    alt_roots = {
        "сто": "сот",
        "сот": "сот",
        "дцать": "дцат",
        "один": "перв",
        "два": "втор",
        "три": "трет",
        "четыре": "четверт",
        "пять": "пят",
        "шесть": "шест",
        "семь": "седьм",
        "восемь": "восьм",
        "девять": "девят",
        "десять": "десят",
        "сорок": "сорок",
    }

    for num, alt_num in alt_roots.items():
        if num in last:
            if num == "три":
                last = re.sub(num, alt_num + "ьего", last)
                break
            else:
                last = re.sub(num, alt_num + "ого", last)
                break

    separate.append(last)
    gen = " ".join(d for d in separate)

    return gen


def ru_fix_ordinals(text):
    """
    Fixes cases of form "10-го / 16-ая etc"
    by putting the corresponding root into its non-nominative form
    and appending the ending.

    """
    term = ["ый", "ой", "ий", "ый", "го", "ая", "ые", "ых"]

    alt_roots = {
        "один": "перв",
        "два": "втор",
        "три": "трет",
        "четыре": "четверт",
        "пять": "пят",
        "шесть": "шест",
        r"\bсемь": "седьм",
        "восемь": "восьм",
        "девять": "девят",
        "десять": "десят",
        "десят": "десят",
        "дцать": "дцат",
        "сорок": "сорок",
        "сто": "сот",
    }

    sep = r"(\s+|\-)"

    for num, alt_num in alt_roots.items():
        if num not in text:
            continue
        for t in term:
            if t not in text:
                continue
            if t == "го":
                if num == "три":
                    text = re.sub(rf"{num}{sep}{t}\b", f"{alt_num}ьего", text)
                else:
                    text = re.sub(rf"{num}{sep}{t}\b", f"{alt_num}ого", text)
            else:
                text = re.sub(rf"{num}{sep}{t}\b", f"{alt_num}{t}", text)

    return text


def ru_convert_dates(text):
    def convert_year(x):
        return "года" if x == "г" else x

    def convert_first_to_digit(x):
        s = x.groups()
        res = undigit(s[0], lang="ru", to="ordinal") + " " + " ".join(map(convert_year, s[1:]))
        return res

    def convert_last_to_digit(x):
        s = x.groups()
        res = " ".join(s[:-1]) + " " + undigit(s[-1], lang="ru", to="ordinal")
        return res

    # Convert day numbers
    text = re.sub(r"\b(\d{1,2})\s+(" + "|".join(_int_to_month["ru"].values()) + r")\b", convert_first_to_digit, text)

    # Convert years number
    text = re.sub(r"\b(" + "|".join(_int_to_month["ru"].values()) + r")\s(\d{4})\b", convert_last_to_digit, text)
    text = re.sub(r"\b(\d{4})\s+(г)", convert_first_to_digit, text)

    return text


_int_to_month = {
    "fr": {
        1: "janvier",
        2: "février",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "août",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "décembre",
    },
    "en": {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    },
    "ar": {
        1: "يناير",
        2: "فبراير",
        3: "مارس",
        4: "أبريل",
        5: "مايو",
        6: "يونيو",
        7: "يوليو",
        8: "أغسطس",
        9: "سبتمبر",
        10: "أكتوبر",
        11: "نوفمبر",
        12: "ديسمبر",
    },
    "ar_islamic": {
        1: "محرم",
        2: "صفر",
        3: "ربيع الأول",
        4: "ربيع الآخر",
        5: "جمادى الأولى",
        6: "جمادى الآخرة",
        7: "رجب",
        8: "شعبان",
        9: "رمضان",
        10: "شوال",
        11: "ذو القعدة",
        12: "ذو الحجة",
    },
    "ar_tn": {
        1: "جانفي",
        2: "فيفري",
        3: "مارس",
        4: "أفريل",
        5: "ماي",
        6: "جوان",
        7: "جويلية",
        8: "أوت",
        9: "سبتمبر",
        10: "أكتوبر",
        11: "نوفمبر",
        12: "ديسمبر",
    },
    "ru": {  # all forms are genetive
        1: "января",
        2: "февраля",
        3: "марта",
        4: "апреля",
        5: "мая",
        6: "июня",
        7: "июля",
        8: "августа",
        9: "сентября",
        10: "октября",
        11: "ноября",
        12: "декабря",
    },
}
_punct_to_word = {
    "fr": {
        ",": "virgule",
        ".": "point",
        ";": "point-virgule",
        ":": "deux-points",
        "?": "point d'interrogation",
        "!": "point d'exclamation",
    },
    "en": {
        ",": "comma",
        ".": "dot",
        ";": "semicolon",
        ":": "colon",
        "?": "question mark",
        "!": "exclamation mark",
    },
    "ar": {
        ",": "فاصيله",
        ".": "فاصيله",  # "نقطه",
        ";": "نقطه وفاصيله",
        ":": "نقطتان",
        "?": "علامه الاستفهام",
        "!": "علامه التعجب",
    },
    "ar_tn": {
        ",": "فاصل",
        ".": "فاصل",  # "نقطه",
        ";": "نقطه و فاصل",
        ":": "نقطتين",
        "?": "علامه الاستفهام",
        "!": "علامه التعجب",
    },
    "ru": {
        ",": "запятая",
        ".": "точка",
        ";": "точка с запятой",
        ":": "двоеточие",
        "?": "вопросительный знак",
        "!": "восклицательный знак",
    },
    "es": {
        ",": "coma",
        ".": "punto",
    },
    "de": {
        ",": "Komma",
        ".": "Punkt",
    },
    "it": {
        ",": "virgola",
        ".": "punto",
    },
    "pt": {
        ",": "vírgula",
        ".": "ponto",
    },
    "nl": {
        ",": "komma",
        ".": "punt",
    },
    "ca": {
        ",": "coma",
        ".": "punt",
    },
    "pl": {
        ",": "przecinek",
        ".": "kropka",
    },
}

_minus = {
    "en": "minus",
    "fr": "moins",
    # "ar": "سالب",
    # "ar_tn": "ناقس",
    "de": "minus",
    "es": "menos",
    "it": "meno",
    "pt": "menos",
    "nl": "min",
    "sv": "minus",
    "da": "minus",
    "nb": "minus",
    "fi": "miinus",
    "tr": "eksi",
    "hu": "mínusz",
    "pl": "minus",
    "cs": "mínus",
    "ru": "минус",
    "uk": "мінус",
    "el": "μείον",
    "bg": "минус",
    "lt": "minus",
    "sl": "minus",
    "hr": "minus",
    "sk": "mínus",
    "et": "miinus",
    "lv": "mīnus",
    "lt": "minus",
    "ro": "minus",
    "he": "מינוס",
    "id": "kurang",
    "vi": "trừ",
    "th": "ลบ",
    "zh": "减",
    "ja": "マイナス",
    "ko": "마이너스",
    "hi": "घटाएं",
    "bn": "কম",
    "gu": "ઘટાવો",
    "ta": "குறைக்க",
    "te": "కనిపించు",
    "kn": "ಕಡಿಮೆ",
    "ml": "കുറയ്ക്കുക",
    "mr": "कमी",
    "pa": "ਘਟਾਓ",
    "ur": "کم کریں",
}
