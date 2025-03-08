import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from lang_trans.arabic import buckwalter as bw

from ssak.utils.text_utils import (
    cardinal_numbers_to_letters,
    collapse_whitespace,
    format_special_characters,
    normalize_arabic_currencies,
    remove_punctuations,
    remove_special_characters,
    symbols_to_letters,
)

# Precompiled regular expressions and constant strings
_regex_arabic_chars = "\u0621-\u063A\u0640-\u064A"
_regex_latin_chars = "a-zA-ZÀ-ÖØ-öø-ÿĀ-ž'"
_arabic_punctuation = "؟!،.؛\"'-_:"
_latin_punctuation = "!?.,:;"
_all_punctuation = "".join(list(set(_latin_punctuation + _arabic_punctuation)))

_regex_arabic_punctuation = re.escape(_arabic_punctuation)
_regex_all_punctuation = re.escape(_all_punctuation)
_regex_not_arabic_neither_punctuation = re.compile(r"(?![" + _regex_arabic_chars + r"])\w")
_regex_arabic = re.compile(r"[" + _regex_arabic_chars + "]")


def is_arabic(word):
    # Use the regular expression to check if the word is Arabic
    return bool(_regex_arabic.match(word))


arabic_diacritics = re.compile(
    """
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """,
    re.VERBOSE,
)

script_dir = os.path.dirname(os.path.realpath(__file__))
assets_path = os.path.join(script_dir, "../../assets")

# Cache for JSON files
_cache = {}


def load_json_file(filepath):
    if filepath not in _cache:
        try:
            with open(filepath, encoding="utf-8") as f:
                _cache[filepath] = json.load(f)
        except Exception as err:
            raise RuntimeError(f"Error loading JSON file '{filepath}'") from err
    return _cache[filepath]


def normalize_chars(text):
    normalization_rules = load_json_file(f"{assets_path}/norm-char-ar-tunisian.json")
    regex = re.compile("[" + "".join(map(re.escape, normalization_rules.keys())) + "]")
    return regex.sub(lambda match: normalization_rules[match.group(0)], text)


# Function to normalize Tunisian words
# def normalize_tunisan_words(text):
#     normalization_words = load_json_file(f'{assets_path}/Tunisian_normalization_words.json')
#     pattern = re.compile(r'\b(' + '|'.join(map(re.escape, normalization_words.keys())) + r')\b', re.UNICODE)
#     return pattern.sub(lambda match: normalization_words[match.group(0)], text)


# Optimized function to normalize Tunisian words.
# This function is designed for efficiency, significantly reducing the time required
# to process large texts. For datasets exceeding 4,000,000 words, this optimized
# version can save approximately 8 hours of normalization time compared to previous implementations.
def normalize_tunisan_words(text):
    normalization_words = load_json_file(f"{assets_path}/norm-words-ar-tunisian.json")

    dict_one_word = {}
    dict_mult_word = {}

    for key, value in normalization_words.items():
        if " " in key:
            dict_mult_word[key] = value
        else:
            dict_one_word[key] = value

    for k, v in dict_mult_word.items():
        text = text.replace(k, v)

    normalized_words = [dict_one_word.get(word, word) for word in text.split()]
    normalized_text = " ".join(normalized_words)

    return normalized_text


# to transliterate arabi into bw
def bw_transliterate(text):
    return bw.transliterate(text)


# This removes the apostrophes and hyphens that are not part of words
def remove_outer_apostrophes_and_hyphens(text):
    pattern = r"(?<!\w)[\'-]|[\'-](?!\w)"  # Regular expression to match apostrophes or hyphens not inside words
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def remove_arabic_diacritics(text):
    return re.sub(arabic_diacritics, "", text)


def convert_hindi_numbers(text):
    return text.translate(str.maketrans("۰۱۲۳٤٥٦۷۸۹", "0123456789"))


def digit2word(text, lang):
    text = convert_hindi_numbers(text)
    return cardinal_numbers_to_letters(text, lang=lang)


def convert_punct_to_arabic(text):
    return text.translate(str.maketrans(";,", "؛،"))


def remove_url(text):
    return re.sub(r"http://\S+|https://\S+", " ", text)


def get_arabic_only(text, keep_punc=False, keep_latin_chars=False):
    what_to_keep = _regex_arabic_chars
    if keep_punc:
        what_to_keep += _regex_all_punctuation if keep_latin_chars else _regex_arabic_punctuation
    if keep_latin_chars:
        what_to_keep += _regex_latin_chars
    return re.sub(r"[^" + what_to_keep + "]+", " ", text)


def unglue_arabic_and_latin_chars(line):
    line = re.sub(r"(" + _regex_arabic.pattern + ")(" + _regex_not_arabic_neither_punctuation.pattern + ")", r"\1 \2", line)
    line = re.sub(r"(" + _regex_not_arabic_neither_punctuation.pattern + ")(" + _regex_arabic.pattern + ")", r"\1 \2", line)
    return re.sub(" {2,}", " ", line)


def remove_repeated_ar_chars(word, maximum=2):
    pattern = "(" + _regex_arabic.pattern + r")\1{" + str(maximum) + ",}"
    return re.sub(pattern, r"\1" * maximum, word)


def remove_long_arabic_words(text, threshold=15):
    words = text.split()
    filtered_words = [word for word in words if not is_arabic(word) or (is_arabic(word) and len(word) < threshold)]
    return " ".join(filtered_words)


def format_text_ar(line, keep_punc=False, keep_latin_chars=True, bw=False, lang="ar", normalize_dialect_words=False):
    try:
        line = remove_url(line)
        line = symbols_to_letters(line, lang=lang, lower_case=False)
        line = normalize_arabic_currencies(line, lang=lang)
        line = remove_arabic_diacritics(line)
        line = normalize_chars(line)
        line = digit2word(line, lang=lang)
        if normalize_dialect_words and lang == "ar_tn":
            line = normalize_tunisan_words(line)
        line = convert_punct_to_arabic(line)
        line = remove_repeated_ar_chars(line)
        line = remove_long_arabic_words(line)
        if not keep_latin_chars:
            line = get_arabic_only(line, keep_punc=keep_punc)
        else:
            line = line.lower()
            line = remove_outer_apostrophes_and_hyphens(line)
            line = unglue_arabic_and_latin_chars(line)
            line = remove_special_characters(line)
            line = format_special_characters(line, remove_ligatures=True)
            if not keep_punc:
                line = remove_punctuations(line, " ")
        if bw:
            line = bw_transliterate(line)
    except Exception as err:
        print(f'Error when processing line: "{line}"')
        raise err
    return collapse_whitespace(line)


def process_lines(lines, kwargs):
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_line = {executor.submit(format_text_ar, line, **kwargs): line for line in lines}
        for future in as_completed(future_to_line):
            line = future_to_line[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Line {line} generated an exception: {exc}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="An input file, or an input string", type=str, nargs="+")
    parser.add_argument("--language", help="Whether to use 'ar or ar_tn'", type=str, default="ar")
    parser.add_argument("--normalize_dialect_words", help="Whether to Normalize Tunisian words", default=False, action="store_true")
    parser.add_argument("--keep_punc", help="Whether to keep punctuations", default=False, action="store_true")
    parser.add_argument(
        "--keep_latin_chars",
        help="Whether to keep latin characters (otherwise, only arabic characters)",
        default=False,
        action="store_true",
    )
    parser.add_argument("--bw", help="Whether to transliterate text into buckwalter encoding.", default=False, action="store_true")
    args = parser.parse_args()

    if args.language == "tn":
        args.language = "ar_tn"

    kwargs = {
        "keep_punc": args.keep_punc,
        "keep_latin_chars": args.keep_latin_chars,
        "bw": args.bw,
        "lang": args.language,
        "normalize_dialect_words": args.normalize_dialect_words,
    }

    input_data = args.input

    if len(input_data) == 1 and os.path.isfile(input_data[0]):
        with open(input_data[0]) as f:
            text = f.read()
            lines = text.splitlines()
            results = process_lines(lines, kwargs)
            for result in results:
                print(result)
    else:
        input_text = " ".join(input_data)
        print(format_text_ar(input_text, **kwargs))
