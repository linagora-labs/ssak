import re

# Some functions are only there to be imported from this module
from .text_basic import (  # noqa: F401
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
from .text_latin import format_text_latin, numbers_and_symbols_to_letters  # noqa: F401
from .text_numbers import base_language, normalize_spoken_numbers
from .text_utils import remove_special_characters  # noqa: F401


def __getattr__(name):
    # Arabic and Russian are imported only when needed (they have specific dependencies)
    if name == "format_text_ar":
        from .text_ar import format_text_ar

        return format_text_ar
    if name == "format_text_ru":
        from .text_ru import format_text_ru

        return format_text_ru
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def format_text(text, language, **kwargs):
    if language in ["fr", "en", "es", "de", "it", "pt", "nl"]:
        return format_text_latin(text, lang=language, **kwargs)
    if language.startswith("ar"):
        from .text_ar import format_text_ar

        if "lang" in kwargs:
            lang = kwargs.pop("lang")
            assert lang == language, f"{lang=} from kwargs, inconsistent with {language=}"
        return format_text_ar(text, lang=language, **kwargs)
    if language == "ru":
        from .text_ru import format_text_ru

        return format_text_ru(text, **kwargs)
    raise NotImplementedError(f"Language {language} not supported yet")


_punctuation_marks = ".,?!;:…؟،؛"


def normalize_text(
    text,
    language,
    case_sensitive=False,
    keep_punctuation=False,
    remove_apostrophes_and_dashes=False,
    keep_disfluencies=False,
    ignore_final_s=False,
    for_wer=False,
):
    """
    Normalize a text, as before computing the WER (see compute_wer, which has the same options) when for_wer is True.

    :param language: language code ("fr", "en", "ar", "ar_tn", ...)
    :param case_sensitive: whether to keep the case
    :param keep_punctuation: whether to keep the punctuation marks. The other marks are removed after the conversion of the numbers,
        acronyms, ... ("1,5" -> "un virgule cinq", "S.N.C.F." -> "SNCF")
    :param remove_apostrophes_and_dashes: whether to replace the characters that are in words, but not letters, by spaces
        ("l'état" -> "l état", "rendez-vous" -> "rendez vous")
    :param keep_disfluencies: whether to keep the disfluencies ("euh", "uh", "hmm", ...). They are only removed in the languages with latin alphabet
    :param ignore_final_s: whether to remove the "s" at the end of the words ("les" -> "le")
    :param for_wer: whether to also do what only makes sense to compare two texts:
        - the same form for what is said the same way, even if it is not correct ("une femme" -> "un femme", "two hundred and fifty" -> "two hundred fifty")
        - the punctuation marks, if kept, separated from the words
        - (Arabic) without any diacritics, and with the same letter for all the alef forms
    """
    language = base_language(language)
    kwargs = {"lower_case": not case_sensitive, "keep_punc": keep_punctuation}
    if language.startswith("ar"):
        kwargs.update(keep_latin_chars=True, normalize_dialect_words=language.endswith("tn"))
    elif language != "ru":
        kwargs.update(remove_disfluencies=not keep_disfluencies)
    text = format_text(text, language, **kwargs)

    if for_wer:
        text = normalize_spoken_numbers(text, language)
        if language.startswith("ar"):
            from .text_ar import normalize_arabic_letters

            text = normalize_arabic_letters(text)
        if keep_punctuation:
            text = re.sub(f"([{_punctuation_marks}])", r" \1 ", text)
    if remove_apostrophes_and_dashes:
        text = re.sub(rf"[^\w{_punctuation_marks}]" if keep_punctuation else r"[^\w]", " ", text)
    if ignore_final_s:
        text = re.sub(r"(\w)s\b", r"\1", text)
    return collapse_whitespace(text)
