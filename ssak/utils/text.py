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
