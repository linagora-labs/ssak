from .text_latin import format_text_latin


def format_text(text, language, **kwargs):
    if language in ["fr", "en"]:
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
