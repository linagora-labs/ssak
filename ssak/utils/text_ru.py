import re

from ssak.utils.text_utils import (
    collapse_whitespace,
    format_special_characters,
    numbers_and_symbols_to_letters,
    remove_parenthesis,
    remove_punctuations,
)


def format_text_ru(
    text,
    lower_case=True,
    keep_punc=False,
    keep_hyphen=False,
    remove_optional_diacritics=True,
    force_transliteration=True,
):
    """

    Args:
        text: input text to normalize
        lower_case: switch to lower case
        keep_punc: keep punctuation or not
        keep_hyphen: keep hyphen even if other punctuations are removed
        remove_optional_diacritics: replaces all ё with е, does not change 'й'
        force_transliteration: transliterates all non-cyrillic sentences to cyrillic

    Returns:
        normalized text

    """
    import cyrtranslit

    lang = "ru"

    if remove_optional_diacritics:
        text = re.sub("Ё", "Е", text)
        text = re.sub("ё", "е", text)

    if force_transliteration:
        if not re.match(r".*[ЁёА-я]", text):
            text = cyrtranslit.to_cyrillic(text, lang)

    text = numbers_and_symbols_to_letters(text, lang="ru")

    if not keep_punc:
        if keep_hyphen:
            text = re.sub("'", " ", text)
        else:
            text = re.sub("[-']", " ", text)
        text = remove_punctuations(text)

    text = format_special_characters(text)

    text = remove_parenthesis(text)

    if lower_case:
        text = text.lower()

    return collapse_whitespace(text)


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1]) as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_ru(line))
    else:
        print(format_text_ru(" ".join(sys.argv[1:])))
