import re
import string
import unicodedata

_whitespace_re = re.compile(r"[\s\r\n]+")


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def remove_parenthesis(text):
    return collapse_whitespace(re.sub(r"\([^)]*\)", "", text))


def regex_escape(text):
    return re.escape(text)


_punctuation_strong = string.punctuation + "。，！？：”、…" + "؟،؛" + "—" + "«°»×‹›•“–‘″‘"
_punctuation = "".join(c for c in _punctuation_strong if c not in ["-", "'"])

_punctuation_strong_regex = r"[" + regex_escape(_punctuation_strong) + "]"
_punctuation_regex = r"[" + regex_escape(_punctuation) + "]"


def remove_punctuations(text, replace_by="", strong=False):
    if strong:
        return re.sub(_punctuation_strong_regex, replace_by, text)
    return re.sub(_punctuation_regex, replace_by, text)


_non_printable_pattern = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"  # r'[\x00-\x1F\x7F-\x9F]'


def format_special_characters(text, remove_ligatures=False, format_whitespace=True):
    for before, after in [
        ("â", "â"),
        ("à", "à"),
        ("á", "á"),
        ("ê", "ê"),
        ("é", "é"),
        ("è", "è"),
        ("ô", "ô"),
        ("û", "û"),
        ("î", "î"),
        ("\x92", "'"),
        ("…", "..."),
        (r"[«“][^\S\r\n]*", '"'),
        (r"[^\S\r\n]*[»”″„]", '"'),
        (r"(``|'')", '"'),
        (r"[’‘‛ʿ]", "'"),
        ("‚", ","),
        (r"–", "-"),
        # non
        ("[  ]", " "),  # weird whitespace
        (_non_printable_pattern, ""),  # non-printable characters
        ("·", "."),
        (r"ᵉʳ", "er"),
        (r"ᵉ", "e"),
    ]:
        text = re.sub(before, after, text)

    if remove_ligatures:
        text = re.sub("œ", "oe", text)
        text = re.sub("æ", "ae", text)
        text = re.sub("ﬁ", "fi", text)
        text = re.sub("ﬂ", "fl", text)
        text = re.sub("ĳ", "ij", text)
        text = re.sub("Œ", "Oe", text)
        text = re.sub("Æ", "Ae", text)

    text = re.sub(" - | -$|^- ", " ", text)
    # text = re.sub('--+',' ', text)
    # text = re.sub('—+',' ', text)
    # text = re.sub('#+',' ', text)
    # text = re.sub('_',' ', text)
    # text = re.sub('\{|\}|\(|\)|\[|\]|"|=',' ',text)
    # text = re.sub('(\.|\?|\!|,|;|:)-',r'\1 ', text)
    # text = re.sub("'+", "'", text)
    # text = re.sub('\*+', ' ', text)

    if format_whitespace:
        text = collapse_whitespace(text)

    return text


def remove_quotes(text):
    text = re.sub(r"[\"]", "", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r" '([^']+)'", r" \1", text)
    return text


def remove_special_words(
    text,
    glue_apostrophe=True,
    glue_dash=None,
):
    """
    Small process designed for text that has ALREADY been processed (ex: "8" -> "huit"), but some special words might still be present (ex: "<noise>")
    """
    # sometimes empty text could have been transformed to None (ex: in CSV)
    if not text:
        return ""

    try:
        text = re.sub(r"<.*?>", "", text)
    except:
        print("PROBLEM WITH TEXT:", text, type(text))
        text = re.sub(r"<.*?>", "", text)

    if glue_apostrophe is True:
        text = re.sub(r"[^\S]+'[^\S]+", "'", text)
    elif glue_apostrophe is False:
        text = re.sub(r"'", "' ", text).strip()

    if glue_dash is True:
        text = re.sub(r"[^\S]+\-[^\S]+", "-", text)
    elif glue_dash is False:
        text = re.sub(r"\-", "- ", text).strip()
    elif glue_dash == "right":
        text = re.sub(r"\-[^\S]+", "-", text)
        text = re.sub("-", " -", text)
    elif glue_dash == "left":
        text = re.sub(r"[^\S]+\-", "-", text)
        text = re.sub("-", "- ", text)

    text = collapse_whitespace(text)

    return text


# this function can split sentences.
def split_around(text, punctuation=_punctuation, must_not_end_with=None, has_to_start_with=None, min_length=0, glue_right=False):
    """
    Split text around punctuation.

    Args:
        text (str): text to split
        punctuation (str): punctuation to split around
        must_not_end_with (str): if the sentence ends with this *regex*, it will be glued to the next sentence
        has_to_start_with (str): if the sentence does not start with this *regex*, it will be glued to the previous sentence
        min_length (int): if the sentence is shorter than this, it will be glued to the next sentence
        glue_right (bool): if True, glue the punctuations to the right (otherwise to the left)
    """
    sentences = re.findall(rf"([^{re.escape(punctuation)}]+)([{re.escape(punctuation)}]+|$)", text)
    if glue_right:
        sentences = zip([""] + [s[1] for s in sentences], [s[0] for s in sentences] + [""])
    sentences = ["".join(s) for s in sentences]
    if must_not_end_with or has_to_start_with or min_length:
        new_sentences = []
        has_to_be_glued = False
        for s in sentences:
            next_has_to_be_glued = False
            if must_not_end_with and re.match(r".*" + must_not_end_with + r"$", s):
                next_has_to_be_glued = True

            if has_to_start_with and len(new_sentences) and len(s) and not re.match(r"^" + has_to_start_with, s):
                has_to_be_glued = True

            if has_to_be_glued:
                new_sentences[-1] += s
            else:
                new_sentences.append(s)

            if min_length and len(new_sentences[-1]) < min_length:
                next_has_to_be_glued = True
            has_to_be_glued = next_has_to_be_glued
        sentences = new_sentences

    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s]
    return sentences


def split_around_apostrophe(text):
    words = text.split("'")
    words[:-1] = [w + "'" for w in words[:-1]]
    return words


def split_around_space_and_apostrophe(text):
    # Note: re.split(r"[' ]", text) does not work (remove the apostrophe)
    words = text.strip().split()
    words = [split_around_apostrophe(w) for w in words if w]
    words = [w for ws in words for w in ws]
    return words


def transliterate(c):
    # Transliterates a character to its closest ASCII equivalent.
    # For example, "é" becomes "e".
    # This is useful for converting Vietnamese text to ASCII.
    # See https://stackoverflow.com/a/517974/446579
    return unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")
