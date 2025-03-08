import os
import re
import shutil
import subprocess

from envsubst import envsubst


def parse_kaldi_wavscp(wavscp):
    # TODO: the reading of wav.scp is a bit crude...
    with open(wavscp) as f:
        wav = {}
        for line in f:
            fields = line.strip().split()
            fields = [f for f in fields if f != "|"]
            wavid = fields[0]
            if line.find("'") >= 0:
                i1 = line.find("'")
                i2 = line.find("'", i1 + 1)
                path = line[i1 + 1 : i2]
            elif len(fields) > 2:
                # examples:
                # sox file.wav -t wav -r 16000 -b 16 - |
                # flac -c -d -s -f file.flac |
                if os.path.basename(fields[1]) == "sox":
                    path = fields[2]
                elif os.path.basename(fields[1]) == "flac":
                    path = fields[-1]
                else:
                    raise RuntimeError(f"Unknown wav.scp format with {fields[1]}")
            else:
                path = fields[1]
            # Look for environment variables in the path
            if "$" in path:
                path = envsubst(path)
            wav[wavid] = path

    return wav


def parse_line(line):
    id_text = line.strip().split(" ", 1)
    if len(id_text) == 1:
        return id_text[0], ""
    return id_text


SPECIAL_CHARS = {
    "en": "",
    "fr": "".join(func("àâäéèêëîïôöùûüÿç") for func in [str.upper, str.lower]),
    "es": "".join(func("áéíóúüñ") for func in [str.upper, str.lower]),
    "de": "".join(func("äöüß") for func in [str.upper, str.lower]),
    "it": "".join(func("àèéìíîòóùú") for func in [str.upper, str.lower]),
    "pt": "".join(func("áâãàéêíóôõúüç") for func in [str.upper, str.lower]),
    "ru": "".join(func("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") for func in [str.upper, str.lower]),
    "tr": "".join(func("âçğıöşü") for func in [str.upper, str.lower]),
    "ar": "".join(func("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويىي") for func in [str.upper, str.lower]),
}


def check_kaldi_dir(dirname, language=None, strict_sort=False):
    strict_sort = "true" if strict_sort else "false"
    tool_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "tools", "kaldi", "utils")

    if os.path.isfile(os.path.join(dirname, "text")):
        with open(os.path.join(dirname, "text")) as f:
            texts = dict(parse_line(line) for line in f)

    p = subprocess.Popen([tool_dir + "/fix_data_dir.sh", dirname, strict_sort])
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ERROR when running: {tool_dir}/fix_data_dir.sh {dirname}")

    if not os.path.isfile(os.path.join(dirname, "utt2dur")):
        p = subprocess.Popen([tool_dir + "/get_utt2dur.sh", dirname], stderr=subprocess.PIPE)
        p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ERROR when running get_utt2dur.sh")

    p = subprocess.Popen([tool_dir + "/validate_data_dir.sh", "--no-feats", "" if strict_sort == "true" else "--no-spk-sort", dirname])
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ERROR when running validate_data_dir.sh")

    # Report
    # - if some ids were filtered out
    # - if some texts are empty
    # - if there were weird characters in the text
    weird_characters = {}
    with open(os.path.join(dirname, "text"), encoding="utf8") as f:
        ids = [s.split(" ", 1)[0] for s in f.read().splitlines()]
    missing_things = False
    if len(texts) != len(ids) or language:
        check_missing = len(texts) != len(ids)
        regex_not_weird = r"[a-zA-Z0-9 \.,\?\!\-\'\:\;" + re.escape(SPECIAL_CHARS.get(language, "")) + r"]"
        for id, text in texts.items():
            if check_missing and id not in ids:
                print("WARNING: Filtered out:", id, text)
                missing_things = True
            elif not text.strip():
                print("WARNING: Empty text:", id, text)
            elif language:
                # Filter out usual characters
                weirdos = re.sub(regex_not_weird, "", text)
                for c in weirdos:
                    if c in weird_characters:
                        continue
                    weird_characters[c] = text
    for c, example in weird_characters.items():
        print(f"WARNING: Got character {c} (example: {example})")

    for tmpdir in ".backup", "log", "split4utt":
        if missing_things and tmpdir == ".backup":
            continue
        tmpdir = os.path.join(dirname, tmpdir)
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check a kaldi folder")
    parser.add_argument("dirname", help="Input kaldi folder", type=str)
    parser.add_argument(
        "--strict_sort",
        default=False,
        action="store_true",
        help="If sort on speakers must be equal to sort on utterances",
    )
    args = parser.parse_args()

    check_kaldi_dir(args.dirname, strict_sort=args.strict_sort)
