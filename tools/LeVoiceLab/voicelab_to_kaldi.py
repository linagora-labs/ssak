import json
import os
import tempfile
import urllib.request
import zipfile
from operator import itemgetter

from slugify import slugify

from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text import format_text_latin, transliterate

char_set_fr = "abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿñæœ-' "


def speakers_to_genders(speakers, default="m"):
    url = "https://www.insee.fr/fr/statistiques/fichier/2540004/nat2021_csv.zip"
    destzip = tempfile.mktemp(suffix=".zip")
    print("Downloading", url)
    urllib.request.urlretrieve(url, destzip)
    with zipfile.ZipFile(destzip, "r") as z:
        with z.open("nat2021.csv") as f:
            data = f.read().decode("utf-8")
    os.remove(destzip)
    print("ok")
    gender_codes = {"1": "m", "2": "f"}
    name_to_genders = {}
    print("Parsing...")
    for line in data.splitlines()[1:]:
        gender, name, year, count = line.split(";")
        gender = gender_codes[gender]
        count = int(count)
        name_to_genders[name] = name_to_genders.get(name, {})
        name_to_genders[name][gender] = name_to_genders[name].get(gender, 0) + count
    for name in name_to_genders:
        d = name_to_genders[name]
        if len(d) == 1:
            name_to_genders[name] = list(d.keys())[0]
        elif d["m"] > d["f"]:
            name_to_genders[name] = "m"
        else:
            name_to_genders[name] = "f"
    speakers = [s.upper() for s in speakers]
    return list(itemgetter(*speakers)(dict((s, name_to_genders.get(transliterate(s), default)) for s in speakers) | name_to_genders))


def convert(dirin, dirout, annotdir=None):
    if annotdir is None:
        for dirname in sorted(os.listdir(dirin)):
            dirname = os.path.join(dirin, dirname)
            if os.path.isdir(dirname):
                annotdir = dirname
                break
    elif os.path.basename(annotdir) == annotdir:
        annotdir = os.path.join(dirin, annotdir)
    print("Using annotations from", annotdir)

    db_name = slugify(os.path.basename(dirin))

    text = {}
    segments = {}
    wavscp = {}
    utt2spk = {}
    utt2dur = {}
    all_speakers = set()
    all_chars = set()

    for filename in sorted(os.listdir(dirin)):
        f = filename.split(".")
        if len(f) < 3 or f[-2] != "audio":
            continue
        pseudo = ".".join(f[:-2])
        wavname = db_name + "_" + pseudo
        filename = os.path.join(dirin, filename)
        annotfile = os.path.join(annotdir, pseudo + ".annotations.json")
        assert os.path.isfile(annotfile), "Missing annotation file: " + annotfile
        print("Processing " + annotfile)
        with open(annotfile) as f:
            annotations = json.load(f)
        wavscp[wavname] = filename
        for i, transcript in enumerate(annotations["transcripts"]):
            speaker = transcript.get("extra", {}).get("speaker", f"spk-{wavname}-{i:03d}")
            speaker_slug = slugify(speaker).replace("-", "")
            utt = f"{db_name}_{speaker_slug}_{pseudo}_{i:03d}"
            annot = transcript["transcript"]
            start = transcript["timestamp_start_milliseconds"] / 1000.0
            end = transcript["timestamp_end_milliseconds"] / 1000.0
            if end <= start:
                print("Warning: end <= start for", utt)
                continue
            assert utt not in text, "Duplicate utterance: " + utt
            text_formatted = format_text_latin(annot)
            text[utt] = text_formatted
            for char in text_formatted:
                all_chars.add(char)
            segments[utt] = (wavname, start, end)
            utt2spk[utt] = speaker_slug
            utt2dur[utt] = end - start
            all_speakers.add((speaker_slug, speaker))

    os.makedirs(dirout, exist_ok=True)
    with open(os.path.join(dirout, "text"), "w") as f:
        for utt, t in text.items():
            f.write(f"{utt} {t}{os.linesep}")
    with open(os.path.join(dirout, "segments"), "w") as f:
        for utt, (prefix, start, end) in segments.items():
            f.write(f"{utt} {prefix} {start:.3f} {end:.3f}{os.linesep}")
    with open(os.path.join(dirout, "wav.scp"), "w") as f:
        for utt, filename in wavscp.items():
            f.write(f"{utt} sox {filename} -t wav -r 16000 -c 1 - |{os.linesep}")
    with open(os.path.join(dirout, "utt2spk"), "w") as f:
        for utt, speaker in utt2spk.items():
            f.write(f"{utt} {speaker}{os.linesep}")
    with open(os.path.join(dirout, "utt2dur"), "w") as f:
        for utt, duration in utt2dur.items():
            f.write(f"{utt} {duration:.3f}{os.linesep}")

    all_speakers = list(all_speakers)
    speaker_genders = speakers_to_genders([s[1].split()[0] for s in all_speakers])

    with open(os.path.join(dirout, "spk2gender"), "w") as f:
        for spk, gender in zip(all_speakers, speaker_genders):
            f.write(f"{spk[0]} {gender}{os.linesep}")

    # Check kaldi directory
    check_kaldi_dir(dirout)

    for char in all_chars:
        if char not in char_set_fr:
            print("WARNING: Weird character:", char)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 convert.py <dirin> <dirout> [annotdir]")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
