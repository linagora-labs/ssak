import argparse
import json
import logging
import shutil

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# En règle générale, les conversations entre adulte se font à un débit de 200 mots/minute.
# Certains montent la cadence jusqu’à 300 mots/minute. Alors que pour des enregistrements audio par exemple, on préconise un rythme de 150 mots/minute pour être bien audible.
# a bit less than 5 characters per word, 4,65 for journalistic, and 4,89 for litterature
# if we add 2 more characters (spaces) it means we have around 7 characters per word (8 to be large)
# fast reading person: 160wpm (2.7wps)
# speaking person: 200wpm (3.3wps)
# ultra fast: 300wpm (5wps) (probably possible to happen for very short duration but not for more than a few sec)
# so for ultra fast 1s we have around 5*8=40 characters
# So, we can safely say that we it can't go over 50
# 5s: 3.3*8*5=132c/ 5*8*5=200
# 10s: 3.3*8*10=264c/ 400c
# 20s: 528c
# 30s: 792c

INCOHERENT_THREEHOLD = {1: 50, 5: 200, 10: 350, 20: 550, 30: 700}


def filter_incoherent_segments(input_file, filtered_out_file):
    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()
        data = [json.loads(l) for l in lines]
    ct_dict = {i: 0 for i in list(INCOHERENT_THREEHOLD.values())}
    ct = 0
    with open(input_file + ".tmp", "w", encoding="utf-8") as f, open(filtered_out_file, "w", encoding="utf-8") as log:
        for i, row in enumerate(tqdm(data, desc="Checking for incoherent texts lengths")):
            dur = float(row["duration"])
            max_text = None
            for k, v in INCOHERENT_THREEHOLD.items():
                if dur < k:
                    max_text = v
                    break
            if max_text is None:
                max_text = list(INCOHERENT_THREEHOLD.values())[-1]
            if len(row["text"]) > max_text:
                ct += 1
                ct_dict[max_text] = ct_dict[max_text] + 1
                json.dump(row, log, ensure_ascii=False)
                log.write("\n")
            else:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    print(f"Find {ct} long texts in {input_file}")
    print(f"Removed: {ct_dict}")
    shutil.move(input_file + ".tmp", input_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines from nemo manifest")
    parser.add_argument("file", help="Input file", type=str)
    parser.add_argument("output", help="output file", type=str)
    # parser.add_argument('--max_char', help="Depends on segments max length", type=int, default=700)
    args = parser.parse_args()
    filter_incoherent_segments(args.file, args.output)
