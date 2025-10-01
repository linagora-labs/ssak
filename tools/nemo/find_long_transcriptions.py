import argparse
import json
import logging
import os
import re
import shutil

import numpy as np
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset

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

french_charset = re.compile(r"^[0-9a-zA-ZàâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸ'’ \-.,;:!?]+$")
spline = None
x = None
y = None


def incoherence_char(duration, text):
    if not french_charset.match(text):
        return True
    return False

def incoherence_curve_too_short(duration, text):
    global spline
    global x
    global y
    if spline is None:
        INCOHERENT_THREEHOLD = {1: 0, 5: 10, 10: 20, 20: 30, 30: 40}
        x = np.array(list(INCOHERENT_THREEHOLD.keys()))
        y = np.array(list(INCOHERENT_THREEHOLD.values()))
        spline = make_interp_spline(x, y, k=3)
    value = None
    if duration <= x[0]:
        value = y[0]
    elif duration >= x[-1]:
        value = y[-1]
    else:
        value = spline(duration)
    return len(text) < value

def incoherence_curve(duration, text):
    global spline
    global x
    global y
    if spline is None:
        INCOHERENT_THREEHOLD = {1: 50, 5: 200, 10: 350, 20: 580, 30: 750}
        x = np.array(list(INCOHERENT_THREEHOLD.keys()))
        y = np.array(list(INCOHERENT_THREEHOLD.values()))
        spline = make_interp_spline(x, y, k=3)
    value = None
    if duration <= x[0]:
        value = y[0]
    elif duration >= x[-1]:
        value = y[-1]
    else:
        value = spline(duration)
    return len(text) > value


def filter_incoherent_segments(input_file, filtered_out_file, mode="charset"):
    if mode == "length":
        incoherence_function = incoherence_curve
    elif mode == "charset":
        incoherence_function = incoherence_char
    elif mode == "too_short":
        incoherence_function = incoherence_curve_too_short
    else:
        raise ValueError(f"Unknown mode {mode}")
    ct = 0
    data = NemoDataset()
    type = data.load(input_file, type=None)
    new_data = NemoDataset()
    removed_data = NemoDataset()
    os.makedirs(os.path.dirname(filtered_out_file), exist_ok=True)
    for i, row in enumerate(tqdm(data, desc="Checking for incoherent texts lengths")):
        if incoherence_function(row.duration, row.answer):
            ct += 1
            removed_data.append(row)
        else:
            new_data.append(row)
    new_data.save(input_file, type=type)
    removed_data.save(filtered_out_file, type=type)
    print(f"Find {ct} long texts in {input_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("file", help="Input file", type=str)
    parser.add_argument("output", help="output file", type=str)
    parser.add_argument("--mode", default="charset", help="length or language", type=str)
    # parser.add_argument('--max_char', help="Depends on segments max length", type=int, default=700)
    args = parser.parse_args()
    filter_incoherent_segments(args.file, args.output, args.mode)
