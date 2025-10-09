import argparse
import logging
import os
import re
from functools import partial

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


def get_charset():
    french_charset = re.compile(r"^[0-9a-zA-ZàâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸ'’ \-.,;:!?]+$")
    return french_charset


def incoherence_char(duration, text, charset=None):
    if not charset.match(text):
        return True
    return False


def get_too_short_args():
    INCOHERENT_THREEHOLD = {0.5: 2, 1: 4, 5: 12, 10: 25, 20: 42, 30: 60}
    x = np.array(list(INCOHERENT_THREEHOLD.keys()))
    y = np.array(list(INCOHERENT_THREEHOLD.values()))
    spline_short = make_interp_spline(x, y, k=3)
    return x, y, spline_short


def get_too_long_args():
    INCOHERENT_THREEHOLD = {0.5: 30, 1: 50, 5: 200, 10: 350, 20: 580, 30: 750}
    x = np.array(list(INCOHERENT_THREEHOLD.keys()))
    y = np.array(list(INCOHERENT_THREEHOLD.values()))
    spline_long = make_interp_spline(x, y, k=3)
    return x, y, spline_long


def incoherence_curve(duration, text, long_mode=True, x=None, y=None, spline=None):
    value = None
    if duration <= x[0]:
        value = y[0]
    elif duration >= x[-1]:
        value = y[-1]
    else:
        value = spline(duration)
    if long_mode:
        return len(text) > value
    return len(text) < value

def filter_incoherent_segments_file(input_file, output_file, mode="charset"):
    data = NemoDataset()
    type = data.load(input_file, type=None)
    if output_file and os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists, skipping")
    elif output_file is None:
        output_file = args.file
    dataset = filter_incoherent_segments(data, mode=mode)
    dataset.save(output_file, type=type)

def filter_incoherent_segments(input_dataset, filtered_out_file, mode="charset"):
    if mode == "too_long":
        x, y, spline_long = get_too_long_args()
        incoherence_function = partial(incoherence_curve, long_mode=True, x=x, y=y, spline=spline_long)
    elif mode == "charset":
        charset = get_charset()
        incoherence_function = partial(incoherence_char, charset=charset)
    elif mode == "too_short":
        x, y, spline_short = get_too_short_args()
        incoherence_function = partial(incoherence_curve, long_mode=False, x=x, y=y, spline=spline_short)
    else:
        raise ValueError(f"Unknown mode {mode}")
    new_data = NemoDataset(name=input_dataset.name, log_folder=input_dataset.log_folder)
    removed_data = NemoDataset()
    os.makedirs(os.path.dirname(filtered_out_file), exist_ok=True)
    for i, row in enumerate(tqdm(input_dataset, desc="Checking for incoherent texts lengths")):
        is_incoherent = incoherence_function(row.duration, row.answer)
        if is_incoherent:
            removed_data.append(row)
        else:
            new_data.append(row)
    removed_data.save(filtered_out_file, type="asr")
    logger.info(f"Find {len(removed_data)} incoherence segments in {input_dataset} using mode {mode} and {incoherence_function.func.__name__}")
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("file", help="Input file", type=str)
    parser.add_argument("output", help="output file", type=str)
    parser.add_argument("--mode", default="charset", help="length or language", type=str)
    # parser.add_argument('--max_char', help="Depends on segments max length", type=int, default=700)
    args = parser.parse_args()
    filter_incoherent_segments_file(args.file, args.output, args.mode)