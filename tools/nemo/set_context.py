import argparse
import json
import logging
import os
import re
import random
import shutil

import numpy as np
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_context(input_file, output_file, context_category):
    dataset = NemoDataset()
    dataset.load(input_file)
    dataset.set_context_if_none(context_category)
    dataset.save(output_file+".tmp")
    shutil.move(output_file+".tmp", output_file)

def set_context_on_folder(folder, context_file):
    with open(context_file, "r") as f:
        contexts_dict = json.load(f)
    for root, dirs, files in tqdm(os.walk(folder)):
        for file in files:
            dataset_name = file.split("_")[1:]
            set_context(os.path.join(root, file), os.path.join(root, file.replace(".json", "_context.json")), context_category=contexts_dict.get(dataset_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("folder", help="Input folder", type=str)
    parser.add_argument("context_file", help="context_file", type=str)
    args = parser.parse_args()
    set_context_on_folder(args.folder, args.context_file)