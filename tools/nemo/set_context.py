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

def set_context(input_file, output_file, context_category, task="asr", language="fr", dataset_name=None):
    dataset = NemoDataset(name=dataset_name)
    dataset.load(input_file)
    dataset.set_context_if_none(context_category, task=task, language=language)
    dataset.save(output_file)

def set_context_on_folder(folder, context_file=None, output_folder=None, task="asr", language="fr"):
    if context_file:
        with open(context_file, "r") as f:
            contexts_dict = json.load(f)
    else:
        contexts_dict = dict()
    for root, dirs, files in tqdm(os.walk(folder)):
        if "intermediate" in root or "audios_merged" in root:
            continue
        for file in files:
            if not file.endswith(".jsonl"):
                continue
            dataset_name = file.replace(".jsonl", "").replace("manifest_", "")
            if output_folder:
                output_file = os.path.join(
                    output_folder,
                    os.path.relpath(root, folder),
                    file
                )
            else:
                output_file = os.path.join(root, file.replace(".jsonl", "_context.jsonl"))
            if not os.path.exists(output_file):
                try:
                    set_context(
                        input_file=os.path.join(root, file),
                        output_file=output_file, 
                        context_category=contexts_dict.get(dataset_name, "default_contexts"),
                        task=task,
                        language=language,
                        dataset_name=dataset_name
                    )
                except Exception as e:
                    raise Exception(f"Error in dataset '{dataset_name}' ({file}): {e}") from e
            else:
                logger.info(f"Skipping {output_file} as it already exists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("input_folder", help="Input folder", type=str)
    parser.add_argument("--context_file", help="context_file", type=str, default=None)
    parser.add_argument("--output_folder", help="", type=str, default=None)
    parser.add_argument("--task", help="", choices=["asr"], default="asr")
    parser.add_argument("--language", help="", choices=["fr", "en"], default="fr")
    args = parser.parse_args()
    set_context_on_folder(args.input_folder, context_file=args.context_file, output_folder=args.output_folder, task=args.task, language=args.language)