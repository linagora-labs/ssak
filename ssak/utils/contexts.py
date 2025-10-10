import argparse
import json
import logging
import os
import re
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_contexts(task="asr", lang="fr"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{lang}_{task}_contexts.json"
    filepath = Path(base_dir, "contexts", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            contexts = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Context file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {filepath}")
    return contexts

def get_available_contexts_categories(task="asr", lang="fr"):
    return list(load_contexts(task=task, lang=lang).keys())

def get_contexts(category, task="asr", lang="fr", add_defaults=True):
    contexts_dict = load_contexts(task=task, lang=lang)
    contexts = contexts_dict.get(category, [])
    if (not contexts or add_defaults) and not category == "default_contexts":
        contexts.extend(contexts_dict.get("default_contexts", []))
    return contexts