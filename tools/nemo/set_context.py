import argparse
import json
import logging
import os

from pathlib import Path
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

def get_contexts(category, task="asr", language="fr", add_defaults=True):
    contexts_dict = load_contexts(task=task, lang=language)
    contexts = contexts_dict.get(category, [])
    if (not contexts or add_defaults) and not category == "default_contexts":
        contexts.extend(contexts_dict.get("default_contexts", []))
    return contexts

def set_context(input_file, output_file, context_category, task="asr", language="fr", dataset_name=None, force_context=False):
    dataset = NemoDataset(name=dataset_name)
    data_type = dataset.load(input_file)
    contexts = get_contexts(context_category, task=task, language=language)
    if data_type=="asr":
        logger.info(f"Input manifest is in asr format, output will be in multiturn format!")
    dataset.set_context_if_none(contexts, force_set_context=force_context)
    dataset.save(output_file, type="multiturn")

def set_context_on_folder(folder, context_file=None, output_folder=None, task="asr", language="fr", force_context=False):
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
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_folder", help="Input folder", type=str)
    parser.add_argument("--context_file", help="context_file", type=str, default=None)
    parser.add_argument("--output_folder", help="", type=str, default=None)
    parser.add_argument("--task", help="", choices=["asr"], default="asr")
    parser.add_argument("--language", help="", choices=["fr", "en"], default="fr")
    parser.add_argument("--force_set_context", help="Set context even if not none", default=False, action="store_true")
    args = parser.parse_args()
    set_context_on_folder(args.input_folder, context_file=args.context_file, output_folder=args.output_folder, task=args.task, language=args.language, force_context=args.force_set_context)