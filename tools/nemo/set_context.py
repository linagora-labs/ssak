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
                        force_context=force_context,
                        dataset_name=dataset_name
                    )
                except Exception as e:
                    raise Exception(f"Error in dataset '{dataset_name}' ({file}): {e}") from e
            else:
                logger.info(f"Skipping {output_file} as it already exists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds context to NeMo manifest file(s).")

    parser.add_argument("input", help="Input folder or manifest file.", type=str)

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output_folder", help="Output folder to save results. If not specified, results will be saved in the same folder as input.", type=str, default=None)
    output_group.add_argument("--output_file", help="Output file path (used when input is a single file). Cannot be used together with --output_folder.", type=str, default=None)

    parser.add_argument("--context_file", help="DEPRECATED", type=str, default=None)    
    parser.add_argument("--force_set_context", help="Force setting context even if one is already defined.", default=False, action="store_true")
    parser.add_argument("--task", help="Task for selecting contexts.", choices=["asr"], default="asr")
    parser.add_argument("--language", help="Language for selecting contexts.", choices=["fr", "en"], default="fr")
    args = parser.parse_args()
    
    if args.context_file and not args.output_folder:
        parser.error("--context_file can only be used when --output_folder is specified.")
    if args.output_file and not args.input.lower().endswith(".jsonl"):
        parser.error("When --output_file is specified, the input must be a .jsonl file.")
    if args.input.lower().endswith(".jsonl") and not args.output_file:
        parser.error("When input is a .jsonl file, --output_file must be specified.")
    if args.output_folder and args.input.lower().endswith(".jsonl"):
        parser.error(f"When --output_folder is specified, the input must be a folder.")
    if args.output_file:
        set_context(
            input_file=args.input,
            output_file=args.output_file, 
            context_category="default_contexts",
            task=args.task,
            language=args.language,
            force_context=args.force_set_context,
            dataset_name=args.input.replace(".jsonl", "").replace("manifest_", "")
        )
    else:
        set_context_on_folder(args.input, context_file=args.context_file, output_folder=args.output_folder, task=args.task, language=args.language, force_context=args.force_set_context)