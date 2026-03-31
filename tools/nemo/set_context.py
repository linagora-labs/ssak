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
    if task=="ast":
        filepath = Path(base_dir, "contexts", "translation", filename)
    else:
        filepath = Path(base_dir, "contexts", filename)
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            contexts = json.load(f)
    else:
        raise FileNotFoundError(f"Context file not found: {filepath}")
    return contexts

def get_available_contexts_categories(task="asr", lang="fr"):
    return list(load_contexts(task=task, lang=lang).keys())

def get_contexts(category, task="asr", language="fr", add_defaults=True):
    contexts = load_contexts(task=task, lang=language)[category]
    if isinstance(contexts, list):
        contexts = {1.0: contexts}
    return contexts

def set_context(input_file, output_file, context_category, task="asr", language="fr", dataset_name=None, force_context=False):
    dataset = NemoDataset(name=dataset_name)
    data_type = dataset.load(input_file)
    contexts = get_contexts(context_category, task=task, language=language)
    if data_type=="asr":
        logger.info(f"Input manifest is in asr format, output will be in multiturn format!")
    dataset.set_context_if_none(contexts, force_set_context=force_context)
    dataset.save(output_file, data_type="multiturn")

def set_context_on_folder(folder, context_category=None, output_folder=None, task="asr", language="fr", force_context=False, force=False):
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
            if not os.path.exists(output_file) or force:
                try:
                    set_context(
                        input_file=os.path.join(root, file),
                        output_file=output_file, 
                        context_category=context_category,
                        task=task,
                        language=language,
                        force_context=force_context,
                        dataset_name=dataset_name
                    )
                except Exception as e:
                    raise Exception(f"Error in dataset '{dataset_name}' ({file}): {e}") from e
            else:
                logger.info(f"Skipping {output_file} as it already exists")
    logger.info(f"Done processing {folder} with context category '{context_category}' to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds context to NeMo manifest file(s).")

    parser.add_argument("input", help="Input folder, manifest file, or YAML config.", type=str)

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output_folder", help="Output folder to save results. If not specified, results will be saved in the same folder as input.", type=str, default=None)
    output_group.add_argument("--output_file", help="Output file path (used when input is a single file). Cannot be used together with --output_folder.", type=str, default=None)

    parser.add_argument("--force", help="Force setting context even if one is already defined.", default=False, action="store_true")
    parser.add_argument("--task", help="Task for selecting contexts.", choices=["asr", "ast"], default="asr")
    parser.add_argument("--context_category", help="", choices=["nocasepunc", "default_contexts"], default="default_contexts")
    def generate_language_choices(languages):
        pairs = [
            f"{src}-{tgt}"
            for src in languages
            for tgt in languages
            if src != tgt
        ]
        return languages + pairs

    parser.add_argument("--language", help="Language for selecting contexts.", choices=generate_language_choices(["fr", "en", "it", "de", "es", "pt", "ar", "nl"]), default="fr")
    args = parser.parse_args()
    
    from ssak.utils.nemo_dataset import resolve_manifest_paths, resolve_output_path
    input_lower = args.input.lower()

    if input_lower.endswith(".jsonl"):
        # Single file mode
        if not args.output_file:
            parser.error("When input is a .jsonl file, --output_file must be specified.")
        set_context(
            input_file=args.input,
            output_file=args.output_file,
            context_category=args.context_category,
            task=args.task,
            language=args.language,
            force_context=args.force,
            dataset_name=Path(args.input).stem.replace("manifest_", "")
        )
    elif input_lower.endswith((".yaml", ".yml")):
        # YAML config mode — resolve manifests from YAML
        if not args.output_folder:
            parser.error("When input is a YAML config, --output_folder must be specified.")
        manifests = resolve_manifest_paths(args.input)
        for mf in manifests:
            out_file = str(resolve_output_path(mf, args.input, args.output_folder))
            dataset_name = mf.stem.replace("manifest_", "")
            try:
                set_context(
                    input_file=str(mf),
                    output_file=out_file,
                    context_category=args.context_category,
                    task=args.task,
                    language=args.language,
                    force_context=args.force,
                    dataset_name=dataset_name
                )
            except Exception as e:
                raise Exception(f"Error in dataset '{dataset_name}' ({mf}): {e}") from e
    else:
        # Folder mode
        if args.output_file:
            parser.error("When input is a folder, use --output_folder instead of --output_file.")
        set_context_on_folder(args.input, context_category=args.context_category, output_folder=args.output_folder, task=args.task, language=args.language, force_context=args.force, force=args.force)