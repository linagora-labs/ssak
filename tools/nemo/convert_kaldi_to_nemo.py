import os
import re
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from ssak.utils.kaldi_dataset import KaldiDataset
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_kaldi_dirs(root_folder):
    """Recursively find directories that contain a wav.scp file."""
    return [p.parent for p in Path(root_folder).rglob("wav.scp")]


def get_dataset_name(kaldi_input_dataset, remove_casing=True, remove_max_duration=True, remove_split=True):
    splitted_path = kaldi_input_dataset.split(os.sep)
    if splitted_path[-1] == "":
        splitted_path = splitted_path[:-1]
    idx = -1
    moved = True
    while moved:
        moved = True
        if splitted_path[idx].startswith("case") or splitted_path[idx].startswith("nocase") or splitted_path[idx].startswith("recase"):
            idx -= 1
        elif splitted_path[idx].startswith("train") or splitted_path[idx].startswith("dev") or splitted_path[idx].startswith("valid") or splitted_path[idx].startswith("test"):
            idx -= 1
        elif splitted_path[idx].startswith("split"):
            idx -= 1
        elif splitted_path[idx].startswith("fr"):
            idx -= 1
        elif splitted_path[idx].startswith("all"):
            idx -= 1
        else:
            moved = False
    name = "_".join(splitted_path[idx:])
    if remove_casing:
        name = name.replace("_casepunc", "").replace("_nocasepunc", "").replace("_recasepunc", "")
    if remove_max_duration:
        name = re.sub(r"_max\d+", "", name)
    if remove_split:
        name = name.replace("_train", "").replace("_dev", "").replace("_test", "").replace("_valid", "").replace("_all", "")
        name = re.sub(r"_split\d+", "", name)
        name = re.sub(r"_fr\d+", "", name)
    return name


def kaldi_to_nemo(input_folder, name, output_folder):
    """Convert a single Kaldi directory to a NemoDataset."""
    kaldi_dataset = KaldiDataset(name=name, log_folder=output_folder, row_checking_kwargs=dict(show_warnings=False))
    kaldi_dataset.load(input_folder)
    nemo_dataset = NemoDataset(name=name, log_folder=output_folder)
    nemo_dataset.kaldi_to_nemo(kaldi_dataset)
    return nemo_dataset


def convert_single(input_folder, output_folder, name=None, nemo_format="multiturn"):
    """Convert a single Kaldi directory to NeMo format."""
    input_folder = str(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_name = name if name else get_dataset_name(input_folder)
    dataset = kaldi_to_nemo(input_folder, dataset_name, str(output_folder))
    output_file = output_folder / f"manifest_{dataset_name}.jsonl"
    dataset.save(output_file, data_type=nemo_format)
    logger.info(f"Saved {len(dataset)} lines to {output_file}")


def convert_recursive(input_folder, output_folder, name=None, nemo_format="multiturn"):
    """Recursively find Kaldi dirs (by wav.scp) and convert each one."""
    input_root = Path(input_folder).resolve()
    output_root = Path(output_folder).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    kaldi_dirs = find_kaldi_dirs(input_root)
    if not kaldi_dirs:
        logger.error("No folders containing wav.scp were found.")
        return

    for kaldi_dir in kaldi_dirs:
        split_name = kaldi_dir.name
        dataset_name = name if name else split_name
        logger.info(f"Processing split: {split_name}")
        dataset = kaldi_to_nemo(str(kaldi_dir), dataset_name, str(output_root))
        dataset.save(output_root / f"{split_name}.jsonl", data_type=nemo_format)

    logger.info("Recursive conversion completed.")


def convert_batch(input_datasets, output_folder, output_wav_folder=None, nemo_format="multiturn"):
    """Convert multiple Kaldi directories from a JSON config or list."""
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(input_datasets, desc="Converting datasets")
    for input_folder in pbar:
        dataset_name = get_dataset_name(input_folder, remove_casing=False, remove_max_duration=False, remove_split=False)
        pbar.set_description(f"Converting {dataset_name}")
        dataset_output_folder = output_folder / get_dataset_name(input_folder)
        dataset_output_folder.mkdir(parents=True, exist_ok=True)

        output_manifest = dataset_output_folder / f"manifest_{dataset_name}.jsonl"
        if output_manifest.exists():
            logger.info(f"Skipping {input_folder} as {output_manifest} already exists")
            continue

        dataset = kaldi_to_nemo(input_folder, dataset_name, str(dataset_output_folder))
        dataset.save(output_manifest, data_type=nemo_format)
        logger.info(f"Saved {len(dataset)} lines to {output_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kaldi datasets to NeMo format (ASR or multiturn)")
    parser.add_argument("datasets", help="Input Kaldi dataset(s). Can be a single dir, multiple dirs, or a JSON config file", type=str, nargs="+")
    parser.add_argument("--output", help="Output directory", type=str, required=True)
    parser.add_argument("--nemo_format", type=str, default="multiturn", choices=["multiturn", "asr"], help="NeMo output format (default: multiturn)")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recursively find Kaldi dirs (by wav.scp) and convert each one")
    parser.add_argument("--name", type=str, default=None, help="Dataset name override")
    parser.add_argument("--input_data_path", help="Root path prefix to prepend to dataset paths", default=None)
    args = parser.parse_args()

    if args.recursive:
        if len(args.datasets) != 1:
            parser.error("--recursive expects exactly one input folder")
        convert_recursive(args.datasets[0], args.output, name=args.name, nemo_format=args.nemo_format)
    elif len(args.datasets) == 1 and os.path.isfile(args.datasets[0]) and args.datasets[0].endswith(".json"):
        # JSON config mode
        with open(args.datasets[0]) as f:
            input_datasets = json.load(f)
        if args.input_data_path:
            input_datasets = {
                os.path.join(args.input_data_path, k): v
                for k, v in input_datasets.items()
            }
        convert_batch(input_datasets, args.output, nemo_format=args.nemo_format)
    elif len(args.datasets) == 1:
        # Single directory
        input_path = args.datasets[0]
        if args.input_data_path:
            input_path = os.path.join(args.input_data_path, input_path)
        convert_single(input_path, args.output, name=args.name, nemo_format=args.nemo_format)
    else:
        # Multiple directories
        input_datasets = {}
        for ds in args.datasets:
            path = os.path.join(args.input_data_path, ds) if args.input_data_path else ds
            input_datasets[path] = {}
        convert_batch(input_datasets, args.output, nemo_format=args.nemo_format)
