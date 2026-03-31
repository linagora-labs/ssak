import argparse
import logging
import os

from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, resolve_manifest_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_manifest(filename):
    if filename.startswith("all_manifests"):
        return False
    if not filename.endswith(".jsonl"):
        return False
    if not filename.startswith("manifest"):
        return False
    if ".tmp" in filename or ".filter" in filename:
        return False
    return True


def merge_manifests(inputs, output):
    if os.path.exists(output):
        raise FileExistsError(f"Output file {output} already exists")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # Resolve inputs: supports files, directories, YAML configs, and lists
    input_files = resolve_manifest_paths(inputs)
    if not input_files:
        logger.warning(f"No manifest files found for inputs: {inputs}")
        return
    merged_data = NemoDataset()
    for input_file in tqdm(input_files, desc="Merging manifest files"):
        name, _ = os.path.splitext(input_file)
        name = os.path.basename(name)
        name = name.split("_")
        split = "all"
        language = "fr"
        name.pop(name.index("manifest"))
        for i in reversed(name):
            if i in ["train", "test", "dev", "validation"]:
                split = i
                name.pop(name.index(i))
            if i in ["fr", "en"]:
                language = i
                name.pop(name.index(i))
        name = "_".join(name)
        data = NemoDataset()
        data_type = data.load(input_file, split=split, language=language, dataset_name=name, show_progress_bar=False)
        merged_data.extend(data)
    merged_data.save(output, data_type=data_type)
    logger.info(f"Saved {len(merged_data)} lines to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge manifest files")
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input manifest files or folder containing manifest files that you want to merge",
    )
    parser.add_argument("output", help="Output file", type=str)
    args = parser.parse_args()
    merge_manifests(args.inputs, args.output)
