import argparse
import logging
import os

from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset

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
    input_files = inputs
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if len(input_files) == 1:
        if os.path.isdir(inputs[0]):
            logger.info("Input is a folder, looking for manifest files in it")
            input_files = []
            for file in os.listdir(inputs[0]):
                if is_manifest(file):
                    input_files.append(os.path.join(inputs[0], file))
        else:
            logger.info("One input file, considering it as containing a list of files")
            with open(inputs[0], encoding="utf-8") as f:
                input_files = [l.strip() for l in f.readlines()]
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
    merged_data.save(output, type=data_type)
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
