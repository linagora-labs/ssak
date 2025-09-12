import argparse
import json
import logging
import os
import shutil

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_manifests(inputs, output):
    if os.path.exists(output):
        raise FileExistsError(f"Output file {output} already exists")
    input_files = inputs
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if len(input_files) == 1:
        if os.path.isdir(inputs[0]):
            logger.info("Input is a folder, looking for manifest files in it")
            input_files = []
            for root, dirs, files in os.walk(inputs[0]):
                input_files.extend([os.path.join(root, f) for f in files if f.endswith(".jsonl") and not f.startswith("all_manifests")])
        else:
            logger.info("One input file, considering it as containing a list of files")
            with open(inputs[0], encoding="utf-8") as f:
                input_files = [l.strip() for l in f.readlines()]
    data = 0
    with open(output + ".tmp", "w", encoding="utf-8") as out:
        for input_file in tqdm(input_files, desc="Merging manifest files"):
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Non-existing file {input_file}")
            elif os.path.isdir(input_file):
                raise IsADirectoryError(f"Directory {input_file}")
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
            with open(input_file, encoding="utf-8") as f:
                lines = f.readlines()
                rows = [json.loads(l) for l in lines]
                for row in rows:
                    if "split" not in row:
                        row["split"] = split
                    if "language" not in row:
                        row["language"] = language
                    if "name" not in row:
                        row["name"] = name
                    json.dump(row, out, ensure_ascii=False)
                    out.write("\n")
                    data += 1
    shutil.move(output + ".tmp", output)
    logger.info(f"Saved {data} lines to {output}")


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
