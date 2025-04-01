import argparse
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dataset_list_files(dataset_list, dataset_folder, dest, mode, subset_pattern):
    if os.path.exists(dest):
        logger.info(f"Reading dataset list from {dest} (already exists)")
        with open(dest) as f:
            return f.read().strip().split("\n")
    new_list = []
    with open(dataset_list) as f:
        datasets = f.read().strip().split("\n")

    patterns = ""
    if mode == "train":
        patterns = r"train$|split\d$"
    elif mode == "dev":
        patterns = r"dev$|split\d_dev$"
    elif mode == "test":
        patterns = r"test$|split\d_test$"
    for i, dataset in enumerate(datasets):
        dataset_path = os.path.join(dataset_folder, dataset)
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset {dataset} not found")
            continue
        dataset_path_subset = os.path.join(dataset_folder, dataset, subset_pattern)
        if os.path.exists(os.path.join(dataset_path_subset, "wav.scp")):
            new_list.append(dataset_path_subset)
        elif os.path.exists(os.path.join(dataset_path, "wav.scp")):
            if "eval" in dataset or "test" in dataset:
                if mode == "test":
                    new_list.append(dataset_path)
            elif "dev" in dataset:
                if mode == "dev":
                    new_list.append(dataset_path)
            elif mode == "train":
                logger.warning(f"Subset {subset_pattern} not found for {dataset}, added {dataset_path} instead")
                new_list.append(dataset_path)
        else:
            subfolders = os.listdir(dataset_path_subset)
            for subfolder in subfolders:
                if re.search(patterns, subfolder):
                    new_list.append(os.path.join(dataset_path_subset, subfolder))
    with open(dest, "w") as f:
        f.write("\n".join(new_list))
        f.write("\n")
    return new_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate a file with the path to the dataset folders")
    parser.add_argument("folder_list", help="Input dataset list")
    parser.add_argument("src", help="Dataset folder")
    parser.add_argument("dest", help="Destination file")
    parser.add_argument("--mode", default="train", choices=["train", "dev", "test"], help="Mode")
    parser.add_argument("--subset_pattern", default="nocasepunc_max30")
    args = parser.parse_args()

    generate_dataset_list_files(args.folder_list, args.src, args.dest, args.mode, args.subset_pattern)
