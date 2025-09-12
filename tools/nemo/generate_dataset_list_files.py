import argparse
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_subset(dataset, mode):
    if "eval" in dataset or "test" in dataset:
        if mode == "test":
            return True
    elif "dev" in dataset:
        if mode == "dev":
            return True
    elif mode == "train":
        # logger.warning(f"Subset {subset_pattern} not found for {dataset}, added {dataset_path} instead")
        return True
    return False


def generate_dataset_list_files(dataset_list, dataset_folder, dest, mode, subset_patterns):
    if os.path.exists(dest):
        logger.info(f"Reading dataset list from {dest} (already exists)")
        with open(dest) as f:
            return json.load(f)
    if dataset_list.endswith(".json"):
        with open(dataset_list) as f:
            data = json.load(f)
        datasets = []
        for d in data:
            if data[d]:
                if mode == "dev" and data[d].get("valid", False):
                    datasets.append(d)
                elif mode == "test" and data[d].get("test", False):
                    datasets.append(d)
                elif mode == "train":
                    datasets.append(d)
            elif mode == "train":
                datasets.append(d)
    else:
        data = None
        with open(dataset_list) as f:
            datasets = f.read().strip().split("\n")
    patterns = ""
    if mode == "train":
        patterns = r"train$|split\d$"
    elif mode == "dev":
        patterns = r"dev$|split\d_dev$"
    elif mode == "test":
        patterns = r"test$|split\d_test$"
    new_list = dict()
    for i, dataset in enumerate(datasets):
        if data:
            dataset_processing = dict()
            dataset_processing["check_audio"] = data[dataset].pop("check_audio", True) if data[dataset] else True
            dataset_processing["check_if_in_audio"] = data[dataset].pop("check_if_in_audio", False) if data[dataset] else False
            dataset_processing["remove_incoherent_texts"] = data[dataset].pop("remove_incoherent_texts", False) if data[dataset] else False
            if data[dataset]:
                if data[dataset].get("min_duration", None):
                    dataset_processing["min_duration"] = data[dataset]["min_duration"]
                dataset = data[dataset].get("kaldi_subpath", dataset)
        else:
            dataset_processing = None
        dataset_path = os.path.join(dataset_folder, dataset)
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset {dataset} ({dataset_path}) not found")
            continue
        for i, subset_pattern in enumerate(subset_patterns):
            dataset_path_subset = os.path.join(dataset_folder, dataset, subset_pattern)
            if not os.path.exists(dataset_path_subset):
                continue
            elif os.path.exists(os.path.join(dataset_path_subset, "wav.scp")):
                new_list[dataset_path_subset] = dataset_processing
            elif os.path.exists(os.path.join(dataset_path, "wav.scp")):
                found = find_subset(dataset, mode)
                if found:
                    new_list[dataset_path] = dataset_processing
                else:
                    logger.warning(f"Found no subset for {dataset} ({dataset_path})")
            else:
                subfolders = os.listdir(dataset_path_subset)
                if len(subfolders) > 1:
                    for subfolder in subfolders:
                        if re.search(patterns, subfolder):
                            new_list[os.path.join(dataset_path_subset, subfolder)] = dataset_processing
                elif len(subfolders) == 1 and find_subset(os.path.join(dataset_path_subset, subfolders[0]), mode):
                    new_list[os.path.join(dataset_path_subset, subfolders[0])] = dataset_processing
                elif i == len(subset_patterns) - 1:
                    logger.warning(f"Found no subfolders for {dataset} ({dataset_path_subset})")
                else:
                    continue
            break
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(new_list, f, indent=2)
    logger.info(f"Wrote to {dest}")
    return new_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate a file with the path to the dataset folders")
    parser.add_argument("folder_list", help="Input dataset list")
    parser.add_argument("src", help="Dataset folder")
    parser.add_argument("dest", help="Destination file")
    parser.add_argument("--mode", default="train", choices=["train", "dev", "test"], help="Mode")
    parser.add_argument("--subset_patterns", nargs="+", default="nocasepunc_max30")
    args = parser.parse_args()

    generate_dataset_list_files(args.folder_list, args.src, args.dest, args.mode, args.subset_patterns)
