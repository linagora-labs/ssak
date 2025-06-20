import json
import logging
import os

from tqdm import tqdm

from ssak.utils.kaldi_dataset import KaldiDataset

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clean_dataset(dataset_name, dataset_path, casepunc=False, subset=None):
    input_dataset_path = os.path.join(dataset_path, "raw", subset if subset else "")
    target = "casepunc" if casepunc else "nocasepunc"
    if os.path.exists(os.path.join(input_dataset_path, "segments")):
        output_dataset_path = os.path.join(dataset_path, target, subset if subset else "")
        if os.path.exists(os.path.join(output_dataset_path, "segments")):
            logger.info(f"Dataset {dataset_name} already exist in {target} version")
            return
        os.makedirs(output_dataset_path, exist_ok=True)
        kaldi_dataset = KaldiDataset()
        kaldi_dataset.load(input_dataset_path, show_progress=False)
        kaldi_dataset.normalize_dataset(keep_case=casepunc, keep_punc=casepunc)
        kaldi_dataset.save(output_dataset_path)
    else:
        logger.warning(f"Dataset {dataset_name} does not exist")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean kaldi datasets")
    parser.add_argument("list", help="JSON file path or comma-separated dataset names", type=str)
    parser.add_argument("path", help="Base path to Kaldi datasets", type=str)
    parser.add_argument("--casepunc", action="store_true", default=False)
    args = parser.parse_args()

    if os.path.isfile(args.list):
        with open(args.list) as f:
            dataset_config = json.load(f)
    else:
        dataset_names = [name.strip() for name in args.list.split(",")]
        dataset_config = {i: None for i in dataset_names}

    pbar = tqdm(dataset_config.items())
    for key, value in pbar:
        pbar.set_description(f"Processing {key}")
        if value and "kaldi_subpath" in value:
            subpath = value["kaldi_subpath"]
        else:
            subpath = key
        dataset_path = os.path.join(args.path, subpath)
        input_dataset_path = os.path.join(dataset_path, "raw")

        if os.path.exists(os.path.join(input_dataset_path, "segments")):
            tqdm.write(f"▶ Cleaning: {key}")
            clean_dataset(key, dataset_path, args.casepunc)
        elif os.path.exists(input_dataset_path):
            subsets = os.listdir(input_dataset_path)
            subsets = [i for i in subsets if os.path.isdir(os.path.join(input_dataset_path, i))]
            for subset in subsets:
                full_name = f"{key}/{subset}"
                pbar.set_description(f"Processing {full_name}")
                tqdm.write(f"▶ Cleaning: {full_name}")
                clean_dataset(full_name, dataset_path, args.casepunc, subset=subset)
        else:
            logger.warning(f"Dataset {key} does not exist")
