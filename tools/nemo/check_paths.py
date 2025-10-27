import argparse
from tqdm import tqdm
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset
import os

def check_path(input_manifest):
    nemo_dataset = NemoDataset()
    nemo_dataset.load(input_manifest)
    for row in tqdm(nemo_dataset):
        assert Path(row.audio_filepath).exists(), f"File not found: {row.audio_filepath}"

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Update audio file paths in a NeMo manifest."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input NeMo manifest file."
    )

    args = parser.parse_args()

    if Path(args.input_path).is_dir():
        for manifest in Path(args.input_path).glob("*.jsonl"):
            print("Checking manifest:", manifest)
            check_path(manifest)
    else:
        check_path(args.input_path)