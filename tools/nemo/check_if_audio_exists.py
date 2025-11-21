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
        description="Check if the audio files in a NeMo manifest (or a folder containing manifests) exist."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input NeMo manifest file or folder to check."
    )

    args = parser.parse_args()

    if Path(args.input_path).is_dir():
        for manifest in Path(args.input_path).glob("*.jsonl"):
            print("Checking manifest:", manifest)
            check_path(manifest)
    else:
        check_path(args.input_path)