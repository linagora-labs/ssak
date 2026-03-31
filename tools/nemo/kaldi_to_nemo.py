import logging
import argparse
from pathlib import Path

from ssak.utils.kaldi_dataset import KaldiDataset
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def find_kaldi_dirs(root_folder: Path):
    """
    Recursively find directories that contain a wav.scp file.
    """
    return [p.parent for p in root_folder.rglob("wav.scp")]

def kaldi_to_nemo(input_folder, name, output_folder):
    kaldi_dataset = KaldiDataset(name=name, log_folder=output_folder, row_checking_kwargs=dict(show_warnings=False))
    kaldi_dataset.load(input_folder)
    nemo_dataset = NemoDataset(name=name, log_folder=output_folder)
    nemo_dataset.kaldi_to_nemo(kaldi_dataset)
    return nemo_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kaldi dataset to Nemo format")
    parser.add_argument("input_folder", type=str, help="Path to the input folder")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--name", type=str, help="Name of the dataset", default=None)

    args = parser.parse_args()

    input_root = Path(args.input_folder).resolve()
    output_root = Path(args.output_folder).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    kaldi_dirs = find_kaldi_dirs(input_root)

    if not kaldi_dirs:
        print("No folders containing wav.scp were found.")
        exit(1)

    for kaldi_dir in kaldi_dirs:
        split_name = kaldi_dir.name
        name = args.name if args.name else split_name

        print(f"Processing split: {split_name}")

        dataset = kaldi_to_nemo(str(kaldi_dir), name, str(output_root))
        dataset.save(output_root / f"{split_name}.jsonl", data_type="multiturn")

    print("Conversion completed.")