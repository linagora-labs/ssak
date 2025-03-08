import argparse
import logging
import os

from convert_kaldi_dataset_to_nemo import convert_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_datasets(inputs: list, output_file, output_wav_dir=None, check_audio=False):
    input_files = inputs
    if len(input_files) == 1:
        logger.warning("One input file, considering it as containing a list of files")
        with open(input_files[0], encoding="utf-8") as f:
            input_files = [l.strip() for l in f.readlines()]
    for input_folder in tqdm(input_files, desc=f"Converting datasets from {inputs} to {output_file}"):
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Non-existing file {input_folder}")
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"File {input_folder} is not a directory")
        convert_dataset(input_folder, output_file, output_wav_dir, check_audio=check_audio)
    logger.info(f"Finished converting datasets from {input_files} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge manifest files")
    parser.add_argument("inputs", help="Input files", type=str, nargs="+")
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--output_wav_dir", type=str, default=None)
    parser.add_argument("--check_audio", action="store_true", default=False)
    args = parser.parse_args()
    convert_datasets(args.inputs, args.output, args.output_wav_dir, args.check_audio)
