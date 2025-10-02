import argparse
import json
import logging
import os

from convert_kaldi_dataset_to_nemo import convert_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def duration_filder(row, min_duration):
    return row.duration >= min_duration


def convert_datasets(inputs: list, output_file, output_wav_dir=None, check_audio=False, check_if_in_audio=False, remove_incoherent_texts=False, output_file_func=None, nemo_dataset_type="multiturn"):
    input_files = inputs
    if len(input_files) == 1 and os.path.isfile(input_files[0]):
        logger.warning("One input file, considering it as containing a list of files")
        with open(input_files[0]) as f:
            input_files = json.load(f)
    for input_folder in tqdm(input_files, desc=f"Converting datasets from {inputs} to {output_file}"):
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Non-existing file {input_folder}")
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"File {input_folder} is not a directory")
        if isinstance(input_files, dict):
            filter = None
            if not input_files[input_folder]:
                input_files[input_folder] = dict()
            if "min_duration" in input_files[input_folder]:
                filter = lambda row: duration_filder(row, input_files[input_folder]["min_duration"])
            convert_dataset(
                input_folder,
                output_file,
                output_wav_dir,
                check_audio=input_files[input_folder].get("check_audio", check_audio),
                check_if_in_audio=input_files[input_folder].get("check_if_in_audio", check_if_in_audio),
                remove_incoherent_texts=input_files[input_folder].get("remove_incoherent_texts", remove_incoherent_texts),
                filter=filter,
                nemo_dataset_type=nemo_dataset_type,
                output_file_func=output_file_func,
                concat_segments=input_files[input_folder].get("concat_segments", False),
                concat_audios=input_files[input_folder].get("concat_audios", False),
            )
        else:
            convert_dataset(input_folder, output_file, output_wav_dir, check_audio=check_audio, check_if_in_audio=check_if_in_audio, remove_incoherent_texts=remove_incoherent_texts)
    logger.info(f"Finished converting datasets from {input_files} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a list of Kaldi datasets to Nemo format")
    parser.add_argument("inputs", help="Input files", type=str, nargs="+")
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--output_wav_dir", type=str, default=None)
    parser.add_argument("--check_audio", action="store_true", default=False)
    parser.add_argument("--input_data_path", default="")
    parser.add_argument("--patterns", type=str, nargs="+", default="")
    args = parser.parse_args()
    input_files = args.inputs
    if len(input_files) == 1 and os.path.isfile(input_files[0]):
        logger.warning("One input file, considering it as containing a list of files")
        with open(input_files[0]) as f:
            input_files = json.load(f)
    new_input_files = dict()
    for input_folder in input_files:
        if input_files[input_folder] is not None and "kaldi_subpath" in input_files[input_folder]:
            new_path = os.path.join(args.input_data_path, input_files[input_folder]["kaldi_subpath"])
        else:
            new_path = os.path.join(args.input_data_path, input_folder)
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Input folder {new_path} does not exist")
        elif not os.path.exists(os.path.join(new_path, "wav.scp")):
            for pattern in args.patterns:
                if os.path.exists(os.path.join(new_path, pattern, "wav.scp")):
                    new_input_files[os.path.join(new_path, pattern)] = input_files[input_folder]
                    break
                elif os.path.exists(os.path.join(new_path, pattern)):
                    dirs = os.listdir(os.path.join(new_path, pattern))
                    for dir in dirs:
                        new_input_files[os.path.join(new_path, pattern, dir)] = input_files[input_folder]
                    break
            else:
                logger.warning(f"Input folder {new_path} does not contain a wav.scp file")

    def get_output_file_in_folder(dataset, output_dir):
        stop_tokens = {"casepunc", "recasepunc", "nocasepunc"}
        parts = dataset.name.split("_")
        collected = []
        for part in parts:
            if part in stop_tokens:
                break
            collected.append(part)
        folder = "_".join(collected)
        file = f"manifest_{dataset.name}.jsonl"
        return os.path.join(output_dir, folder, file)

    convert_datasets(new_input_files, args.output, args.output_wav_dir, args.check_audio, output_file_func=get_output_file_in_folder)
