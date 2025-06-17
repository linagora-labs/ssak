import argparse
import json
import logging
import os

from find_long_transcriptions import filter_incoherent_segments
from tqdm import tqdm

from ssak.utils.kaldi_dataset import KaldiDataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Convert Kaldi dataset to Nemo format")
    parser.add_argument("kaldi_dataset", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--output_wav_dir",
        type=str,
        default=None,
        help="Output folder for converted audio files (if check_audio is True)",
    )
    parser.add_argument("--check_audio", action="store_true", default=False, help="Check audio files for correct format")
    return parser.parse_args()


def get_output_file(dataset, output_dir):
    file = f"manifest_{dataset.name}.jsonl" if dataset.name else "manifest.jsonl"
    return os.path.join(output_dir, file)


def kaldi_to_nemo(kaldi_dataset, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(kaldi_dataset):
            row_data = vars(row)
            row_data.pop("id")
            row_data.pop("end")
            row_data.pop("audio_id")
            row_data["audio_filepath"] = row_data.pop("audio_path")
            row_data["offset"] = row_data.pop("start")
            row_data["text"] = row_data.pop("text")
            row_data.pop("normalized_text")
            if row_data.get("gender") is None:
                row_data.pop("gender")
            json.dump(row_data, f, ensure_ascii=False)
            f.write("\n")


def convert_dataset(kaldi_input_dataset, output_dir, new_audio_folder=None, check_audio=False, check_if_in_audio=False, remove_incoherent_texts=False):
    logger.info(f"Converting Kaldi dataset {kaldi_input_dataset} to NeMo format")
    splitted_path = kaldi_input_dataset.split(os.sep)
    if splitted_path[-1] == "":
        splitted_path = splitted_path[:-1]
    idx = -1
    moved = True
    while moved:
        moved = True
        if splitted_path[idx].startswith("case") or splitted_path[idx].startswith("nocase"):
            idx -= 1
        elif splitted_path[idx].startswith("train") or splitted_path[idx].startswith("dev") or splitted_path[idx].startswith("valid") or splitted_path[idx].startswith("test"):
            idx -= 1
        elif splitted_path[idx].startswith("split"):
            idx -= 1
        elif splitted_path[idx].startswith("fr"):
            idx -= 1
        else:
            moved = False
    kaldi_dataset = KaldiDataset(name="_".join(splitted_path[idx:]))
    file = get_output_file(kaldi_dataset, output_dir)
    if os.path.exists(file):
        logger.warning(f"File {file} already exists. Abording conversion to NeMo...")
        return
    kaldi_dataset.load(kaldi_input_dataset)
    if check_audio:
        logger.info("Checking (and transforming if needed) audio files")
        kaldi_dataset.normalize_audios(
            os.path.join(new_audio_folder, kaldi_dataset.name.replace("_casepunc", "").replace("_nocasepunc", "")),
            target_sample_rate=16000,
            target_extension="wav",
            num_workers=6,
        )  # wavs are faster to load than mp3
    if check_if_in_audio:
        logger.info("Check if segments are in audios")
        kaldi_dataset.check_if_segments_in_audios()
    logger.info(f"Writing to {file}")
    os.makedirs(output_dir, exist_ok=True)
    kaldi_to_nemo(kaldi_dataset, file)
    if remove_incoherent_texts:
        logger.info("Check for incoherent texts (very long text with a short audio segment)")
        filter_incoherent_segments(file, file + "_removed_lines")
    logger.info(f"Conversion done (saved to {len(kaldi_dataset)} lines to {file})")


if __name__ == "__main__":
    args = get_args()
    convert_dataset(args.kaldi_dataset, args.output_dir, args.output_wav_dir, args.check_audio)
