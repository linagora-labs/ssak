import argparse
import json
import logging
import os
import re
import shutil

from concat_segments import concat_segments as f_concat_segments
from find_incoherent_transcriptions import filter_incoherent_segments
from tqdm import tqdm

from ssak.utils.kaldi_dataset import KaldiDataset
from ssak.utils.nemo_dataset import NemoDataset

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
    parser.add_argument("--check_if_in_audio", action="store_true", default=False, help="Check if segment is part of the audio")
    parser.add_argument("--remove_incoherent_texts", action="store_true", default=False, help="Remove text with incoherent length")

    return parser.parse_args()


def get_output_file(dataset, output_dir):
    file = f"manifest_{dataset.name}.jsonl" if dataset.name else "manifest.jsonl"
    return os.path.join(output_dir, file)


def kaldi_to_nemo(kaldi_dataset, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file + ".tmp", "w", encoding="utf-8") as f:
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
    shutil.move(output_file + ".tmp", output_file)


def get_dataset_name(kaldi_input_dataset, remove_casing=True, remove_max_duration=True, remove_split=True):
    splitted_path = kaldi_input_dataset.split(os.sep)
    if splitted_path[-1] == "":
        splitted_path = splitted_path[:-1]
    idx = -1
    moved = True
    while moved:
        moved = True
        if splitted_path[idx].startswith("case") or splitted_path[idx].startswith("nocase") or splitted_path[idx].startswith("recase"):
            idx -= 1
        elif splitted_path[idx].startswith("train") or splitted_path[idx].startswith("dev") or splitted_path[idx].startswith("valid") or splitted_path[idx].startswith("test"):
            idx -= 1
        elif splitted_path[idx].startswith("split"):
            idx -= 1
        elif splitted_path[idx].startswith("fr"):
            idx -= 1
        elif splitted_path[idx].startswith("all"):
            idx -= 1
        else:
            moved = False
    name = "_".join(splitted_path[idx:])
    if remove_casing:
        name = name.replace("_casepunc", "").replace("_nocasepunc", "").replace("_recasepunc", "")
    if remove_max_duration:
        name = re.sub(r"_max\d+", "", name)
    if remove_split:
        name = name.replace("_train", "").replace("_dev", "").replace("_test", "").replace("_valid", "").replace("_all", "")
        name = re.sub(r"_split\d+", "", name)
        name = re.sub(r"_fr\d+", "", name)
    return name

def convert_dataset(
    kaldi_input_dataset, output_dir, new_audio_folder=None, check_audio=False, check_if_in_audio=False, remove_incoherent_texts=False, filter=None, nemo_dataset_type="asr", output_file_func=None, concat_segments=False, concat_audios=False
):
    logger.info(f"Converting Kaldi dataset {kaldi_input_dataset} to NeMo format")
    logger.info(f"check_audio : {check_audio}, check_if_in_audio : {check_if_in_audio}, remove_incoherent_texts : {remove_incoherent_texts}")

    cache_folder = os.path.join(output_dir, ".cache")
    os.makedirs(cache_folder, exist_ok=True)
    name = get_dataset_name(kaldi_input_dataset=kaldi_input_dataset)
    kaldi_dataset = KaldiDataset(name=name, log_folder=os.path.join(cache_folder, f"{name}_log_folder"), row_checking_kwargs=dict(show_warnings=False))
    if output_file_func:
        file = output_file_func(kaldi_dataset, output_dir)
    else:
        file = get_output_file(kaldi_dataset, output_dir)
    if os.path.exists(file):
        logger.warning(f"File {file} already exists. Abording conversion to NeMo...")
        return
    kaldi_dataset.load(kaldi_input_dataset)
    if filter:
        kaldi_dataset.apply_filter(filter, filter_out=False)
    if check_audio and new_audio_folder:
        logger.info("Checking (and transforming if needed) audio files")
        kaldi_dataset.normalize_audios(
            os.path.join(new_audio_folder, kaldi_dataset.name.replace("_casepunc", "").replace("_nocasepunc", "").replace("_recasepunc", "")),
            target_sample_rate=16000,
            target_extension="wav",
            num_workers=6,
        )  # wavs are faster to load than mp3
    if check_if_in_audio:
        logger.info("Check if segments are in audios")
        kaldi_dataset.check_if_segments_in_audios()
    logger.info(f"Writing to {file}")
    nemo_dataset = NemoDataset()
    nemo_dataset.kaldi_to_nemo(kaldi_dataset)
    if remove_incoherent_texts:
        file_filter_cache = os.path.join(kaldi_dataset.log_folder, os.path.basename(file))
        nemo_dataset.save(file_filter_cache + ".tofilter", type=nemo_dataset_type)
        logger.info("Check for incoherent texts (very long text with a short audio segment)")
        filter_incoherent_segments(file_filter_cache + ".tofilter", os.path.join(kaldi_dataset.log_folder, "filtered_out_incoherent_segments_charset.jsonl"), output_file=file_filter_cache + ".charset")
        filter_incoherent_segments(file_filter_cache + ".charset", os.path.join(kaldi_dataset.log_folder, "filtered_out_incoherent_segments_time_long.jsonl"), output_file=file_filter_cache + ".long", mode="too_long")
        filter_incoherent_segments(file_filter_cache + ".long", os.path.join(kaldi_dataset.log_folder, "filtered_out_incoherent_segments_time_short.jsonl"), output_file=file_filter_cache + ".short", mode="too_short")
        shutil.copyfile(file_filter_cache + ".short", file)
    else:
        # kaldi_to_nemo(kaldi_dataset, file)
        nemo_dataset.save(file, type=nemo_dataset_type)
    if concat_segments:
        os.makedirs(kaldi_dataset.log_folder, exist_ok=True)
        file_concat_cache = os.path.join(kaldi_dataset.log_folder, os.path.basename(file) + ".toconcat")
        shutil.move(file, file_concat_cache)
        logger.info("Concatenating segments")
        f_concat_segments(
            file_concat_cache,
            output_file=file,
            max_duration=30,
            acceptance=1.0,
            acceptance_punc=0.2,
            merge_audios=concat_audios,
            merged_audio_folder=os.path.join(new_audio_folder, kaldi_dataset.name.replace("_casepunc", "").replace("_nocasepunc", "").replace("_recasepunc", "") + "_merged"),
            keep_audio_structure=True,
            num_threads=8,
        )
    logger.info(f"Conversion done (saved to {len(nemo_dataset)} lines to {file})")


if __name__ == "__main__":
    args = get_args()
    convert_dataset(args.kaldi_dataset, args.output_dir, args.output_wav_dir, args.check_audio)
