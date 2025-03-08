#!/usr/bin/env python3

import csv
import hashlib
import logging
import os
import random
import re

from tqdm import tqdm

from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text_utils import format_special_characters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_integers_for_safe_kaldi(s):
    return re.sub(r"(\d+)", _add_zeros, s)


def _add_zeros(match):
    content = int(match.group(1))
    return f"{content:07d}"


def md5_file(file_path, block_size=2**20):
    """Calculate the MD5 hash of file."""
    with open(file_path, "rb") as bf:
        md5 = hashlib.md5()
        while True:
            data = bf.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.digest()


def find_audio_path(name, path):
    found_files = []
    for root, dirs, files in os.walk(path):
        if name in files:
            found_files.append(os.path.join(root, name))
    if len(found_files) == 0:
        raise FileNotFoundError(f"No audio file found for {name}")
    if len(found_files) > 1:
        print(f"{len(found_files)} audio files found for {name}. Checking if they are identical...")
        hash_files = [md5_file(file) for file in found_files]
        unique_hashes = set(hash_files)
        if len(unique_hashes) > 1:
            found_files_str = "\n".join(found_files)
            raise RuntimeError(f"Multiple audio files found for {name}, and some of them are not identical:\n{found_files_str}")
    return found_files[0]


def generate_examples(filepath, path_to_clips, ignore_missing_gender, ignore_missing_files=False):
    """
    Yields examples as dictionaries
    {
        "filename": ...,
        "text": ...,
        "gender": ..., # "m" / "f" / "other" / ""
        "client_id": ..., # unique id for each speaker / sentence
    }

    filepath: path to the TSV or CSV file
    path_to_clips: path to the folder containing the audio files
    """

    has_warning_on_unexisting_file_given = False

    # data_fields =     ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"]
    # data_fields_old = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"]
    # data_fields_csv = ["filename", "text", "up_votes", "down_votes", "age", "gender", "accent", "duration"]
    is_csv = filepath.endswith(".csv")
    delimiter = "," if is_csv else "\t"

    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)

        column_names = next(reader)

        aliases = {
            "path": ["filename", "audio_filepath", "filepath", "file_id", "UTTRANS_ID", "wav"],
            "accents": ["accent"],
            "text": ["sentence", "raw_transcription", "transcription", "TRANSCRIPTION", "wrd"],
            "client_id": ["id", "worker_id", "SPEAKER_ID"],
        }

        for k, v in aliases.items():
            if k not in column_names:
                for alias in v:
                    if alias in column_names:
                        column_names[column_names.index(alias)] = k
                        break

        assert "path" in column_names, f"No path or filename column found in {filepath}."
        assert "text" in column_names, f"No sentence or text column found in {filepath}."
        # assert "client_id" in column_names, f"No client_id column found in {filepath}."
        must_create_client_id = "client_id" not in column_names
        if not ignore_missing_gender:
            assert "gender" in column_names, f"No gender column found in {filepath}."

        if must_create_client_id:
            column_names.append("client_id")

        path_idx = column_names.index("path")

        for field_values in reader:
            # if data is incomplete, fill with empty values
            if len(field_values) < len(column_names):
                field_values += (len(column_names) - len(column_names)) * [None]

            filename_relative = field_values[path_idx]
            filename_absolute = os.path.join(path_to_clips, field_values[path_idx])
            if not os.path.isfile(filename_absolute):
                filename_relative = os.path.basename(field_values[path_idx].replace("\\", "/"))
                filename_absolute = os.path.join(path_to_clips, filename_relative)
            if not os.path.isfile(filename_absolute):
                filename_relative2 = filename_relative.replace(":", "_")  # Hack for TunSwitchTO (1/2)
                filename_absolute = os.path.join(path_to_clips, filename_relative2)
            if not os.path.isfile(filename_absolute):
                if not has_warning_on_unexisting_file_given:
                    print(f"WARNING: Audio file not found: {filename_relative}. Searching in sub-folders (this might take a while)")
                    has_warning_on_unexisting_file_given = True
                try:
                    filename_absolute = find_audio_path(filename_relative, path_to_clips)
                except FileNotFoundError:
                    try:
                        filename_absolute = find_audio_path(filename_relative2, path_to_clips)
                    except FileNotFoundError as err:
                        if ignore_missing_files:
                            print(err)
                            continue
                        raise err

            # set an id if not present
            if must_create_client_id:
                field_values.append(convert_integers_for_safe_kaldi(os.path.splitext(filename_relative)[0].replace("/", "--")))

            # set absolute path for mp3 audio file
            # field_values[path_idx] = filename_absolute
            field_values[path_idx] = os.path.realpath(filename_absolute)

            yield {key: value for key, value in zip(column_names, field_values)}


def tsv2kaldi(
    input_file,
    audio_folder,
    output_folder,
    ignore_missing_gender,
    language=None,
    prefix=None,
    compute_duration=False,
    check_kaldi=True,
):
    rows = generate_examples(input_file, audio_folder, ignore_missing_gender)

    os.makedirs(output_folder, exist_ok=True)

    has_duration = None

    speakers = []
    with open(output_folder + "/utt2spk", "w") as utt2spk_file, open(output_folder + "/text", "w") as text_file, open(output_folder + "/wav.scp", "w") as wavscp_file, open(output_folder + "/utt2dur", "w") as utt2dur_file:
        uniq_spks = []
        for row in tqdm(rows, desc="Converting tsv to kaldi format"):
            if has_duration is None:
                has_duration = "duration" in row
            else:
                assert has_duration == ("duration" in row), "All rows must have the duration or not"

            file_id = os.path.splitext(os.path.basename(row["path"]))[0]
            spk_id = row["client_id"]
            if prefix:
                spk_id = prefix + spk_id
            utt_id = spk_id
            if True:  # file_id not in utt_id:
                utt_id += "_" + file_id
            if spk_id not in uniq_spks:
                uniq_spks.append(spk_id)
                if row.get("gender") == "other":
                    row["gender"] = random.choice(["m", "f"])
                gender = row["gender"][0].lower() if row.get("gender", "") != "" else random.choice(["m", "f"])
                if gender not in ["m", "f"]:
                    raise RuntimeError("Unexpected gender: " + row["gender"])
                speakers.append({"id": spk_id, "gender": gender})

            text = row["text"]
            text = format_special_characters(text)
            if text:
                utt2spk_file.write(utt_id + " " + spk_id + "\n")
                text_file.write(utt_id + " " + text + "\n")
                wavscp_file.write(utt_id + " sox " + os.path.abspath(row["path"]) + " -t wav -r 16k -b 16 -c 1 - |\n")
                if has_duration:
                    utt2dur_file.write(utt_id + " " + row["duration"] + "\n")
                elif compute_duration:
                    # from pydub.utils import mediainfo
                    # dur = float(mediainfo(os.path.abspath(row['path']))['duration'])
                    import librosa

                    dur = librosa.get_duration(path=os.path.abspath(row["path"]))
                    l = f"{utt_id} {dur:.3f}"
                    utt2dur_file.write(l + "\n")
    if not has_duration and not compute_duration:
        os.remove(output_folder + "/utt2dur")

    with open(output_folder + "/spk2gender", "w") as spk2gender_file:
        for speaker in speakers:
            spk2gender_file.write(speaker["id"] + " " + speaker["gender"] + "\n")

    if check_kaldi:
        return check_kaldi_dir(output_folder, language=language)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a dataset in CSV or TSV format (like CommonVoice) into kaldi format")
    parser.add_argument("input_file", type=str, help="Input TSV or CSV file")
    parser.add_argument("audio_folder", type=str, help="Input folder with audio files inside")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--prefix", default=None, type=str, help="A prefix for all ids (ex: MyDatasetName_)")
    parser.add_argument("--language", default=None, type=str, help="Main language (only for checking the charset and giving warnings)")
    parser.add_argument("--ignore_missing_gender", default=False, action="store_true", help="True if there's no gender column")
    parser.add_argument("--compute_duration", default=False, action="store_true", help="True if there's no gender column")
    parser.add_argument("--no_kaldi_check", default=False, action="store_true", help="")
    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder
    audio_folder = args.audio_folder

    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isfile(input_file), f"Input file not found: {input_file}"
    assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"

    logger.info(f"Converting {input_file} and {audio_folder} to {output_folder}")
    logger.info(f"Checking kaldi folder: {not args.no_kaldi_check}")
    logger.info(f"Ignoring missing gender: {args.ignore_missing_gender}")
    logger.info(f"Computing duration if not found: {args.compute_duration}")
    logger.info(f"Checking charset for language: {args.language}")
    logger.info(f"Adding prefix to ids: {args.prefix}")

    tsv2kaldi(
        input_file,
        audio_folder,
        output_folder,
        language=args.language,
        ignore_missing_gender=args.ignore_missing_gender,
        prefix=args.prefix,
        compute_duration=args.compute_duration,
        check_kaldi=not args.no_kaldi_check,
    )
