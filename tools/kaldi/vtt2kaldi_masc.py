#!/usr/bin/env python3

import csv
import os
import random
import warnings

import webvtt

from ssak.utils.audio import get_audio_total_duration
from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text_utils import format_special_characters

warnings.simplefilter(action="ignore", category=FutureWarning)  # remove warning : the 'mangle_dupe_cols' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'mangle_dupe_cols'


def time_to_seconds(time_str):
    # Split the time string into components
    hours, minutes, seconds_milliseconds = time_str.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")

    # Convert components to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    # Calculate total time in seconds
    total_seconds = (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)
    return total_seconds


def vtt2kaldi(transcription_folder, audio_folder, output_folder, meta_data_folder=None, ignore_missing_gender=False):
    extension = ["wav", "mp3", "ogg", "flac"]
    dataset_splits = ["clean_dev", "clean_test", "clean_train", "noisy_dev", "noisy_test", "noisy_train"]

    for folder in [audio_folder, transcription_folder]:
        assert os.path.isdir(folder), f"Input folder {folder} does not exist."
    os.makedirs(output_folder, exist_ok=True)

    if meta_data_folder is not None:
        gender_map = {"Male": "m", "Female": "f"}
        for dataset_split in dataset_splits:
            meta_file = os.path.join(meta_data_folder, f"{dataset_split}_meta.csv")
            dialects = set()

            assert meta_file is not None, f"{meta_file} dos not exist."

            with open(meta_file) as meta_csv:
                csv_reader = csv.DictReader(meta_csv)
                for row in csv_reader:
                    dialects.add(row["dialect"])

            for dialect in dialects:
                dialect_output_folder = os.path.join(output_folder, dataset_split, dialect)
                os.makedirs(dialect_output_folder, exist_ok=True)

                with open(os.path.join(dialect_output_folder, "utt2spk"), "w") as utt2spk_f, open(os.path.join(dialect_output_folder, "spk2utt"), "w") as spk2utt_f, open(os.path.join(dialect_output_folder, "text"), "w") as text_f, open(
                    os.path.join(dialect_output_folder, "segments"), "w"
                ) as segments_f, open(os.path.join(dialect_output_folder, "wav.scp"), "w") as wav_scp_f, open(os.path.join(dialect_output_folder, "spk2gender"), "w") as spk2gender_f, open(
                    os.path.join(dialect_output_folder, "utt2dur"), "w"
                ) as utt2dur_f:
                    with open(meta_file) as meta_csv:
                        csv_reader = csv.DictReader(meta_csv)
                        for row in csv_reader:
                            if row["dialect"] == dialect:
                                wav_id = row["video_id"]
                                audio_path = os.path.realpath(os.path.join(audio_folder, f"{wav_id}.{extension}"))
                                wav_scp_f.write(f"{wav_id} sox {audio_path}  -t wav -r 16k -b 16 -c 1 - |\n")

                                transcription_file = None
                                for filename in os.listdir(transcription_folder):
                                    if filename.startswith(wav_id) and filename.endswith(".vtt"):
                                        transcription_file = filename
                                        break

                                if transcription_file is None:
                                    warnings.warn(f"No matching VTT file found for {wav_id}.")
                                    continue
                                for _id, caption in enumerate(webvtt.read(transcription_folder + f"/{transcription_file}")):
                                    utt_id = f"MASC_{wav_id}-seg{_id:04d}"

                                    start = time_to_seconds(caption.start)
                                    end = time_to_seconds(caption.end)
                                    duration = round(float(end - start), 3)

                                    text = caption.text.replace("\n", " ")
                                    text = format_special_characters(text.strip())
                                    if text and duration > 0:
                                        text_f.write(f"{utt_id} {text}\n")
                                        segments_f.write(f"{utt_id} {wav_id} {start:.2f} {end:.2f}\n")
                                        utt2dur_f.write(f"{utt_id} {duration}\n")
                                        utt2spk_f.write(f"{utt_id} {utt_id}\n")
                                        spk2utt_f.write(f"{utt_id} {utt_id}\n")
                                        if not ignore_missing_gender:
                                            gender = gender_map.get(row["gender"], random.choice(["m", "f"]))
                                            spk2gender_f.write(f"{utt_id} {gender}\n")
                                        else:
                                            os.remove(output_folder + "/spk2gender")

                                    else:
                                        if not text:
                                            warning_message = f"No text found for {utt_id}."
                                        elif duration <= 0:
                                            warning_message = f"Non-positive duration found for {utt_id}."
                                        warnings.warn(f"{warning_message} Please check your data.", UserWarning)
                                        continue
    else:
        with open(os.path.join(output_folder, "utt2spk"), "w") as utt2spk_f, open(os.path.join(output_folder, "spk2utt"), "w") as spk2utt_f, open(os.path.join(output_folder, "text"), "w") as text_f, open(
            os.path.join(output_folder, "segments"), "w"
        ) as segments_f, open(os.path.join(output_folder, "wav.scp"), "w") as wav_scp_f, open(os.path.join(output_folder, "spk2gender"), "w") as spk2gender_f, open(os.path.join(output_folder, "utt2dur"), "w") as utt2dur_f:
            for audio_file in sorted(os.listdir(audio_folder)):
                ex = audio_file.split(".")[-1]
                if ex not in extension:
                    warnings.warn(
                        f"Ignoring {audio_file} because it is not in the expected format (no extension of those {extension}).",
                        UserWarning,
                    )
                    continue

                wav_id = os.path.splitext(audio_file)[0]
                original_wav_id = os.path.splitext(audio_file)[0][:11]
                audio_path = os.path.realpath(os.path.join(audio_folder, audio_file))
                _, audio_duration = get_audio_total_duration(audio_path)

                transcription_file = None
                for filename in os.listdir(transcription_folder):
                    if filename.startswith(original_wav_id) and filename.endswith(".vtt"):
                        transcription_file = filename
                        break

                if transcription_file is None:
                    warnings.warn(f"No matching VTT file found for {wav_id}.")
                    continue
                wav_scp_f.write(f"{wav_id} sox {audio_path}  -t wav -r 16k -b 16 -c 1 - |\n")
                for _id, caption in enumerate(webvtt.read(transcription_folder + f"/{transcription_file}")):
                    utt_id = f"{wav_id}-seg{_id:04d}"

                    start = time_to_seconds(caption.start)
                    end = time_to_seconds(caption.end)
                    duration = round(float(end - start), 3)

                    text = caption.text.replace("\n", " ").replace("&lrm;", "")
                    text = format_special_characters(text)

                    if audio_duration > start:
                        if text and duration > 0:
                            text_f.write(f"{utt_id} {text}\n")
                            segments_f.write(f"{utt_id} {wav_id} {start:.2f} {end:.2f}\n")
                            utt2dur_f.write(f"{utt_id} {duration}\n")
                            utt2spk_f.write(f"{utt_id} {utt_id}\n")
                            spk2utt_f.write(f"{utt_id} {utt_id}\n")
                            gender = random.choice(["m", "f"])
                            if not ignore_missing_gender:
                                spk2gender_f.write(f"{utt_id} {gender}\n")
                            else:
                                os.remove(output_folder + "/spk2gender")
                        else:
                            if not text:
                                warning_message = f"No text found for {utt_id}."
                            elif duration <= 0:
                                warning_message = f"Non-positive duration found for {utt_id}."
                            warnings.warn(f"{warning_message} Please check your data.", UserWarning)
                            continue
                    else:
                        warnings.warn(
                            f"{utt_id} there is audio duration:{audio_duration} less than segment start duration {start} Please check your data.",
                            UserWarning,
                        )
                        continue
    check_kaldi_dir(output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a dataset from  vtt format (like YouTube subtitles files) into kaldi format ")
    parser.add_argument("transcription_folder", help="Input subtitels folder that conatin vtt files", type=str)
    parser.add_argument("audio_path", help="Path of Audio files", type=str)
    parser.add_argument("output_folder", help="Output kaldi folder", type=str)
    parser.add_argument("--metadata_folder", help="input Meta Folder that contain the CSV files", type=str, default=None)
    parser.add_argument(
        "--ignore_missing_gender",
        type=bool,
        default=False,
        help="True if there's no gender2spk file if there's no meta data",
    )
    args = parser.parse_args()

    audio_folder = args.audio_path
    output_folder = args.output_folder
    transcription_folder = args.transcription_folder
    meta_data_folder = args.metadata_folder

    if meta_data_folder is not None:
        assert os.path.isdir(meta_data_folder), f"Input folder not found: {meta_data_folder}"

    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isdir(transcription_folder), f"Input file not found: {transcription_folder}"
    assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"

    vtt2kaldi(
        transcription_folder,
        audio_folder,
        output_folder,
        meta_data_folder=meta_data_folder,
        ignore_missing_gender=args.ignore_missing_gender,
    )
