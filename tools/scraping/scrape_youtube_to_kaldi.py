#!/usr/bin/env python3

import argparse
import csv
import os
import warnings

from ssak.utils.audio import get_audio_duration
from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text_utils import format_special_characters


def generate_kaldi_data(
    audio_folder,
    transcription_folder,
    output_folder,
    extension="mp3",
    audio_suffix=None,
    utt_prefix="youtube",
):
    for folder in [audio_folder, transcription_folder]:
        assert os.path.isdir(folder), f"Input folder {folder} does not exist."

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    if len(utt_prefix) > 0:
        utt_prefix += "_"
    segments_file = os.path.join(output_folder, "segments")
    wav_scp_file = os.path.join(output_folder, "wav.scp")
    utt2spk_file = os.path.join(output_folder, "utt2spk")
    spk2utt_file = os.path.join(output_folder, "spk2utt")
    utt2dur_file = os.path.join(output_folder, "utt2dur")
    spk2gender_file = os.path.join(output_folder, "spk2gender")
    text_file = os.path.join(output_folder, "text")
    with open(segments_file, "w", encoding="utf-8") as segments, open(wav_scp_file, "w", encoding="utf-8") as wav_scp, open(utt2spk_file, "w", encoding="utf-8") as utt2spk, open(spk2utt_file, "w", encoding="utf-8") as spk2utt, open(
        spk2gender_file, "w", encoding="utf-8"
    ) as spk2gender, open(utt2dur_file, "w", encoding="utf-8") as utt2dur, open(text_file, "w", encoding="utf-8") as txt:
        for transcription_file in sorted(os.listdir(transcription_folder)):
            if not transcription_file.endswith(".csv"):
                print(f"Skipping {transcription_file}")
                continue

            audio_name = os.path.splitext(transcription_file)[0]
            if audio_suffix:
                audio_name = f"{audio_name}{audio_suffix}"

            audio_file = f"{audio_name}.{extension}"
            audio_path = os.path.realpath(os.path.join(audio_folder, audio_file))
            audio_duration = get_audio_duration(audio_path)
            assert os.path.isfile(audio_path), f"Missing audio file: {audio_path}"

            transcription_path = os.path.join(transcription_folder, transcription_file)

            try:
                with open(transcription_path) as f:
                    reader = csv.reader(f, delimiter=";")
                    next(reader)  # skip the first row (headers)
                    has_written = False
                    for _id, row in enumerate(reader):
                        if len(row) == 0:
                            continue

                        utt_id = f"{utt_prefix}{audio_name}-seg_{_id:05d}"
                        try:
                            text, start, duration = row
                        except ValueError as err:
                            raise RuntimeError(f"Error on line {_id}: {row}") from err
                        text = format_special_characters(text)
                        if not text:
                            continue
                        start = round(float(start), 3)
                        end = round(start + float(duration), 3)
                        duration = round(end - start, 3)

                        if audio_duration < start:
                            warnings.warn(
                                f"{utt_id} there is audio duration:{audio_duration} less than segment start duration {start} Please check your data.",
                                UserWarning,
                            )
                            continue

                        if duration == 0:
                            warnings.warn(f"Duration is 0 for {utt_id}", UserWarning)
                            continue
                        segments.write(f"{utt_id} {audio_name} {start:.3f} {end:.3f}\n")
                        txt.write(f"{utt_id} {text}\n")
                        utt2dur.write(f"{utt_id} {duration}\n")
                        utt2spk.write(f"{utt_id} {utt_id}\n")
                        spk2utt.write(f"{utt_id} {utt_id}\n")
                        spk2gender.write(f"{utt_id} m\n")
                        has_written = True
                    if has_written:
                        wav_scp.write(f"{audio_name} sox {audio_path} -t wav -r 16000  -b 16 -c 1 - |\n")
            except Exception as e:
                raise RuntimeError(f"Error while reading {transcription_path}") from e

    check_kaldi_dir(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audios", help="Path to folder that contain the audios", type=str)
    parser.add_argument("transcription", help="Path to folder contain the transcription", type=str)
    parser.add_argument("output", help="Path to kaldi data folder", type=str)
    parser.add_argument("--extension", help="The file extension should be one of: [.mp3, .wav, .ogg]", type=str, default="mp3")
    parser.add_argument(
        "--audio_suffix",
        help="Specify whether there are audio files with tags different from their corresponding transcription files",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    generate_kaldi_data(
        args.audios,
        args.transcription,
        args.output,
        extension=args.extension,
        audio_suffix=args.audio_suffix,
    )
