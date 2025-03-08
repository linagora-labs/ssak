import os
import random
import re

import pandas as pd


def stm_to_dataframe(stm_file):
    # Read the STM file
    with open(stm_file) as stm_file:
        stm_data = stm_file.read()
    # Split the data into lines
    lines = stm_data.strip().split("\n")

    # Define a pattern to extract information
    pattern = re.compile(r"(\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (.+)")

    # Initialize lists to store data
    segments = []
    for line in lines:
        match = pattern.match(line)
        if match:
            segments.append(match.groups())
    # Create a DataFrame
    columns = ["filename", "channel", "speaker_id", "start", "end", "gender", "details", "transcription"]
    stmframe = pd.DataFrame(segments, columns=columns)
    # Convert numeric columns to float
    numeric_columns = ["start", "end"]
    stmframe[numeric_columns] = stmframe[numeric_columns].astype(float)

    # Replace underscores with hyphens and add '.wav'
    stmframe["filename"] = stmframe["filename"].str.replace("_", "-") + ".wav"
    stmframe["filename"]
    # Extract gender information
    stmframe["gender"] = stmframe["gender"].str.extract(r"<\S+,\S+,(.*?)>").replace("female", "f").replace("male", "")
    stmframe = stmframe.sort_values(by=["speaker_id"])

    return stmframe


def stmframe_to_kaldi(stm_file, audio_path, output_folder, prefix=None):
    stmframe = stm_to_dataframe(stm_file)
    if os.path.isdir(output_folder):
        raise RuntimeError(f"Output folder {output_folder} already exists. Please remove it to overwrite.")
    os.makedirs(output_folder, exist_ok=True)

    unique_wav_paths = set()

    with open(f"{output_folder}/utt2spk", "w", encoding="utf-8") as utt2spk, open(f"{output_folder}/spk2utt", "w", encoding="utf-8") as spk2utt, open(f"{output_folder}/text", "w", encoding="utf-8") as text, open(
        f"{output_folder}/wav.scp", "w", encoding="utf-8"
    ) as wav_scp, open(f"{output_folder}/segments", "w", encoding="utf-8") as segments, open(f"{output_folder}/utt2dur", "w", encoding="utf-8") as utt2dur, open(f"{output_folder}/spk2gender", "w", encoding="utf-8") as spk2gender:
        for index, row in stmframe.iterrows():
            utt_id = row["filename"].split(".")[0]
            uniq_utt_id = f"{utt_id}-seg_{index:04d}"
            spk_id = f"{row['speaker_id']}-seg_{index:04d}"
            if prefix:
                spk_id = prefix + spk_id
                uniq_utt_id = prefix + uniq_utt_id
            transcription = row["transcription"]

            if transcription is None:
                raise RuntimeError(f"text is non for {uniq_utt_id} utt!!!")

            # Assuming 'filename' contains the path to the audio file
            audio_name = os.path.realpath(os.path.join(audio_path, row["filename"]))
            assert audio_name, f'This Audio file {row["filename"]} is not exist in {audio_path}'

            # Write utt2spk
            utt2spk.write(f"{uniq_utt_id} {spk_id}\n")
            spk2utt.write(f"{spk_id} {uniq_utt_id}\n")
            # Write text
            text.write(f"{uniq_utt_id} {transcription}\n")

            # Assuming 'start' and 'end' columns are available
            if not pd.isna(row["start"]) and not pd.isna(row["end"]):
                start_time = round(float(row["start"]), 3)
                end_time = round(float(row["end"]), 3)
                duration = round(end_time - start_time, 3)
                if duration == 0:
                    raise RuntimeError(f"Audio file not found for {utt_id} (in {audio_path})")
                # Write segments
                segments.write(f"{uniq_utt_id} {utt_id} {start_time} {end_time}\n")
                utt2dur.write(f"{uniq_utt_id} {duration}\n")

            # Assuming 'gender' column is available
            if "gender" in stmframe.columns:
                if row.get("gender") == "other":
                    row["gender"] = random.choice(["m", "f"])
                gender = row["gender"]
                # Write spk2gender
                spk2gender.write(f"{spk_id} {gender}\n")
            else:
                gender = random.choice(["m", "f"])
                spk2gender.write(f"{spk_id} {gender}\n")

            # Write wav.scp
            if os.path.exists(audio_name):
                if utt_id not in unique_wav_paths:
                    wav_scp.write(f"{utt_id} sox {audio_name} -t wav -r 16k -b 16 -c 1 - |\n")
                    unique_wav_paths.add(utt_id)
    # check_kaldi_dir(output_folder, language=None)

    # check_kaldi_dir(output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a dataset in Transcriber format (.xml with extension .trs) into kaldi format")
    parser.add_argument("stmfile", help="Folder with trs files")
    parser.add_argument("audio_folder", help="Folder with audio files (if different from trs folder)", nargs="?", default=None)
    parser.add_argument("output_folder", help="output directory")
    parser.add_argument("--prefix", default=None, type=str, help="A prefix for all ids (ex: MyDatasetName_)")
    args = parser.parse_args()

    assert os.path.isdir(args.audio_folder), f"Folder {args.audio_folder} does not exist"
    assert not os.path.isdir(args.output_folder), f"Folder {args.output_folder} already exists"

    stmframe_to_kaldi(args.stmfile, args.audio_folder, args.output_folder, args.prefix)
