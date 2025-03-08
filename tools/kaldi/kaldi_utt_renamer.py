#!/usr/bin/env python3

import os

from tqdm import tqdm


def rename_audios_based_on_wavscp(input_folder, char_to_replace, replacement_char):
    with open(os.path.join(input_folder, "wav.scp")) as f:
        content = f.readlines()
        for line in tqdm(content):
            line = line.strip().split(" ", 1)
            file_path = line[1]
            new_file = file_path.split("/")[-1].replace(char_to_replace, replacement_char)
            new_file_path = os.path.join("/".join(file_path.split("/")[:-1]), new_file)
            os.rename(file_path, new_file_path)


def replace_char_in_file(file_path, char_to_replace, replacement_char, column=-1, delimiter=" ", slices=1):
    new_content = ""
    if slices is None:
        slices = -1
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            if column >= 0:
                line = line.strip().split(delimiter, slices)
                line[column] = line[column].replace(char_to_replace, replacement_char)
                new_content += delimiter.join(line) + "\n"
            else:
                new_content += line.replace(char_to_replace, replacement_char)
    with open(file_path, "w") as f:
        f.write(new_content)
    if os.path.exists(file_path):
        print(f"File '{file_path}' has been modified ('{char_to_replace}' -> '{replacement_char}')")
    else:
        raise Exception(f"File {file_path} missing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replace a character in utterances or audio_ids (or both) of a kaldi dataset.")
    parser.add_argument("input_folder", type=str, help="Input folder with kaldi files")
    parser.add_argument("--char_to_replace", type=str, default=":", help="Character to replace")
    parser.add_argument("--replacement_char", type=str, default="-", help="Replacement character")
    parser.add_argument("--delimiter", type=str, default=" ", help="Delimiter for kaldi files")
    parser.add_argument(
        "--rename_wavs",
        action="store_true",
        default=False,
        help="Rename wavs based on wav.scp",
    )
    parser.add_argument(
        "--rename_mode",
        action="store_true",
        default="all",
        choices=["all", "audio_ids", "utt_ids"],
        help="Rename mode for files: all, audio_ids, utt_ids",
    )
    args = parser.parse_args()

    files = None

    if args.rename_mode == "utt_ids":
        files = [
            ("utt2dur", 0),
            ("utt2spk", 0),
            ("spk2utt", 0),
            ("segments", 0),
            ("text", 0),
        ]
    elif args.rename_mode == "audio_ids":
        files = [("wav.scp", 0), ("segments", 1)]
    elif args.rename_mode == "all":
        files = [
            ("utt2dur", 0),
            ("utt2spk", 0),
            ("spk2utt", 0),
            ("segments", 0),
            ("text", 0),
            ("wav.scp", 0),
            ("segments", 1),
        ]

    if args.rename_wavs:
        rename_audios_based_on_wavscp(args.input_folder, args.char_to_replace, args.replacement_char)

    for filename, column in files:
        filepath = os.path.join(args.input_folder, filename)
        if os.path.exists(filepath):
            replace_char_in_file(
                filepath,
                args.char_to_replace,
                args.replacement_char,
                column=column,
                delimiter=args.delimiter,
                slices=-1 if filename != "text" else 1,
            )
