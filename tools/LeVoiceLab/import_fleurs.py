#!/usr/bin/env python3

import csv
import os
import shutil

import soxbindings as sox
from datasets import load_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts Fleur dataset in TSV format (like CommonVoice)")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--language", type=str, help="Language code", default="all")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    main_language = args.language
    OTHER_GENDER = "other"

    # self copy
    os.makedirs(args.output_folder, exist_ok=True)
    shutil.copy(__file__, os.path.join(args.output_folder, os.path.basename(__file__)))

    split = ["validation", "test", "train"]
    for i in split:
        print("Collecting", i)

        processed_languages = []

        dataset = load_dataset("google/fleurs", main_language, split=i, streaming=True)

        all_languages = dataset.features["lang_id"].names

        output_audio_folder = os.path.join(args.output_folder, "clips")

        for sample in dataset:
            audio = sample["audio"]
            array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            path = audio["path"]
            id = sample["id"]
            language = all_languages[sample["lang_id"]]
            raw_transcription = sample["raw_transcription"]
            transcription = sample["transcription"]
            gender = {0: "male", 1: "female"}.get(sample["gender"], OTHER_GENDER)
            if gender == OTHER_GENDER:
                print("WARNING: got gender " + str(sample["gender"]))
            language_full = sample["language"]
            duration = sample["num_samples"] / sampling_rate

            id = f"google_fleurs_{language}_{int(id):06d}"
            path = os.path.join(language, path)

            row = [
                id,
                path,
                raw_transcription,
                transcription,
                duration,
                language,
                gender,
            ]

            output_wav = os.path.join(output_audio_folder, path)
            if not os.path.isfile(output_wav):
                os.makedirs(os.path.dirname(output_wav), exist_ok=True)
                sox.write(output_wav, array, sampling_rate)

            output_tsvs = {language: os.path.join(args.output_folder, language, f"{i}.tsv")}
            if main_language == "all":
                output_tsvs["all"] = os.path.join(args.output_folder, f"{i}.tsv")

            if args.verbose:
                print(" | ".join([str(r) for r in row]))

            for lan, output_tsv in output_tsvs.items():
                if lan not in processed_languages:
                    processed_languages.append(lan)

                    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
                    with open(output_tsv, "w", encoding="utf8") as file:
                        writer = csv.writer(file, delimiter="\t")
                        field = [
                            "id",
                            "path",
                            "raw_transcription",
                            "transcription",
                            "duration",
                            "language",
                            "gender",
                        ]

                        writer.writerow(field)

                with open(output_tsv, "a", encoding="utf8") as file:
                    writer = csv.writer(file, delimiter="\t")
                    writer.writerow(row)
                    file.flush()
