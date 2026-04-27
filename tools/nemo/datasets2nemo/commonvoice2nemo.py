import argparse
import logging
import os
import shutil
from functools import partial
from ssak.utils.kaldi_converter import *
from pathlib import Path
from ssak.utils.nemo_converter import Reader2Nemo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CommonVoice dataset to NeMo format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--language", type=str, default="it", choices=["es", "en", "fr", "it", "de"])
    args = parser.parse_args()

    if args.input is None:
        args.input = f"/data-server/datasets/audio/raw/transcript/{args.language}/COMMONVOICE/cv-corpus-23.0-2025-09-05/{args.language}"
        print(f"Input path not specified, using default: {args.input}")
    if args.output is None:
        args.output = f"/data-server/datasets/audio/nemo/asr/{args.language}/nocontext/CommonVoice"
        print(f"Output path not specified, using default: {args.output}")

    input_dataset = args.input
    output_path = Path(args.output)

    durations = CsvFile2Kaldi(
        input="clip_durations.tsv",
        return_columns=["audio_path", "duration"],
        separator="\t",
        header=True,
        execute_order=1,
        merge_on="audio_path"
    )

    def get_gender(row):
        if row["gender"].startswith("male"):
            return "m"
        elif row["gender"].startswith("female"):
            return "f"
        else:
            return None
    
    genders = RowApplyFunction(return_columns="gender", execute_order=2, function=get_gender)

    def get_path(dataset_folder):
        return lambda row : os.path.join(dataset_folder, "clips", row['audio_path'])
    
    # paths = RowApplyFunction(return_columns="audio_path", execute_order=2, function=partial(get_path, dataset_folder=args.input))
    paths = RowApplyFunction(return_columns="audio_path", execute_order=2, function=get_path(args.input))

    
    def get_dur(row):
        return float(row["duration"])/1000
    dur_to_s = RowApplyFunction(return_columns="duration", execute_order=2, function=get_dur)
    
    for split in ["train", "dev", "test"]:
        transcripts = CsvFile2Kaldi(
            input=split+".tsv",
            return_columns=["speaker", "audio_path", "id", "answer", None, None, None, None, "gender", None],
            separator="\t",
            header=True,
            execute_order=0,
        )
    
        reader = Reader2Nemo(input_dataset, processors=[transcripts, durations, paths, dur_to_s, genders])
        dataset = reader.load(debug=False, custom_metadata_to_keep=["gender"])
        dataset.save(output_path / Path(split+".jsonl"))
    