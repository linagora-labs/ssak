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
    parser = argparse.ArgumentParser(description="Convert VoxPopuli dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--language", type=str, default="it", choices=["es", "en", "fr", "it", "de"])
    args = parser.parse_args()

    if args.input is None:
        args.input = f"/data-server/datasets/audio/transcript/multilang/VoxPopuli_labelled/data/{args.language}"
        print(f"Input path not specified, using default: {args.input}")
    if args.output is None:
        args.output = f"/data-server/datasets/audio/nemo/multi-turn/asr/{args.language}/nocontext/VoxPopuli"
        print(f"Output path not specified, using default: {args.output}")

    input_dataset = args.input
    output_path = Path(args.output)
    
    if (output_path / Path("test.jsonl")).exists()  and not args.force:
        print("Output path already exists, use --force to overwrite")
        exit()

    def get_gender(row):
        if row["gender"].lower().startswith("male"):
            return "m"
        elif row["gender"].lower().startswith("female"):
            return "f"
        else:
            return None
    
    genders = RowApplyFunction(return_columns="gender", execute_order=2, function=get_gender)

    def get_path(dataset_folder, split):
        return lambda row : os.path.join(dataset_folder, split, "audios", row["id"][:4], row["id"]+".wav")
    
    
    ids = RowApplyFunction(return_columns="id", execute_order=3, function=lambda row: row["id"].replace(":","_"))
    answers = RowApplyFunction(return_columns="answer", execute_order=2, function=lambda row: row["raw_text"] if row["raw_text"] != "" else row["normalized_text"])
    durations = Row2Duration(execute_order=4, max_workers=12)
    
    for split in ["train", "dev", "test"]:
        print()
        logger.info(f"Processing {split}")
        transcripts = CsvFile2Kaldi(
            input="asr_"+split+".tsv",
            return_columns=["id", "raw_text", "normalized_text", "speaker", None, "gender", None, None],
            separator="\t",
            header=True,
            execute_order=0,
        )
        paths = RowApplyFunction(return_columns="audio_path", execute_order=2, function=get_path(args.input, split))
        reader = Reader2Nemo(input_dataset, processors=[transcripts, paths, answers, genders, ids, durations])
        dataset = reader.load(debug=False, custom_metadata_to_keep=["gender"])
        dataset.save(output_path / Path(split+".jsonl"))