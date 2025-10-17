import argparse
import logging
import os
import shutil
from functools import partial
from ssak.utils.kaldi_converter import *
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LesVocaux dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/data-server/datasets/audio/transcript/en/COMMONVOICE/cv-corpus-18.0-2024-06-14/en")
    parser.add_argument("--output", type=str, default="/data-server/datasets/audio/kaldi/en/CommonVoice")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw = os.path.join(output_path, "casepunc")

    # if os.path.exists(raw) and not args.force:
    #     raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    # elif os.path.exists(raw):
    #     shutil.rmtree(raw)



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
        elif row["gender"].strip() == "": 
            return None
        elif row["gender"].startswith("transgender"):
            return None
        else:
            raise ValueError(f"Unknown gender: {row['gender']} ({row})")
    genders = RowApplyFunction(return_columns="gender", execute_order=2, function=get_gender)

    def get_path(row, dataset_folder):
        return os.path.join(dataset_folder, "clips", row['audio_path'])
    
    paths = RowApplyFunction(return_columns="audio_path", execute_order=2, function=partial(get_path, dataset_folder=args.input))
    
    def get_dur(row):
        return float(row["duration"])/1000
    dur_to_s = RowApplyFunction(return_columns="duration", execute_order=2, function=get_dur)
    
    for input in ["train", "dev", "test"]:
        transcripts = CsvFile2Kaldi(
            input=input+".tsv",
            return_columns=["speaker", "audio_path", "id", "text", None, None, None, None, "gender", None],
            separator="\t",
            header=True,
            execute_order=0,
        )
    
        reader = Reader2Kaldi(input_dataset, processors=[transcripts, durations, genders, paths, dur_to_s])
        dataset = reader.load(debug=False, accept_missing_speaker=True)
        dataset.save(os.path.join(raw, input), True)
    