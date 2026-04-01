import argparse
import logging
import os
import shutil
from pathlib import Path
from ssak.utils.kaldi_converter import *
from ssak.utils.nemo_converter import Reader2Nemo
from ssak.utils.nemo_dataset import NemoDataset

# ln -s ../../multilang/FLEURS/clips/it_it clips
# ln -s ../../multilang/FLEURS/it_it/test.tsv test.tsv
# ln -s ../../multilang/FLEURS/it_it/train.tsv train.tsv
# ln -s ../../multilang/FLEURS/it_it/validation.tsv validation.tsv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FLEURS dataset to NeMo format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--language", type=str, default="es", choices=["es", "en", "fr", "it", "de", "pt", "ar", "nl"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"/data-server/datasets/audio/nemo/asr/{args.language}/nocontext/FLEURS"
    
    language = f"{args.language}_{args.language}"
    if args.language == "pt":
        language = "pt_br"
    elif args.language == "ar":
        language = "ar_eg"
    elif args.language == "es":
        language = "es_419"
    elif args.language == "en":
        language = "en_us"
    
    if args.input is None:
        args.input = f"/data-server/datasets/audio/raw/transcript/{language}/FLEURS"

    input_dataset = args.input
    output_path = Path(args.output)

    splits = ["train", "validation", "test"]
    
    for split in splits:
        print()
        logger.info(f"Processing {split}")
        transcripts = CsvFile2Kaldi(
            input=split+".tsv",
            return_columns=["id", "audio_path", "answer", None, "duration", "language", "gender"],
            separator="\t",
            header=True,
            execute_order=0,
        )
        def get_audio_path(row):
            audio_path = Path(input_dataset) / Path("clips") / Path(split) / Path(row["audio_path"]).name
            audio_path = str(audio_path)
            return audio_path
        
        audio_paths = RowApplyFunction(function=get_audio_path, return_columns=["audio_path"], execute_order=1)
        
        duration = RowApplyFunction(function=lambda row: float(row["duration"])/16000, return_columns=["duration"], execute_order=2)
        
        reader = Reader2Nemo(input_dataset, processors=[transcripts, audio_paths, duration])
        dataset = reader.load(debug=False, dataset_name="fleurs")
        dataset.save(output_path / Path(split+".jsonl"))