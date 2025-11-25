import argparse
import logging
import os
import shutil
from pathlib import Path
from ssak.utils.kaldi_converter import *
from ssak.utils.nemo_converter import Reader2Nemo
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Clotho AQA dataset to NeMo format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/data-server/datasets/audio/sounds/clotho_aqa")
    parser.add_argument("--output", type=str, default="/data-server/datasets/audio/nemo/multi-turn/sound/audio-question-answering/en/clotho_aqa")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = Path(args.output)

    # if os.path.exists(raw) and not args.force:
    #     raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    # elif os.path.exists(raw):
    #     shutil.rmtree(raw)


    audios = AudioFolder2Kaldi("audio_files", execute_order=3, extracted_id="audio_id", audio_extensions=[".wav"], sort_merging=False)
    durations = Row2Duration(execute_order=4)
    audio_ids = Row2Info(input="audio_id", return_columns=["audio_id"], execute_order=1, separator=".", info_position=[0, -1])
    
    class RowApplySetFunction(Row2KaldiInfo):
        def __init__(self, function, return_columns, execute_order, sort_merging=True, input=None) -> None:
            super().__init__(input, return_columns, execute_order=execute_order, sort_merging=sort_merging)
            if not function:
                raise ValueError(f"Function should be passed")
            self.function = function
            self.id_set = set()
        
        def __call__(self, row):
            new_id, self.id_set = self.function(row, self.id_set)
            return {self.return_columns[0]: new_id}

    def get_id(row, id_set):
        q = 1
        base = (
            row["audio_id"]
            .replace(" ", "_")
            .replace("&", "_")
            .replace(",", "_")
            .replace("#", "_")
        )
        
        new_id = f"{base}_q{q}"
        while new_id in id_set:
            q += 1
            new_id = f"{base}_q{q}"

        id_set.add(new_id)
        return new_id, id_set

    ids = RowApplySetFunction(return_columns="id", execute_order=2, function=get_id)
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        print()
        logger.info(f"Processing {split}")
        transcripts = CsvFile2Kaldi(
            input="clotho_aqa_"+split+".csv",
            return_columns=["audio_id", "context", "answer", "confidence"],
            separator=",",
            header=True,
            execute_order=0,
        )
    
        reader = Reader2Nemo(input_dataset, processors=[transcripts, audio_ids, ids, audios, durations])
        dataset = reader.load(debug=False, dataset_name="clotho_aqa", custom_metadata_to_keep={"confidence"})
        
        def filter_confidence(dataset):
            maybe_dataset = NemoDataset()
            yes_dataset = NemoDataset()
            for row in dataset:
                confidence = row.custom_metadata["confidence"].strip().lower()
                row.custom_metadata = None
                if confidence=="yes":
                    yes_dataset.append(row)
                else:
                    maybe_dataset.append(row)
            return yes_dataset, maybe_dataset
        
        yes, maybe = filter_confidence(dataset)
        
        yes.save(output_path / Path(split+".jsonl"))
        maybe.save(output_path / Path("maybe_"+split+".jsonl"))
        
    shutil.move(os.path.join(output_path, "val.jsonl"), os.path.join(output_path, "dev.jsonl"))