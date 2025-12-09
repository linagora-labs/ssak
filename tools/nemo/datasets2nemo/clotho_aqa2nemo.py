import argparse
import logging
import os
import shutil
from pathlib import Path
from ssak.utils.kaldi_converter import *
from ssak.utils.nemo_converter import Reader2Nemo

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
    
    class Vote2Kaldi(DatasetProcessor2Kaldi):
        def __init__(self, input, execute_order, vote_columns=["vote1", "vote2"]) -> None:
            super().__init__(input, execute_order)
            self.vote_columns = vote_columns
        
        def process(self, dataset, debug=False):
            grouped = dict()
            # group by audio_id and context
            # for each group, find the most voted
            for row in dataset:
                grouped_id = "_".join(row[i] for i in self.input).replace(" ", "_")
                grouped[grouped_id] = grouped.get(grouped_id, []) + [row]
            weight = {"yes": 1, "maybe": 0.5, "no": 0.1}
            new_data = []
            removed_data = 0
            for group, rows in grouped.items():
                scores = {}
                answer_to_rows = {}

                for row in rows:
                    answer = row[self.vote_columns[0]].lower()
                    conf = row[self.vote_columns[1]].lower()
                    w = weight.get(conf, 0)

                    scores[answer] = scores.get(answer, 0) + w
                    answer_to_rows.setdefault(answer, []).append(row)

                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                if not sorted_scores:
                    removed_data += 1
                    continue
                if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
                    removed_data += 1
                    continue
                if sorted_scores[0][1]<=1:
                    removed_data += 1
                    # print(sorted_scores)
                    # input()
                    continue
                winning_answer = sorted_scores[0][0]
                winner_row = answer_to_rows[winning_answer][0]
                new_data.append(winner_row)
            logger.info(f"Removed {removed_data} rows using vote")
            return new_data
    
    deduplicater = Vote2Kaldi(input=["audio_id", "context"], execute_order=5, vote_columns=["answer", "confidence"])
    
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
    
        reader = Reader2Nemo(input_dataset, processors=[transcripts, audio_ids, deduplicater, ids, audios, durations])
        dataset = reader.load(debug=False, dataset_name="clotho_aqa")
        dataset.save(output_path / Path(split+".jsonl"))
        
    shutil.move(os.path.join(output_path, "val.jsonl"), os.path.join(output_path, "dev.jsonl"))