import logging
import os
import re
import random
import shutil
import json
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass

from tqdm import tqdm

# from ssak.utils.kaldi import check_kaldi_dir

logger = logging.getLogger(__name__)

CONTEXTS = [
    "Transcrivez l'audio en français suivant de la manière la plus précise possible.",
    "Écrivez exactement ce qui est dit dans cet audio en français.",
    "Fournissez une transcription complète du discours en français dans ce clip audio.",
    "Écoutez l'audio en français et transcrivez-le mot pour mot.",
    "Écrivez la transcription complète de ce texte parlé en français.",
    # "Transcrivez cet audio en français en incluant toutes les pauses et hésitations.",
    # "Transcrivez avec précision l'audio en français, y compris tous les bruits de fond ou interjections.",
    # "Générez une transcription écrite de ce discours en français, en vous assurant qu'aucun détail n'est omis.",
    "Transcrivez le discours en français en respectant la ponctuation pour plus de clarté.",
    "Fournissez une transcription propre de cette conversation en français."
]

@dataclass
class NemoDatasetRow:
    """
    Dataclass for a row (/segment) in a nemo dataset

    Attributes:
        id (str): Segment id
    """

    id: str
    context: str = None
    answer: str = None
    audio_filepath: str = None
    duration: float = None
    offset: float = None
    dataset_name: str = None
    speaker: str = None
    language: str = None
    split: str = None


class NemoDataset:
    """
    Iterator class for nemo datasets.
    You can load, save, add, iterate over and normalize the dataset.

    Main attributes:
        name (str): Name of the dataset
        dataset (list): 

    Main methods:
    """

    def __init__(self, name=None, log_folder=None):
        if name:
            self.name = name
        self.log_folder = log_folder if log_folder else "nemo_data_processing"
        self.dataset = list()
        self.splits = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> NemoDatasetRow:
        return self.dataset[index]

    def __next__(self):
        for row in self.dataset:
            yield row

    def __iter__(self) -> Iterator["NemoDatasetRow"]:
        return self.__next__()

    def extend(self, dataset):
        """
        Extend the dataset with another dataset. Do not make any checks on the dataset.

        Args:
            dataset (KaldiDataset): Dataset to append to the current dataset
        """
        self.dataset.extend(dataset.dataset)

    def append(self, row):
        """
        Append a row to the dataset

        Args:
            row (dict or NemoDatasetRow): Row to append to the dataset. If a dict, the keys must be : {id, audio_id, audio_path, text, duration, start, end, speaker}
        """
        if not isinstance(row, NemoDatasetRow):
            row = NemoDatasetRow(**row)
        self.dataset.append(row)

    def kaldi_to_nemo(self, kaldi_dataset):
        for row in tqdm(kaldi_dataset):
            offset = row.start if row.start else 0
            if row.duration:
                duration = row.duration
            elif row.end:
                duration = row.end - offset
            else:
                duration = None
            nemo_row = NemoDatasetRow(
                id=row.id,
                audio_filepath=row.audio_path,
                offset=offset,
                duration=duration,
                answer=row.text,
                dataset_name=kaldi_dataset.name,
                speaker=row.speaker,
                split=row.split,
            )
            self.append(nemo_row)
    
    def load(self, input_file, type=None, debug=False):
        if debug and isinstance(debug, bool):
            debug = 10
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading dataset")):
                if debug and i>=debug:
                    break
                json_row = json.loads(line)
                if type is None:
                    if "conversations" in json_row:
                        type = "multiturn"
                    else:
                        type = "asr"
                if type=="asr":
                    row = NemoDatasetRow(
                        id=json_row["id"],
                        dataset_name=json_row.get("dataset_name", None),
                        audio_filepath=json_row["audio_filepath"],
                        offset=json_row["offset"],
                        duration=json_row["duration"],
                        answer=json_row["text"],
                        speaker=json_row.get("speaker", None),
                        language=json_row.get("language", None),
                        split=json_row.get("split", None),
                    )
                elif type=="multiturn":
                    row = NemoDatasetRow(
                        id=json_row["id"],
                        dataset_name=json_row.get("dataset_name", None),
                        audio_filepath=json_row["conversations"][1]["value"],
                        offset=json_row["conversations"][1]["offset"],
                        duration=json_row["conversations"][1]["duration"],
                        answer=json_row["conversations"][2]["value"],
                        context=json_row["conversations"][0]["value"],
                    )
                else:
                    raise ValueError(f"Unkown type {type} for saving nemo dataset. Should be 'asr' or 'multiturn")
                self.append(row)
        return type
    
    def save(self, output_file, type="multiturn"):
        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file.with_suffix(output_file.suffix + ".tmp"), "w", encoding="utf-8") as f:
            for row in tqdm(self, desc="Saving dataset"):
                if type=="asr":
                    row_data = vars(row)
                    row_data['text'] = row_data.pop("answer")
                    row_data['audio_filepath'] = str(row_data['audio_filepath'])
                    row_data.pop("context")
                elif type=="multiturn":
                    row_data = {"id": row.id, "conversations":
                        [
                            {"from": "User", "value": row.context, "type": "text"},
                            {"from": "User", "value": str(row.audio_filepath), "type": "audio", "duration": row.duration, "offset": row.offset},
                            {"from": "Assistant", "value": row.answer, "type": "text"},
                        ]
                    }
                    if row.dataset_name is not None:
                        row_data["dataset_name"] = row.dataset_name
                else:
                    raise ValueError(f"Unkown type {type} for saving nemo dataset. Should be 'asr' or 'multiturn")
                json.dump(row_data, f, ensure_ascii=False, indent=None)
                f.write("\n")
        shutil.move(output_file.with_suffix(output_file.suffix + ".tmp"), output_file)