import json
import logging
import shutil
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ssak.utils.kaldi_dataset import audio_checks

logger = logging.getLogger(__name__)

@dataclass
class NemoTurn:
    
    role: str = None
    value: str = None
    turn_type: str = None
    duration: float = None
    offset: float = 0.0
    
    @property
    def audio_filepath(self) -> str:
        """Alias for audio_filepath"""
        if self.turn_type == "audio":
            return self.value
        return None
    
    def to_json(self) -> dict:
        if self.turn_type == "audio":
            return {
                "from": self.role,
                "value": self.value,
                "type": self.turn_type,
                "duration": self.duration,
                "offset": self.offset,
            }
        else:
            return {
                "from": self.role,
                "value": self.value,
                "type": self.turn_type,
            }
        
    
    @classmethod
    def from_json(cls, data: dict):
        return cls(
            role=data.get("from"),
            value=data.get("value"),
            turn_type=data.get("type"),
            duration=data.get("duration"),
            offset=data.get("offset", 0.0),
        )

@dataclass
class NemoDatasetRow:
    """
    Dataclass for a row (/segment) in a nemo dataset

    Attributes:
        id (str): Segment id
    """

    id: str
    turns: list[NemoTurn] = None
    dataset_name: str = None
    speaker: str = None
    language: str = None
    split: str = None
    custom_metadata: dict = None

    @property
    def audio_filepath(self) -> str:
        """Alias for audio_filepath"""
        if self.turns[0].turn_type == "audio":
            return self.turns[0].value
        elif self.turns[1].turn_type == "audio":
            return self.turns[1].value
        return None

    @property
    def context(self) -> str:
        """Alias for context"""
        if self.turns[0].turn_type == "text":
            return self.turns[0].value
        return None

    @property
    def answer(self) -> str:
        """Alias for answer"""
        if self.turns[1].turn_type == "text":
            return self.turns[1].value
        elif self.turns[2].turn_type == "text":
            return self.turns[2].value
        return None

    @property
    def text(self) -> str:
        """Alias for text"""
        return self.answer

    def to_json(self, data_type="multiturn") -> dict:
        """Convert to json"""
        
        row_data = vars(self).copy()

        if data_type == "asr":
            row_data["audio_filepath"] = self.turns[0].value
            row_data["duration"] = self.turns[0].duration
            row_data["offset"] = self.turns[0].offset
            row_data["text"] = self.turns[1].value

        elif data_type == "multiturn":
            row_data["conversations"] = [t.to_json() for t in self.turns]

        row_data.pop("turns", None)
        row_data = {k: v for k, v in row_data.items() if v is not None}

        return row_data


    def get_audio_turns(self) -> list:
        """Get all audios in the conversation"""
        return [t for t in self.turns if t.turn_type == "audio"]

    @classmethod
    def from_json(cls, json_row: dict, data_type: str, dataset_name=None):
        if data_type == "asr":
            audio_turn = NemoTurn(
                role="User",
                value=json_row["audio_filepath"],
                duration=json_row["duration"],
                offset=json_row.get("offset", 0.0),
                turn_type="audio"
            )
            text_turn = NemoTurn(role="Assistant", value=json_row["text"], turn_type="text")
            turns = [audio_turn, text_turn]
        elif data_type == "multiturn":
            turns_json = json_row["conversations"]
            turns = [NemoTurn.from_json(t) for t in turns_json]

        return cls(
            id=json_row.get("id", json_row.get("utt_id")),
            dataset_name=json_row.get("dataset_name", dataset_name),
            turns=turns,
            speaker=json_row.get("speaker"),
            language=json_row.get("language"),
            split=json_row.get("split"),
        )


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
        self.name = name
        self.log_folder = Path(log_folder) if log_folder else "nemo_data_processing"
        self.dataset = list()
        self.splits = set()
    
    def __repr__(self):
        default_repr = object.__repr__(self)
        first_row = self.dataset[0] if self.dataset else "No data"
        if self.name:
            return f"{self.name} ({default_repr}, len={len(self.dataset)}): {first_row}"
        else:
            return f"{default_repr} (len={len(self.dataset)}): {first_row}"

    def __str__(self):
        default_repr = object.__repr__(self)
        if self.name:
            return f"{self.name}"
        else:
            return f"{default_repr}"

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
            dataset (NemoDataset): Dataset to append to the current dataset
        """
        self.dataset.extend(dataset.dataset)

    def get_audio_paths(self, unique=True):
        """
        Get the audio paths of the dataset

        Returns:
            set (or list if unique is False): Set of audio paths
        """
        if unique:
            return {turn.value for i in self.dataset for turn in i.turns if turn.turn_type == "audio"}
        return [turn.value for i in self.dataset for turn in i.turns if turn.turn_type == "audio"]

    def append(self, row):
        """
        Append a row to the dataset

        Args:
            row (dict or NemoDatasetRow): Row to append to the dataset. If a dict, the keys must be : TODO
        """
        if not isinstance(row, NemoDatasetRow):
            row = NemoDatasetRow(**row)
        self.dataset.append(row)

    def kaldi_to_nemo(self, kaldi_dataset):
        for row in tqdm(kaldi_dataset, desc="Converting kaldi to nemo"):
            offset = row.start if row.start else 0.0
            if row.duration:
                duration = row.duration
            elif row.end:
                duration = row.end - offset
            else:
                duration = None
            audio_turn = NemoTurn(
                role="User",
                value=row.audio_path,
                duration=duration,
                offset=offset,
                turn_type="audio"
            )
            text_turn = NemoTurn(role="Assistant", value=row.text, turn_type="text")
            turns = [audio_turn, text_turn]
            nemo_row = NemoDatasetRow(
                id=row.id,
                turns=turns,
                dataset_name=kaldi_dataset.name,
                speaker=row.speaker,
                split=row.split,
            )
            self.append(nemo_row)

    def load(self, input_file, data_type=None, debug=False, split=None, language=None, dataset_name=None, show_progress_bar=True):
        if debug and isinstance(debug, bool):
            debug = 10
        with open(input_file, encoding="utf-8") as f:
            if show_progress_bar:
                pbar = tqdm(f, desc="Loading dataset")
            else:
                pbar = f
            for i, line in enumerate(pbar):
                if debug and i >= debug:
                    break
                json_row = json.loads(line)
                if data_type is None:
                    if "conversations" in json_row:
                        data_type = "multiturn"
                    else:
                        data_type = "asr"
                row = NemoDatasetRow.from_json(json_row, data_type=data_type, dataset_name=dataset_name)
                self.append(row)
        return data_type

    def save(self, output_file, data_type="multiturn", keep_minimal=True):
        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file.with_suffix(output_file.suffix + ".tmp"), "w", encoding="utf-8") as f:
            for row in tqdm(self, desc="Saving dataset"):
                if data_type == "asr":
                    row_data = row.to_json(data_type)
                    if keep_minimal:
                        row_data = {k: v for k, v in row_data.items() if k in ["id", "audio_filepath", "text", "offset", "duration"]}
                elif data_type == "multiturn":
                    row_data = row.to_json(data_type)
                    if row.dataset_name is not None:
                        row_data["dataset_name"] = row.dataset_name
                else:
                    raise ValueError(f"Unkown type {data_type} for saving nemo dataset. Should be 'asr' or 'multiturn")
                json.dump(row_data, f, ensure_ascii=False, indent=None)
                f.write("\n")
        shutil.move(output_file.with_suffix(output_file.suffix + ".tmp"), output_file)

    def check_if_segments_in_audios(self, acceptance_end_s=0.25):
        from pydub.utils import mediainfo

        new_data = []
        removed_lines = []
        files_duration = dict()
        for row in tqdm(self, desc="Check if segments are in audios"):
            if row.audio_filepath not in files_duration:
                dur = round(float(mediainfo(row.audio_filepath)["duration"]), 3)
                files_duration[row.audio_filepath] = dur
            dur = files_duration[row.audio_filepath]
            if row.offset >= dur:
                removed_lines.append(row)
            elif row.offset + row.duration > dur + acceptance_end_s:
                removed_lines.append(row)
            else:
                new_data.append(row)
        self.dataset = new_data
        logger.info(f"Removed {len(removed_lines)} segments that were not in audios (start or end after audio), check removed_lines_not_in_audios file")
        self.log_folder.mkdir(exist_ok=True, parents=True)
        with open(self.log_folder / "filtered_out_not_in_audios.jsonl", "w") as f:
            for row in removed_lines:
                json.dump(row.to_json(), f, ensure_ascii=False, indent=None)
                f.write("\n")
    
    def set_context_if_none(self, contexts, force_set_context=False):
        for row in tqdm(self, desc="Set context if none"):
            new_turn = NemoTurn(role="User", turn_type="text", value=random.choice(contexts))
            if row.turns[0].turn_type=="audio":
                row.turns.insert(0, new_turn)
            elif force_set_context:
                row.turns[0] = new_turn
    
    def normalize_audios(self, output_wavs_conversion_folder, target_sample_rate=16000, target_extension=None, num_workers=1):
        """
        Check audio files sample rate and number of channels and convert them if they don't match the target sample rate/number of channels.

        Updates the audio_path in the dataset with the new path if the audio file was converted.

        Args:
            output_wavs_conversion_folder (str): Folder where to save the transformed audio files
            target_sample_rate (int): Target sample rate for the audio files
            target_extension (str): Optional. Target extension for the audio files (wav, mp3...). If set to None, it will keep the original extension
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        updated_audio_paths = dict()
        audio_paths = list(self.get_audio_paths(unique=True))
        errors = False
        if num_workers == 1:
            for audio_path in tqdm(audio_paths, total=len(audio_paths), desc="Checking audio files"):
                new_path = audio_checks(audio_path, output_wavs_conversion_folder, target_sample_rate, target_extension)
                if new_path != audio_path:
                    updated_audio_paths[audio_path] = new_path
                    if new_path == "error":
                        errors = True
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        audio_checks,
                        audio_path,
                        output_wavs_conversion_folder,
                        target_sample_rate,
                        target_extension,
                    ): audio_path
                    for audio_path in audio_paths
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Checking audio files"):
                    audio_path = futures[future]
                    try:
                        new_path = future.result()  # Get the result of audio_checks for each audio file
                        if new_path != audio_path:
                            updated_audio_paths[audio_path] = new_path
                            if new_path == "error":
                                errors = True
                    except Exception as e:
                        raise RuntimeError(f"Error processing {audio_path}: {e}")
        if len(updated_audio_paths) > 0:
            for row in self.dataset:
                for turn in row.turns:
                    if turn.type=="audio":
                        turn.audio_filepath = updated_audio_paths.get(row.audio_filepath, row.audio_filepath)
        if errors:
            new_dataset = []
            removed_lines = []
            for row in self.dataset:
                valid_row = True
                for turn in row.turns:
                    if turn.type=="audio" and turn.value=="error":
                        valid_row = False
                if valid_row:
                    new_dataset.append(row)
                else:
                    removed_lines.append(row)
            self.dataset = new_dataset
            self.log_folder.mkdir(exist_ok=True, parents=True)
            with open(self.log_folder / "filtered_out_audio_empty.jsonl", "w") as f:
                for row in removed_lines:
                    json.dump(row.to_json("asr"), f, ensure_ascii=False, indent=None)
                    f.write("\n")