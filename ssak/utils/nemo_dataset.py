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

    @property
    def text(self) -> str:
        """Alias for answer"""
        return self.answer
    
    @text.setter
    def text(self, value: str):
        self.answer = value
    
    def to_json(self) -> dict:
        """Convert to json"""
        row_data = vars(self)
        row_data["text"] = row_data.pop("answer")
        row_data["audio_filepath"] = str(row_data["audio_filepath"])
        row_data.pop("context")
        return row_data

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
        # If dataset is a list of lists:
        default_repr = object.__repr__(self)
        first_row = self.dataset[0] if self.dataset else "No data"
        if self.name:
            return f"{self.name} ({default_repr}, len={len(self.dataset)}): {first_row}"
        else:
            return f"{default_repr} (len={len(self.dataset)}): {first_row}"

    def __str__(self):
        # If dataset is a list of lists:
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
        for row in tqdm(kaldi_dataset, desc="Converting kaldi to nemo"):
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

    def load(self, input_file, type=None, debug=False, split=None, language=None, dataset_name=None, show_progress_bar=True):
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
                if type is None:
                    if "conversations" in json_row:
                        type = "multiturn"
                    else:
                        type = "asr"
                if type == "asr":
                    row = NemoDatasetRow(
                        id=json_row.get("id", json_row.get("utt_id", None)),
                        dataset_name=json_row.get("dataset_name", dataset_name),
                        audio_filepath=json_row["audio_filepath"],
                        offset=json_row.get("offset", 0),
                        duration=json_row["duration"],
                        answer=json_row["text"],
                        speaker=json_row.get("speaker", None),
                        language=json_row.get("language", language),
                        split=json_row.get("split", split),
                    )
                elif type == "multiturn":
                    row = NemoDatasetRow(
                        id=json_row.get("id", json_row.get("utt_id", None)),
                        dataset_name=json_row.get("dataset_name", dataset_name),
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
                if type == "asr":
                    row_data = row.to_json()
                elif type == "multiturn":
                    row_data = {
                        "id": row.id,
                        "conversations": [
                            {"from": "User", "value": row.context, "type": "text"},
                            {"from": "User", "value": str(row.audio_filepath), "type": "audio", "duration": row.duration, "offset": row.offset},
                            {"from": "Assistant", "value": row.answer, "type": "text"},
                        ],
                    }
                    if row.dataset_name is not None:
                        row_data["dataset_name"] = row.dataset_name
                else:
                    raise ValueError(f"Unkown type {type} for saving nemo dataset. Should be 'asr' or 'multiturn")
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
            if row.context is None or force_set_context:
                row.context = random.choice(contexts)
    
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
                row.audio_filepath = updated_audio_paths.get(row.audio_filepath, row.audio_filepath)
        if errors:
            new_dataset = []
            removed_lines = []
            for row in self.dataset:
                if row.audio_filepath != "error":
                    new_dataset.append(row)
                else:
                    removed_lines.append(row)
            self.dataset = new_dataset
            self.log_folder.mkdir(exist_ok=True, parents=True)
            with open(self.log_folder / "filtered_out_audio_empty.jsonl", "w") as f:
                for row in removed_lines:
                    json.dump(row.to_json(), f, ensure_ascii=False, indent=None)
                    f.write("\n")