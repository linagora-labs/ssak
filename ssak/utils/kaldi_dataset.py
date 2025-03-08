import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

from tqdm import tqdm

from ssak.utils.kaldi import check_kaldi_dir

logger = logging.getLogger(__name__)


@dataclass
class KaldiDatasetRow:
    """
    Dataclass for a row (/segment) in a kaldi dataset

    Attributes:
        id (str): Segment id
        text (str): Text of the segment
        audio_id (str): Audio id
        audio_path (str): Path to the audio file
        normalized_text (str) : Optional. Normalized text of the segment
        duration (float): Optional if start and end are specified. Duration of the segment
        start (float): Start time of the segment
        end (float): Optional if start and duration are specified. End time of the segment
        speaker (str): Speaker id
        gender (str): Optional. Must be "M" or "F".
    """

    id: str
    text: str = None
    audio_id: str = None
    audio_path: str = None
    normalized_text: str = None
    duration: float = None
    start: float = None
    end: float = None
    speaker: str = None
    gender: str = None
    split: str = None

    def check_row(
        self,
        show_warnings=True,
        accept_warnings=False,
        warn_if_shorter_than=0.05,
        warn_if_longer_than=3600,
        check_if_segments_in_audio=False,
        accept_missing_speaker=False,
    ):
        """
        Check if the row is valid and fill missing attributes if possible. If not, it will log (or throw an error) a warning and skip.
        """
        if self.duration is not None:
            self.duration = float(self.duration)
        if self.start is not None:
            self.start = float(self.start)
        if self.end is not None:
            self.end = float(self.end)
        if self.duration is None and self.start is not None and self.end is not None:
            self.duration = self.end - self.start
        elif self.end is None and self.start is not None and self.duration is not None:
            self.end = self.start + self.duration
        elif self.duration is not None and self.start is None and self.end is None:
            self.start = 0
            self.end = self.duration
        if self.audio_id is None:
            self.audio_id = self.id
        if self.audio_path is None:
            raise ValueError(f"Audio path must be specified for {self.id} ({self})")
        if self.duration is None:
            raise ValueError(f"Duration (or end and start) must be specified for row {self}")
        if warn_if_shorter_than is not None and self.duration < warn_if_shorter_than:
            if show_warnings:
                logger.warning(f"{'Skipping: ' if not accept_warnings else ''}Duration too short for {self.id}: {self.duration:.3f} ({self.start}->{self.end}) (file: {self.audio_id})")
            if not accept_warnings:
                return False
        if warn_if_longer_than is not None and self.duration >= warn_if_longer_than:
            if show_warnings:
                logger.warning(f"{'Skipping: ' if not accept_warnings else ''}Duration too long for {self.id}: {self.duration:.3f} ({self.start}->{self.end}) (file: {self.audio_id})")
            if not accept_warnings:
                return False
        if self.text is not None:  # should not check if None (Should only be None when load_text is False)
            self.text = re.sub(r"\s+", " ", self.text).strip()
            BOM = "\ufeff"
            self.text = self.text.replace(BOM, "")
            if len(self.text) == 0:
                if show_warnings:
                    logger.warning(f"{'Skipping: ' if not accept_warnings else ''}Empty text for {self.id} (with file: {self.audio_id})")
                if not accept_warnings:
                    return False
        if self.gender is not None:
            self.gender = self.gender.lower()
            if self.gender == "h":
                self.gender = "m"
            if self.gender != "m" and self.gender != "f":
                raise ValueError(f"Gender must be 'm' or 'f' not {self.gender} for {self.id} (with file: {self.audio_id})")
        if check_if_segments_in_audio:
            from ssak.utils.audio import get_audio_duration

            dur = round(get_audio_duration(self.audio_path), 3)
            if self.start > dur:
                if show_warnings:
                    logger.warning(f"{'Skipping: ' if not accept_warnings else ''}Start time is greater than audio duration ({self.start}>{dur}) for {self.id} (with file: {self.audio_id})")
                if not accept_warnings:
                    return False
            elif self.end > dur + 0.1:
                if show_warnings:
                    logger.warning(f"{'Skipping: ' if not accept_warnings else ''}End time is greater than audio duration ({self.end}>{dur}) for {self.id} (with file: {self.audio_id})")
                if not accept_warnings:
                    return False
        if self.speaker is None and not accept_missing_speaker:
            raise ValueError(f"Speaker must be specified for {self.id} (with file: {self.audio_id})")
        return True


class KaldiDataset:
    """
    Iterator class for kaldi datasets.
    You can load, save, add, iterate over and normalize the dataset.

    Main attributes:
        name (str): Name of the dataset
        dataset (list): List of KaldiDatasetRow objects (See KaldiDatasetRow doc for more info)

    Main methods:
        append(row): Append a row to the dataset
        save(output_dir): Save the dataset to a kaldi directory
        load(input_dir): Load a kaldi dataset from a directory and adds it to the dataset
        normalize_dataset(apply_text_normalization): Normalize the texts in the dataset using the format_text_latin function from ssak.utils.text_latin
        normalize_audios(output_wavs_conversion_folder, target_sample_rate): Check audio files sample rate and number of channels and convert them if they don't match the target sample rate/number of channels
    """

    def __init__(self, name=None, row_checking_kwargs=dict(), accept_missing_speaker=False):
        if name:
            self.name = name
        self.row_checking_kwargs = row_checking_kwargs
        self.accept_missing_speaker = accept_missing_speaker
        self.dataset = list()
        self.splits = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> KaldiDatasetRow:
        return self.dataset[index]

    def __next__(self):
        for row in self.dataset:
            yield row

    def __iter__(self) -> Iterator["KaldiDatasetRow"]:
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
            row (dict or KaldiDatasetRow): Row to append to the dataset. If a dict, the keys must be : {id, audio_id, audio_path, text, duration, start, end, speaker}
        """
        if not isinstance(row, KaldiDatasetRow):
            row = KaldiDatasetRow(**row)
        if row.check_row(accept_missing_speaker=self.accept_missing_speaker, **self.row_checking_kwargs):
            self.dataset.append(row)

    def get_ids(self, unique=True):
        """
        Get the ids of the dataset

        Args:
            unique (bool): If True, it will return a set of ids, otherwise a list
        Returns:
            set: Set of ids
        """
        if unique:
            return set([i.id for i in self.dataset])
        return [i.id for i in self.dataset]

    def get_speakers(self, unique=True):
        """
        Get the speakers of the dataset

        Args:
            unique (bool): If True, it will return a set of speakers, otherwise a list
        Returns:
            set (or list if unique is False): Set of speakers
        """
        if unique:
            return set([i.speaker for i in self.dataset])
        return [i.speaker for i in self.dataset]

    def get_audio_ids(self, unique=True):
        """
        Get the audio ids of the dataset

        Returns:
            set (or list if unique is False): Set of audio ids
        """
        if unique:
            return set([i.audio_id for i in self.dataset])
        return [i.audio_id for i in self.dataset]

    def get_audio_paths(self, unique=True):
        """
        Get the audio paths of the dataset

        Returns:
            set (or list if unique is False): Set of audio paths
        """
        if unique:
            return set([i.audio_path for i in self.dataset])
        return [i.audio_path for i in self.dataset]

    def get_speaker_segments(self, speaker):
        """
        Get the segments of a speaker

        Args:
            speaker (str): Speaker id

        Returns:
            list: List of KaldiDatasetRow objects
        """
        return [i for i in self.dataset if i.speaker == speaker]

    def get_duration(self, mode=sum, target="segment"):
        if mode == "sum" or mode == "min" or mode == "max":
            mode = eval(mode)
        if target == "wav" or target == "audio":
            from ssak.utils.audio import get_audio_duration

            durations = []
            for i in self.get_audio_paths(unique=True):
                durations.append(get_audio_duration(i))
            return mode(durations)
        return mode([i.duration for i in self.dataset])

    def filter_by_audio_ids(self, audio_ids):
        """
        Filter the dataset by audio ids

        Args:
            audio_ids (list): List of audio ids

        Returns:
            KaldiDataset: New KaldiDataset object with the filtered dataset
        """
        new_dataset = KaldiDataset()
        for row in self.dataset:
            if row.audio_id in audio_ids:
                new_dataset.append(row)
        return new_dataset

    def filter_by_speakers(self, speakers):
        """
        Filter the dataset by speakers

        Args:
            speakers (list or set): Set of speaker ids

        Returns:
            KaldiDataset: New KaldiDataset object with the filtered dataset
        """
        if not isinstance(speakers, set):  # set are infinitely faster for lookups
            speakers = set(speakers)
        new_dataset = KaldiDataset()
        for row in self.dataset:
            if row.speaker in speakers:
                new_dataset.append(row)
        return new_dataset

    def normalize_dataset(self, apply_text_normalization=True):
        """
        Normalize the texts in the dataset using the format_text_latin function from ssak.utils.text_latin

        Args:
            apply_text_normalization (bool): If True, the normalized text will replace the original text in the dataset, otherwise it will be stored in the normalized_text attribute
        """
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        if self.dataset[0].normalized_text is not None:
            logger.warning("Dataset is already normalized (or at least first segment), skipping normalization")
            return
        for row in tqdm(self.dataset, total=len(self.dataset), desc="Normalizing texts"):
            from ssak.utils.text_latin import format_text_latin

            row.normalized_text = format_text_latin(row.text)
            if apply_text_normalization:
                row.text = row.normalized_text

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
                new_path = self.audio_checks(audio_path, output_wavs_conversion_folder, target_sample_rate, target_extension)
                if new_path != audio_path:
                    updated_audio_paths[audio_path] = new_path
                    if new_path == "error":
                        errors = True
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self.audio_checks,
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
                row.audio_path = updated_audio_paths.get(row.audio_path, row.audio_path)
        if errors:
            new_dataset = []
            removed_lines = []
            for row in self.dataset:
                if row.audio_path != "error":
                    new_dataset.append(row)
                else:
                    removed_lines.append(row)
            self.dataset = new_dataset
            with open("removed_lines", "w") as f:
                for row in removed_lines:
                    f.write(str(row) + "\n")

    def add_splits(self, splits, function_id_to_id=None):
        for row in tqdm(self.dataset, total=len(self.dataset), desc="Adding splits"):
            id = row.id
            if function_id_to_id is not None:
                id = function_id_to_id(id)
            if id not in splits:
                print(f"Missing split for {id}")
                continue
            if splits[id] not in self.splits:
                self.splits.add(splits[id])
            row.split = splits[id]

    def save(self, output_dir, check_durations_if_missing=False):
        """
        Save the dataset to a kaldi directory

        Args:
            output_dir (str): Path to the output directory
            check_durations_if_missing (bool): If True, it will check the duration of the audio files if it is not specified in the dataset
        """
        speakers_to_gender = dict()
        no_spk = True
        saved_wavs = set()
        output_dirs = [output_dir]
        total_saved_rows = 0
        if self.splits is not None and len(self.splits) > 0:
            output_dirs = [os.path.join(output_dir, i) for i in self.splits]
        for output_dir in output_dirs:
            nb_rows = 0
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "text"), "w", encoding="utf-8") as text_file, open(os.path.join(output_dir, "wav.scp"), "w") as wav_file, open(os.path.join(output_dir, "utt2spk"), "w") as uttspkfile, open(
                os.path.join(output_dir, "utt2dur"), "w"
            ) as uttdurfile, open(os.path.join(output_dir, "segments"), "w") as segmentfile:
                for row in tqdm(self.dataset, total=len(self.dataset), desc=f"Saving kaldi to {output_dir}"):
                    if row.split is not None and row.split != os.path.basename(output_dir):
                        continue
                    nb_rows += 1
                    text_file.write(f"{row.id} {row.text}\n")
                    if row.audio_id not in saved_wavs:
                        audio_path = row.audio_path
                        if " " in audio_path:
                            audio_path = f"'{audio_path}'"
                        wav_file.write(f"{row.audio_id} {row.audio_path}\n")
                        saved_wavs.add(row.audio_id)
                    if row.speaker is not None:
                        no_spk = False
                        uttspkfile.write(f"{row.id} {row.speaker}\n")
                        if row.gender is not None:
                            speakers_to_gender[row.speaker] = row.gender
                    duration = row.duration if row.duration is not None else None
                    if duration is None and row.end is not None and row.start is not None:
                        duration = row.end - row.start
                    elif duration is None:
                        if check_durations_if_missing:
                            import torchaudio

                            infos = torchaudio.info(row.audio_path)
                            duration = infos.num_frames / infos.sample_rate
                        else:
                            raise ValueError(f"Duration (or end and start) must be specified for row {row.id}")
                    uttdurfile.write(f"{row.id} {duration:.3f}\n")
                    start = row.start if row.start is not None else 0
                    end = row.end if row.end is not None else start + duration
                    segmentfile.write(f"{row.id} {row.audio_id} {start:.3f} {end:.3f}\n")
            if no_spk:
                os.remove(os.path.join(output_dir, "utt2spk"))
            if len(speakers_to_gender) > 0:
                with open(os.path.join(output_dir, "spk2gender"), "w") as f:
                    for i in speakers_to_gender:
                        f.write(f"{i} {speakers_to_gender[i].lower()}\n")
            logger.info(f"Validating dataset {output_dir}")
            if self.accept_missing_speaker:
                logger.info("Skipping check_kaldi_dir because it will raise and error since accept_missing_speaker is True")
            else:
                check_kaldi_dir(output_dir)
            logger.info(f"Saved {nb_rows} rows to {output_dir}")
            total_saved_rows += nb_rows
        if len(self.splits) > 0:
            logger.info(f"Saved {total_saved_rows} rows in total")
        if total_saved_rows != len(self.dataset):
            logger.warning(f"Saved {total_saved_rows} rows but dataset has {len(self.dataset)} rows")

    def load(self, input_dir, show_progress=True, load_texts=True, load_speakers=True):
        """
        Load a kaldi dataset from a directory and adds it to the dataset

        Args:
            input_dir (str): Path to the kaldi dataset directory
        """
        texts = None
        if load_texts:
            texts = parse_text_file(os.path.join(input_dir, "text"))
        spks = dict()
        if load_speakers:
            spks = parse_utt2spk_file(os.path.join(input_dir, "utt2spk"))
        else:
            self.accept_missing_speaker = True
            logger.info("Accept_missing_speaker is set to True since load_speakers is set to False")
        if not os.path.exists(os.path.join(input_dir, "segments")):
            file = "wav.scp"
            durations = parse_utt2dur_file(os.path.join(input_dir, "utt2dur"))
        else:
            file = "segments"
            wavs = parse_wav_scp_file(os.path.join(input_dir, "wav.scp"))
        with open(os.path.join(input_dir, file)) as f:
            if show_progress:
                loop = tqdm(f.readlines(), desc=f"Loading {input_dir}")
            else:
                loop = f.readlines()
            for line in loop:
                line = line.strip().split()
                if file == "segments":
                    start, end = round(float(line[2]), 3), round(float(line[3]), 3)
                    duration = round(end - start, 3)
                    seg_id = line[0]
                    audio_id = line[1]
                    wav_path = wavs[audio_id]
                else:
                    seg_id = audio_id = line[0]
                    wav_path = get_audio_from_wav_scp_line(line)
                    if durations is not None:
                        duration = durations[seg_id]
                    else:
                        from ssak.utils.audio import get_audio_duration

                        duration = get_audio_duration(wav_path)
                    start, end = 0, duration
                self.append(
                    KaldiDatasetRow(
                        id=seg_id,
                        text=texts[seg_id] if texts else None,
                        audio_path=wav_path,
                        duration=duration,
                        audio_id=audio_id,
                        start=start,
                        end=end,
                        speaker=spks.get(seg_id, None),
                    )
                )
        logger.info(f"Loaded {len(self.dataset)} rows from {input_dir}")

    def audio_checks(self, audio_path, new_folder, target_sample_rate=16000, target_extension=None, max_channel=1):
        """
        Check audio file sample rate and number of channels and convert it if it doesn't match the target sample rate/number of channels.

        Args:
            audio_path (str): Path to the audio file
            new_folder (str): Folder where to save the transformed audio file
            target_sample_rate (int): Target sample rate for the audio file
            target_extension (str): Optional. Target extension for the audio file. If set to None, it will keep the original extension
            max_channel (int): Maximum number of channels for the audio file. If the audio file has more channels, it will keep only the first channel. TODO: Add option to keep all channels in different files
        """
        from pydub import AudioSegment
        from pydub.utils import mediainfo

        if new_folder:
            if target_extension:
                if not target_extension.startswith("."):
                    target_extension = "." + target_extension
                new_path = os.path.join(new_folder, os.path.basename(audio_path).replace(os.path.splitext(audio_path)[1], target_extension))
            else:
                new_path = os.path.join(new_folder, os.path.basename(audio_path))
        else:
            raise ValueError("New folder must be specified for audio conversion")
        if not os.path.exists(new_path):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} does not exist")
            infos = mediainfo(audio_path)
            src_sample_rate = int(infos["sample_rate"])
            src_n_channels = int(infos["channels"])
            try:
                if infos["duration"] == "N/A":  # or float(infos['duration'])<0.01:
                    logger.error(f"Audio file {audio_path} has no duration: {infos['duration']}. It is probably corrupted!")
                    return "error"
                elif src_n_channels > max_channel or src_sample_rate != target_sample_rate or (target_extension is not None and not audio_path.endswith(target_extension)):
                    waveform = AudioSegment.from_file(audio_path)
                    if src_n_channels > max_channel:
                        logger.debug(f"Audio file {audio_path} has {src_n_channels} channels. Converting to 1 channel...")
                        waveform = waveform.set_channels(1)
                    if src_sample_rate != target_sample_rate:
                        logger.debug(f"Audio file {audio_path} has sample rate of {src_sample_rate}. Converting to {target_sample_rate}Hz...")
                        waveform = waveform.set_frame_rate(target_sample_rate)
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder, exist_ok=True)
                    waveform.export(new_path, format="wav")
                    return new_path
                elif not audio_path.endswith(target_extension):
                    logger.debug(f"Audio file has the wrong extension {audio_path}. Converting to {target_extension}...")
                    waveform = AudioSegment.from_file(audio_path)
                    waveform.export(new_path, format="wav")
                    return new_path
                else:
                    return audio_path
            except Exception as e:
                raise Exception(f"Error with {audio_path} with infos: {infos}") from e
        return new_path

    def apply_filter(self, filter, filter_out=True):
        new_data = []
        removed_lines = []
        for row in self.dataset:
            if filter_out and filter(row):
                removed_lines.append(row)
            elif not filter_out and filter(row):
                new_data.append(row)
            elif filter_out:
                new_data.append(row)
            else:
                removed_lines.append(row)
        self.dataset = new_data
        with open("filtered_out", "w") as f:
            for row in removed_lines:
                f.write(str(row) + "\n")


def get_audio_from_wav_scp_line(line):
    line = line[1:]
    if line[0] == "sox" or line[0] == "flac" or line[0] == "/usr/bin/sox":
        line = line[1:]
    audio_path = line[0]
    if audio_path.startswith("'") and not audio_path.endswith("'"):  # in case of spaces in the path
        for i in range(1, len(line)):
            audio_path += " " + line[i]
            if audio_path.endswith("'"):
                break
        audio_path = audio_path[1:-1]
    return audio_path


def parse_wav_scp_file(file):
    wavs = dict()
    with open(file) as f:
        for line in f.readlines():
            line = line.strip().split()
            audio_id = line[0]
            wavs[audio_id] = get_audio_from_wav_scp_line(line)
    return wavs


def parse_text_file(file):
    texts = dict()
    with open(file, encoding="utf-8") as f:
        text_lines = f.readlines()
        for line in text_lines:
            line = line.strip().split()
            texts[line[0]] = " ".join(line[1:])
    return texts


def parse_utt2spk_file(file):
    spks = dict()
    with open(file) as f:
        for line in f.readlines():
            line = line.strip().split()
            spks[line[0]] = line[1]
    return spks


def parse_utt2dur_file(file):
    durs = dict()
    with open(file) as f:
        for line in f.readlines():
            line = line.strip().split()
            durs[line[0]] = round(float(line[1]), 3)
    return durs
