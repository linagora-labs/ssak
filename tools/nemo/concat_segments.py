import argparse
import logging
import shutil
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_audio_files(audios_paths, output_dir, dataset_name="YODAS", keep_structure=True):
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not isinstance(audios_paths, str):
        first = Path(audios_paths[0])
        audio_id = "-".join(first.stem.split("-")[:-3])
        start = first.stem.split("-")[-2]
        end = Path(audios_paths[-1]).stem.split("-")[-1]
        file_name = f"{audio_id}-{first.stem.split('-')[-3]}-{start}-{end}.wav"
        if keep_structure:
            new_path = output_dir / Path(str(first.parent).split(f"{dataset_name}/")[1]) / file_name
            new_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            new_path = output_dir / file_name
        if not new_path.exists():
            merged_audio = AudioSegment.empty()
            for audio_path in audios_paths:
                audio = AudioSegment.from_wav(audio_path)
                merged_audio += audio
            merged_audio.export(new_path, format="wav")
    else:
        file_path = Path(audios_paths)
        if keep_structure:
            new_path = output_dir / Path(str(file_path.parent).split(f"{dataset_name}/")[1]) / file_path.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            new_path = output_dir / file_path.name
        shutil.copy(audios_paths, new_path)
    return new_path


def merge_segments_and_audios(prev, row, max_duration, acceptance):
    # for yodas
    if isinstance(prev.audio_filepath, str):
        prev_audio_path = Path(prev.audio_filepath)
    else:
        prev_audio_path = Path(prev.audio_filepath[-1])
    row_audio_path = Path(row.audio_filepath)

    same_file = prev_audio_path.stem.split("-")[0] == row_audio_path.stem.split("-")[0]

    prev_end = prev_audio_path.stem.split("-")[-1]
    row_start = row_audio_path.stem.split("-")[-2]
    follow = float(prev_end) / 1000 + acceptance >= float(row_start) / 1000
    merged_duration = prev.duration + row.duration

    if same_file and follow and merged_duration <= max_duration:
        sep = " "
        first_char = row.answer.lstrip()[0] if row.answer else ""
        last_char = prev.answer.rstrip()[-1] if prev.answer else ""
        if last_char in string.punctuation:
            sep = " "
        else:
            sep = ". " if first_char.isupper() else ", "

        prev.duration = round(merged_duration, 3)
        prev.answer = (prev.answer + sep + row.answer).strip()
        if isinstance(prev.audio_filepath, str):
            prev.audio_filepath = [prev.audio_filepath, row.audio_filepath]
        else:
            prev.audio_filepath.append(row.audio_filepath)
        return True, prev
    return False, None


def merge_segments(prev, row, max_duration, acceptance, acceptance_punc):
    # for youtube
    same_file = prev.audio_filepath == row.audio_filepath
    gap = row.offset - (prev.offset + prev.duration)
    merged_duration = (row.offset + row.duration) - prev.offset

    if same_file and gap <= acceptance and merged_duration <= max_duration:
        sep = " "
        if gap > acceptance_punc:  # only consider punctuation when there is a noticeable pause
            first_char = row.answer.lstrip()[0] if row.answer else ""
            last_char = prev.answer.rstrip()[-1] if prev.answer else ""
            if last_char in string.punctuation:
                sep = " "
            else:
                sep = ". " if first_char.isupper() else ", "

        # Extend previous segment to cover the new one
        prev.duration = round(merged_duration, 3)
        prev.answer = (prev.answer + sep + row.answer).strip()
        return True, prev
    return False, None


def concat_segments(input_file, output_file=None, max_duration=30, acceptance=1.0, acceptance_punc=0.2, merge_audios=False, merged_audio_folder="audio", keep_audio_structure=True, num_threads=4):
    if not isinstance(input_file, Path):
        input_file = Path(input_file)
    if not output_file:
        output_folder = input_file.parent.parent / (input_file.parent.name + "_merged")
        output_file = output_folder / input_file.name
    nemo_data = NemoDataset()
    nemo_dataset_type = nemo_data.load(input_file)
    input_data = nemo_data
    if not merged_audio_folder.startswith("/"):
        merged_audio_folder = output_folder / Path(merged_audio_folder)
        merged_audio_folder.mkdir(parents=True, exist_ok=True)
    else:
        merged_audio_folder = Path(merged_audio_folder)
        merged_audio_folder.mkdir(parents=True, exist_ok=True)
    new_data = NemoDataset()

    logger.info("Sorting dataset")
    rows = sorted(input_data, key=lambda r: (r.audio_filepath, r.offset))
    logger.info("Finished sorting dataset")
    prev = None
    number_of_merge = 0
    for i, row in enumerate(tqdm(rows, desc="Merging segments")):
        if prev is None:
            prev = row
            continue
        if not merge_audios:
            merged, previous = merge_segments(prev, row, max_duration, acceptance, acceptance_punc)
        else:
            merged, previous = merge_segments_and_audios(prev, row, max_duration, acceptance)
        if merged:
            number_of_merge += 1
            prev = previous
        else:
            new_data.append(prev)
            prev = row
    new_data.append(prev)
    if merge_audios:
        logger.info(f"Merging audios with {num_threads} threads")

        def process_row(row):
            row.audio_filepath = merge_audio_files(row.audio_filepath, merged_audio_folder, keep_structure=keep_audio_structure)
            row.id = Path(row.audio_filepath).stem
            return row

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_row, row): row for row in new_data}
            merged_data = NemoDataset()
            for future in tqdm(as_completed(futures), total=len(futures), desc="Merging audios"):
                merged_data.append(future.result())
        new_data = merged_data
    logger.info(f"Old number of segments: {len(input_data)}")
    logger.info(f"New number of segments: {len(new_data)}")
    logger.info(f"Made {number_of_merge} merges")
    new_data.save(output_file, type=nemo_dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove incoherent lines by looking at the number of words and segment duration from nemo manifest")
    parser.add_argument("file", help="Input manifest", type=str)
    parser.add_argument("--max_duration", help="", type=float, default=30.0)
    parser.add_argument("--acceptance", help="", type=float, default=1.0)
    parser.add_argument("--merge_audios", help="", default=False, action="store_true")
    parser.add_argument("--merged_audio_folder", help="", default="audio")
    args = parser.parse_args()

    concat_segments(args.file, max_duration=args.max_duration, acceptance=args.acceptance, merge_audios=args.merge_audios, merged_audio_folder=args.merged_audio_folder)
