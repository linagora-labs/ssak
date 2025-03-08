import logging
import os
import re

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ssak.utils.kaldi_dataset import KaldiDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_seconds(time_str):
    # if last character is a digit, add an 's' to the end
    if time_str[-1].isdigit():
        if "h" and "m" in time_str:
            time_str += "s"
        elif "h" in time_str:
            time_str += "m"
        else:
            raise ValueError("Invalid time format, should be in the format 'XhYmZs'")

    # Define a regex pattern to extract hours, minutes, and seconds
    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"

    # Search for matches using the regex
    match = re.match(pattern, time_str)

    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    else:
        raise ValueError("Invalid time format")


def print_result_infos(train_dataset, test_dataset, dev_dataset=None):
    print(" TRAIN SPLIT ".center(40, "-"))
    print(f"ROWS: {len(train_dataset)}")
    print(f"SPEAKERS: {len(train_dataset.get_speakers())}")
    print(f"AUDIOS: {len(train_dataset.get_audio_ids())}")
    hours = train_dataset.get_duration() / 3600
    minutes = (hours - int(hours)) * 60
    seconds = (minutes - int(minutes)) * 60
    print(f"DURATION: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    if dev_dataset is not None:
        print(" DEV SPLIT ".center(40, "-"))
        print(f"ROWS: {len(dev_dataset)}")
        print(f"SPEAKERS: {len(dev_dataset.get_speakers())}")
        print(f"AUDIOS: {len(dev_dataset.get_audio_ids())}")
        hours = dev_dataset.get_duration() / 3600
        minutes = (hours - int(hours)) * 60
        seconds = (minutes - int(minutes)) * 60
        print(f"DURATION: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    print(" TEST SPLIT ".center(40, "-"))
    print(f"ROWS: {len(test_dataset)}")
    print(f"SPEAKERS: {len(test_dataset.get_speakers())}")
    print(f"AUDIOS: {len(test_dataset.get_audio_ids())}")
    hours = test_dataset.get_duration() / 3600
    minutes = (hours - int(hours)) * 60
    seconds = (minutes - int(minutes)) * 60
    print(f"DURATION: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("-" * 40)


def split(dataset, test_size, random_seed=42, show_tqdm=False):
    spks_ids = set()
    spks = list()  # need a list to have reproducible results
    for i in dataset.get_speakers(unique=False):
        if i not in spks_ids:
            spks.append(i)
            spks_ids.add(i)

    train_spks, test_spks = train_test_split(spks, test_size=test_size, random_state=random_seed)
    del spks_ids  # free some ram in case of big dataset
    del spks  # free some ram in case of big dataset

    logger.debug(f"Base splits done: {len(train_spks)} speakers in train and {len(test_spks)} speakers in test")

    train_dataset = dataset.filter_by_speakers(train_spks)
    test_dataset = dataset.filter_by_speakers(test_spks)

    logger.debug(f"Generated new datasets (based on splits): {len(train_dataset)} rows in train and {len(test_dataset)} rows in test")
    logger.debug("Starting algorithm to remove common speakers and files between train and test")

    pbar = None
    if show_tqdm:
        pbar = tqdm()
    changes = True
    while changes:
        changes = False
        train_audio_ids = train_dataset.get_audio_ids()
        test_audio_ids = test_dataset.get_audio_ids()
        if pbar:
            pbar.update(1)
        if len(train_audio_ids.intersection(test_audio_ids)) > 0:
            changes = True
            common_audio_ids = set(train_audio_ids).intersection(test_audio_ids)
            common_train_dataset = train_dataset.filter_by_audio_ids(common_audio_ids)
            common_train_spks = common_train_dataset.get_speakers()
            test_dataset.extend(dataset.filter_by_speakers(common_train_spks))
            train_dataset = train_dataset.filter_by_speakers(set(train_spks).difference(common_train_spks))
            if train_dataset is None or len(train_dataset) == 0:
                logger.warning("Fail to split the dataset.")
                raise Exception("Fail to split the dataset.")
    if pbar:
        pbar.close()

    return train_dataset, test_dataset


def split_by_speakers(kaldi_dir, output, test_size, random_seed=42):
    dataset = KaldiDataset("dataset")
    dataset.load(kaldi_dir)
    logger.debug(f"Found {len(dataset.get_speakers())} speakers")
    test_size = int(test_size) if test_size > 1.0 else test_size
    logger.info(f"Splitting dataset with test_size of: {test_size}")
    train_dataset, test_dataset = split(dataset, test_size, random_seed, show_tqdm=True)
    print_result_infos(train_dataset, test_dataset)
    train_dataset.save(os.path.join(output, "train"))
    test_dataset.save(os.path.join(output, "test"))


def grid_search(dataset, target_test_duration, speakers_to_test, seed_to_test):
    closest_duration = None
    closest_duration_speakers = None
    closest_duration_seed = 1

    for number_of_speakers in tqdm(speakers_to_test):
        _, test_dataset = split(dataset, number_of_speakers, closest_duration_seed)
        test_duration = test_dataset.get_duration()
        if closest_duration is None or abs(target_test_duration - test_duration) < abs(target_test_duration - closest_duration):
            closest_duration = test_duration
            closest_duration_speakers = number_of_speakers
        if abs(target_test_duration - test_duration) > abs(target_test_duration - closest_duration) and test_duration > target_test_duration:
            break
    logger.info(f"Best split found with {closest_duration_speakers} speakers and {closest_duration} seconds (target: {target_test_duration} seconds)")

    for seed in tqdm(seed_to_test):
        _, test_dataset = split(dataset, closest_duration_speakers, seed)
        test_duration = test_dataset.get_duration()
        if abs(target_test_duration - test_duration) < abs(target_test_duration - closest_duration):
            closest_duration = test_duration
            closest_duration_seed = seed
    logger.info(f"Best split found with seed {closest_duration_seed} ({closest_duration} seconds, target: {target_test_duration} seconds)")

    train_dataset, test_dataset = split(dataset, closest_duration_speakers, closest_duration_seed)
    return train_dataset, test_dataset


def split_by_duration(
    kaldi_dir,
    output,
    target_test_duration,
    target_dev_duration,
    speakers_to_test=range(1, 20),
    seed_to_test=range(2, 12),
):
    dataset = KaldiDataset("dataset")
    dataset.load(kaldi_dir)

    logger.info(f"Starting grid search for test set with target duration of {target_test_duration} seconds")
    train_dataset, test_dataset = grid_search(dataset, target_test_duration, speakers_to_test, seed_to_test)
    dev_dataset = None
    if target_dev_duration is not None:
        logger.info(f"Starting grid search for dev set with target duration of {target_dev_duration} seconds")
        train_dataset, dev_dataset = grid_search(train_dataset, target_dev_duration, speakers_to_test, seed_to_test)
        dev_dataset.save(os.path.join(output, "dev"))
    print_result_infos(train_dataset, test_dataset, dev_dataset)
    train_dataset.save(os.path.join(output, "train"))
    test_dataset.save(os.path.join(output, "test"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a kaldi folder with only a subset of utterances from a kaldi folder")
    parser.add_argument("input_folder", type=str, help="Input kaldi folder")
    parser.add_argument("output_folder", type=str, help="Output folder where the split will be saved")
    parser.add_argument("--test_size", type=float, default=None, help="Wanted number of speakers in test. Example: 5")
    parser.add_argument("--test_duration", type=str, default=None, help="Duration of the test set. Example: 5h15m")
    parser.add_argument("--dev_duration", type=str, default=None, help="Duration of the dev set. Example: 5h15m")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (used if test_size is specified)")
    args = parser.parse_args()
    if args.test_size is None and args.test_duration is None:
        raise Exception("You must specify either test_size or test_duration")
    elif args.test_size is not None and args.test_duration is not None:
        raise Exception("You must specify either test_size or test_duration, not both")
    if args.test_duration is not None:
        test_duration = convert_to_seconds(args.test_duration)
        dev_duration = convert_to_seconds(args.dev_duration) if args.dev_duration is not None else None
        split_by_duration(args.input_folder, args.output_folder, test_duration, dev_duration)
    else:
        split(args.input_folder, args.output_folder, args.test_size, args.random_seed)
