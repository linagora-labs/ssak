import argparse
import json
import logging
import os

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(input, output, split_on_speaker=True, seed=42, train=0.8, validation=0.1, test=0.1):
    if os.path.exists(os.path.join(output, "train.jsonl")):
        raise FileExistsError(f'Output folder "{output}" already exists')
    if round(train + validation + test, 2) > 1.0:
        raise ValueError(f"Train, validation and test ratios must be between 0 and 1 (current {train}+{validation}+{test})")
    elif not 0 < round(train, 2) <= 1.0:
        raise ValueError("Train ratio must be 0<X<=1")
    elif not 0 < round(validation, 2) < 1.0:
        raise ValueError("Validation ratio must be 0<X<1")
    elif not 0 <= round(test, 2) < 1.0:
        raise ValueError("Test ratio must be 0<=X<1")
    input_file = input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Non-existing file {input_file}")
    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()
        data = [json.loads(l) for l in lines]
    splits = {"train": [], "validation": [], "test": []}
    remaining = list()
    remaining_with_spks = dict()
    for row in data:
        split = row["split"]
        if split == "dev":
            split = "validation"
        if split != "all":
            splits[split].append(row)
        else:
            if "speaker" in row:
                if row["speaker"] not in remaining_with_spks:
                    remaining_with_spks[row["speaker"]] = list()
                remaining_with_spks[row["speaker"]].append(row)
            else:
                remaining.append(row)
    if len(remaining_with_spks) > 0:
        logger.info(f"Found {len(remaining_with_spks)} speakers ({sum(len(data_list) for data_list in remaining_with_spks.values())} rows) with no splits defined, splitting on speakers using ratios arguments")
        speakers = list(remaining_with_spks.keys())

        train_speakers, remaining_speakers = train_test_split(speakers, test_size=1 - train, random_state=seed)
        train = [data for speaker in train_speakers for data in remaining_with_spks[speaker]]

        if round(test, 2) == 0:
            dev = [data for speaker in remaining_speakers for data in remaining_with_spks[speaker]]
            test = []
        else:
            try:
                dev, test = train_test_split(remaining_speakers, test_size=test / (test + validation), random_state=seed)
            except ValueError as e:
                logger.warning(f"Could not split speakers on test and dev sets because there are {len(remaining_speakers)} speakers, trying to split them")
                if len(remaining_speakers) == 1:
                    dev = []
                    test = remaining_speakers
                elif len(remaining_speakers) < 10:
                    if test > validation:
                        dev, test = train_test_split(remaining_speakers, test_size=0.6, random_state=seed)
                    else:
                        dev, test = train_test_split(remaining_speakers, test_size=0.4, random_state=seed)
                else:
                    raise e
            dev = [data for speaker in dev for data in remaining_with_spks[speaker]]
            test = [data for speaker in test for data in remaining_with_spks[speaker]]
        splits["train"].extend(train)
        splits["validation"].extend(dev)
        splits["test"].extend(test)
    if len(remaining) > 0:
        logger.info(f"Found {len(remaining)} rows with no splits and speaker defined, splitting them using ratios arguments")
        train, remaining = train_test_split(remaining, test_size=1 - train, random_state=seed)
        if round(test, 2) == 0:
            dev = remaining
            test = []
        else:
            dev, test = train_test_split(remaining, test_size=test / (test + validation), random_state=seed)
        splits["train"].extend(train)
        splits["validation"].extend(dev)
        splits["test"].extend(test)
    os.makedirs(output, exist_ok=True)
    for i in splits:
        with open(os.path.join(output, i + ".jsonl"), "w", encoding="utf-8") as f:
            for row in splits[i]:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Saved {len(splits[i])} lines to {os.path.join(output, i+'.jsonl')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a manifest file into train/dev/test splits")

    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="Output folder to save splits", type=str)
    parser.add_argument(
        "--split_on_speaker",
        help="NOT IMPLEMENTED : Split on speaker for the data where the info is available",
        action="store_true",
    )
    parser.add_argument("--seed", help="Seed for random split", type=int, default=42)
    parser.add_argument(
        "--train",
        help="Train split ratio for the remaning rows (the ones where set is not defined)",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--validation",
        help="Dev split ratio for the remaning rows (the ones where set is not defined)",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test",
        help="Test split ratio for the remaning rows (the ones where set is not defined)",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    split_dataset(args.input, args.output, args.split_on_speaker, args.seed, args.train, args.validation, args.test)
