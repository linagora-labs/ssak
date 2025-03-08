#!/usr/bin/env python3

import os

import regex as re

from ssak.utils.kaldi import check_kaldi_dir


def kaldi_split(
    input_folders,
    output_folder_base,
    splits=None,
    max_duration=False,
    throw_if_output_exists=True,
    split_names=None,
):
    """
    Split a kaldi folder into multiple folders, with a given ratio, explicit split, or maximum duration per split

    Parameters
    ----------
    input_folders : list of str
        List of input folders
    output_folder_base : str
        Output folder base
    splits : list of float or list of list of str
        List of ratios for each split. If a list of list of str, it is a list of explicit regex for ids
    max_duration : float
        Maximum duration per split (either provide max_duration or splits, not both)
    throw_if_output_exists : bool
        If True, raise an error if the output folder already exists
    """

    explicit_splits = None
    if max_duration:
        assert not splits, "Cannot provide both max_duration and splits"
    else:
        assert splits, "Have to provide either max_duration or splits"
        if min([isinstance(r, float) for r in splits]):
            # List of floats
            assert sum(splits) <= 1.0, "splits must sum to at most 1"
            assert min(splits) > 0.0, "splits must be all positive"
        elif min([isinstance(ids, list) for ids in splits]) and min([min([isinstance(id, str) for id in ids]) for ids in splits]):
            # List of explicit regex for ids
            explicit_splits = splits
        else:
            raise RuntimeError("splits must be a list of floats")

    # Make checks
    has_segments = None
    has_genders = None
    for input_folder in input_folders:
        for file in ["text", "wav.scp", "utt2dur", "spk2utt", "utt2spk"]:
            assert os.path.isfile(input_folder + "/" + file), f"Missing file: {input_folder}/{file}"

        if os.path.isfile(input_folder + "/segments"):
            _has_segments = True
        else:
            _has_segments = False
            print(f"WARNING: {input_folder} has no segments file")
        if has_segments is None:
            has_segments = _has_segments
        else:
            assert has_segments == _has_segments, f"Some folders have segments files and not others: {input_folder}"

        if os.path.isfile(input_folder + "/spk2gender"):
            _has_genders = True
        else:
            _has_genders = False
            print(f"WARNING: {input_folder} has no spk2gender file")
        if has_genders is None:
            has_genders = _has_genders
        else:
            assert has_genders == _has_genders, f"Some folders have spk2gender files and not others: {input_folder}"

        if throw_if_output_exists and os.path.isdir(output_folder_base):
            raise RuntimeError(f"Output folder already exists. Please remove it first if you want to regenerate it:\n#\trm -R {output_folder_base}")

    utt2wav = {}  # Which audio file is each utt
    wav2dur = {}  # How long is each wav

    for input_folder in input_folders:
        if has_segments:
            with open(input_folder + "/segments") as f:
                for line in f:
                    id, wav_id, _ = line.strip().split(" ", 2)
                    utt2wav[id] = wav_id

        with open(input_folder + "/utt2dur") as f:
            for line in f:
                id, duration = line.strip().split(" ")
                duration = float(duration)
                wav = utt2wav.get(id, id)
                wav2dur[wav] = wav2dur.get(wav, 0.0) + float(duration)

    # Which split is each wav
    wav2split = {}

    if max_duration:
        # Split by duration
        isplit = 0
        accu_duration = 0.0
        for i, (wav, duration) in enumerate(wav2dur.items()):
            accu_duration += duration
            if accu_duration > max_duration:
                if i > 0:
                    isplit += 1
                accu_duration = duration
            wav2split[wav] = isplit

    elif explicit_splits:
        # Split with explicit subsets
        for isplit, ids in enumerate(explicit_splits):
            for id in ids:
                for _id in wav2dur.keys():
                    if re.match(r".*" + id + r"(\.[a-zA-Z]+)?$", _id):
                        assert _id not in wav2split, f"ERROR: {id} is already in a split (caught twice by the regex)"
                        wav2split[_id] = isplit
                        wav2dur.pop(_id)
                        break
                else:
                    raise RuntimeError(f"ERROR: No match for {id}. Ignoring.")

    else:
        # Split by ratio
        total_duration = sum(wav2dur.values())
        isplit = 0
        split_max_duration = splits[0] * total_duration
        accu_duration = 0
        for i, (wav, duration) in enumerate(wav2dur.items()):
            accu_duration += duration
            if accu_duration >= split_max_duration + 1e-3:
                total_accu_duration = sum(wav2dur[w] for w in wav2split.keys())
                if i > 0:
                    isplit += 1
                    if isplit >= len(splits):
                        print(f"WARNING: Ignoring next wav because it exceeded the given ratio, by {split_max_duration-accu_duration:.3f} seconds (taking {total_accu_duration/3600:.3f} hours / {total_duration/3600:.3f}).")
                        break
                accu_duration = duration
                remaining_ratio = sum(splits[isplit:])
                remaining_duration = total_duration - total_accu_duration
                split_max_duration = (splits[isplit] / remaining_ratio) * remaining_duration
            wav2split[wav] = isplit

    isplit = max(wav2split.values())

    splits = range(isplit + 1)

    # Create the splits
    def get_split_folder(isplit):
        if split_names:
            return output_folder_base + f"/{split_names[isplit]}"
        else:
            return output_folder_base + f"/split{isplit}"

    def create_split(isplit):
        output_folder = get_split_folder(isplit)

        os.makedirs(output_folder, exist_ok=True)
        return {
            "text": open(output_folder + "/text", "a"),
            "wav.scp": open(output_folder + "/wav.scp", "a"),
            "utt2dur": open(output_folder + "/utt2dur", "a"),
            "utt2spk": open(output_folder + "/utt2spk", "a"),
            "segments": open(output_folder + "/segments", "a") if has_segments else None,
            "spk2gender": open(output_folder + "/spk2gender", "a") if has_genders else None,
        }

    split2info = [create_split(i) for i in splits]

    # Helper to write the information in the right folder
    def get_split(id):
        wav = utt2wav.get(id, id)
        return wav2split.get(wav, None)

    def write_in_split(id, line, what):
        isplit = get_split(id)
        if isplit is None:
            return
        split = split2info[isplit]
        split[what].write(line)

    try:
        for input_folder in input_folders:
            with open(input_folder + "/text") as f:
                for i, line in enumerate(f):
                    id = _get_first_field(line)
                    write_in_split(id, line, "text")

            if has_segments:
                with open(input_folder + "/segments") as f:
                    for line in f:
                        id = _get_first_field(line)
                        write_in_split(id, line, "segments")

            with open(input_folder + "/wav.scp") as f:
                for line in f:
                    id = _get_first_field(line)
                    write_in_split(id, line, "wav.scp")

            with open(input_folder + "/utt2dur") as f:
                for line in f:
                    id = _get_first_field(line)
                    write_in_split(id, line, "utt2dur")

            spk2splits = {}
            with open(input_folder + "/utt2spk") as f:
                for line in f:
                    id = _get_first_field(line)
                    spk = line.strip().split(" ")[1]
                    isplit = get_split(id)
                    if isplit is not None:
                        if isplit not in spk2splits.get(spk, []):
                            spk2splits[spk] = spk2splits.get(spk, []) + [isplit]
                        write_in_split(id, line, "utt2spk")

            if os.path.isfile(input_folder + "/spk2gender"):
                with open(input_folder + "/spk2gender") as f:
                    for line in f:
                        spk = _get_first_field(line)
                        for isplit in spk2splits.get(spk, []):
                            split2info[isplit]["spk2gender"].write(line)

    finally:
        # Close files
        for split in split2info:
            for f in split.values():
                if f is None:
                    continue
                f.close()

    for isplit in splits:
        check_kaldi_dir(get_split_folder(isplit))


def _get_first_field(line):
    f = line.split(" ", 1)
    assert len(f), "Got an empty line"
    return f[0]


def file_to_ids(filename):
    if os.path.splitext(filename)[1] in [".txt", ".lst"]:
        with open(filename) as f:
            return [line.strip() for line in f]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a kaldi folder with only a subset of utterances from a kaldi folder")
    parser.add_argument("input_folder", type=str, help="Input folder(s) with kaldi files", nargs="+")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument(
        "--splits",
        type=str,
        help="Wanted ratio. Example: 0.6 0.2 0.2. Or filenames with explicit split IDs.",
        nargs="*",
    )
    parser.add_argument("--max_duration", default=None, type=float, help="Maximum duration for a split (in seconds)")
    parser.add_argument("--find_index_in", default=None, type=str, help="A file to find ids corresponding to indices")
    args = parser.parse_args()

    # Conform to the expected format
    split_names = None
    if args.splits:
        try:
            splits = [float(s) for s in args.splits]
        except:
            splits = []
            split_names = []
            for s in args.splits:
                if os.path.isfile(s):
                    splits.append(file_to_ids(s))
                    split_names.append(os.path.splitext(os.path.basename(s))[0])
                else:
                    raise RuntimeError(f"Cannot convert {s} to either ratio or a filename with list of ids")
            if args.find_index_in:
                all_ids = file_to_ids(args.find_index_in)
                splits = [[all_ids[int(id)] for id in ids] for ids in splits]

    kaldi_split(
        args.input_folder,
        args.output_folder,
        splits=splits,
        max_duration=args.max_duration,
        split_names=split_names,
    )
