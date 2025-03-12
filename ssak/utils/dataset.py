import logging
import math
import os
import pathlib
import random
from operator import itemgetter

import datasets
import numpy as np
import pandas as pd
import transformers

from ssak.utils.kaldi import parse_kaldi_wavscp
from ssak.utils.misc import commonprefix, get_cache_dir
from ssak.utils.monitoring import logger

from .audio import array_to_bytes, load_audio
from .misc import hashmd5
from .text_basic import remove_special_words

try:
    # Avoid VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes)
    # is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except AttributeError:
    pass
import torch
from envsubst import envsubst


def kaldi_folder_to_dataset(
    kaldi_path,
    online=False,
    n_shards=1,
    shuffle=False,
    max_data=None,
    min_duration=None,
    max_duration=None,
    max_text_length=None,
    choose_data_with_max_duration=False,
    sort_by_len=0,
    weights=1,
    split=None,
    return_format="dataset",
    include_duration=False,
    # TODO include_speaker = False,
    verbose=True,
    logstream=None,
    do_cache=True,
    seed=69,
):
    """
    Take a kaldi folder and returns a tuple (metadata dict, dataset)

    The class of dataset depends on option return_format

    Parameters
    ----------
    kaldi_path : str
        Path to kaldi folder, or list of paths to kaldi folders
    online : bool
        If True, audio files will be loaded and processed on-the-fly (out of core).
        If False, audio files will be loaded and processed at first, and *cached* (typically in ~/.cache/linacache).
    n_shards : int
        Number of shards to use for caching. If > 1, the dataset will be split in n_shards parts, and each part will be cached separately.
        This is useful when using dataloader_num_workers > 1 in transformers
    shuffle : bool
        If True, the dataset will be shuffled.
    max_data : int
        Maximum number of data to use. If None, use all files in the kaldi folder
    min_duration : int
        Minimum length in seconds of audio. If None, no limit.
    max_duration : int
        Maximum length in seconds of audio. If None, no limit.
    max_text_length : int or tuple (function, int)
        Maximum length of text, in number of characters. Or a tuple (tokenizer function, maximum number of tokens).
    weights : float
        Weight of this dataset. Has an interest if several datasets are specified.
        Data will be duplicated (upsampled when weights > 1).
    choose_data_with_max_duration : bool
        If True and max_data is not None, the longest utterances will be chosen (good for testing if everything fits in memory).
    return_format : "dataset" / "csv" / "pandas"
        Output format:
        - "dataset": a HuggingFace dataset
        - "csv": a csv file
        - "pandas": a pandas pandas
    include_duration : bool
        If True, include the duration of the audio in the dataset.
        Warning: does not have any effect if return_format="dataset".
    split : str
        Split to use ("train", "dev", "test"). If None, unspecified.
    verbose : bool
        Whether to print some steps
    logstream : file
        Stream to print some logs to
    do_cache : bool
        Internal. Do not use
    seed : int
        Seed for random shuffling

    Returns
    -------
    dataset : datasets.Dataset
    """

    opts = dict((k, v) for k, v in locals().items() if "kaldi_path" not in k)

    assert return_format in [
        "dataset",
        "csv",
        "pandas",
    ], f"return_format {return_format} must be 'dataset', 'csv' or 'pandas'"

    if max_text_length is None:
        text_tokenizer = None
    elif isinstance(max_text_length, int):
        text_tokenizer = lambda x: x
    else:
        assert isinstance(max_text_length, tuple) and len(max_text_length) == 2 and isinstance(max_text_length[1], int), "max_text_length must be an integer or a tuple (function, integer)"
        text_tokenizer, max_text_length = max_text_length
    assert max_text_length is None or max_text_length > 0, f"max_text_length must be > 0 (got {max_text_length})"

    if return_format == "pandas":
        opts["return_format"] = "csv"
        meta, csv_file = kaldi_folder_to_dataset(kaldi_path, **opts)
        ds = pd.read_csv(csv_file)
        os.remove(csv_file)
        return meta, ds

    use_csv = return_format in ["csv"]

    # Logging business (start)
    ds_progress_bar = datasets.utils.is_progress_bar_enabled()
    loggers = [datasets.builder.logger, datasets.info.logger, datasets.arrow_dataset.logger]
    ds_log_level = [l.getEffectiveLevel() for l in loggers]
    if verbose:
        datasets.utils.enable_progress_bar()
        for l in loggers:
            l.setLevel(logging.WARNING)
    else:
        datasets.utils.disable_progress_bar()
        for l in loggers:
            l.setLevel(logging.ERROR)

    empty_dataset = (
        {
            "samples": 0,
            "h duration": 0,
            "samples (with duplicates)": 0,
            "h duration (with duplicates)": 0,
            "weight": weights,
        },
        datasets.Dataset.from_dict({}),
    )

    if not isinstance(kaldi_path, str):
        if not isinstance(weights, list):
            weights = [weights] * len(kaldi_path)

        opts["online"] = False
        opts["shuffle"] = False
        opts["do_cache"] = False
        ds = [kaldi_folder_to_dataset(p, **(opts | {"weights": w})) for p, w in zip(kaldi_path, weights)]

        dataset = datasets.concatenate_datasets([d[1] for d in ds])
        if do_cache:
            dataset = make_cachable(
                dataset,
                online=online,
                shuffle=shuffle,
                seed=seed,
                n_shards=n_shards,
                verbose=verbose,
                logstream=logstream,
                return_csv=use_csv,
            )

        meta = {
            "samples": sum([d[0]["samples"] for d in ds]),
            "h duration": sum([d[0]["h duration"] for d in ds]),
            "samples (with duplicates)": sum([d[0]["samples (with duplicates)"] for d in ds]),
            "h duration (with duplicates)": sum([d[0]["h duration (with duplicates)"] for d in ds]),
        }

        return meta, dataset

    if os.path.isfile(kaldi_path):
        # Parse a file listing folders

        new_kaldi_path = []
        new_weights = []
        assert isinstance(weights, (int, float))

        with open(kaldi_path) as f:
            for line in f:
                words = line.strip().split()
                for w in words:
                    # Look for environment variables in the path
                    if "$" in w:
                        w = envsubst(w)

                    if os.path.isdir(w):
                        new_kaldi_path.append(w)
                        new_weights.append(weights)
                    else:
                        try:
                            w = float(w)
                        except ValueError:
                            raise RuntimeError("Could not find folder %s" % w)
                        assert len(new_weights) > 0, "File cannot start with a weight (first a folder name, then a weight)"
                        new_weights[-1] *= w

        opts.pop("weights", None)
        return kaldi_folder_to_dataset(new_kaldi_path, weights=new_weights, **opts)

    elif not os.path.isdir(kaldi_path):
        if "," in kaldi_path:
            # Parse a comma-separated list of folders
            return kaldi_folder_to_dataset(kaldi_path.split(","), **opts)

        raise RuntimeError("Could not find folder %s" % kaldi_path)

    for fname in "text", "wav.scp":
        if not os.path.isfile(kaldi_path + "/" + fname):
            raise RuntimeError("Could not find file %s in folder %s" % (fname, kaldi_path))

    has_segment = os.path.isfile(kaldi_path + "/segments")
    if verbose:
        print("Parsing", kaldi_path, "(no segments)" if not has_segment else "")

    with open(kaldi_path + "/text", encoding="utf8") as f:

        def split_line(line):
            res = line.strip().split(" ", 1)
            if len(res) == 1:
                res = [res[0], ""]
            return res

        try:
            uttids, annots = zip(*map(split_line, f))
            uttids = list(uttids)
            annots = list(annots)
        except Exception as err:
            raise RuntimeError("Error while parsing %s/text" % (kaldi_path)) from err

    if max_text_length:
        print("Filtering too long texts")
        num_annots = len(annots)
        is_short_enough = lambda text: len(text_tokenizer(text)) <= max_text_length
        # if verbose:
        #     def is_short_enough(text):
        #         text = remove_special_words(text)
        #         l = len(text_tokenizer(text))
        #         res = l <= max_text_length
        #         if verbose and not res:
        #             print(f"Discarding '{text}' of length {l} > {max_text_length}")
        #         return res
        uttids, annots = zip(*[(uttid, annot) for uttid, annot in zip(uttids, annots) if is_short_enough(annot)])
        uttids = list(uttids)
        annots = list(annots)
        if len(annots) == 0:
            print(f"WARNING: No data selected! (with max_text_length: {max_text_length})")
            return empty_dataset
        if len(annots) != num_annots:
            print(f"WARNING: filtered out {num_annots - len(annots)}/{num_annots} utterances with text longer than {max_text_length}.")

    if not choose_data_with_max_duration and max_data and max_data < len(uttids):
        random.seed(seed)
        random.shuffle(uttids)
        random.seed(seed)
        random.shuffle(annots)
        uttids = uttids[:max_data]
        annots = annots[:max_data]

    wav = parse_kaldi_wavscp(kaldi_path + "/wav.scp")

    total_duration = None
    if has_segment:
        segments = {}
        with open(kaldi_path + "/segments") as f:
            for line in f:
                fields = line.strip().split()
                uttid = fields[0]
                wavid = fields[1]
                start = float(fields[2])
                end = float(fields[3])
                duration = end - start
                assert duration > 0, f"Error in {kaldi_path}/segments:\nDuration of utterance {uttid} is negative: {duration}"
                discarded = (max_duration and duration > max_duration) or (min_duration and duration < min_duration)
                if discarded:
                    try:
                        i = uttids.index(uttid)
                    except ValueError:
                        continue
                    uttids.pop(i)
                    annots.pop(i)
                    continue
                segments[uttid] = [wavid, start, end]

        if (choose_data_with_max_duration and max_data and max_data < len(uttids)) or sort_by_len not in [0, None]:
            # We select the longest utterances
            uttids, annots = zip(*sorted(zip(uttids, annots), key=lambda i: (segments[i[0]][2] - segments[i[0]][1], len(i[1]))))
            if max_data and max_data < len(uttids):
                uttids = uttids[-max_data:]
                annots = annots[-max_data:]
            uttids = list(uttids)
            annots = list(annots)
            # Longest utterances first
            if sort_by_len < 0:
                uttids = list(uttids)
                annots = list(annots)
                uttids.reverse()
                annots.reverse()

        if len(uttids) == 0:
            print(f"WARNING: No data selected! (with min-max duration: {min_duration}-{max_duration})")
            return empty_dataset
        wavids, starts, ends = zip(*map(lambda id: segments[id], uttids))
        paths = list(map(lambda id: wav[id], wavids))

        dataset = {
            "ID": uttids,
            "path": paths,
            "start": starts,
            "end": ends,
            "text": annots,
        }
        durations = [end - start for start, end in zip(starts, ends)]

    else:  # No segments (take full audio)
        for fname in ("utt2dur",):
            if not os.path.isfile(kaldi_path + "/" + fname):
                raise RuntimeError("Could not find file %s in folder %s" % (fname, kaldi_path))

        def parse_line(line):
            f = line.strip().split()
            f[1] = float(f[1])
            return f

        with open(kaldi_path + "/utt2dur") as f:
            durations = dict([parse_line(line) for line in f if line.strip()])

        if max_duration or min_duration or (choose_data_with_max_duration and max_data) or sort_by_len:
            uttids, annots = zip(*sorted(zip(uttids, annots), key=lambda i: (durations[i[0]], len(i[1]))))
        durations = itemgetter(*uttids)(durations)
        durations = [durations] if isinstance(durations, float) else list(durations)  # Tuple to list conversion
        if max_duration or min_duration:
            a = 0
            b = 0
            for d in durations:
                if min_duration and d < min_duration:
                    a += 1
                if max_duration and d > max_duration:
                    break
                b += 1
            if b <= a:
                print(f"WARNING: No data selected! (with min-max duration: {min_duration}-{max_duration})")
                return empty_dataset
            uttids = uttids[a:b]
            annots = annots[a:b]
            durations = durations[a:b]
        if choose_data_with_max_duration and max_data and max_data < len(uttids):
            uttids = uttids[-max_data:]
            annots = annots[-max_data:]
            durations = durations[-max_data:]

        # Longest utterances first
        if sort_by_len and sort_by_len < 0:
            uttids = list(uttids)
            annots = list(annots)
            durations = list(durations)
            uttids.reverse()
            annots.reverse()
            durations.reverse()

        paths = list(map(lambda id: wav[id], uttids))
        dataset = {
            "ID": uttids,
            "path": paths,
            "text": annots,
            "start": [0] * len(uttids),
            "end": durations,
        }

    total_duration = sum(durations)
    if include_duration:
        dataset["duration"] = durations
    if verbose:
        print(f"    minmax(duration) = {min(durations)}-{max(durations)}")

    if weights != 1:
        # Duplicate all entries of the dictionary
        # print("Duplicating dataset with weights", weights)
        l = len(dataset["ID"])
        dataset = {k: list(v) * int(weights) + random.Random(45).sample(v, int(len(v) * (weights % 1))) for k, v in dataset.items()}
        if weights > 1:
            # Assign different identifiers, as speechbrain require all ids to be different
            for i in range(1, math.ceil(weights) + 1):
                a = i * l
                b = min((i + 1) * l, len(dataset["ID"]))
                dataset["ID"][a:b] = [id + "_%d" % i for id in dataset["ID"][a:b]]

    nsamples_dup = len(dataset["ID"])
    if include_duration:
        total_duration_dup = sum(dataset["duration"])
    else:
        total_duration_dup = total_duration * weights

    dataset = datasets.Dataset.from_dict(
        dataset,
        split={
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "valid": datasets.Split.VALIDATION,
            "validation": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }.get(split, split),
    )

    if do_cache:
        dataset = make_cachable(
            dataset,
            online=online,
            shuffle=shuffle,
            seed=seed,
            n_shards=n_shards,
            verbose=verbose,
            logstream=logstream,
            return_csv=use_csv,
        )

    meta = {
        "samples": len(uttids),
        "h duration": total_duration / 3600,
        "samples (with duplicates)": nsamples_dup,
        "h duration (with duplicates)": total_duration_dup / 3600,
        "weight": weights,
    }

    if nsamples_dup != len(uttids):
        metastr = ", ".join(f"{v} {k}" for k, v in meta.items())
    else:
        metastr = ", ".join(f"{v} {k}" for k, v in meta.items() if k not in ["samples (with duplicates)", "h duration (with duplicates)", "weight"])
    if verbose:
        print("   ", metastr)
    if logstream:
        print(kaldi_path, ":", metastr, file=logstream)

    # Logging business (end)
    if ds_progress_bar:
        datasets.utils.enable_progress_bar()
    else:
        datasets.utils.disable_progress_bar()
    for l, ll in zip(loggers, ds_log_level):
        l.setLevel(ll)

    return meta, dataset


def make_cachable(dataset, online=False, shuffle=False, seed=69, n_shards=1, return_csv=False, verbose=True, logstream=None):
    assert n_shards >= 1, "n_shards must be >= 1"
    # Make sure that all IDs are unique
    if len(set(dataset["ID"])) < len(dataset["ID"]):
        all_ids = []

        def make_id_unique(d):
            id = d["ID"]
            if id in all_ids:
                j = 2
                while f"{id}_{j}" in all_ids:
                    j += 1
                d["ID"] = f"{id}_{j}"
            all_ids.append(id)
            return d

        dataset = dataset.map(make_id_unique)
    if shuffle:
        if verbose:
            print("Shuffling dataset")
        dataset = dataset.shuffle(seed)
    cache_file_dir = os.path.join(get_cache_dir("linacache"), dataset._fingerprint)
    if not os.path.isdir(cache_file_dir):
        os.makedirs(cache_file_dir)
    if n_shards > 1:
        cache_filenames = [os.path.join(cache_file_dir, f"shard{n_shards}-{i+1}.csv") for i in range(n_shards)]
    else:
        cache_filenames = [os.path.join(cache_file_dir, "full.csv")]
    assert len(cache_filenames)
    cache_file_regex = cache_filenames[0][:-4] + "*.csv"
    if verbose:
        print("Caching CSV dataset in ", cache_file_regex)
    if logstream:
        logstream.write("- CSV cached in %s\n" % cache_file_regex)
    if len(cache_filenames) == 1:
        dataset.to_csv(cache_filenames[0])
    else:
        for i, filename in enumerate(cache_filenames):
            if not os.path.isfile(filename):
                dataset.shard(n_shards, i).to_csv(filename)
    if return_csv:
        if n_shards > 1:
            raise NotImplementedError("return_csv not implemented with n_shards > 1")
        return cache_filenames[0]
    # Use this to pass "origin_metadata = None" so that the caching mechanism will be OK
    data_files = datasets.data_files.DataFilesDict({"train": datasets.data_files.DataFilesList([pathlib.PosixPath(filename) for filename in cache_filenames], origin_metadata=None)})
    res = datasets.load_dataset(
        "csv",
        data_files=data_files,
        streaming=online,
        split=dataset.split,  # TODO WTF This fails if split is not "train"...
        cache_dir=cache_file_dir,
        keep_default_na=False,  # This is important to avoid pandas to convert "nan" to NaN
    )
    if not online and logstream:
        logstream.write(f"- Huggingface cached CSV in {format_cache_files(res.cache_files)}\n")
    if isinstance(res, dict) and len(res) == 1:
        res = list(res.values())[0]
    return res


def format_cache_files(cache_files):
    if isinstance(cache_files, list):
        res = list(set([format_cache_files(f) for f in cache_files]))
        pref = commonprefix(res)
        return pref + " | ".join(sorted([f[len(pref) :] for f in res if f != pref]))
    elif isinstance(cache_files, dict):
        if "filename" in cache_files:
            return cache_files["filename"]
        elif len(cache_files) == 1:
            return format_cache_files(list(cache_files.values())[0])
        else:
            return str(cache_files)
    else:
        return str(cache_files)


def process_dataset(
    processor,
    dataset,
    batch_size=32,
    num_proc=1,
    data_augmenter=None,
    text_augmenter=None,
    text_processor=remove_special_words,
    verbose=True,
    force_cache=True,
    logstream=None,
):
    """
    Process a dataset with a HuggingFace processor.

    Parameters
    ----------
    processor : HuggingFace.Preprocessor
        Processor to use
    dataset : datasets.Dataset
        Dataset to process
    batch_size : int (default: 32)
        Batch size to use
    text_processor: function to normalize the text (input: string / output: string), applied *after* text augmentation if any
    data_augmenter: function to augment audio (input: raw audio signal / output: raw audio signal)
    text_augmenter: function to augment text (input: string / output: string)
    num_proc : int (default: 1)
        Number of processes to use (not: may be disabled in online mode).
        WARNING: using more than 1 process may lead to hang
    verbose : bool
        Whether to print some steps
    force_cache : bool
        Whether to force the use of the cache (except in the case of online, where it will be disabled)
        Tricky things in HuggingFace's datasets cache mechanism...
        Maybe this is not needed anymore
    logstream : file
        Stream to print some logs to
    """

    sample_rate = processor.feature_extractor.sampling_rate
    is_iterable = isinstance(dataset, datasets.IterableDataset)  # or not hasattr(dataset, "_fingerprint")
    force_cache = force_cache and not is_iterable
    if force_cache:
        datasets.enable_caching()
        cache_file_dir = os.path.join(get_cache_dir("linacache"), dataset._fingerprint)
        if not os.path.isdir(cache_file_dir):
            os.makedirs(cache_file_dir)
    if is_iterable:
        for e in dataset:
            column_names = list(e.keys())
            break
        map_kwargs = {}
    else:
        column_names = dataset.column_names
        map_kwargs = {
            "num_proc": num_proc,
        }
        if force_cache:
            map_kwargs.update(
                {
                    "cache_file_name": os.path.join(cache_file_dir, "loaded.arrow"),
                    "load_from_cache_file": True,
                }
            )
    has_segment = "start" in column_names
    if verbose and hasattr(dataset, "_fingerprint"):
        print("Loading audios", dataset._fingerprint)

    if text_processor is None:
        text_processor = lambda x: x
    if data_augmenter or text_augmenter:
        if not data_augmenter:
            data_augmenter = lambda x: x
        if not text_augmenter:
            text_augmenter = lambda x: x
        processed = dataset.map(
            lambda row: {"input_values": np.array([1.0], dtype=np.float32), "labels": "e"}
            if (hasattr(transformers.trainer, "SKIPPING") and transformers.trainer.SKIPPING)
            else {
                "input_values": data_augmenter(
                    load_audio(
                        row["path"],
                        start=row["start"] if has_segment else None,
                        end=row["end"] if has_segment else None,
                        sample_rate=sample_rate,
                    )
                ),
                "labels": text_processor(text_augmenter(row["text"])),
            },
            remove_columns=column_names,
            **map_kwargs,
        )
    else:
        processed = dataset.map(
            lambda row: {"input_values": np.array([1.0], dtype=np.float32), "labels": "e"}
            if (hasattr(transformers.trainer, "SKIPPING") and transformers.trainer.SKIPPING)
            else {
                "input_values": load_audio(
                    row["path"],
                    start=row["start"] if has_segment else None,
                    end=row["end"] if has_segment else None,
                    sample_rate=sample_rate,
                ),
                "labels": text_processor(row["text"]),
            },
            remove_columns=column_names,
            **map_kwargs,
        )

    if logstream and hasattr(processed, "cache_files"):
        logstream.write(f"- Huggingface cached dataset with loaded audio in {format_cache_files(processed.cache_files)}\n")

    # Check characters
    def extract_all_chars(batch):
        all_text = " ".join(batch)
        vocab = sorted(list(set(all_text)))
        return {"vocab": vocab}

    subset = processed
    if hasattr(subset, "__len__"):
        if len(subset) > 100:
            subset = processed.select(random.sample(range(min(len(processed), 100000)), 100))
        chars = subset.map(
            lambda batch: extract_all_chars(batch["labels"]),
            batched=True,
            batch_size=-1,
            remove_columns=processed.column_names,
        )["vocab"]
    else:
        subset = processed.take(100)
        text = [sample["labels"] for sample in subset]
        chars = extract_all_chars(text)["vocab"]
    vocab = processor.tokenizer.get_vocab()
    character_level_vocab = len(vocab) < 100
    if character_level_vocab:
        no_warning = True
        for char in chars:
            if char not in vocab and char != " ":
                if verbose:
                    print(f"WARNING: character {char} not in vocabulary")
                no_warning = False
        if no_warning and verbose:
            print("GOOD: All characters seem to be in vocabulary")

    if not is_iterable:
        if "cache_file_name" in map_kwargs:
            map_kwargs.pop("cache_file_name")  # Will be done automatically at this point
        map_kwargs.update({"num_proc": num_proc})  # Batch size is used
    if verbose and hasattr(dataset, "_fingerprint"):
        print("Processing audios", processed._fingerprint)
    processed = processed.map(lambda batch: apply_processor(processor, batch, sample_rate), batch_size=batch_size, batched=True, **map_kwargs)
    if logstream and hasattr(processed, "cache_files"):
        logstream.write(f"- Huggingface cached pre-processed dataset in {format_cache_files(processed.cache_files)}\n")

    if verbose and hasattr(processor, "_fingerprint"):
        print("Audio processed", processed._fingerprint)

    return processed


def apply_processor(processor, batch, sample_rate):
    batch_inputs = batch["input_values"]
    processed = processor(batch_inputs, sampling_rate=sample_rate)
    if hasattr(processed, "input_values"):
        batch["input_values"] = processed.input_values
    elif hasattr(processed, "input_features"):
        batch["input_features"] = processed.input_features
    else:
        raise NotImplementedError(f"Could not find any known key among {list(processed.keys())}")
    batch["input_length"] = [len(audio) / sample_rate for audio in batch_inputs]
    if hasattr(processor, "tokenizer"):
        batch["labels"] = processor.tokenizer(batch["labels"]).input_ids
    else:
        with processor.as_target_processor():
            batch["labels"] = processor(batch["labels"]).input_ids
    return batch


def to_audio_batches(
    input,
    batch_size=0,
    sample_rate=16_000,
    mono=True,
    return_format="array",
    sort_by_len=False,
    output_ids=False,
):
    """
    Convert a filename, a kaldi folder, or a list of those into batches of audio

    return_format : str
        Output format. Possible values: 'array', 'torch' or 'bytes'
    """
    if isinstance(input, str):
        if os.path.isdir(input):
            _, dataset = kaldi_folder_to_dataset(input, sort_by_len=-1 if sort_by_len else 0)
            batch = []
            for data in dataset:
                audio = load_audio(
                    data["path"],
                    data.get("start"),
                    data.get("end"),
                    sample_rate=sample_rate,
                    mono=mono,
                    return_format=return_format,
                )
                if output_ids:
                    audio = (audio, data["ID"])
                if batch_size == 0:
                    yield audio
                else:
                    batch.append(audio)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            if len(batch) > 0:
                yield batch

        elif os.path.isfile(input):
            audio = load_audio(input, sample_rate=sample_rate, mono=mono, return_format=return_format)
            if output_ids:
                audio = (audio, os.path.basename(input))
            if batch_size == 0:
                yield audio
            else:
                yield [audio]

        elif input_start_end := parse_input_file_with_start_end(input):
            input, start, end = input_start_end
            audio = load_audio(input, start=start, end=end, sample_rate=sample_rate, mono=mono, return_format=return_format)
            if output_ids:
                audio = (audio, os.path.basename(input))
            if batch_size == 0:
                yield audio
            else:
                yield [audio]

        elif input_starts_ends := parse_input_file_with_starts_ends(input):
            input, starts_ends = input_starts_ends
            batch = []
            for start, end in starts_ends:
                audio = load_audio(input, start, end, sample_rate=sample_rate, mono=mono, return_format=return_format)
                if output_ids:
                    audio = (audio, os.path.basename(input) + ":" + str(start) + "-" + str(end))
                if batch_size == 0:
                    yield audio
                else:
                    batch.append(audio)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            if len(batch) > 0:
                yield batch

        else:
            raise ValueError(f"Cannot interpret {input} as an existing audio file, kaldi folder (or audiofile.wav:start-end)")

    elif isinstance(input, list):
        batch = []
        for data in input:
            batches = to_audio_batches(
                data,
                batch_size=batch_size,
                sample_rate=sample_rate,
                mono=mono,
                return_format=return_format,
                sort_by_len=sort_by_len,
                output_ids=output_ids,
            )
            for b in batches:
                if batch_size == 0 or len(b) == batch_size:
                    yield b
                    continue
                for sample in b:
                    if batch_size == 0:
                        yield sample
                    else:
                        batch.append(sample)
                        if len(batch) == batch_size:
                            yield batch
                            batch = []
        if len(batch) > 0:
            yield batch

    elif isinstance(input, np.ndarray) and len(input.shape) == 1:
        if return_format == "torch":
            input = torch.Tensor(input)
        elif return_format == "bytes":
            input = array_to_bytes(input)
        if output_ids:
            input = (input, hashmd5(input))
        if batch_size == 0:
            yield input
        else:
            yield [input]

    else:
        raise NotImplementedError("Unsupported type: %s" % type(input))


def parse_input_file_with_start_end(input_file):
    """
    Parse an input that is XXX.wav:start-end
    Return None if it fails
    """
    if ":" in input_file:
        f = input_file.split(":")
        input_file = ":".join(f[:-1])
        if not os.path.isfile(input_file):
            logger.warning(f"Cannot find file {input_file}")
            return None
        start_end = f[-1].split("-")
        if len(start_end) == 2:
            start, end = start_end
            try:
                start = float(start)
                end = float(end)
            except ValueError:
                return None
            return (input_file, start, end)
    return None


def parse_input_file_with_starts_ends(input_file):
    """
    Parse an input that is XXX.wav:start1-end1,start2-end2,...
    Return None if it fails
    """
    if ":" in input_file:
        f = input_file.split(":")
        input_file = ":".join(f[:-1])
        if not os.path.isfile(input_file):
            logger.warning(f"Cannot find file {input_file}")
            return None
        starts_ends_str = f[-1].split(",")
        starts_ends = []
        for start_end in starts_ends_str:
            start_end = start_end.split("-")
            if len(start_end) != 2:
                return None
            start, end = start_end
            try:
                start = float(start)
                end = float(end)
            except ValueError:
                return None
            starts_ends.append((start, end))
        return input_file, starts_ends
    return None


def to_annotation_text(input):
    """
    Convert a filename, a kaldi folder, or a list of those into batches of audio

    return_format : str
        Output format. Possible values: 'array', 'torch' or 'bytes'
    """
    if isinstance(input, str):
        if os.path.isdir(input) or os.path.isfile(input):
            _, dataset = kaldi_folder_to_dataset(input)
            for data in dataset:
                yield data["text"]

        else:
            raise ValueError(f"Cannot interpret {input} as a file or a directory")

    elif isinstance(input, list):
        for data in input:
            for text in to_annotation_text(data):
                yield text

    else:
        raise NotImplementedError("Unsupported type: %s" % type(input))
