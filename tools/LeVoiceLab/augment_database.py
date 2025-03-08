import glob
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import librosa
import numpy as np
import soxbindings as sox


def json_dump(dic, f):
    json.dump(dic, f, indent=2, ensure_ascii=False)


import sys

sys.path = [os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_augmentation")] + sys.path
import warnings

from augmentation import SpeechAugment, transform2description, transform2genericstr, transform2str

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="audiomentations")
warnings.filterwarnings("ignore", message="formats: mp3 can't encode")


def load_audio(path, start=None, end=None, sampling_rate=None, mono=True):
    """
    Load an audio file and return the data as a numpy array.
    """
    if start is not None:
        sr = sox.get_info(path)[0].rate
        offset = int(start * sr)
        nframes = 0
        if end is not None:
            nframes = int((end - start) * sr)
        audio, sr = sox.read(path, offset=offset, nframes=nframes)
    else:
        audio, sr = sox.read(path)
    audio = np.float32(audio)
    if mono:
        if audio.shape[1] == 1:
            audio = audio.reshape(audio.shape[0])
        else:
            audio = librosa.to_mono(audio.transpose())
    if sampling_rate is not None and sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return (audio, sr)


def save_audio(path, audio, sampling_rate):
    """
    Save an audio file from a numpy array.
    """
    try:
        sox.write(path, audio, sampling_rate)
    except Exception as e:
        print("Error while saving audio file:", e)
        print("Retrying in 1 minute...")
        time.sleep(60)
        return save_audio(path, audio, sampling_rate)


augmenter8k = None
augmenter16k = None


def get_augmenter(sample_rate=16_000):
    global augmenter8k, augmenter16k
    if sample_rate <= 10000:
        if augmenter8k is None:
            print("Loading augmenter for 8kHz...")
            augmenter8k = SpeechAugment(
                noise_dir="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises",
                rir_dir="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise",
                rir_lists=[
                    "simulated_rirs_8k/smallroom/rir_list",
                    "simulated_rirs_8k/mediumroom/rir_list",
                    "simulated_rirs_8k/largeroom/rir_list",
                ],
                apply_prob=1,
            )
        return augmenter8k
    else:
        if augmenter16k is None:
            print("Loading augmenter for 16kHz...")
            augmenter16k = SpeechAugment(
                noise_dir="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises",
                rir_dir="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise",
                rir_lists=[
                    "simulated_rirs_16k/smallroom/rir_list",
                    "simulated_rirs_16k/mediumroom/rir_list",
                    "simulated_rirs_16k/largeroom/rir_list",
                ],
                apply_prob=1,
            )
        return augmenter16k


def augment_audio(path_in, dir_out, target_sr=None, append_transform_to_dir=False):
    (audio, sr) = load_audio(path_in)

    if target_sr and target_sr != sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    augmenter = get_augmenter(sr)

    transform, audio = augmenter(audio, sr)

    transform_str = transform2str(transform, True)
    generic_transform_str = transform2genericstr(transform)

    fname = os.path.basename(path_in)
    fname, ext = fname.split(".", 1)
    if append_transform_to_dir:
        dir_out = os.path.join(dir_out, os.path.basename(dir_out) + "_" + generic_transform_str)
    path_out = os.path.join(dir_out, fname + "." + transform_str + "." + ext)

    if not os.path.isfile(path_out):
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        save_audio(path_out, audio, sr)

    return transform, path_out


def time2str(t):
    if t is None:
        return ""
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def uncamel(s):
    # Add spaces before capital letters
    return "".join([" " + c if c.isupper() else c for c in s]).strip()


def run_command(command, doit=True, verbose=True):
    if isinstance(command, list):
        command = " ".join(command)

    if verbose or not doit:
        print(f"... Running command ...\n{command}")
    if not doit:
        return

    return subprocess.run(command, shell=True, check=True)


def post_to_minio_and_clean(dir_in, s3server, s3user, initialize=False):
    USERNAME = os.environ.get("LVL_MINIO_USERNAME")
    PASSWD = os.environ.get("LVL_MINIO_PASSWD")
    if not USERNAME or not PASSWD:
        raise ValueError("Please set LVL_MINIO_USERNAME and LVL_MINIO_PASSWD environment variables")

    alias_command = f"mc alias set voicelab {s3server} {USERNAME} {PASSWD}"
    sync_command = f"mc mirror --overwrite {dir_in} voicelab/upload-data-linagora/{s3user}/{os.path.basename(dir_in)}"
    if initialize:
        print("### Initialize mc:\n" + alias_command)
        run_command(alias_command)
        print("### Upload to minio will be done with command:\n" + sync_command)
        return

    run_command(sync_command)

    # Clean data
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if any([file.endswith(ext) for ext in [".audio.mp3", ".annotations.json", ".meta.json"]]):
                os.remove(os.path.join(root, file))


if __name__ == "__main__":
    import argparse
    import random
    import sys

    import audiomentations
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Augment a database")
    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument(
        "--split_by_augmentation_type",
        default=False,
        action="store_true",
        help="Split in sub-folders depending on the augmentation type",
    )
    parser.add_argument("--continue", dest="continue_", default=False, action="store_true", help="Continue from where we stopped")
    parser.add_argument("--to16k", default=False, action="store_true", help="Convert audio to 16kHz")
    parser.add_argument(
        "--s3user",
        default=None,
        help="A minio username. If specified, data will be uploaded while being generated, and cleaned. Example: linagora-jerome-louradour",
    )
    parser.add_argument("--s3server", default="https://s3.levoicelab.org", help="A minio server. Example: https://s3.levoicelab.org")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print more information")
    parser.add_argument("--extension", default=None, help="Only process files with this extension")
    args = parser.parse_args()

    SEED = 1
    MAX = None
    random.seed(SEED)

    dir_in = args.input
    dir_out = args.output
    subdirs_out = {}

    if os.path.isfile(dir_out):
        raise RuntimeError(f"Output directory {dir_out} is a file. Please remove it or specify another directory.")
    # NOCOMMIT
    # if not args.continue_ and os.path.exists(dir_out):
    #     print(f"Output directory {dir_out} already exists. Please remove it or use --continue.")
    #     sys.exit(0)
    os.makedirs(dir_out, exist_ok=True)

    upload_and_clean = bool(args.s3user)
    post_to_minio_each = 20
    if upload_and_clean:
        post_to_minio_and_clean(dir_out, args.s3server, args.s3user, initialize=True)

    print("Scanning folder")
    all_items = os.listdir(dir_in)
    print(f"* {len(all_items)} items found in folder")

    def is_folder(fname):
        if "." in fname:
            return False
        return os.path.isdir(os.path.join(dir_in, fname))

    annotation_dirs = sorted([f for f in all_items if is_folder(f)])
    print(f"* {len(annotation_dirs)} annotation folders found")

    if args.extension:
        _ending = ".audio." + args.extension.lstrip(".")

        def is_audio(fname):
            return fname.endswith(_ending)
    else:

        def is_audio(fname):
            f = fname.split(".")
            return len(f) >= 3 and f[-2] == "audio"

    audio_files = sorted([f for f in all_items if is_audio(f)])
    print(f"* {len(audio_files)} audio files found")

    NUM_MULTI_AUG = get_augmenter().get_num_transforms()
    print(f"{NUM_MULTI_AUG} transformations (augmentation types)")

    now = time2str(datetime.now())

    done = 0
    for ifile, fname in enumerate(tqdm(audio_files, desc="Augmenting audio files")):
        # NOCOMMIT
        if ifile < 107358:
            continue
        f = fname.split(".")
        base = ".".join(f[:-2])
        audio = os.path.join(dir_in, fname)
        meta = os.path.join(dir_in, base + ".meta.json")
        assert os.path.isfile(audio), f"File {audio} does not exist"
        assert os.path.isfile(meta), f"File {meta} does not exist"

        meta = json.load(open(meta))
        meta["is_natural"] = False
        meta["is_augmented"] = True
        meta["date_created"] = now
        meta["collection_date"] = now

        if args.continue_:
            # Basic heuristic to check if things were computed already
            bname = os.path.splitext(os.path.splitext(fname)[0])[0]
            if args.split_by_augmentation_type:
                outputs = glob.glob(os.path.join(dir_out, "**", bname + ".*.audio.mp3"), recursive=True)
            else:
                outputs = glob.glob(os.path.join(dir_out, bname + ".*.audio.mp3"))
            if len(outputs) >= NUM_MULTI_AUG:
                continue

        for _ in range(NUM_MULTI_AUG):
            tic = time.time()

            transform, fname = augment_audio(
                audio,
                dir_out,
                target_sr=16_000 if args.to16k else None,
                append_transform_to_dir=args.split_by_augmentation_type,
            )
            if args.verbose:
                print(fname, f"({time.time() - tic} sec)")
            subdir_out = os.path.dirname(fname)
            if subdir_out not in subdirs_out:
                description = transform2description(transform)
                subdirs_out[subdir_out] = {
                    "short": uncamel(transform2genericstr(transform)),
                    "description": description,
                }
            f = os.path.basename(fname).split(".")
            base_out = ".".join(f[:-2])

            meta["extra"] = {"augmentation_technic": transform2str(transform, False)}
            # If we change the speed...
            # meta["duration_milliseconds"] = ...
            json_dump(meta, open(os.path.join(subdir_out, base_out + ".meta.json"), "w"))

            for annotation_dir in annotation_dirs:
                annot = os.path.join(dir_in, annotation_dir, base + ".annotations.json")
                assert os.path.isfile(annot)
                if not os.path.isdir(os.path.join(subdir_out, annotation_dir)):
                    os.makedirs(os.path.join(subdir_out, annotation_dir))
                shutil.copy(annot, os.path.join(subdir_out, annotation_dir, base_out + ".annotations.json"))

        done += 1
        if MAX is not None and done >= MAX:
            break

        if upload_and_clean and (done == 1 or (done % post_to_minio_each == 0)):
            post_to_minio_and_clean(dir_out, args.s3server, args.s3user)

    RATIO = 1 if args.split_by_augmentation_type else NUM_MULTI_AUG

    for dir_out, augmentation_meta in subdirs_out.items():
        # global meta.json
        meta = json.load(open(os.path.join(dir_in, "meta.json")))

        # new in version 0.0.2
        version = "0.0.2"
        meta["version"] = version
        meta["format_specification_uri"] = f"http://levoicelab.org/schemas/{version}/main-db.schema.json"
        meta["languages"] = meta.get("languages", ["fr"])

        # from version 0.0.1
        meta["description"] = f"Augmentation de la base {meta['name']}."
        if args.split_by_augmentation_type:
            description = augmentation_meta["description"]
            description = description[0].lower() + description[1:]
            meta["description"] = meta["description"][:-1] + f", avec {description}."
        meta["name"] += " Augmented"
        if args.split_by_augmentation_type:
            meta["name"] += f" ({augmentation_meta['short']})"
        meta["num_audio_files"] = meta["num_audio_files"] * RATIO
        meta["date_created"] = now
        meta["collection_date_from"] = now
        meta["collection_date_to"] = now
        if args.to16k:
            meta["sample_rate"] = 16_000
        meta["num_channels"] = 1
        meta["bit_depth"] = 16
        meta["contains_natural_speech"] = False
        meta["contains_augmented_speech"] = True
        meta["total_duration_seconds"] = meta["total_duration_seconds"] * RATIO
        meta["augmented_speech_duration_seconds"] = meta["natural_speech_duration_seconds"] * RATIO
        meta["natural_speech_duration_seconds"] = 0
        if "extra" not in meta:
            meta["extra"] = {}
        meta["extra"]["augmentation_technic"] = f"Augmentation was done using python audiomentations package, version {audiomentations.__version__}"
        # Always 1 channel and 16-bit after augmentation
        if "extra" in meta and "num_channels" in meta["extra"]:
            meta["extra"].pop("num_channels")
        if "extra" in meta and "bit_depth" in meta["extra"]:
            meta["extra"].pop("bit_depth")
        if args.to16k:
            if "extra" in meta and "sample_rate" in meta["extra"]:
                meta["extra"].pop("sample_rate")
        if "extra" in meta and "notes" in meta["extra"]:
            if args.to16k or "sample_rate" not in meta["extra"]["notes"]:
                meta["extra"].pop("notes")
            else:
                meta["extra"]["notes"] = meta["extra"]["notes"].replace("bit_depth / ", "").replace(" / bit_depth", "").replace("num_channels / ", "").replace(" / num_channels", "")
        json_dump(meta, open(os.path.join(dir_out, "meta.json"), "w"))

        for annotation_dir in annotation_dirs:
            shutil.copy(os.path.join(dir_in, annotation_dir, "meta.json"), os.path.join(dir_out, annotation_dir, "meta.json"))

    if upload_and_clean:
        post_to_minio_and_clean(dir_out, args.s3server, args.s3user)
