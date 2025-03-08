import argparse
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rewrite_wavscp(filepath, backup=True):
    with open(filepath) as f:
        lines = f.readlines()
    if backup:
        shutil.move(filepath, filepath + ".bak")
    with open(filepath, "w") as f:
        for line in lines:
            line = line.replace(src, dest)
            f.write(line)


def rewrite_kaldi_annotations(directory, backup=True):
    object_paths = os.listdir(directory)
    wav_scp = os.path.join(directory, "wav.scp")
    if os.path.exists(wav_scp):
        rewrite_wavscp(wav_scp, backup)
        logger.info(f'Updated {wav_scp} {"(with backup)" if backup else ""}')
    else:
        for folder in object_paths:
            wav_scp = os.path.join(directory, folder, "wav.scp")
            if os.path.exists(wav_scp):
                rewrite_wavscp(wav_scp, backup)
                logger.info(f'Updated {wav_scp} {"(with backup)" if backup else ""}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move dataset to another directory")
    parser.add_argument("dataset_folder", type=str, help="source directory")
    parser.add_argument("target_folder", type=str, help="destination directory")
    parser.add_argument("--kaldi_annotations", default=None, type=str, help="kaldi dataset folders")
    parser.add_argument("--kaldi_target_folder", default=None, type=str, help="destination directory for kaldi dataset")
    parser.add_argument("--no_backup", default=False, action="store_true", help="backup wav.scp files in case dataset move fails")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    target_folder = args.target_folder
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"{dataset_folder} does not exist")

    path_in_dataset = os.listdir(dataset_folder)
    kaldi_annotations = args.kaldi_annotations
    kaldi_target_folder = args.kaldi_target_folder
    if kaldi_annotations is not None:
        logger.info(f"Will not check for kaldi dataset folders in {dataset_folder} because kaldi_annotations is provided")

    src = os.path.abspath(dataset_folder)
    dest = os.path.abspath(target_folder)

    logger.info(f"Moving {src} to {dest}")
    inter = list({"annotations", "kaldi"} & set(path_in_dataset))
    if inter and kaldi_annotations is None:
        rewrite_kaldi_annotations(os.path.join(dataset_folder, inter[0]), backup=not args.no_backup)
    elif kaldi_annotations is not None and not os.path.exists(kaldi_annotations):
        raise FileNotFoundError(f"{kaldi_annotations} does not exist")
    elif kaldi_annotations is not None:
        rewrite_kaldi_annotations(kaldi_annotations, backup=not args.no_backup)

    shutil.move(dataset_folder, target_folder)

    if kaldi_annotations is not None and kaldi_target_folder is not None:
        shutil.move(kaldi_annotations, kaldi_target_folder)
        logger.info(f"Moved {kaldi_annotations} to {kaldi_target_folder}")
