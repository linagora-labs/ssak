import argparse
import logging
import os
import subprocess

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rsync")
    parser.add_argument("folder_list", help="Source directory")
    parser.add_argument("src", help="Source directory")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("--only_audios", action="store_true", help="Only rsync audio files")

    args = parser.parse_args()
    with open(args.folder_list) as f:
        folders = f.read().strip().split("\n")
    for i, folder in enumerate(folders):
        folders[i] = os.path.join(args.src, folder)
        if folders[i][-1] == "/":
            folders[i] = folders[i][:-1]
    logger.info(f"Found {len(folders)} folders to rsync to {args.dest}")
    os.makedirs("logs", exist_ok=True)
    for folder in tqdm(folders):
        log_file = os.path.join("logs", os.path.basename(folder) + ".log")
        if os.path.exists(log_file):
            logger.info(f"Skipping {folder}, already rsynced")
            continue
        logger.info(f"Rsyncing {folder}")
        if args.only_audios:
            cmd = f"rsync -rlDvz --chmod=u=rwX,g=rX,o= --size-only --copy-links --include='*.wav' --include='*.flac' --include='*.mp3' --include='*/' --exclude='*' {folder} {args.dest} > {log_file}"
        else:
            cmd = f"rsync -rlDvz --chmod=u=rwX,g=rX,o= --size-only --copy-links --exclude='*.zip' --exclude='*.tar' --exclude='*.gz' {folder} {args.dest} > {log_file}"
        # cmd = f"ls {folder} > {log_file}"
        subprocess.run(cmd, shell=True)
        logger.info(f"Rsynced {folder}")
    logger.info("All done")
