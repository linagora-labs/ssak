import argparse
import logging
import os

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Touch everything")
    parser.add_argument("input_folder", help="Source directory")
    args = parser.parse_args()
    dirs = os.listdir(args.input_folder)
    logger.info(f"Touching everything in {args.input_folder}")
    pbar = tqdm(dirs)
    for dir in pbar:
        pbar.set_description(f"Touching {dir}")
        for root, _, files in os.walk(os.path.join(args.input_folder, dir)):
            for file in files:
                try:
                    os.utime(os.path.join(root, file))
                except Exception as e:
                    logger.info(f"Failed to touch {os.path.join(root, file)}: {e}")
    logger.info("All done")
