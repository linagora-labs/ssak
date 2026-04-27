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
    permission_denied = []
    other_errors = []
    pbar = tqdm(dirs)
    for dir in pbar:
        pbar.set_description(f"Touching {dir}")
        for root, _, files in os.walk(os.path.join(args.input_folder, dir)):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    os.utime(filepath)
                except PermissionError:
                    permission_denied.append(filepath)
                except Exception as e:
                    other_errors.append((filepath, e))
    if permission_denied:
        logger.warning(f"Permission denied for {len(permission_denied)} files (first 5: {permission_denied[:5]})")
    for filepath, e in other_errors:
        logger.error(f"Failed to touch {filepath}: {e}")
    logger.info("All done")
