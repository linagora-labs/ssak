import argparse
import json
import logging
import os

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rename_audio_path(input, old_path, new_path):
    with open(input + ".tmp", "w", encoding="utf-8") as out:
        with open(input, encoding="utf-8") as fin:
            for line in tqdm(fin.readlines()):
                json_line = json.loads(line)
                if old_path in json_line["audio_filepath"]:
                    json_line["audio_filepath"] = json_line["audio_filepath"].replace(old_path, new_path)
                json.dump(json_line, out, ensure_ascii=False)
                out.write("\n")
    os.rename(input + ".tmp", input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge manifest files")
    parser.add_argument("input", type=str, help="")
    parser.add_argument("search", type=str, default="/data-server/datasets/audio/transcript/fr/YODAS/", help="")
    parser.add_argument("replace_with", type=str, help="")
    args = parser.parse_args()
    rename_audio_path(args.input, args.search, args.replace_with)
