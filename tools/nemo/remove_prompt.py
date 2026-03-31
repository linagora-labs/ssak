import argparse
import json
import logging
import shutil

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_prompt_from_line(line):
    """Remove the first text prompt turn from a conversation line if present."""
    row = json.loads(line)
    conversations = row.get("conversations")
    if conversations and conversations[0].get("from") == "User" and conversations[0].get("type") == "text":
        row["conversations"] = conversations[1:]
    return row


def remove_prompt_from_file(filepath):
    """Remove text prompts from all lines in a JSONL file, in-place."""
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    count = 0
    with open(filepath, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = remove_prompt_from_line(line)
            json.dump(row, fout, ensure_ascii=False)
            fout.write("\n")
            count += 1
    shutil.move(tmp_path, filepath)
    logger.info(f"Processed {filepath}: {count} lines")


def remove_prompt_from_directory(directory, recursive=False):
    """Remove text prompts from all JSONL files in a directory."""
    directory = Path(directory)
    if recursive:
        jsonl_files = sorted(directory.rglob("*.jsonl"))
    else:
        jsonl_files = sorted(directory.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {directory}")
        return

    logger.info(f"Found {len(jsonl_files)} .jsonl files")
    for filepath in jsonl_files:
        remove_prompt_from_file(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove text prompts from multiturn JSONL conversation files.")
    parser.add_argument("input", help="Directory containing .jsonl files, or a single .jsonl file.", type=str)
    parser.add_argument("--recursive", "-r", help="Search for .jsonl files recursively.", default=False, action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        remove_prompt_from_file(input_path)
    elif input_path.is_dir():
        remove_prompt_from_directory(input_path, recursive=args.recursive)
    else:
        logger.error(f"Input path does not exist: {input_path}")
