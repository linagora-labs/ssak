import argparse
import logging
import random
from pathlib import Path

from ssak.utils.nemo_dataset import NemoDataset, resolve_manifest_paths, resolve_output_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_row_duration(row):
    """Sum of all audio turn durations in a row."""
    return sum(t.duration for t in row.turns if t.turn_type == "audio" and t.duration is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a smaller manifest from a larger one.")
    parser.add_argument("input", help="Input JSONL manifest, directory, or YAML config.")
    parser.add_argument("--output", default=None, help="Output JSONL manifest or directory (default: <input>_subsampled.jsonl).")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for manifest files when input is a directory.")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recursively search directories.")
    parser.add_argument("--head", type=float, help="Keep first N rows (int) or proportion (0-1).")
    parser.add_argument("--tail", type=float, help="Keep last N rows (int) or proportion (0-1).")
    parser.add_argument("--random", type=float, help="Keep N random rows (int) or proportion (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (only used with --random, default: 42).")
    parser.add_argument("--min_duration", type=float, help="Minimum audio duration in seconds.")
    parser.add_argument("--max_duration", type=float, help="Maximum audio duration in seconds.")

    args = parser.parse_args()

    manifests = resolve_manifest_paths(args.input, pattern=args.pattern, recursive=args.recursive)
    if not manifests:
        logger.error(f"No manifest files found for input: {args.input}")
        exit(1)

    for mf in manifests:
        if args.output is not None:
            out_file = str(resolve_output_path(mf, args.input, args.output))
        else:
            out_file = str(mf.with_suffix("")) + "_subsampled.jsonl"

        dataset = NemoDataset()
        data_type = dataset.load(str(mf))
        logger.info(f"Loaded {len(dataset)} rows from {mf}")

        # Duration filtering
        if args.min_duration is not None or args.max_duration is not None:
            before = len(dataset)
            filtered = []
            for row in dataset:
                dur = get_row_duration(row)
                if args.min_duration is not None and dur < args.min_duration:
                    continue
                if args.max_duration is not None and dur > args.max_duration:
                    continue
                filtered.append(row)
            dataset.dataset = filtered
            logger.info(f"Duration filter: {before} -> {len(dataset)} rows")

        # Convert proportion to count
        def to_count(value):
            if value is not None and 0 < value < 1:
                return max(1, int(len(dataset) * value))
            return int(value) if value is not None else None

        # Slicing
        if args.head is not None:
            dataset.dataset = dataset.dataset[:to_count(args.head)]
        elif args.tail is not None:
            dataset.dataset = dataset.dataset[-to_count(args.tail):]
        elif args.random is not None:
            random.seed(args.seed)
            n = min(to_count(args.random), len(dataset))
            dataset.dataset = random.sample(dataset.dataset, n)

        logger.info(f"Saving {len(dataset)} rows to {out_file}")
        dataset.save(out_file, data_type=data_type)
