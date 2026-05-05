#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import re
import signal
import sys


def find_train_jsonl(root):
    """Find all train*jsonl files under root, skipping paths containing 'nocontext'."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "/nocontext/" in dirpath + "/" or "/Nvidia/" in dirpath + "/" or "/old/" in dirpath + "/" or "Multitask-National-Speech-Corpus" in dirpath:
            continue
        for f in filenames:
            if f.endswith("_orig.jsonl"):
                continue
            if f.endswith(".jsonl"): # and f.startswith("train"):
                if not f.endswith("_randomorder.jsonl"):
                    results.append(os.path.join(dirpath, f))
    return sorted(results)


def deterministic_seed(filepath):
    """Return a deterministic integer seed based on the first line of the file.

    Absolute paths are normalized to /PATH/filename.ext so the seed is
    consistent across machines with different directory layouts.
    """
    with open(filepath, encoding="utf-8") as fh:
        first_line = fh.readline()
    normalized = re.sub(r"/[^\s\"',}]+/([^/\s\"',}]+\.\w+)", r"/PATH/\1", first_line)
    return int(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), 16) % (2**32)


def randomize_conversations(conversations, rng):
    """Randomly swap consecutive User text/audio turn pairs (50% chance).

    Returns a new list with turns possibly reordered.
    """
    result = list(conversations)
    i = 0
    while i < len(result) - 1:
        a, b = result[i], result[i + 1]
        if (
            a.get("from") == "User"
            and b.get("from") == "User"
            and {a.get("type"), b.get("type")} == {"text", "audio"}
        ):
            if rng.random() < 0.5:
                result[i], result[i + 1] = result[i + 1], result[i]
            i += 2
        else:
            i += 1
    return result


def output_path_for(filepath):
    """Return the _randomorder.jsonl path for a given input file."""
    base, ext = os.path.splitext(filepath)
    return f"{base}_randomorder{ext}"


def process_file(filepath):
    """Process a single file: randomize turns and write output.

    Returns True on success, False on failure.
    """
    outpath = output_path_for(filepath)
    seed = deterministic_seed(filepath)
    rng = random.Random(seed)

    tmppath = os.path.join(
        os.path.dirname(outpath),
        f".tmp_randomorder_{os.getpid()}_{os.path.basename(outpath)}",
    )
    try:
        with open(tmppath, "w", encoding="utf-8") as out, open(
            filepath, encoding="utf-8"
        ) as inp:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "conversations" in entry:
                    entry["conversations"] = randomize_conversations(
                        entry["conversations"], rng
                    )
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        try:
            src_gid = os.stat(filepath).st_gid
            os.chown(tmppath, -1, src_gid)
        except (OSError, PermissionError):
            pass
        os.replace(tmppath, outpath)
        return True
    except BaseException:
        # Clean up partial output on any failure (including KeyboardInterrupt)
        try:
            os.unlink(tmppath)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Randomly permute User text/audio turn order in train JSONL files."
    )
    parser.add_argument(
        "--root",
        default="/data-server/datasets/audio/nemo",
        help="Root directory to search (default: /data-server/datasets/audio/nemo)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing _randomorder.jsonl files (default: skip them)",
    )
    args = parser.parse_args()

    files = find_train_jsonl(args.root)
    print(f"Found {len(files)} train*.jsonl file(s) under {args.root}")

    processed = 0
    skipped = 0
    failed = 0
    for filepath in files:
        outpath = output_path_for(filepath)
        if os.path.exists(outpath) and not args.overwrite:
            skipped += 1
            continue
        try:
            process_file(filepath)
            processed += 1
            print(f"OK: {outpath}")
        except KeyboardInterrupt:
            print(f"\nInterrupted while processing {filepath}")
            sys.exit(1)
        except Exception as e:
            print(f"FAIL: {filepath} — {e}")
            failed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
