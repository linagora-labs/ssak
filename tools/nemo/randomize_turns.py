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
        if "/nocontext/" in dirpath + "/" or "/old/" in dirpath + "/" or "Multitask-National-Speech-Corpus" in dirpath:
            continue
        if dirpath.endswith("_shards") or "_shards_" in os.path.basename(dirpath):
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
    """Randomly reorder maximal runs of consecutive User turns (50% chance per run).

    For each maximal run of consecutive User turns of length >= 2, one
    `rng.random()` call is made; if it is < 0.5 the run is "flipped":
    the run is split into maximal contiguous same-type chunks and the
    order of chunks is reversed while order within each chunk is kept.
    If the whole run shares a single type, the elements themselves are
    reversed instead (fallback that makes the length-2 same-type case
    behave like a plain swap).

    Examples (when flipped):
      [text, audio]                  -> [audio, text]
      [audio, audio]                 -> [audio_2, audio_1]
      [text, audio_1, audio_2]       -> [audio_1, audio_2, text]
      [audio_1, audio_2, text]       -> [text, audio_1, audio_2]

    Length-2 runs consume one `rng.random()` call and either swap or not,
    matching the previous implementation exactly so a given seed produces
    the same output on datasets that only contain length-2 User runs.

    Returns a new list with turns possibly reordered.
    """
    result = list(conversations)
    n = len(result)
    i = 0
    while i < n:
        if result[i].get("from") != "User":
            i += 1
            continue
        j = i
        while j < n and result[j].get("from") == "User":
            j += 1
        if j - i >= 2 and rng.random() < 0.5:
            # Split run [i:j] into maximal same-type chunks.
            chunks = []
            k = i
            while k < j:
                m = k + 1
                while m < j and result[m].get("type") == result[k].get("type"):
                    m += 1
                chunks.append(result[k:m])
                k = m
            if len(chunks) > 1:
                result[i:j] = [t for chunk in reversed(chunks) for t in chunk]
            else:
                # Single chunk (all same type): reverse the elements.
                # For length 2 this is just a swap, matching the legacy behavior.
                result[i:j] = list(reversed(result[i:j]))
        i = j
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
            if os.path.getmtime(outpath) >= os.path.getmtime(filepath):
                skipped += 1
                continue
            print(f"STALE: {outpath} is older than {filepath}, regenerating")
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
