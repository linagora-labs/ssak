#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _resolve_oc_env(path_str):
    """Resolve ${oc.env:VAR_NAME} patterns using environment variables."""
    def replacer(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable '{var_name}' not set (needed for: {path_str})")
        return value
    return re.sub(r'\$\{oc\.env:([^}]+)\}', replacer, path_str)


def _update_manifest_paths_in_yaml(cfg, path_to_output):
    """Recursively replace manifest_filepath values in a YAML config structure.

    Args:
        cfg: The parsed YAML structure (list of dicts with input_cfg / manifest_filepath).
        path_to_output: dict mapping resolved manifest path (str) -> _randomorder path (str).
    """
    if isinstance(cfg, list):
        for entry in cfg:
            if isinstance(entry, dict):
                if "input_cfg" in entry:
                    _update_manifest_paths_in_yaml(entry["input_cfg"], path_to_output)
                if "manifest_filepath" in entry:
                    raw = entry["manifest_filepath"]
                    try:
                        resolved = _resolve_oc_env(raw)
                        resolved_abs = str(Path(resolved).resolve())
                        if resolved_abs in path_to_output:
                            entry["manifest_filepath"] = path_to_output[resolved_abs]
                    except ValueError:
                        pass


def randomize_from_yaml(yaml_path, overwrite=False):
    """Randomize all manifests referenced in a YAML config and produce an updated YAML."""
    import yaml

    from ssak.utils.nemo_dataset import resolve_manifest_paths

    yaml_path = Path(yaml_path)
    # Already a _randomorder config (its manifests are the *_randomorder.jsonl): nothing
    # to randomize, and we must NOT emit a doubly-suffixed *_randomorder_randomorder.yaml.
    if yaml_path.stem.endswith("_randomorder"):
        logger.info(f"Config is already a _randomorder config, nothing to randomize: {yaml_path}")
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    manifest_files = resolve_manifest_paths(yaml_path)
    if not manifest_files:
        logger.warning(f"No manifest files found in YAML: {yaml_path}")
        return

    path_to_output = {}
    errors = []
    generated = 0
    for mf in manifest_files:
        outpath = output_path_for(str(mf))
        if os.path.exists(outpath) and not overwrite and os.path.getmtime(outpath) >= os.path.getmtime(mf):
            logger.info(f"SKIP (up to date): {outpath}")
        else:
            try:
                process_file(str(mf))
                generated += 1
                logger.info(f"OK: {outpath}")
            except Exception as e:
                logger.error(f"FAIL: {mf} — {e}")
                errors.append((mf, e))
                continue
        path_to_output[str(mf.resolve())] = str(Path(outpath).resolve())

    if errors:
        summary = "\n".join(f"  - {mf}: {e}" for mf, e in errors)
        raise RuntimeError(
            f"{len(errors)}/{len(manifest_files)} manifest(s) failed to process:\n{summary}"
        )

    output_yaml = yaml_path.parent / f"{yaml_path.stem}_randomorder{yaml_path.suffix}"
    if output_yaml.exists() and generated == 0:
        logger.info(f"No new manifests generated and output YAML exists, leaving unchanged: {output_yaml}")
        return

    _update_manifest_paths_in_yaml(cfg, path_to_output)

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Saved updated YAML config: {output_yaml}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Randomly permute User text/audio turn order in train JSONL files."
    )
    parser.add_argument(
        "--root",
        default="/data-server/datasets/audio/nemo",
        help="Root directory to search (default: /data-server/datasets/audio/nemo)",
    )
    parser.add_argument(
        "--yaml",
        default=None,
        help="YAML config listing manifests: randomize each and write an updated "
        "<name>_randomorder.yaml (like shard_manifest.py). Overrides --root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing _randomorder.jsonl files (default: skip them)",
    )
    args = parser.parse_args()

    if args.yaml:
        randomize_from_yaml(args.yaml, overwrite=args.overwrite)
        return

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
