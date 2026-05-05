import argparse
import itertools
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from ssak.utils.nemo_dataset import resolve_manifest_paths

class TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        tqdm.write(self.format(record))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmHandler())

STAT_KEYS = ("total", "missing", "unreadable", "duration_mismatch", "segment_out_of_bounds", "wrong_channels", "wrong_sample_rate", "invalid_field", "wrong_last_turn", "long_text", "ok")

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


def empty_stats():
    return {k: 0 for k in STAT_KEYS} | {"errors": []}


def total_errors(stats):
    return sum(stats[k] for k in STAT_KEYS if k not in ("total", "ok"))


def empty_overall():
    return empty_stats() | {"manifests_checked": 0, "manifests_missing": 0}


def merge_stats(dst, src):
    for k in STAT_KEYS:
        dst[k] += src[k]
    dst["errors"].extend(src["errors"])
    dst.setdefault("bad_rows", []).extend(src.get("bad_rows", []))


def merge_overall(dst, src):
    merge_stats(dst, src)
    dst["manifests_checked"] += src.get("manifests_checked", 0)
    dst["manifests_missing"] += src.get("manifests_missing", 0)


def resolve_oc_env(path_str):
    """Resolve ${oc.env:VAR_NAME} patterns using environment variables."""
    def replacer(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set (needed for path: {path_str})")
        return value
    return re.sub(r'\$\{oc\.env:([^}]+)\}', replacer, path_str)


def extract_manifest_paths(cfg):
    """Recursively extract all manifest_filepath values from a nested input_cfg structure."""
    paths = []
    if isinstance(cfg, list):
        for entry in cfg:
            if isinstance(entry, dict):
                if "input_cfg" in entry:
                    paths.extend(extract_manifest_paths(entry["input_cfg"]))
                if "manifest_filepath" in entry:
                    paths.append(entry["manifest_filepath"])
    return paths


def _check_audio_fields(path, duration, offset):
    """Return number of invalid fields found."""
    errors = 0
    if duration is None:
        logger.error(f"{path}: missing duration")
        errors += 1
    elif not isinstance(duration, (int, float)):
        logger.error(f"{path}: duration is not a float: {duration!r}")
        errors += 1
    if offset is not None and not isinstance(offset, (int, float)):
        logger.error(f"{path}: offset is not a float: {offset!r}")
        errors += 1
    return errors


def extract_audio_entries(row):
    """Extract (path, duration, offset) tuples from a parsed JSON row (ASR or multiturn)."""
    if "conversations" in row:
        return [
            (t["value"], t.get("duration"), t.get("offset"))
            for t in row["conversations"] if t.get("type") == "audio"
        ]
    elif "audio_filepath" in row:
        return [(row["audio_filepath"], row.get("duration"), row.get("offset"))]
    return []


def _check_long_text(text, max_text_length, label, stats, row_errors):
    """If text exceeds max_text_length characters, record a long_text error."""
    if not max_text_length or not isinstance(text, str):
        return
    if len(text) > max_text_length:
        logger.error(f"{label}: text length {len(text)} exceeds max_text_length={max_text_length}")
        stats["long_text"] += 1
        stats["errors"].append({
            "status": "long_text", "path": label,
            "expected": f"<= {max_text_length} chars", "actual": len(text),
        })
        if "long_text" not in row_errors:
            row_errors.append("long_text")


def check_row(row, stats, max_text_length=None):
    """Run field validation and last-turn check, updating stats in place. Returns list of error types."""
    row_errors = []
    row_id = row.get("id", "<no-id>")
    if "conversations" in row:
        turns = row["conversations"]
        if turns and turns[-1].get("from") != "Assistant":
            logger.error(f"Last turn is not from Assistant: {turns[-1].get('from')!r} (id={row.get('id')})")
            stats["wrong_last_turn"] += 1
            row_errors.append("wrong_last_turn")
        for t in turns:
            if t.get("type") == "audio":
                n = _check_audio_fields(t["value"], t.get("duration"), t.get("offset"))
                stats["invalid_field"] += n
                if n:
                    row_errors.append("invalid_field")
            elif t.get("type") == "text":
                _check_long_text(t.get("value"), max_text_length, f"row={row_id}", stats, row_errors)
    elif "audio_filepath" in row:
        n = _check_audio_fields(row["audio_filepath"], row.get("duration"), row.get("offset"))
        stats["invalid_field"] += n
        if n:
            row_errors.append("invalid_field")
        _check_long_text(row.get("text"), max_text_length, row.get("audio_filepath", f"row={row_id}"), stats, row_errors)
    return row_errors


def parallel_map(func, items, num_threads, desc):
    """Run func on items, optionally threaded. Yields results."""
    if num_threads > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(func, item): item for item in items}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                yield future.result()
    else:
        yield from (func(item) for item in items)


def probe_file(audio_path):
    """Check existence and read metadata via soundfile. Returns (path, info_dict)."""
    import soundfile as sf
    try:
        info = sf.info(audio_path)
        return audio_path, {
            "status": "ok",
            "duration": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }
    except FileNotFoundError:
        return audio_path, {"status": "missing"}
    except Exception as e:
        return audio_path, {"status": "unreadable", "error": str(e)}


def _iter_rows(manifest_path, num_rows, label=None):
    """Yield parsed JSON rows from a manifest. num_rows=None means all rows."""
    label = label or manifest_path
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = itertools.islice(f, num_rows)
        if num_rows is None:
            lines = tqdm(f, desc=f"Reading {label}", leave=False)
        for i, line in enumerate(lines):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{label} line {i+1} ({line}): could not parse JSON: {e}")


def check_manifest(manifest_path, num_rows=None, disable_audio_check=False,
                   disable_channel_check=False, disable_rate_check=False,
                   duration_tolerance=0.5, num_threads=1, label=None,
                   max_text_length=None, max_errors=10):
    manifest_path = Path(manifest_path)
    label = label or str(manifest_path)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {label}")
        return empty_stats()

    def hit_error_limit():
        return max_errors and max_errors > 0 and total_errors(stats) >= max_errors

    # Pass 1: Read rows, row-level checks, collect unique audio paths
    stats = empty_stats()
    row_infos = []  # (row_dict, audio_entries, row_errors)
    unique_paths = set()
    total = 0
    stopped_early = False
    for row in _iter_rows(manifest_path, num_rows, label):
        row_errors = check_row(row, stats, max_text_length=max_text_length)
        audio_entries = extract_audio_entries(row)
        for path, dur, offset in audio_entries:
            unique_paths.add(path)
            total += 1
        row_infos.append((row, audio_entries, row_errors))
        if hit_error_limit():
            logger.warning(f"{label}: reached --max_errors={max_errors}, stopping checks for this manifest")
            stopped_early = True
            break
    stats["total"] = total

    # Pass 2: Probe unique audio files → metadata cache
    if disable_audio_check:
        file_info = {}
    else:
        file_info = {p: info for p, info in parallel_map(
            probe_file, list(unique_paths), num_threads, f"Probing {label}"
        )}

    # Pass 3: Check each row's audio entries against file_info, collect bad rows
    bad_rows = []
    for row, audio_entries, row_errors in row_infos:
        if hit_error_limit():
            if not stopped_early:
                logger.warning(f"{label}: reached --max_errors={max_errors}, stopping checks for this manifest")
                stopped_early = True
            break
        row_error_types = set(row_errors)
        for path, expected_dur, expected_offset in audio_entries:
            if disable_audio_check:
                stats["ok"] += 1
                continue
            info = file_info[path]
            if info["status"] != "ok":
                stats[info["status"]] += 1
                err_entry = {"status": info["status"], "path": path}
                if "error" in info:
                    err_entry["error"] = info["error"]
                stats["errors"].append(err_entry)
                row_error_types.add(info["status"])
                continue

            entry_ok = True
            if expected_offset is not None:
                offset = expected_offset
                if (offset + (expected_dur or 0)) > info["duration"] + duration_tolerance:
                    segment_end = offset + (expected_dur or 0)
                    stats["segment_out_of_bounds"] += 1
                    stats["errors"].append({
                        "status": "segment_out_of_bounds", "path": path,
                        "expected": f"offset={offset}+dur={expected_dur}={round(segment_end, 3)}",
                        "actual": round(info["duration"], 3),
                    })
                    row_error_types.add("segment_out_of_bounds")
                    entry_ok = False
            elif expected_dur is not None and expected_dur > info["duration"] + duration_tolerance:
                stats["duration_mismatch"] += 1
                stats["errors"].append({
                    "status": "duration_mismatch", "path": path,
                    "expected": expected_dur, "actual": round(info["duration"], 3),
                })
                row_error_types.add("duration_mismatch")
                entry_ok = False

            if not disable_channel_check and info.get("channels") != EXPECTED_CHANNELS:
                stats["wrong_channels"] += 1
                stats["errors"].append({
                    "status": "wrong_channels", "path": path,
                    "expected": EXPECTED_CHANNELS, "actual": info.get("channels"),
                })
                row_error_types.add("wrong_channels")
                entry_ok = False

            if not disable_rate_check and info.get("sample_rate") != EXPECTED_SAMPLE_RATE:
                stats["wrong_sample_rate"] += 1
                stats["errors"].append({
                    "status": "wrong_sample_rate", "path": path,
                    "expected": EXPECTED_SAMPLE_RATE, "actual": info.get("sample_rate"),
                })
                row_error_types.add("wrong_sample_rate")
                entry_ok = False

            if entry_ok:
                stats["ok"] += 1

        if row_error_types:
            row["_errors"] = sorted(row_error_types)
            row["_manifest"] = label
            bad_rows.append(row)
    stats["bad_rows"] = bad_rows

    extra = ""
    if stats["invalid_field"]:
        extra += f", {stats['invalid_field']} invalid fields"
    if stats["wrong_last_turn"]:
        extra += f", {stats['wrong_last_turn']} wrong last turn"
    if stats["long_text"]:
        extra += f", {stats['long_text']} long texts"

    if disable_audio_check:
        logger.info(
            f"{label}: {stats['total']} audio refs ({len(unique_paths)} unique) — "
            f"audio checks disabled{extra}"
        )
    else:
        msg = (
            f"{label}: {stats['total']} audio refs ({len(unique_paths)} unique) — "
            f"{stats['missing']} missing, {stats['unreadable']} unreadable, "
            f"{stats['duration_mismatch']} duration mismatches, "
            f"{stats['segment_out_of_bounds']} segments OOB"
        )
        if not disable_channel_check:
            msg += f", {stats['wrong_channels']} wrong channels"
        if not disable_rate_check:
            msg += f", {stats['wrong_sample_rate']} wrong sample rate"
        logger.info(msg + extra)
    return stats


# ---------------------------------------------------------------------------
# YAML / directory / path dispatch
# ---------------------------------------------------------------------------

def process_path(path, recursive=False, **kwargs):
    """Resolve input path (file, dir, or YAML) and check all found manifests."""
    manifests = resolve_manifest_paths(path, recursive=recursive)
    if not manifests:
        logger.warning(f"No manifest files found for: {path}")
        return None

    # Compute common prefix for shorter labels
    str_paths = [str(p) for p in manifests]
    base = os.path.commonpath(str_paths) if len(str_paths) > 1 else None

    overall = empty_overall()
    pbar = tqdm(manifests, desc="Manifests", unit="file")
    for mf in pbar:
        label = os.path.relpath(mf, base) if base else str(mf)
        pbar.set_postfix_str(label)
        stats = check_manifest(mf, label=label, **kwargs)
        overall["manifests_checked"] += 1
        merge_stats(overall, stats)

    return overall


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ERROR_FMT = {
    "missing": lambda e: f"  MISSING: {e['path']}",
    "unreadable": lambda e: f"  UNREADABLE: {e['path']} ({e.get('error', '')})",
    "duration_mismatch": lambda e: f"  DURATION MISMATCH: {e['path']} (expected={e['expected']}s, actual={e['actual']}s)",
    "segment_out_of_bounds": lambda e: f"  SEGMENT OOB: {e['path']} (segment_end={e['expected']}, file_duration={e['actual']}s)",
    "wrong_channels": lambda e: f"  WRONG CHANNELS: {e['path']} (expected={e['expected']}, actual={e['actual']})",
    "wrong_sample_rate": lambda e: f"  WRONG SAMPLE RATE: {e['path']} (expected={e['expected']}, actual={e['actual']})",
    "long_text": lambda e: f"  LONG TEXT: {e['path']} (length={e['actual']}, expected {e['expected']})",

}


def print_summary(overall):
    print("\n=== Overall Summary ===")
    print(f"Manifests checked:    {overall['manifests_checked']}")
    if overall["manifests_missing"] > 0:
        print(f"Manifests missing:    {overall['manifests_missing']}")
    print(f"Total audio refs:     {overall['total']}")
    print(f"OK:                   {overall['ok']}")
    for key, label in [("missing", "Missing files"), ("unreadable", "Unreadable files"),
                       ("duration_mismatch", "Duration mismatches"), ("segment_out_of_bounds", "Segments OOB"),
                       ("wrong_channels", "Wrong channels"), ("wrong_sample_rate", "Wrong sample rate"),
                       ("invalid_field", "Invalid fields"),
                       ("wrong_last_turn", "Wrong last turn"),
                       ("long_text", "Long texts")]:
        if overall[key] > 0:
            print(f"{label + ':':22}{overall[key]}")

    if overall["errors"]:
        print(f"\n--- Errors ({len(overall['errors'])}) ---")
        seen = set()
        for err in overall["errors"]:
            key = (err["status"], err["path"])
            if key not in seen:
                seen.add(key)
                print(ERROR_FMT[err["status"]](err))
    else:
        print("\nNo errors found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check audio files referenced in NeMo YAML training configs or JSONL manifests.\n\n"
                    "By default, checks all rows and validates audio (existence, readability,\n"
                    "duration/offset, mono channel, 16kHz sample rate).\n"
                    "  --rows                    Check only 1 row\n"
                    "  --rows N                  Check N rows\n"
                    "  --disable_audio_check     Skip all audio file checks\n"
                    "  --disable_channel_check   Skip mono-channel check\n"
                    "  --disable_rate_check      Skip 16kHz sample-rate check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_paths", nargs="+", help="YAML configs, JSONL manifests, or directories.")
    parser.add_argument("--rows", nargs="?", const=1, type=int, default=None, help="Limit rows to check. No flag = all rows, --rows = 1 row, --rows N = N rows.")
    parser.add_argument("--disable_audio_check", action="store_true", default=False, help="Disable all audio file checks (existence, readability, duration, channels, sample rate).")
    parser.add_argument("--disable_channel_check", action="store_true", default=False, help=f"Disable check that audio files are mono ({EXPECTED_CHANNELS} channel).")
    parser.add_argument("--disable_rate_check", action="store_true", default=False, help=f"Disable check that audio files have sample rate {EXPECTED_SAMPLE_RATE} Hz.")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recursively search directories for JSONL files.")
    parser.add_argument("--duration_tolerance", type=float, default=0.5, help="Tolerance in seconds for duration checks (default: 0.5).")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel checking.")
    parser.add_argument("--output_errors", type=str, default=None, help="Write problematic rows to this JSONL file.")
    parser.add_argument("--max_text_length", type=int, default=5000, help="Flag rows whose text exceeds this many characters (default: 5000). Set to 0 to disable. Checks `text` for ASR rows and text-type turns in conversations.")
    parser.add_argument("--max_errors", type=int, default=10, help="Stop checking a manifest once this many errors have been found (default: 10). Set to 0 to disable.")

    args = parser.parse_args()

    check_kwargs = dict(
        num_rows=args.rows,
        disable_audio_check=args.disable_audio_check,
        disable_channel_check=args.disable_channel_check,
        disable_rate_check=args.disable_rate_check,
        duration_tolerance=args.duration_tolerance,
        num_threads=args.num_threads,
        max_text_length=args.max_text_length,
        max_errors=args.max_errors,
    )

    combined = empty_overall()
    for input_path in args.input_paths:
        result = process_path(input_path, recursive=args.recursive, **check_kwargs)
        if result:
            merge_overall(combined, result)

    print_summary(combined)

    if args.output_errors and combined.get("bad_rows"):
        with open(args.output_errors, "w", encoding="utf-8") as f:
            for row in combined["bad_rows"]:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Wrote {len(combined['bad_rows'])} problematic rows to {args.output_errors}")

    error_keys = [k for k in STAT_KEYS if k not in ("total", "ok")]
    if any(combined[k] for k in error_keys):
        sys.exit(1)
