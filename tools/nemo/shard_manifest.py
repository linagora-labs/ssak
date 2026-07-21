import argparse
import logging
import math
import os
import re
import shutil
from pathlib import Path

import yaml

from ssak.utils.nemo_dataset import resolve_manifest_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def shard_single_manifest(manifest_path, shard_size=1000, min_lines=0, shuffle=False, seed=42, force=False):
    """Shard a single JSONL manifest into smaller files.

    Returns the NeMo glob pattern for the shards, or None if the manifest was
    skipped (empty or below min_lines threshold).
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        logger.warning(f"Manifest is empty, nothing to shard: {manifest_path}")
        return None

    if total < min_lines:
        logger.info(f"SKIP (too small, {total} lines < {min_lines} threshold): {manifest_path}")
        return None

    num_shards = math.ceil(total / shard_size)
    output_dir = manifest_path.parent / f"{manifest_path.stem}_shards"
    glob_pattern = str(output_dir / f"{manifest_path.stem}__OP_0..{num_shards - 1}_CL_.jsonl")

    if output_dir.exists() and any(output_dir.iterdir()):
        if force:
            logger.info(f"Removing existing shard folder: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            existing_shards = sorted(output_dir.glob(f"{manifest_path.stem}_*.jsonl"))
            if len(existing_shards) == num_shards:
                logger.info(f"SKIP (up to date, the {num_shards} expected shards already exists): {output_dir}")
                return glob_pattern
            raise FileExistsError(
                f"Output folder already exists with {len(existing_shards)} shards "
                f"but {num_shards} were expected: {output_dir} (use --force to overwrite)"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(lines)

    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, total)
        shard_name = f"{manifest_path.stem}_{i}.jsonl"
        shard_path = output_dir / shard_name
        with open(shard_path, "w", encoding="utf-8") as out:
            out.writelines(lines[start:end])

    logger.info(
        f"OK: Sharded {total} lines into {num_shards} files of ~{shard_size} lines "
        f"in {output_dir}"
    )
    logger.info(f"NeMo glob pattern: {glob_pattern}")
    return glob_pattern


def _resolve_oc_env(path_str):
    """Resolve ${oc.env:VAR_NAME} patterns using environment variables."""
    def replacer(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable '{var_name}' not set (needed for: {path_str})")
        return value
    return re.sub(r'\$\{oc\.env:([^}]+)\}', replacer, path_str)


def _update_manifest_paths_in_yaml(cfg, path_to_pattern):
    """Recursively replace manifest_filepath values in a YAML config structure.

    Args:
        cfg: The parsed YAML structure (list of dicts with input_cfg / manifest_filepath).
        path_to_pattern: dict mapping resolved manifest path (str) -> NeMo glob pattern (str).
    """
    if isinstance(cfg, list):
        for entry in cfg:
            if isinstance(entry, dict):
                if "input_cfg" in entry:
                    _update_manifest_paths_in_yaml(entry["input_cfg"], path_to_pattern)
                if "manifest_filepath" in entry:
                    raw = entry["manifest_filepath"]
                    try:
                        resolved = _resolve_oc_env(raw)
                        resolved_abs = str(Path(resolved).resolve())
                        if resolved_abs in path_to_pattern:
                            entry["manifest_filepath"] = path_to_pattern[resolved_abs]
                    except ValueError:
                        pass


def shard_from_yaml(yaml_path, shard_size=1000, min_lines=0, shuffle=False, seed=42, force=False):
    """Shard all manifests referenced in a YAML config and produce an updated YAML."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    manifest_files = resolve_manifest_paths(yaml_path)
    if not manifest_files:
        # A config that references no resolvable manifest is broken, not empty — raising
        # keeps the caller (and the slurm exit code) from reporting a successful no-op.
        raise ValueError(f"No manifest files found in YAML: {yaml_path}")

    # Shard each manifest and collect the mapping from resolved path -> glob pattern
    path_to_pattern = {}
    errors = []
    for mf in manifest_files:
        try:
            pattern = shard_single_manifest(mf, shard_size=shard_size, min_lines=min_lines, shuffle=shuffle, seed=seed, force=force)
        except Exception as e:
            logger.error(f"Failed to shard {mf}: {e}")
            errors.append((mf, e))
            continue
        if pattern:
            path_to_pattern[str(mf.resolve())] = pattern

    if errors:
        summary = "\n".join(f"  - {mf}: {e}" for mf, e in errors)
        raise RuntimeError(
            f"{len(errors)}/{len(manifest_files)} manifest(s) failed to shard:\n{summary}"
        )

    # Update the YAML structure with glob patterns
    _update_manifest_paths_in_yaml(cfg, path_to_pattern)

    # The output YAML is a cheap derived artifact — always (re)write it, even when the
    # expensive shard folders already exist and were skipped above. (--force only gates
    # regenerating the shard folders themselves.)
    output_yaml = yaml_path.parent / f"{yaml_path.stem}_sharded{yaml_path.suffix}"
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Saved updated YAML config: {output_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shard large NeMo manifests into smaller JSONL files for better dataloader randomization."
    )
    parser.add_argument("input", help="Input manifest file, directory, or YAML config", type=str)
    parser.add_argument(
        "--shard-size",
        help="Maximum number of lines per shard (default: 5000)",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--no-shuffle",
        help="Disable shuffling of lines before sharding (shuffling is enabled by default, uses --seed for reproducibility)",
        dest="shuffle",
        action="store_false",
        default=True,
    )
    parser.add_argument("--seed", help="Random seed for shuffling (default: 42)", type=int, default=42)
    parser.add_argument(
        "--min-lines",
        help="Only shard manifests with at least this many lines (default: 0, shard all)",
        type=int,
        default=25000,
    )
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern to filter manifest files when using a directory (default: *.jsonl)")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recursively search directories")
    parser.add_argument("--force", action="store_true", default=False, help="Overwrite existing shard folders and output YAML")
    args = parser.parse_args()

    input_path = Path(args.input)

    # A path that does not exist is a caller mistake (typo, wrong config base), never an
    # empty-input case. Without this a bad .yaml path falls through to the directory
    # branch below, globs nothing and used to exit(0) — so the job "succeeded" having
    # sharded nothing. Fail loudly instead.
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        exit(1)

    if input_path.is_file() and input_path.suffix in (".yaml", ".yml"):
        shard_from_yaml(input_path, shard_size=args.shard_size, min_lines=args.min_lines, shuffle=args.shuffle, seed=args.seed, force=args.force)
    else:
        manifest_files = resolve_manifest_paths(args.input, pattern=args.pattern, recursive=args.recursive)
        if not manifest_files:
            logger.error(f"No manifest files found for input: {args.input}")
            exit(1)
        logger.info(f"Found {len(manifest_files)} manifest file(s)")
        errors = []
        for mf in manifest_files:
            try:
                shard_single_manifest(mf, shard_size=args.shard_size, min_lines=args.min_lines, shuffle=args.shuffle, seed=args.seed, force=args.force)
            except Exception as e:
                logger.error(f"Failed to shard {mf}: {e}")
                errors.append((mf, e))
        if errors:
            summary = "\n".join(f"  - {mf}: {e}" for mf, e in errors)
            raise RuntimeError(
                f"{len(errors)}/{len(manifest_files)} manifest(s) failed to shard:\n{summary}"
            )
