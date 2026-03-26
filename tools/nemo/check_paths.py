import argparse
import os
import re
from tqdm import tqdm
from pathlib import Path
import json
import yaml
from ssak.utils.nemo_dataset import NemoDataset


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


def process_yaml(yaml_path, all_paths, show_errors):
    """Load a YAML input_cfg file and check all manifest paths it references."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    manifest_paths = extract_manifest_paths(cfg)
    print(f"Found {len(manifest_paths)} manifest paths in {yaml_path}")
    for manifest_str in manifest_paths:
        try:
            resolved = resolve_oc_env(manifest_str)
        except ValueError as e:
            if show_errors:
                print(f"!!! {e}")
            continue
        manifest = Path(resolved)
        if not manifest.exists():
            print(f"Missing manifest: {resolved}")
            continue
        try:
            check_manifest(manifest, all_paths)
        except Exception as e:
            if show_errors:
                print(f"!!! Failed while processing {manifest} ({str(e)})")

def check_manifest(manifest_path, all_paths):
    if all_paths:
        nemo_dataset = NemoDataset()
        data_type = nemo_dataset.load(str(manifest_path))
        for row in tqdm(nemo_dataset, desc=f"Checking {manifest_path.name}"):
            for audio_turn in row.get_audio_turns():
                if not Path(audio_turn.value).exists():
                    print(f"Invalid path: {audio_turn.value} in manifest: {manifest_path}")
    else:
        with open(manifest_path, "r") as f:
            data = json.loads(f.readline())
        audio_turn = [i for i in data["conversations"] if i["type"] == "audio"][0]
        if not Path(audio_turn["value"]).exists():
            print(f"Invalid path: {audio_turn['value']} in manifest: {manifest_path}")

def process_path(path, recursive=False, all_paths=False, show_errors=False):
    path = Path(path)
    if path.is_dir():
        if recursive:
            manifests = sorted(path.rglob("*.jsonl"))
            print("Recursively checking directories")
        else:
            manifests = sorted(path.glob("*.jsonl"))
        if not manifests:
            print(f"No .jsonl files found in directory: {path}")
        for manifest in manifests:
            try:
                check_manifest(manifest, all_paths)
            except Exception as e:
                if show_errors:
                    print(f"!!! Failed while processing {manifest} ({str(e)})")
    elif path.is_file() and path.suffix in (".yaml", ".yml"):
        process_yaml(path, all_paths, show_errors)
    elif path.is_file() and path.suffix == ".jsonl":
        check_manifest(path, all_paths)
    else:
        print(f"Skipping invalid path: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update audio file paths in one or more NeMo manifest (.jsonl) files."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more files or directories containing .jsonl manifests."
    )
    parser.add_argument("--recursive", default=False, action="store_true", help="If input path is a folder, recursively search for jsonl")
    parser.add_argument("--all", default=False, action="store_true", help="Check all paths")
    parser.add_argument("--show_errors", default=False, action="store_true", help="Show errors")

    args = parser.parse_args()

    for input_path in args.input_paths:
        process_path(input_path, recursive=args.recursive, all_paths=args.all, show_errors=args.show_errors)
