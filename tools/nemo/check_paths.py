import argparse
from tqdm import tqdm
from pathlib import Path
import json
from ssak.utils.nemo_dataset import NemoDataset

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
