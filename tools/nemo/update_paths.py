import argparse
from tqdm import tqdm
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset

def update_path_in_manifest(manifest_path, str_in, str_out):
    nemo_dataset = NemoDataset()
    data_type = nemo_dataset.load(str(manifest_path))

    for row in tqdm(nemo_dataset, desc=f"Updating {manifest_path.name}"):
        for audio_turn in row.get_audio_turns():
            audio_turn.value = audio_turn.value.replace(str_in, str_out)

    nemo_dataset.save(str(manifest_path), data_type=data_type)

def process_path(path, str_in, str_out, recursive=False):
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
            update_path_in_manifest(manifest, str_in, str_out)
    elif path.is_file() and path.suffix == ".jsonl":
        update_path_in_manifest(path, str_in, str_out)
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
    parser.add_argument("--str_in", help="Substring to replace in paths.")
    parser.add_argument("--str_out", help="Replacement substring.")
    parser.add_argument("--recursive", default=False, action="store_true", help="If input path is a folder, recursively search for jsonl")

    args = parser.parse_args()

    for input_path in args.input_paths:
        process_path(input_path, args.str_in, args.str_out, recursive=args.recursive)
