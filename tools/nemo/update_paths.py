import argparse
from tqdm import tqdm
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset

def update_path_in_manifest(input_manifest, str_in, str_out):
    nemo_datset = NemoDataset()
    data_type = nemo_datset.load(input_manifest)
    for row in tqdm(nemo_datset):
        for turn in row.turns:
            if turn.turn_type=="audio":
                turn.value = turn.value.replace(str_in, str_out)
    nemo_datset.save(input_manifest, data_type=data_type)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Update audio file paths in a NeMo manifest."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input NeMo manifest file."
    )
    parser.add_argument(
        "str_in",
        type=str,
        help="Substring in the current paths to be replaced."
    )
    parser.add_argument(
        "str_out",
        type=str,
        help="Substring to replace with."
    )

    args = parser.parse_args()

    print(f"Updating paths in {args.input_path} from {args.str_in} to {args.str_out}")
    if Path(args.input_path).is_dir():
        for manifest in Path(args.input_path).rglob("*.jsonl"):
            print(f"Updating manifest: {manifest}")
            update_path_in_manifest(manifest, args.str_in, args.str_out)
    else:
        update_path_in_manifest(args.input_path, args.str_in, args.str_out)
    print(f"Done!")