import argparse
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset

def update_path(input_manifest, str_in, str_out):
    nemo_datset = NemoDataset()
    data_type = nemo_datset.load(input_manifest)
    for row in tqdm(nemo_datset):
        row.audio_filepath = row.audio_filepath.replace(str_in, str_out)
    nemo_datset.save(input_manifest, type=data_type)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Update audio file paths in a NeMo manifest."
    )
    parser.add_argument(
        "input_manifest",
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

    update_path(args.input_manifest, args.str_in, args.str_out)