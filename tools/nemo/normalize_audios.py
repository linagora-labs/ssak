import argparse
from tqdm import tqdm
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a list of Kaldi datasets to Nemo format")
    parser.add_argument("input_manifest", help="Input dataset", type=str)
    parser.add_argument("output_wav_folder", type=str)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()
    
    nemo_dataset = NemoDataset()
    data_type = nemo_dataset.load(args.input_manifest)
    nemo_dataset.normalize_audios(args.output_wav_folder, target_sample_rate=16000, target_extension="wav", num_workers=args.num_threads)
    nemo_dataset.save(args.input_manifest+".normalized", type=data_type)
    