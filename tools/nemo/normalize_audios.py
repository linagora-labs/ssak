import argparse
import shutil
from ssak.utils.nemo_dataset import NemoDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize audios from a dataset if needed. Target audio is monochannel, 16Khz wav format.")
    parser.add_argument("input_manifest", help="Input manifest", type=str)
    parser.add_argument("output_wav_folder", help="The folder to save the normalized audios", type=str)
    parser.add_argument("--num_threads", help="Number of threads to use for normalizing audios", type=int, default=8)
    args = parser.parse_args()
    
    nemo_dataset = NemoDataset()
    data_type = nemo_dataset.load(args.input_manifest)
    nemo_dataset.normalize_audios(args.output_wav_folder, target_sample_rate=16000, target_extension="wav", num_workers=args.num_threads)
    shutil.move(args.input_manifest, args.input_manifest+".original")
    nemo_dataset.save(args.input_manifest, data_type=data_type)
    