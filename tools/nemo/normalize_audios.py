import argparse
import logging
import shutil
from pathlib import Path
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize audios from a dataset if needed. Target audio is monochannel, 16Khz wav format.")
    parser.add_argument("input_manifest", help="Input manifest file or directory containing manifests", type=str)
    parser.add_argument("output_wav_folder", help="The folder to save the normalized audios", type=str)
    parser.add_argument("--num_threads", help="Number of threads to use for normalizing audios", type=int, default=8)
    parser.add_argument("--one_segment_per_audio", help="If set, one segment per audio is created", action="store_true", default=False)
    parser.add_argument("--relative-to", help="Strip this prefix from audio paths to preserve directory structure (e.g. /data/dataset)")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern to filter manifest files when using a directory (default: *.jsonl)")
    args = parser.parse_args()

    # Collect manifest files
    manifest_path = Path(args.input_manifest)
    if manifest_path.is_dir():
        manifest_files = sorted(str(p) for p in manifest_path.rglob(args.pattern))
        logger.info(f"Found {len(manifest_files)} manifest files matching '{args.pattern}' in {args.input_manifest}")
        if not manifest_files:
            logger.warning(f"No files matching '{args.pattern}' found in directory")
            exit(0)
    else:
        manifest_files = [args.input_manifest]

    for mf in manifest_files:
        logger.info(f"Processing {mf}")
        nemo_dataset = NemoDataset()
        data_type = nemo_dataset.load(mf)
        if args.one_segment_per_audio:
            nemo_dataset.extract_one_segment_per_audio(args.output_wav_folder, target_sample_rate=16000, target_extension="wav", num_workers=args.num_threads, relative_to=args.relative_to)
        else:
            nemo_dataset.normalize_audios(args.output_wav_folder, target_sample_rate=16000, target_extension="wav", num_workers=args.num_threads, relative_to=args.relative_to)
        shutil.move(mf, mf + ".original")
        nemo_dataset.save(mf, data_type=data_type)
        logger.info(f"Saved {mf} (original backed up as {mf}.original)")
