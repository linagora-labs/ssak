
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from prompts_captioning import SHORT_CAPTION_PROMPTS as prompts
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

random.seed(42)
np.random.seed(42)

# AudioCaps captions are short (~8 words) -> short/neutral instruction pool (see
# prompts_captioning.py). One is picked at random per row by ``set_context_if_none``
# to build the context version of the manifest.

# Columns read from the parquet shards. The heavy embedded ``audio`` column is
# deliberately left out: we reference the already-downloaded wavs instead.
# (``audio_length`` is skipped too — it is a sample count at the original sr, not
# seconds; durations are computed from the actual wav.)
_COLUMNS = ["audiocap_id", "youtube_id", "start_time", "caption"]

# Parquet split prefix -> output split name (AudioCaps ships validation/ as the dev split).
_SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def _iter_rows(data_dir: Path, split_prefix: str):
    """Yield row dicts (selected columns only) from all shards of one split."""
    shards = sorted(data_dir.glob(f"{split_prefix}-*.parquet"))
    for shard in shards:
        table = pq.read_table(shard, columns=_COLUMNS)
        for row in table.to_pylist():
            yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AudioCaps (sound captioning) to NeMo format")
    parser.add_argument("--raw_input", type=str, default=None,
                        help="AudioCaps HF repo folder (contains data/ and audio_files/).")
    parser.add_argument("--audio_subdir", type=str, default="audio_files",
                        help="Sub-folder of raw_input holding {youtube_id}.wav files.")
    parser.add_argument("--resampled_dir", type=str, default=None,
                        help="Root where converted 16 kHz mono FLACs are written (raw sub-structure is preserved under it).")
    parser.add_argument("--nemo_no_context", type=str, default=None,
                        help="Output folder for the no-context NeMo manifests.")
    parser.add_argument("--nemo_context", type=str, default=None,
                        help="Output folder for the context NeMo manifests.")
    parser.add_argument("--no_resample", action="store_true",
                        help="Reference the original wavs as-is instead of normalizing to 16 kHz mono FLAC.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers for audio normalization.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many rows per split (debug).")
    args = parser.parse_args()

    data_folder = os.environ.get("DATA_FOLDER", "/data-server/datasets/audio")
    if args.raw_input is None:
        args.raw_input = f"{data_folder}/raw/sounds/AudioCaps"
    if args.resampled_dir is None:
        args.resampled_dir = f"{data_folder}/converted_audios/sounds/AudioCaps"
    if args.nemo_no_context is None:
        args.nemo_no_context = f"{data_folder}/nemo/sounds/audio-captioning/en/nocontext/AudioCaps"
    if args.nemo_context is None:
        args.nemo_context = f"{data_folder}/nemo/sounds/audio-captioning/en/context/AudioCaps"

    raw_input = Path(args.raw_input)
    data_dir = raw_input / "data"
    audio_dir = raw_input / args.audio_subdir
    resampled_dir = Path(args.resampled_dir)
    nemo_no_context = Path(args.nemo_no_context)
    nemo_context = Path(args.nemo_context)
    for d in (resampled_dir, nemo_no_context, nemo_context):
        d.mkdir(parents=True, exist_ok=True)

    for split_prefix, split_name in _SPLITS.items():
        dataset = NemoDataset(name="AudioCaps")
        n_total = n_missing = n_kept = 0

        for row in _iter_rows(data_dir, split_prefix):
            if args.limit is not None and n_total >= args.limit:
                break
            n_total += 1

            youtube_id = row["youtube_id"]
            src = audio_dir / f"{youtube_id}.wav"
            if not src.exists():
                n_missing += 1
                continue

            # Duration is taken from the source and stays valid after normalization
            # (resampling / channel down-mix does not change duration).
            info = sf.info(str(src))
            dataset.append(NemoDatasetRow(
                id=str(row["audiocap_id"]),
                turns=[
                    NemoTurn(role="User", value=str(src), turn_type="audio",
                             duration=round(info.frames / info.samplerate, 3)),
                    NemoTurn(role="Assistant", value=row["caption"], turn_type="text"),
                ],
                dataset_name="AudioCaps",
                language="en",
                split=split_name,
                custom_metadata={"youtube_id": youtube_id, "start_time": row["start_time"]},
            ))
            n_kept += 1
            if n_kept % 1000 == 0:
                print(f"[{split_name}] processed {n_kept} kept ({n_missing} missing audio)...")

        # Normalize to 16 kHz mono FLAC, but keep already-good wav/flac files untouched.
        if not args.no_resample:
            dataset.normalize_audios(
                str(resampled_dir), target_sample_rate=16000, target_extension="flac",
                accepted_extensions=["wav", "flac"], num_workers=args.num_workers,
                relative_to=str(raw_input))

        # No-context manifest, then a context version with a random prompt per row.
        dataset.save(nemo_no_context / f"{split_name}.jsonl")
        dataset.set_context_if_none(prompts)
        dataset.save(nemo_context / f"{split_name}.jsonl")

        cover = 100 * n_kept / n_total if n_total else 0
        print(f"[{split_name}] kept {n_kept}/{n_total} ({cover:.1f}% audio coverage, "
              f"{n_missing} missing) -> {nemo_context / (split_name + '.jsonl')}")

    print("Done")
