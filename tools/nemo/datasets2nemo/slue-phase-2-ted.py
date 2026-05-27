import argparse
import glob
import io
import logging
import os
import random
import re
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "slue-phase-2-TED"
SPLITS = ["train", "validation", "test"]

_PROMPTS = [
    "Provide an abstract for this clip.",
    "Summarize this talk.",
    "What is this talk about?",
    "Give a brief abstract of the presentation.",
    "Write a short summary of this audio.",
    "Briefly describe what this talk covers.",
    "Provide a concise summary of this presentation.",
    "Describe the key ideas presented in this audio."
]


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def decode_audio(hf_audio: dict) -> tuple[np.ndarray, int]:
    """Decode an HF audio entry loaded with decode=False."""
    if hf_audio.get("bytes") is not None:
        arr, sr = sf.read(io.BytesIO(hf_audio["bytes"]))
    else:
        arr, sr = sf.read(hf_audio["path"])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr, int(sr)


def write_flac_if_missing(path: Path, arr: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(path, arr, sr)
    return {"path": path, "duration": len(arr) / sr}


def load_source_dataset(cache_dir: str | None, from_disk: str | None, parquet_dir: str | None):
    if parquet_dir:
        logger.info(f"Loading parquet files from: {parquet_dir}")
        data_files = {}
        for split, prefix in [("train", "train"), ("validation", "validation"), ("test", "test")]:
            files = sorted(glob.glob(os.path.join(parquet_dir, f"{prefix}-*.parquet")))
            if files:
                data_files[split] = files
        if not data_files:
            raise FileNotFoundError(f"No train/validation/test parquet files found in {parquet_dir}")
        return datasets.load_dataset("parquet", data_files=data_files)
    if from_disk:
        logger.info(f"Loading dataset from disk: {from_disk}")
        return datasets.load_from_disk(from_disk)
    logger.info("Loading asapp/slue-phase-2 (ted)"
                + (f" from cache_dir={cache_dir}" if cache_dir else ""))
    return datasets.load_dataset("asapp/slue-phase-2", "ted", cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert slue-phase-2 ted to NeMo manifest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir (reuses files already downloaded via load_dataset).")
    parser.add_argument("--from-disk", type=str, default=None,
                        help="Path to a dataset saved via save_to_disk (takes precedence over --cache-dir).")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="Path to a folder containing {train,validation,test}-*.parquet (e.g. an HF hub snapshot dir).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--raw-manifest-path", type=str, default=None,
                        help="Override raw manifest output folder (with custom_metadata).")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override the metadata-free manifest output folder.")
    parser.add_argument("--audio-path", type=str, default=None,
                        help="Override raw audio output folder.")
    args = parser.parse_args()

    if args.raw_manifest_path is None:
        args.raw_manifest_path = f"{os.environ['DATA_DIR']}/raw/summary/en/slue-phase-2-ted"
    if args.manifest_path is None:
        args.manifest_path = f"{os.environ['DATA_DIR']}/nemo/summary/en/slue-phase-2-ted"
    if args.audio_path is None:
        args.audio_path = f"{os.environ['DATA_DIR']}/raw/summary/en/slue-phase-2-ted/audios"

    RAW_MANIFEST_PATH = Path(args.raw_manifest_path)
    MANIFEST_PATH = Path(args.manifest_path)
    AUDIO_PATH = Path(args.audio_path)

    AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    RAW_MANIFEST_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.mkdir(parents=True, exist_ok=True)

    ds = load_source_dataset(args.cache_dir, args.from_disk, args.parquet_dir)

    for split in SPLITS:
        if split not in ds:
            logger.info(f"Split {split!r} not in dataset, skipping")
            continue

        manifest_file_raw = RAW_MANIFEST_PATH / f"{split}.jsonl"
        manifest_file = MANIFEST_PATH / f"{split}.jsonl"
        if manifest_file_raw.exists() and manifest_file.exists() and not args.force:
            logger.info(f"[{split}] manifests already exist, skipping: {manifest_file_raw}, {manifest_file}")
            continue

        subset = ds[split].cast_column("audio", datasets.Audio(decode=False))
        logger.info(f"[{split}] total={len(subset)}")

        out_raw = NemoDataset(name=DATASET_NAME)
        out = NemoDataset(name=DATASET_NAME)

        for row in tqdm(subset, desc=split):
            uid = sanitize(str(row["id"]))

            try:
                arr, sr = decode_audio(row["audio"])
            except Exception as e:
                logger.warning(f"[{split}] {row}: audio decode failed ({e}), skipping")
                continue

            audio = write_flac_if_missing(
                AUDIO_PATH / split / f"{uid}.flac",
                arr, sr,
            )

            abstract = row.get("abstract")
            if not abstract:
                logger.warning(f"[{split}] {uid}: no abstract, skipping")
                continue

            audio_turn = NemoTurn(role="User", value=str(audio["path"]), turn_type="audio",
                                  duration=round(audio["duration"], 3))
            summary_turn = NemoTurn(role="Assistant", value=abstract, turn_type="text")

            prompt_turn = NemoTurn(role="User", value=random.choice(_PROMPTS), turn_type="text")

            out_raw.append(NemoDatasetRow(
                id=uid,
                dataset_name=DATASET_NAME,
                split=split,
                language="en",
                turns=[audio_turn, summary_turn],
                custom_metadata={
                    "talk_id":    row.get("id"),
                    "speaker":    row.get("speaker"),
                    "title":      row.get("title"),
                    "abstract":   abstract,
                    "transcript": row.get("transcript"),
                },
            ))
            out.append(NemoDatasetRow(
                id=uid,
                dataset_name=DATASET_NAME,
                split=split,
                language="en",
                turns=[prompt_turn, audio_turn, summary_turn],
            ))


        out_raw.save(manifest_file_raw)
        out.save(manifest_file)
        logger.info(f"[{split}] wrote {len(out_raw)} rows → {manifest_file_raw}")
        logger.info(f"[{split}] wrote {len(out)} rows → {manifest_file}")


if __name__ == "__main__":
    main()
