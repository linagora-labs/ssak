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

DATASET_NAME = "nutshell"
SPLITS = ["train", "dev", "test"]

_PROMPTS = [
    "Provide the abstract of this scientific conference presentation.",
    "Summarize the research presented in this talk.",
    "What research problem does this paper address in the audio?",
    "Write the abstract for this academic conference talk.",
    "Briefly describe the contributions of the research paper given in the clip.",
    "Summarize the key findings presented in this scientific talk.",
    "What are the main results discussed in this paper presentation?",
    "Provide a concise summary of the research presentation provided in the audio.",
]


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def decode_audio(hf_audio: dict) -> tuple[np.ndarray, int]:
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


def load_source_dataset(cache_dir: str | None, parquet_dir: str | None):
    if parquet_dir:
        logger.info(f"Loading parquet files from: {parquet_dir}")
        data_files = {}
        for split in SPLITS:
            files = sorted(glob.glob(os.path.join(parquet_dir, split, f"{split}_*.parquet")))
            if files:
                data_files[split] = files
        if not data_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}/{{train,dev,test}}/")
        return datasets.load_dataset("parquet", data_files=data_files)
    logger.info("Loading maikezu/nutshell"
                + (f" from cache_dir={cache_dir}" if cache_dir else ""))
    return datasets.load_dataset("maikezu/nutshell", cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert nutshell to NeMo manifest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir.")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="Path to folder containing train/, dev/, test/ subdirs with parquet files.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--raw-manifest-path", type=str, default=None,
                        help="Override raw manifest output folder (with custom_metadata).")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override the metadata-free manifest output folder.")
    parser.add_argument("--audio-path", type=str, default=None,
                        help="Override raw audio output folder.")
    parser.add_argument("--min-duration", type=float, default=120.0,
                        help="Skip rows with audio shorter than this (seconds). Default: 60.")
    parser.add_argument("--nemo-only", action="store_true",
                        help="Regenerate only nemo manifests (manifest-path) from existing raw manifests. "
                             "Skips audio extraction entirely.")
    args = parser.parse_args()

    if args.raw_manifest_path is None:
        args.raw_manifest_path = f"{os.environ['DATA_DIR']}/raw/summary/en/nutshell"
    if args.manifest_path is None:
        args.manifest_path = f"{os.environ['DATA_DIR']}/nemo/summary/en/nutshell"
    if args.audio_path is None:
        args.audio_path = f"{os.environ['DATA_DIR']}/raw/summary/en/nutshell/audios"

    RAW_MANIFEST_PATH = Path(args.raw_manifest_path)
    MANIFEST_PATH = Path(args.manifest_path)
    AUDIO_PATH = Path(args.audio_path)

    RAW_MANIFEST_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.mkdir(parents=True, exist_ok=True)

    if args.nemo_only:
        for split in SPLITS:
            manifest_file_raw = RAW_MANIFEST_PATH / f"{split}.jsonl"
            manifest_file = MANIFEST_PATH / f"{split}.jsonl"
            if not manifest_file_raw.exists():
                logger.info(f"[{split}] raw manifest not found: {manifest_file_raw}, skipping")
                continue
            if manifest_file.exists() and not args.force:
                logger.info(f"[{split}] nemo manifest already exists, skipping: {manifest_file}")
                continue

            raw = NemoDataset(name=DATASET_NAME)
            raw.load(str(manifest_file_raw))
            out = NemoDataset(name=DATASET_NAME)
            for row in raw.dataset:
                if row.duration is not None and row.duration < args.min_duration:
                    logger.info(f"[{split}] {row.id}: skipping, duration {row.duration:.1f}s < {args.min_duration}s")
                    continue
                prompt_turn = NemoTurn(role="User", value=random.choice(_PROMPTS), turn_type="text")
                out.append(NemoDatasetRow(
                    id=row.id,
                    dataset_name=DATASET_NAME,
                    split=row.split,
                    language=row.language,
                    turns=[prompt_turn] + row.turns,
                ))
            out.save(manifest_file)
            logger.info(f"[{split}] wrote {len(out.dataset)} rows → {manifest_file}")
        return

    AUDIO_PATH.mkdir(parents=True, exist_ok=True)

    ds = load_source_dataset(args.cache_dir, args.parquet_dir)

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
            video_path = row.get("video_path", "")
            uid = sanitize(video_path.rsplit("/", 1)[-1].removesuffix(".mp4")) if video_path else sanitize(str(id(row)))

            try:
                arr, sr = decode_audio(row["audio"])
            except Exception as e:
                logger.warning(f"[{split}] {uid}: audio decode failed ({e}), skipping")
                continue

            duration = len(arr) / sr
            if duration < args.min_duration:
                logger.info(f"[{split}] {uid}: skipping, duration {duration:.1f}s < {args.min_duration}s")
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
                    "video_path":  video_path,
                    "conference":  row.get("conference"),
                    "year":        row.get("year"),
                    "duration":    row.get("duration"),
                    "sr":          row.get("sr"),
                    "abstract":    abstract,
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
