import argparse
import logging
import os
import re
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "TED-LIUM"
SPLITS = ["train", "validation", "test"]

TARBALL_LAYOUT = {
    "release1": {
        "train":      ["TEDLIUM_release1/train.tar.gz"],
        "validation": ["TEDLIUM_release1/dev.tar.gz"],
        "test":       ["TEDLIUM_release1/test.tar.gz"],
    },
    "release2": {
        "train":      ["TEDLIUM_release2/train_1.tar.gz", "TEDLIUM_release2/train_2.tar.gz"],
        "validation": ["TEDLIUM_release2/dev.tar.gz"],
        "test":       ["TEDLIUM_release2/test.tar.gz"],
    },
    "release3": {
        "train":      ["TEDLIUM_release3/legacy/train_1.tar.gz", "TEDLIUM_release3/legacy/train_2.tar.gz"],
        "validation": ["TEDLIUM_release3/legacy/dev.tar.gz"],
        "test":       ["TEDLIUM_release3/legacy/test.tar.gz"],
    },
}


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def write_flac_if_missing(path: Path, arr: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(path, arr, sr)
    return {"path": path, "duration": len(arr) / sr}


def extract_tarball_if_needed(tar_path: Path, dest: Path) -> None:
    marker = dest / ".extracted"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {tar_path} → {dest}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    marker.touch()


def _maybe_trim_stm_suffix(transcript: str) -> str:
    parts = transcript.rsplit(" ", 1)
    if len(parts) > 1 and parts[-1].startswith("("):
        return parts[0]
    return transcript


def _parse_gender(label_str: str) -> str:
    g = re.split(r",|_", label_str)[-1][:-1]
    if not g or g == "<NA":
        return "unknown"
    if g == "F":
        return "female"
    if g == "M":
        return "male"
    return g


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.replace(" '", "'")
    return re.sub(r"\s+", " ", text).strip()


def iter_tarball_split(release: str, split: str, tarball_dir: Path, extract_cache: Path):
    """Yield per-utterance dicts from extracted TEDLIUM tarballs."""
    stm_files = []
    for rel in TARBALL_LAYOUT[release][split]:
        tar_path = tarball_dir / rel
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing tarball: {tar_path}")
        shard_dest = extract_cache / release / split / Path(rel).stem.replace(".tar", "")
        extract_tarball_if_needed(tar_path, shard_dest)
        stm_files.extend(sorted(shard_dest.rglob("*.stm")))

    for stm_path in stm_files:
        sph_path = stm_path.parent.parent / "sph" / (stm_path.stem + ".sph")
        if not sph_path.exists():
            alt = stm_path.with_suffix(".sph")
            if alt.exists():
                sph_path = alt
            else:
                logger.warning(f"sph not found for {stm_path}; skipping")
                continue
        try:
            segment, sr = sf.read(str(sph_path), dtype=np.int16)
        except Exception as e:
            logger.warning(f"failed to read {sph_path}: {e}; skipping")
            continue
        with open(stm_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    _fn, _channel, speaker, start, end, label, transcript = line.split(" ", 6)
                except ValueError:
                    continue
                s = int(float(start) * sr)
                e = min(int(float(end) * sr), segment.shape[0])
                yield {
                    "talk_id":       stm_path.stem,
                    "id":            "-".join([speaker, start, end, label]),
                    "text":          _maybe_trim_stm_suffix(transcript),
                    "speaker_id":    speaker,
                    "gender":        _parse_gender(label),
                    "file":          str(sph_path),
                    "samples":       segment[s:e],
                    "sampling_rate": sr,
                }


def main():
    parser = argparse.ArgumentParser(description="Convert TED-LIUM (ASR) to NeMo manifests")
    parser.add_argument("--release", type=str, nargs="+",
                        default=["release1", "release2", "release3"],
                        choices=["release1", "release2", "release3"],
                        help="TED-LIUM release(s) to convert (default: all).")
    parser.add_argument("--tarball-dir", type=str, required=True,
                        help="Folder containing TEDLIUM_release{1,2,3}/ with the raw .tar.gz archives.")
    parser.add_argument("--extract-cache", type=str, default=None,
                        help="Where to extract tarballs (default: <tarball-dir>/_extracted).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--raw-manifest-path", type=str, default=None,
                        help="Override raw manifest folder (with custom_metadata).")
    parser.add_argument("--with-nemo-manifest", action="store_true",
                        help="Also write the metadata-free NeMo ASR manifest (default: off).")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override metadata-free NeMo ASR manifest folder (with --with-nemo-manifest).")
    parser.add_argument("--audio-path", type=str, default=None,
                        help="Override extracted per-utterance FLAC output folder.")
    parser.add_argument("--keep-ignored", action="store_true",
                        help="Keep utterances marked 'ignore_time_segment_in_scoring' (default: skip).")
    args = parser.parse_args()

    tarball_dir = Path(args.tarball_dir)
    extract_cache = Path(args.extract_cache) if args.extract_cache else tarball_dir / "_extracted"
    data_dir = os.environ["DATA_FOLDER"]

    for release in args.release:
        logger.info(f"=== Processing {release} ===")

        raw_dir = Path(args.raw_manifest_path or f"{data_dir}/raw/transcript/en/TEDLIUM/{release}")
        audio_dir = Path(args.audio_path or f"{data_dir}/raw/transcript/en/TEDLIUM/{release}/audios")
        raw_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        if args.with_nemo_manifest:
            nemo_dir = Path(args.manifest_path or f"{data_dir}/nemo/asr/en/nocontext/TEDLIUM/{release}")
            nemo_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            raw_file = raw_dir / f"{split}.jsonl"
            nemo_file = (nemo_dir / f"{split}.jsonl") if args.with_nemo_manifest else None
            done = raw_file.exists() and (not args.with_nemo_manifest or nemo_file.exists())
            if done and not args.force:
                logger.info(f"[{split}] manifests already exist, skipping")
                continue

            out_raw = NemoDataset(name=DATASET_NAME)
            out = NemoDataset(name=DATASET_NAME) if args.with_nemo_manifest else None

            n_skipped_empty = 0
            n_skipped_ignored = 0
            for i, row in enumerate(tqdm(iter_tarball_split(release, split, tarball_dir, extract_cache), desc=split)):
                text = clean_text(row["text"])
                if not text:
                    n_skipped_empty += 1
                    continue
                if not args.keep_ignored and "ignore_time_segment_in_scoring" in text:
                    n_skipped_ignored += 1
                    continue

                uid = sanitize(f"{row['id']}_{i}")
                arr = np.asarray(row["samples"], dtype=np.float32) / 32768.0
                audio = write_flac_if_missing(audio_dir / split / f"{uid}.flac", arr, row["sampling_rate"])

                audio_turn = NemoTurn(role="User", value=str(audio["path"]), turn_type="audio",
                                      duration=round(audio["duration"], 3))
                text_turn = NemoTurn(role="Assistant", value=text, turn_type="text")
                base_kwargs = dict(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[audio_turn, text_turn],
                )
                out_raw.append(NemoDatasetRow(
                    **base_kwargs,
                    custom_metadata={
                        "talk_id":    row["talk_id"],
                        "speaker_id": row["speaker_id"],
                        "gender":     row["gender"],
                        "file":       row["file"],
                        "release":    release,
                    },
                ))
                if out is not None:
                    out.append(NemoDatasetRow(**base_kwargs))

            if n_skipped_empty:
                logger.warning(f"[{split}] skipped {n_skipped_empty} rows with empty text")
            if n_skipped_ignored:
                logger.warning(f"[{split}] skipped {n_skipped_ignored} rows marked ignore_time_segment_in_scoring")

            out_raw.save(raw_file)
            logger.info(f"[{split}] wrote {len(out_raw)} rows → {raw_file}")
            if out is not None:
                out.save(nemo_file)
                logger.info(f"[{split}] wrote {len(out)} rows → {nemo_file}")


if __name__ == "__main__":
    main()
