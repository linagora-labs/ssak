import argparse
import gzip
import json
import logging
import random
from pathlib import Path

import numpy as np
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

FINEGRAINED_PROMPTS = [
    "Describe in detail what you hear in this speech audio, including voice qualities, delivery, and any temporal dynamics.",
    "Give a fine-grained caption of the speech: voice characteristics, accent, pitch, speaking rate, tone, and any changes over time.",
    "Provide a detailed narrative description of the speakers and how they speak in this recording.",
    "Listen to this speech and write a rich description covering voice timbre, prosody, delivery, and emotional tone.",
    "Caption this speech audio with fine-grained detail about the speaker(s), their voices, and how the delivery evolves.",
]

GLOBAL_PROMPTS = [
    "Describe the speaker in this audio: their voice, accent, gender, pitch, and overall delivery.",
    "Give a holistic profile of the speaker you hear.",
    "Summarize the speaker's voice characteristics in this recording.",
    "Provide a global caption describing the speaker and their voice.",
    "What can you say about the speaker's voice in this clip?",
]

# Subsets we want to keep (skip Emilia for now).
ALLOWED_SUBSETS = {"ears", "expresso", "voxceleb1", "voxceleb2"}


def detect_subset(source_path: str) -> str | None:
    """Return the subset name (ears/expresso/voxceleb1/voxceleb2) from an audio source path."""
    parts = Path(source_path).parts
    for p in parts:
        low = p.lower()
        if low in ALLOWED_SUBSETS:
            return low
    # Fallback: try heuristic on path string
    low = source_path.lower()
    for s in ALLOWED_SUBSETS:
        if f"/{s}/" in low or low.startswith(f"{s}/"):
            return s
    return None


def resolve_audio_path(source: str, input_root: Path) -> Path:
    """Resolve a Lhotse source path against the FCaps root (paths typically start with 'download/')."""
    p = Path(source)
    if p.is_absolute():
        return p
    return input_root / p


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


CAPTION_FIELDS = ("global_captions", "finegrained_captions")


def build_row(cut: dict, input_root: Path, caption_field: str):
    sources = cut.get("recording", {}).get("sources", [])
    if not sources:
        return None, None
    source = sources[0].get("source")
    if not source:
        return None, None

    subset = detect_subset(source)
    if subset is None or subset not in ALLOWED_SUBSETS:
        return None, None

    supervisions = cut.get("supervisions", [])
    if not supervisions:
        return None, None
    sup = supervisions[0]
    custom = sup.get("custom", {}) or {}

    captions = custom.get(caption_field) or []
    if not captions:
        return None, None
    caption = random.choice(captions) if len(captions) > 1 else captions[0]
    if not caption or not caption.strip():
        return None, None

    audio_path = resolve_audio_path(source, input_root)
    duration = cut.get("duration")
    if duration is None:
        return None, None

    offset = float(cut.get("start", 0.0) or 0.0)

    audio_turn = NemoTurn(
        role="User",
        value=str(audio_path),
        turn_type="audio",
        duration=round(float(duration), 3),
        offset=round(offset, 3),
    )
    text_turn = NemoTurn(role="Assistant", value=caption.strip(), turn_type="text")

    metadata = {
        "subset": subset,
        "speaker": sup.get("speaker"),
        "gender": sup.get("gender"),
        "accent": custom.get("accent"),
        "pitch": custom.get("pitch"),
        "speaking_rate": custom.get("speaking_rate"),
        "intrinsic_tags": custom.get("intrinsic_tags"),
        "situational_tags": custom.get("situational_tags"),
        "transcript": sup.get("text"),
        "caption_type": caption_field,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    row = NemoDatasetRow(
        id=cut.get("id"),
        turns=[audio_turn, text_turn],
        speaker=sup.get("speaker"),
        custom_metadata=metadata,
    )
    return row, subset


def process_split(jsonl_gz: Path, input_root: Path):
    datasets = {field: NemoDataset() for field in CAPTION_FIELDS}
    kept = {field: 0 for field in CAPTION_FIELDS}
    skipped = 0
    for idx, cut in enumerate(iter_jsonl_gz(jsonl_gz)):
        any_kept = False
        for field in CAPTION_FIELDS:
            row, _subset = build_row(cut, input_root, field)
            if row is None:
                continue
            datasets[field].append(row)
            kept[field] += 1
            any_kept = True
        if not any_kept:
            skipped += 1
        if (idx + 1) % 10000 == 0:
            logger.info(f"  {jsonl_gz.name}: read {idx + 1}, kept {kept}, skipped {skipped}")
    logger.info(f"Done {jsonl_gz.name}: kept {kept}, skipped {skipped}")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FCaps dataset to NeMo format")
    parser.add_argument(
        "--input",
        type=str,
        default="/data-server/datasets/audio/raw/misc/en/FCaps",
        help="Root of the FCaps download (containing data/ and download/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data-server/datasets/audio/nemo/misc/en",
        help="Output folder for the NeMo manifests.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Skip writing the version with prompts.",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    download_dir = input_root / "download"
    data_dir = input_root / "data"

    output_root = Path(args.output)
    nocontext_out = output_root / "nocontext" / "FCaps"
    context_out = output_root / "context" / "FCaps"
    nocontext_out.mkdir(parents=True, exist_ok=True)
    context_out.mkdir(parents=True, exist_ok=True)

    prompts_by_field = {
        "global_captions": GLOBAL_PROMPTS,
        "finegrained_captions": FINEGRAINED_PROMPTS,
    }
    suffix_by_field = {
        "global_captions": "",
        "finegrained_captions": "_finegrained",
    }

    splits = {
        "train": data_dir / "fcaps-pscbase-train_base.jsonl.gz",
        "dev": data_dir / "fcaps-dev.jsonl.gz",
        "test": data_dir / "fcaps-test.jsonl.gz",
    }

    for split_name, jsonl_gz in splits.items():
        if not jsonl_gz.exists():
            logger.warning(f"Missing split file: {jsonl_gz} -- skipping {split_name}")
            continue
        logger.info(f"Processing split '{split_name}' from {jsonl_gz}")
        datasets = process_split(jsonl_gz, input_root)
        for field, dataset in datasets.items():
            if len(dataset) == 0:
                continue
            suffix = suffix_by_field[field]
            # dataset.save(nocontext_out / f"{split_name}{suffix}.jsonl")
            if not args.no_context:
                dataset.set_context_if_none(prompts_by_field[field])
                # dataset.save(context_out / f"{split_name}{suffix}.jsonl")
                print(dataset[0])

    logger.info("Done")
