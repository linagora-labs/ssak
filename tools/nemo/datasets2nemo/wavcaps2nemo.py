
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from prompts_captioning import SHORT_CAPTION_PROMPTS as prompts
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

random.seed(42)
np.random.seed(42)

# WavCaps captions are short (~8 words) -> short/neutral instruction pool (see
# prompts_captioning.py). One is picked at random per row by ``set_context_if_none``
# to build the context version of the manifest.

# WavCaps sub-corpora: source folder -> (json basename, blacklist key).
# Audio files live in audio_files/{source}/{id_stem}.flac. The AudioSet_SL ids carry a
# ".wav" suffix (e.g. "Yb0RFKhbpFJA.wav"); everything else is a bare id. Only AudioSet and
# FreeSound overlap the AudioCaps/eval sets, hence only those have a blacklist key.
_SOURCES = {
    "FreeSound":         ("fsd_final.json", "FreeSound"),
    "AudioSet_SL":       ("as_final.json",  "AudioSet"),
    "BBC_Sound_Effects": ("bbc_final.json", None),
    "SoundBible":        ("sb_final.json",  None),
}

# All WavCaps data is training data (no official test split).
_SPLIT_NAME = "train"


def _load_blacklist(json_dir: Path, name: str) -> dict:
    """Return {source_key -> set(ids)} of samples to exclude (test-set / eval overlap)."""
    if name is None:
        return {}
    path = json_dir / "blacklist" / name
    with open(path) as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


def _resolve_audio(audio_dir: Path, stem: str) -> Path | None:
    """Locate the audio file for one entry, tolerating .flac/.wav mismatches."""
    flac = audio_dir / f"{stem}.flac"
    if flac.exists():
        return flac
    wav = audio_dir / f"{stem}.wav"
    if wav.exists():
        return wav
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WavCaps (sound captioning) to NeMo format")
    parser.add_argument("--raw_input", type=str, default=None,
                        help="WavCaps repo folder (contains json_files/ and audio_files/).")
    parser.add_argument("--resampled_dir", type=str, default=None,
                        help="Root where converted 16 kHz mono FLACs are written when --resample is set "
                             "(the audio_files/{source} sub-structure is preserved under it).")
    parser.add_argument("--nemo_no_context", type=str, default=None,
                        help="Output folder for the no-context NeMo manifest.")
    parser.add_argument("--nemo_context", type=str, default=None,
                        help="Output folder for the context NeMo manifest.")
    parser.add_argument("--blacklist", type=str, default="blacklist_exclude_all_ac.json",
                        help="Blacklist json under json_files/blacklist to exclude AudioCaps/eval overlap. "
                             "Pass 'none' to disable.")
    parser.add_argument("--sources", nargs="+", default=list(_SOURCES.keys()),
                        help="Subset of WavCaps sources to convert.")
    parser.add_argument("--resample", action="store_true",
                        help="Normalize clips to 16 kHz mono FLAC into resampled_dir (heavy: 400k files); files already "
                             "16 kHz mono are kept as-is. Default: reference originals and take durations from the json.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers for audio normalization.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many entries per source (debug).")
    args = parser.parse_args()

    data_folder = os.environ.get("DATA_FOLDER", "/data-server/datasets/audio")
    if args.raw_input is None:
        args.raw_input = f"{data_folder}/raw/sounds/WavCaps"
    if args.resampled_dir is None:
        args.resampled_dir = f"{data_folder}/converted_audios/sounds/WavCaps"
    if args.nemo_no_context is None:
        args.nemo_no_context = f"{data_folder}/nemo/sounds/audio-captioning/en/nocontext/WavCaps"
    if args.nemo_context is None:
        args.nemo_context = f"{data_folder}/nemo/sounds/audio-captioning/en/context/WavCaps"

    raw_input = Path(args.raw_input)
    json_dir = raw_input / "json_files"
    audio_root = raw_input / "audio_files"
    resampled_root = Path(args.resampled_dir)
    nemo_no_context = Path(args.nemo_no_context)
    nemo_context = Path(args.nemo_context)
    for d in (nemo_no_context, nemo_context):
        d.mkdir(parents=True, exist_ok=True)

    blacklist_name = None if str(args.blacklist).lower() == "none" else args.blacklist
    blacklist = _load_blacklist(json_dir, blacklist_name)

    dataset = NemoDataset(name="WavCaps")
    n_total = n_missing = n_blacklisted = n_kept = 0

    for source in args.sources:
        json_name, bl_key = _SOURCES[source]
        audio_dir = audio_root / source
        bl_ids = blacklist.get(bl_key, set()) if bl_key else set()

        with open(json_dir / source / json_name) as f:
            entries = json.load(f)["data"]

        n_src = 0
        for entry in entries:
            if args.limit is not None and n_src >= args.limit:
                break
            n_src += 1
            n_total += 1

            raw_id = entry["id"]
            stem = Path(raw_id).stem  # AudioSet ids carry a ".wav" suffix

            # Blacklist keys use the raw id for AudioSet (".wav" kept) and the bare id for FreeSound.
            if raw_id in bl_ids or stem in bl_ids:
                n_blacklisted += 1
                continue

            src = _resolve_audio(audio_dir, stem)
            if src is None:
                n_missing += 1
                continue

            # Reference the original audio; durations come from the json (valid after
            # normalization, which preserves duration). Optional resampling happens as a
            # parallel post-pass below.
            duration = entry.get("duration")
            if duration is None:
                info = sf.info(str(src))
                duration = info.frames / info.samplerate

            dataset.append(NemoDatasetRow(
                id=f"{source}_{stem}",
                turns=[
                    NemoTurn(role="User", value=str(src), turn_type="audio",
                             duration=round(float(duration), 3)),
                    NemoTurn(role="Assistant", value=entry["caption"], turn_type="text"),
                ],
                dataset_name="WavCaps",
                language="en",
                split=_SPLIT_NAME,
                custom_metadata={"source": source, "wavcaps_id": raw_id},
            ))
            n_kept += 1
            if n_kept % 5000 == 0:
                print(f"[{source}] kept {n_kept} total ({n_missing} missing, {n_blacklisted} blacklisted)...")

        print(f"[{source}] done ({n_src} entries scanned)")

    # Optional normalization to 16 kHz mono FLAC (keeps already-good wav/flac files as-is).
    if args.resample:
        dataset.normalize_audios(
            str(resampled_root), target_sample_rate=16000, target_extension="flac",
            accepted_extensions=["wav", "flac"], num_workers=args.num_workers,
            relative_to=str(audio_root))

    # No-context manifest, then a context version with a random prompt per row.
    dataset.save(nemo_no_context / f"{_SPLIT_NAME}.jsonl")
    dataset.set_context_if_none(prompts)
    dataset.save(nemo_context / f"{_SPLIT_NAME}.jsonl")

    cover = 100 * n_kept / n_total if n_total else 0
    print(f"[{_SPLIT_NAME}] kept {n_kept}/{n_total} ({cover:.1f}% kept, "
          f"{n_missing} missing audio, {n_blacklisted} blacklisted) -> {nemo_context / (_SPLIT_NAME + '.jsonl')}")
    print("Done")
