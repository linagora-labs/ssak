"""Convert the ESLO corpus into NeMo manifests.

ESLO ships several sub-corpora in incompatible layouts/formats. This converter
currently implements only the **diarization** family, and only over the
**ESLO-orleans `.trs` subset**, which is the cleanest self-contained,
audio-paired set: each `record-*/` folder holds one Transcriber `.trs`
transcript (UTF-8, turn-level timing + speaker labels + overlap markers) next to
its audio file.

Expected layout (Cocoon/Orléans "demande" export):

    <root>/
        ESLO-orleans/
            record-1/   <ID>_C.trs   <something>.mp3|wav|mp4
            record-2/   ...
            ...

Produces (each opt-in via a flag):
  - diarization/  : windowed "who spoke when" rows in asr/timestamps/timestamps_asr
                    variants, several output formats (RTTM, SRT, VTT, ...) and
                    with/without backchannels.

ASR (over *all* available ESLO audio-paired data: ESLO-md + ESLO-orleans, both
`.trs` and `crdo*.xml`) is intentionally not wired up yet. The `--asr` flag and
its dispatch hook are present so it can be filled in later without reshaping the
CLI; for now passing it errors out.

Note on language: ESLO is French (language="fr"), so rows draw the French
prompts from diar_prompts (including the French-only "Locuteur" formats); any
prompt list French is missing falls back to English automatically.
"""

import argparse
import hashlib
import logging
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn
from diar_prompts import (
    DIAR_VARIANTS,
    choose_format,
    formats_for,
    clean_window_pieces,
    make_diar_lean_row,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "ESLO"
LANGUAGE = "fr"

AUDIO_EXTS = (".mp3", ".wav", ".mp4", ".flac", ".m4a", ".ogg")
# Trailing transcription-tier marker on the .trs basename (ESLO1_ENT_001_C.trs).
_TIER_RE = re.compile(r"_[A-Z]$")


# ESLO has no canonical train/dev/test split, so we derive a deterministic one by
# hashing the recording id (stable across runs and independent of file order).
def split_for(rec_id: str, val_ratio: float, test_ratio: float) -> str:
    h = (int(hashlib.md5(rec_id.encode("utf-8")).hexdigest(), 16) % 1000) / 1000.0
    if h < test_ratio:
        return "test"
    if h < test_ratio + val_ratio:
        return "validation"
    return "train"


def audio_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        logger.warning(f"Could not read duration for {path}: {e}")
        return None


def build_converted_index(conv_dataset_dir: Path) -> dict:
    """Map lowercased filename stem -> converted audio path (wav preferred over
    flac) for every audio under the dataset's converted tree. The converted tree
    may reorganise files into different subdirs/extension relative to raw (e.g. ESLO
    raw 'ESLO-orleans/record-N/<REC>.mp3' vs converted
    'ESLO_800H/<n>/<CAT>/<REC>/<REC>.wav'), so we match by filename stem, not path."""
    index: dict[str, Path] = {}
    if not conv_dataset_dir.is_dir():
        return index
    for ext in (".flac", ".wav"):  # wav indexed last so it wins on a stem clash
        for fp in conv_dataset_dir.rglob(f"*{ext}"):
            if "@eaDir" in fp.parts:
                continue
            index[fp.stem.lower()] = fp
    return index


def resolve_audio(audio: Path, conv_index: dict, prefer: bool) -> Path:
    """Return the converted (wav/flac, 16 kHz mono) copy for this recording if one
    exists in the converted tree (matched by filename stem); else the raw audio."""
    if not prefer:
        return audio
    return conv_index.get(audio.stem.lower(), audio)


# --------------------------- text cleanup ---------------------------

# Transcriber events are separate <Event> elements, so .trs tail text is mostly
# clean; this just normalises whitespace and drops any stray inline [..] markers.
_INLINE_TAG_RE = re.compile(r"\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")
_DETOK_RE = re.compile(r"\s+([,.;:!?])")


def clean_text(text: str) -> str:
    text = _INLINE_TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return _DETOK_RE.sub(r"\1", text)


# --------------------------- .trs parsing ---------------------------

def parse_trs(path: Path):
    """Return (speakers, root) where speakers maps spk id -> display name.

    ET respects the XML declaration's encoding (ESLO-md is ISO-8859-1, orleans is
    UTF-8) and does not fetch the declared external DTD."""
    root = ET.parse(str(path)).getroot()
    speakers = {}
    for s in root.iter("Speaker"):
        sid = s.get("id")
        if sid:
            speakers[sid] = s.get("name") or sid
    return speakers, root


def iter_trs_utterances(root, speakers: dict):
    """Yield {'speaker', 'start', 'end', 'text'} for every speaker-attributed turn.

    Transcriber stores spoken text in the `.tail` of <Sync>/<Who> elements. A
    <Sync time=t> opens a time-anchored sub-segment; a <Who nb=k> switches the
    active speaker within an overlapping turn (turn @speaker is a space-separated
    list, k is 1-based into it). Each sub-segment ends at the next Sync time, or
    the turn's endTime. Overlapping chunks share the enclosing sub-segment span.
    """
    for turn in root.iter("Turn"):
        spk_list = (turn.get("speaker") or "").split()
        if not spk_list:  # silence / non-speech turn
            continue
        try:
            t_start = float(turn.get("startTime"))
            t_end = float(turn.get("endTime"))
        except (TypeError, ValueError):
            continue

        items: list[list] = []  # [start, who_idx, text]
        sync_times: list[float] = []
        cur_start = t_start
        cur_who = 0
        if turn.text and turn.text.strip():
            items.append([cur_start, cur_who, turn.text.strip()])
        for c in turn:
            tag = c.tag
            if tag == "Sync":
                try:
                    cur_start = float(c.get("time"))
                    sync_times.append(cur_start)
                except (TypeError, ValueError):
                    pass
            elif tag == "Who":
                try:
                    cur_who = int(c.get("nb")) - 1
                except (TypeError, ValueError):
                    pass
            tail = (c.tail or "").strip()
            if tail:
                items.append([cur_start, cur_who, tail])
        if not items:
            continue

        boundaries = sorted(set(sync_times + [t_start, t_end]))

        def end_for(s: float) -> float:
            for b in boundaries:
                if b > s + 1e-6:
                    return b
            return t_end

        # Concatenate fragments sharing the same (start, speaker), preserving order.
        merged: dict[tuple, list[str]] = {}
        order: list[tuple] = []
        for s, who, txt in items:
            key = (s, who)
            if key not in merged:
                merged[key] = []
                order.append(key)
            merged[key].append(txt)

        for s, who in order:
            idx = who if 0 <= who < len(spk_list) else 0
            spk_id = spk_list[idx]
            text = clean_text(" ".join(merged[(s, who)]))
            if not text:
                continue
            e = end_for(s)
            if e <= s:
                e = t_end
            if e <= s:
                continue
            yield {
                "speaker": speakers.get(spk_id, spk_id),
                "start": s,
                "end": e,
                "text": text,
            }


# --------------------------- discovery ---------------------------

def find_audio(folder: Path) -> Path | None:
    for fp in sorted(folder.iterdir()):
        if fp.suffix.lower() in AUDIO_EXTS:
            return fp
    return None


def pick_trs(folder: Path) -> Path | None:
    trs = sorted(folder.glob("*.trs"))
    if not trs:
        return None
    # Prefer the final '_C' tier when several tiers are present.
    for t in trs:
        if t.stem.endswith("_C"):
            return t
    return trs[0]


def discover_orleans_trs(orleans_root: Path) -> list[tuple[str, Path, Path]]:
    """Return [(rec_id, audio_path, trs_path)] for orleans records with both."""
    out = []
    for d in sorted(orleans_root.glob("record-*")):
        if not d.is_dir():
            continue
        trs = pick_trs(d)
        if trs is None:
            continue
        audio = find_audio(d)
        if audio is None:
            logger.debug(f"{d.name}: .trs but no audio, skipping")
            continue
        rec_id = _TIER_RE.sub("", trs.stem) or d.name
        out.append((rec_id, audio, trs))
    return out


# --------------------------- diarization windowing ---------------------------
# (Shared shape with ami2nemo.py / icsi2nemo.py; kept local per the per-converter
# convention. The only ESLO-specific part is how `units` are built, above.)

def build_units(root, speakers: dict, merge_gap: float | None,
                mix_duration: float | None = None) -> list[dict]:
    """Chronological speaker turns: {'speaker','start','end','text','words'}.

    `words` carries one fragment per source utterance (its own start/end/text) so
    a too-long merged turn can be split back at utterance boundaries, and so
    same-speaker merges accumulate cleanly."""
    units: list[dict] = []
    for u in iter_trs_utterances(root, speakers):
        start, end = u["start"], u["end"]
        if mix_duration is not None:
            end = min(end, mix_duration)
        if end <= start:
            continue
        units.append({
            "speaker": u["speaker"], "start": start, "end": end, "text": u["text"],
            "words": [{"text": u["text"], "start": start, "end": end}],
        })
    units.sort(key=lambda u: (u["start"], u["end"]))
    if merge_gap is not None:
        units = _merge_same_speaker(units, merge_gap)
    return units


def _merge_same_speaker(units: list[dict], gap: float) -> list[dict]:
    out: list[dict] = []
    for u in units:
        prev = out[-1] if out else None
        if prev and prev["speaker"] == u["speaker"] and u["start"] - prev["end"] <= gap:
            prev["end"] = max(prev["end"], u["end"])
            prev["text"] = clean_text(prev["text"] + " " + u["text"])
            prev["words"].extend(u["words"])
        else:
            out.append(dict(u, words=list(u["words"])))
    return out


def _mk_piece(speaker: str, words: list[dict]) -> dict:
    text = clean_text(" ".join(w["text"] for w in words if w.get("text")))
    return {"speaker": speaker, "start": words[0]["start"],
            "end": words[-1]["end"], "text": text, "words": list(words)}


def _split_long_turn(unit: dict, max_dur: float) -> list[dict]:
    """Split a turn longer than max_dur at fragment boundaries; else return [unit]."""
    if unit["end"] - unit["start"] <= max_dur:
        return [unit]
    words = [w for w in unit["words"]
             if w.get("start") is not None and w.get("end") is not None and w.get("text")]
    if not words:
        pieces, s = [], unit["start"]
        while s < unit["end"]:
            e = min(s + max_dur, unit["end"])
            pieces.append({"speaker": unit["speaker"], "start": s, "end": e,
                           "text": unit["text"] if s == unit["start"] else "", "words": []})
            s = e
        return pieces
    pieces, cur, cur_start = [], [], words[0]["start"]
    for w in words:
        if cur and w["end"] - cur_start > max_dur:
            pieces.append(_mk_piece(unit["speaker"], cur))
            cur, cur_start = [], w["start"]
        cur.append(w)
    if cur:
        pieces.append(_mk_piece(unit["speaker"], cur))
    return pieces


def window_turns(units: list[dict], max_dur: float,
                 max_turns: int | None = None) -> list[list[dict]]:
    windows: list[list[dict]] = []
    cur: list[dict] = []
    cur_start = None
    for u in units:
        for piece in _split_long_turn(u, max_dur):
            fits = (cur
                    and piece["end"] - cur_start <= max_dur
                    and (max_turns is None or len(cur) < max_turns))
            if fits:
                cur.append(piece)
            else:
                if cur:
                    windows.append(cur)
                cur, cur_start = [piece], piece["start"]
    if cur:
        windows.append(cur)
    return windows


def _pieces_to_segs(pieces: list[dict], win_start: float):
    labels: dict[str, int] = {}
    segs = []
    for p in pieces:
        if p["speaker"] not in labels:
            labels[p["speaker"]] = len(labels) + 1
        segs.append({
            "n": labels[p["speaker"]],
            "s": round(p["start"] - win_start, 2),
            "e": round(p["end"] - win_start, 2),
            "text": p["text"],
        })
    return segs, labels


def build_diar_rows(meeting_id: str, mix_audio: Path, units: list[dict],
                    max_dur: float, min_turns: int, min_dur: float,
                    split: str, max_turns: int | None = None,
                    format_variety: bool = True, backchannel_versions: bool = True,
                    generic_ratio: float = 0.4, backchannel_ratio: float = 0.5,
                    mixed_variants: tuple | None = None) -> dict:
    """Build diarization rows for every variant from one shared windowing.

    See ami2nemo.build_diar_rows for the full description of the per-window
    variant/format/backchannel logic; this is the same machinery."""
    out: dict[str, list] = {v: [] for v in DIAR_VARIANTS}
    if mixed_variants:
        out["mixed"] = []
    for i, window in enumerate(window_turns(units, max_dur, max_turns)):
        if len(window) < min_turns:
            continue
        win_start = window[0]["start"]
        win_end = max(p["end"] for p in window)
        dur = win_end - win_start
        if dur < min_dur or dur <= 0:
            continue

        if backchannel_versions:
            clean = clean_window_pieces(window, LANGUAGE)
            versions = []
            if clean:
                versions.append(("clean", clean, False))
            if not clean:
                versions.append(("full", window, True))
            elif len(clean) < len(window):
                rng_bc = random.Random(f"{meeting_id}.{i}.bc")
                if rng_bc.random() < backchannel_ratio:
                    versions.append(("full", window, True))
        else:
            versions = [("all", window, None)]

        for tag, pieces, include_bc in versions:
            segs, labels = _pieces_to_segs(pieces, win_start)
            if not segs:
                continue
            seg_meta = [{
                "speaker": p["speaker"],
                "label": f"Speaker {labels[p['speaker']]}",
                "start": round(p["start"], 3),
                "end": round(p["end"], 3),
                "text": p["text"],
            } for p in pieces]
            built: dict[str, NemoDatasetRow] = {}
            for variant in DIAR_VARIANTS:
                formats = formats_for(LANGUAGE, variant)
                rng = random.Random(f"{meeting_id}.{i}.{variant}.{tag}")
                fmt, prompt_style = choose_format(variant, formats, rng,
                                                  format_variety, generic_ratio)
                target = formats[fmt]["render"](segs, meeting_id)
                if not target.strip():
                    continue
                audio_turn = NemoTurn(role="User", value=str(mix_audio), turn_type="audio",
                                      duration=round(dur, 3), offset=round(win_start, 3))
                target_turn = NemoTurn(role="Assistant", value=target, turn_type="text")
                md = {
                    "meeting": meeting_id,
                    "variant": variant,
                    "format": fmt,
                    "prompt_style": prompt_style,
                    "num_speakers": len(labels),
                    "segments": seg_meta,
                }
                if include_bc is not None:
                    md["backchannels"] = include_bc
                row = NemoDatasetRow(
                    id=f"{meeting_id}.diar.{i}.{tag}",
                    dataset_name=DATASET_NAME,
                    split=split,
                    language=LANGUAGE,
                    turns=[audio_turn, target_turn],
                    custom_metadata=md,
                )
                out[variant].append(row)
                built[variant] = row
            if mixed_variants:
                pool = [v for v in mixed_variants if v in built]
                if pool:
                    rng_m = random.Random(f"{meeting_id}.{i}.{tag}.mixed")
                    out["mixed"].append(built[rng_m.choice(pool)])
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert ESLO to NeMo manifests.")

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--root", type=str, default=None,
                          help="ESLO dataset root (contains ESLO-orleans/, ...). "
                               "Default: $DATA_FOLDER/raw/transcript/fr/ESLO.")
    io_group.add_argument("--force", action="store_true",
                          help="Overwrite existing manifest .jsonl files.")
    io_group.add_argument("--limit", type=int, default=None,
                          help="Process at most N recordings (debug).")
    io_group.add_argument("--val-ratio", type=float, default=0.05,
                          help="Fraction of recordings hashed into the validation split.")
    io_group.add_argument("--test-ratio", type=float, default=0.05,
                          help="Fraction of recordings hashed into the test split.")
    io_group.add_argument("--prefer-converted", action=argparse.BooleanOptionalAction, default=True,
                          help="Prefer the converted (wav/flac, 16 kHz mono) audio mirroring the "
                               "raw path under --converted-audio-root when present "
                               "(--no-prefer-converted to always use the raw audio).")
    io_group.add_argument("--raw-audio-root", type=str, default=None,
                          help="Root of the raw audio tree, for raw->converted path mapping. "
                               "Default: $DATA_FOLDER/raw or /data-server/datasets/audio/raw.")
    io_group.add_argument("--converted-audio-root", type=str, default=None,
                          help="Root of the converted audio tree. Default: "
                               "$DATA_FOLDER/converted_audios or "
                               "/data-server/datasets/audio/converted_audios.")

    # --asr is declared so the CLI shape is final; the implementation (over all
    # ESLO audio-paired data) is not wired up yet.
    asr_group = parser.add_argument_group("asr manifest (not implemented yet)")
    asr_group.add_argument("--asr", action="store_true",
                           help="[NOT IMPLEMENTED] ASR manifest over all ESLO data.")
    asr_group.add_argument("--asr-min-duration", type=float, default=0.2)
    asr_group.add_argument("--asr-max-duration", type=float, default=60.0)

    diar_group = parser.add_argument_group("diarization manifest")
    diar_group.add_argument("--diarization", action="store_true",
                            help="Enable the speaker-diarization manifests (off by default), "
                                 "built from the ESLO-orleans .trs subset. Emits three "
                                 "variants under diarization/: 'asr' (speaker-labeled "
                                 "transcript), 'timestamps' (who-spoke-when), and "
                                 "'timestamps_asr' (labels + times + words).")
    diar_group.add_argument("--diar-max-duration", type=float, default=90.0,
                            help="Max window length in seconds.")
    diar_group.add_argument("--diar-min-duration", type=float, default=10.0,
                            help="Drop diarization windows shorter than this many seconds.")
    diar_group.add_argument("--diar-min-turns", type=int, default=3,
                            help="Drop windows with fewer than this many speaker turns.")
    diar_group.add_argument("--diar-max-turns", type=int, default=10,
                            help="Start a new window once it reaches this many speaker turns "
                                 "(0 disables the cap).")
    diar_group.add_argument("--diar-format-variety", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Vary the output format per row with a prompt that specifies "
                                 "it. --no-diar-format-variety forces the default format.")
    diar_group.add_argument("--diar-generic-prompt-ratio", type=float, default=0.5,
                            help="Fraction of rows using a generic, no-format prompt.")
    diar_group.add_argument("--diar-cross-lingual-prompt-ratio", type=float, default=0.05,
                            help="Probability of drawing the prompt from a different language than "
                                 "the audio (e.g. a French prompt on English data and vice versa); "
                                 "only formats defined in both languages are eligible.")
    diar_group.add_argument("--diar-backchannel-versions", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Emit clean (no-backchannel) and, when present, full "
                                 "(backchannel-included) versions per window.")
    diar_group.add_argument("--diar-backchannel-ratio", type=float, default=0.5,
                            help="Probability of also emitting the backchannel-included version.")
    diar_group.add_argument("--diar-mixed", action=argparse.BooleanOptionalAction, default=True,
                            help="Also emit a 'mixed/' folder (one randomly-chosen variant per window).")
    diar_group.add_argument("--diar-mixed-variants", nargs="+", choices=list(DIAR_VARIANTS),
                            default=list(DIAR_VARIANTS),
                            help="Pool of variants to sample from for the 'mixed' folder.")
    diar_group.add_argument("--diar-merge-gap", type=float, default=1.0,
                            help="Merge consecutive same-speaker turns separated by <= this "
                                 "many seconds. Negative disables merging.")
    diar_group.add_argument("--diar-manifest-path", type=str, default=None,
                            help="Training diarization manifest folder. "
                                 "Default: $DATA_FOLDER/nemo/diarization/fr/eslo.")

    args = parser.parse_args()

    if args.asr:
        parser.error("--asr is not implemented yet for ESLO; only --diarization is available.")
    if not args.diarization:
        parser.error("Nothing to do: pass --diarization.")

    data_dir = os.environ.get("DATA_FOLDER")
    if args.root is None:
        if not data_dir:
            parser.error("--root not set and DATA_FOLDER env var is missing")
        args.root = f"{data_dir}/raw/transcript/fr/ESLO"
        print(f"Input path not specified, using default: {args.root}")
    root = Path(args.root)
    orleans_root = root / "ESLO-orleans"
    if not orleans_root.is_dir():
        parser.error(f"Missing directory: {orleans_root}")

    if args.diar_manifest_path is None:
        if not data_dir:
            parser.error("--diar-manifest-path not set and DATA_FOLDER env var is missing")
        args.diar_manifest_path = f"{data_dir}/nemo/diarization/fr/eslo"

    diar_out_root = Path(args.diar_manifest_path)
    diar_outputs = list(DIAR_VARIANTS) + (["mixed"] if args.diar_mixed else [])
    for v in diar_outputs:
        (diar_out_root / v).mkdir(parents=True, exist_ok=True)

    raw_audio_root = Path(args.raw_audio_root
                          or (f"{data_dir}/raw" if data_dir else "/data-server/datasets/audio/raw"))
    conv_audio_root = Path(args.converted_audio_root
                           or (f"{data_dir}/converted_audios" if data_dir
                               else "/data-server/datasets/audio/converted_audios"))
    conv_dataset_dir = (Path(str(root).replace(str(raw_audio_root), str(conv_audio_root), 1))
                        if str(root).startswith(str(raw_audio_root)) else conv_audio_root)
    conv_index = build_converted_index(conv_dataset_dir) if args.prefer_converted else {}

    recordings = discover_orleans_trs(orleans_root)
    if args.limit:
        recordings = recordings[:args.limit]
    logger.info(f"Found {len(recordings)} ESLO-orleans .trs recordings under {orleans_root}")
    if args.prefer_converted:
        logger.info(f"Indexed {len(conv_index)} converted audio files under {conv_dataset_dir}")

    diar_buckets: dict[str, dict[str, NemoDataset]] = {
        v: {} for v in diar_outputs
    }

    def get(buckets, split):
        if split not in buckets:
            buckets[split] = NemoDataset(name=DATASET_NAME)
        return buckets[split]

    n_converted = 0
    for rec_id, audio, trs in tqdm(recordings, desc="recordings"):
        split = split_for(rec_id, args.val_ratio, args.test_ratio)
        try:
            speakers, root_xml = parse_trs(trs)
        except ET.ParseError as e:
            logger.warning(f"{rec_id}: bad .trs ({e}), skipping")
            continue

        audio = resolve_audio(audio, conv_index, args.prefer_converted)
        if str(audio).startswith(str(conv_audio_root)):
            n_converted += 1
        mix_dur = audio_duration(audio)  # None for unreadable audio (e.g. some mp3)

        if args.diarization:
            units = build_units(
                root_xml, speakers,
                None if args.diar_merge_gap < 0 else args.diar_merge_gap,
                mix_duration=mix_dur,
            )
            if not units:
                continue
            per_variant = build_diar_rows(
                rec_id, audio, units,
                args.diar_max_duration, args.diar_min_turns, args.diar_min_duration,
                split=split,
                max_turns=None if args.diar_max_turns <= 0 else args.diar_max_turns,
                format_variety=args.diar_format_variety,
                backchannel_versions=args.diar_backchannel_versions,
                generic_ratio=args.diar_generic_prompt_ratio,
                backchannel_ratio=args.diar_backchannel_ratio,
                mixed_variants=tuple(args.diar_mixed_variants) if args.diar_mixed else None,
            )
            for variant, rows in per_variant.items():
                if not rows:
                    continue
                lean = get(diar_buckets[variant], split)
                for r in rows:
                    lean.append(make_diar_lean_row(r, args.diar_cross_lingual_prompt_ratio))

    if args.prefer_converted:
        logger.info(f"Used converted audio for {n_converted}/{len(recordings)} recordings")

    def dump(buckets, kind, out_base):
        for split, lean in buckets.items():
            if not len(lean):
                continue
            lean_file = out_base / kind / f"{split}.jsonl"
            if lean_file.exists() and not args.force:
                logger.info(f"[{kind}/{split}] exists, skipping (use --force)")
                continue
            lean.save(lean_file)
            logger.info(f"[{kind}/{split}] wrote {len(lean)} rows → {lean_file}")

    if args.diarization:
        for variant in diar_outputs:
            dump(diar_buckets[variant], variant, out_base=diar_out_root)


if __name__ == "__main__":
    main()
