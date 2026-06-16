"""Shared diarization prompt/format machinery for the meeting-corpus converters.

This module is the single source of truth for the diarization output variants,
their per-format renderers, the prompts (per language), the backchannel handling,
and the lean-row prompt selection. Both ami2nemo.py and icsi2nemo.py (and any
other meeting converter) import from here, so a prompt/format edit applies
everywhere.

`seg` fields used by the renderers: n (speaker number, 1..), s/e (start/end
seconds, relative to the clip), text (words).

-------------------------------------------------------------------------------
ADDING A NEW LANGUAGE
-------------------------------------------------------------------------------
Prompts are keyed by language (`_DIAR_PROMPTS[<lang>]`). The output *formats*
themselves (the render functions and their keys) are language-independent and
live in `_DIAR_FORMATS`; only the natural-language instruction text differs per
language. To add, say, French:

    1. Add a "fr" entry to `_DIAR_PROMPTS` mirroring the "en" structure:
         _DIAR_PROMPTS["fr"] = {
             "generic": { "asr": [...], "timestamps": [...], "timestamps_asr": [...] },
             "formats": { variant: { fmt: [...] for every fmt }, ... },
             "backchannel_suffixes": [...],
         }
    2. That's it. Rows whose `language == "fr"` will draw French prompts; any
       prompt list a language is missing falls back to "en" automatically
       (see `_prompts_for` / `_backchannel_suffixes_for`).

The format keys a language provides must be a subset of `_DIAR_FORMATS[variant]`;
`_validate_prompts()` checks at import time that "en" (the fallback) is complete.
"""

import json
import random
import re

from ssak.utils.nemo_dataset import NemoDatasetRow, NemoTurn

DIAR_VARIANTS = ("asr", "timestamps", "timestamps_asr")

# Fallback language: every other language falls back to this one for any prompt
# list it does not define, and this one must be complete.
DEFAULT_LANGUAGE = "en"

# Each diarization variant can be rendered in several output formats. A row picks
# one format, renders its target in that format, and is paired with a prompt drawn
# from that same format's prompt list (in the row's language) — so the prompt
# teaches the model which format to produce. The first format of each variant is
# the "default": generic, format-agnostic prompts map to it.

# Speaker-label styles shared across all three variants. `n` is 1-based.
#   letter -> 'A', 'B', ...    s_num -> 'S1', 'S2', ...    upper -> 'SPEAKER_00', ...
def _lbl_letter(n): return chr(ord("A") + n - 1)
def _lbl_s(n): return f"S{n}"
def _lbl_upper(n): return f"SPEAKER_{n - 1:02d}"

def _r_asr_speaker_colon(segs, meeting):
    return "\n".join(f"Speaker {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_bare_letter(segs, meeting):
    return "\n".join(f"{_lbl_letter(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_s_num(segs, meeting):
    return "\n".join(f"{_lbl_s(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_upper(segs, meeting):
    return "\n".join(f"{_lbl_upper(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_dash(segs, meeting):
    return "\n".join(f"Speaker {s['n']} - {s['text']}" for s in segs if s["text"])

def _r_asr_letter(segs, meeting):
    return "\n".join(f"Speaker {chr(ord('A') + s['n'] - 1)}: {s['text']}"
                     for s in segs if s["text"])

def _r_ts_plain(segs, meeting):
    return "\n".join(f"Speaker {s['n']} {s['s']:.2f} {s['e']:.2f}" for s in segs)

def _r_ts_rttm(segs, meeting):
    # Classic NIST RTTM: SPEAKER <uri> <chan> <onset> <dur> <NA> <NA> <spk> <NA> <NA>
    return "\n".join(
        f"SPEAKER {meeting} 1 {s['s']:.2f} {s['e'] - s['s']:.2f} <NA> <NA> Speaker_{s['n']} <NA> <NA>"
        for s in segs)

def _r_ts_bracket(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] Speaker {s['n']}" for s in segs)

def _r_ts_bare_letter(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_letter(s['n'])}" for s in segs)

def _r_ts_s_num(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_s(s['n'])}" for s in segs)

def _r_ts_speaker_upper(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_upper(s['n'])}" for s in segs)

def _r_tsa_bracket_colon(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] Speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_bare_letter(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_letter(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_s_num(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_s(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_upper(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_upper(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_inline(segs, meeting):
    return "\n".join(f"Speaker {s['n']} ({s['s']:.2f}-{s['e']:.2f}): {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_arrow(segs, meeting):
    return "\n".join(f"{s['s']:.2f} --> {s['e']:.2f}  Speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _hms(t, sep):
    """Seconds -> 'HH:MM:SS<sep>mmm' (sep is ',' for SRT, '.' for WebVTT)."""
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

def _r_tsa_srt(segs, meeting):
    blocks, idx = [], 1
    for s in segs:
        if not s["text"]:
            continue
        blocks.append(f"{idx}\n{_hms(s['s'], ',')} --> {_hms(s['e'], ',')}\n"
                      f"Speaker {s['n']}: {s['text']}")
        idx += 1
    return "\n\n".join(blocks)

def _r_tsa_vtt(segs, meeting):
    cues = [f"{_hms(s['s'], '.')} --> {_hms(s['e'], '.')}\nSpeaker {s['n']}: {s['text']}"
            for s in segs if s["text"]]
    return "WEBVTT\n\n" + "\n\n".join(cues)


# JSON: a structured, schema-heavy output (kept to a small slice of the rows, see
# JSON_FORMAT_RATIO). Speakers are identified as 'spk<N>'. Only fields derivable
# from the source segments (n/s/e/text) are emitted: there is no per-word timing,
# no confidence, no raw-vs-processed distinction and no per-segment language in the
# pipeline, so those schema fields are deliberately omitted rather than fabricated.
def _spk(n): return f"spk{n}"

def _r_asr_json(segs, meeting):
    # ASR + speaker, no timestamps.
    result = "\n".join(f"{_spk(s['n'])}: {s['text']}" for s in segs if s["text"])
    segments = [{"segment": s["text"], "spk_id": _spk(s["n"])}
                for s in segs if s["text"]]
    return json.dumps({"transcription_result": result, "segments": segments},
                      ensure_ascii=False, indent=2)

def _r_ts_json(segs, meeting):
    # Speaker + timestamps, no transcript: a per-speaker summary plus the segments.
    speakers = {}
    for s in segs:
        spk = _spk(s["n"])
        agg = speakers.setdefault(spk, {"spk_id": spk, "duration": 0.0, "nbr_seg": 0})
        agg["duration"] = round(agg["duration"] + (s["e"] - s["s"]), 2)
        agg["nbr_seg"] += 1
    segments = [{"seg_id": i, "spk_id": _spk(s["n"]),
                 "seg_begin": round(s["s"], 2), "seg_end": round(s["e"], 2)}
                for i, s in enumerate(segs, 1)]
    return json.dumps({"speakers": list(speakers.values()), "segments": segments},
                      ensure_ascii=False, indent=2)

def _r_tsa_json(segs, meeting):
    # ASR + speaker + timestamps.
    result = "\n".join(f"{_spk(s['n'])}: {s['text']}" for s in segs if s["text"])
    segments = [{"segment": s["text"],
                 "start": round(s["s"], 2), "end": round(s["e"], 2),
                 "duration": round(s["e"] - s["s"], 2), "spk_id": _spk(s["n"])}
                for s in segs if s["text"]]
    return json.dumps({"transcription_result": result, "segments": segments},
                      ensure_ascii=False, indent=2)


# Whole-word label-casing styles. French (Locuteur/LOCUTEUR/locuteur) back the
# French-only formats in `_LANG_EXTRA_FORMATS["fr"]`; English SPEAKER/speaker are
# base formats (distinct from the zero-padded 'SPEAKER_00' style).
def _r_asr_locuteur_colon(segs, meeting):
    return "\n".join(f"Locuteur {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_dash(segs, meeting):
    return "\n".join(f"Locuteur {s['n']} - {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_caps(segs, meeting):
    return "\n".join(f"LOCUTEUR {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_lower(segs, meeting):
    return "\n".join(f"locuteur {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_caps(segs, meeting):
    return "\n".join(f"SPEAKER {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_lower(segs, meeting):
    return "\n".join(f"speaker {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_ts_locuteur_bracket(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] Locuteur {s['n']}" for s in segs)

def _r_ts_locuteur_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] LOCUTEUR {s['n']}" for s in segs)

def _r_ts_locuteur_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] locuteur {s['n']}" for s in segs)

def _r_ts_speaker_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] SPEAKER {s['n']}" for s in segs)

def _r_ts_speaker_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] speaker {s['n']}" for s in segs)

def _r_tsa_locuteur_bracket_colon(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] Locuteur {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_inline(segs, meeting):
    return "\n".join(f"Locuteur {s['n']} ({s['s']:.2f}-{s['e']:.2f}): {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] LOCUTEUR {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] locuteur {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] SPEAKER {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])


# --------------------------------------------------------------------------- #
# Output formats: language-independent. Maps variant -> format key -> renderer. #
# The first format key of each variant is its default (used by generic prompts).#
# --------------------------------------------------------------------------- #
_DIAR_FORMATS = {
    "asr": {
        "speaker_colon": {"render": _r_asr_speaker_colon},
        "bare_letter":   {"render": _r_asr_bare_letter},
        "s_num":         {"render": _r_asr_s_num},
        "speaker_upper": {"render": _r_asr_speaker_upper},
        "dash":          {"render": _r_asr_dash},
        "letter":        {"render": _r_asr_letter},
        "speaker_caps":  {"render": _r_asr_speaker_caps},
        "speaker_lower": {"render": _r_asr_speaker_lower},
        "json":          {"render": _r_asr_json},
    },
    "timestamps": {
        "plain":         {"render": _r_ts_plain},
        "rttm":          {"render": _r_ts_rttm},
        "bracket":       {"render": _r_ts_bracket},
        "bare_letter":   {"render": _r_ts_bare_letter},
        "s_num":         {"render": _r_ts_s_num},
        "speaker_upper": {"render": _r_ts_speaker_upper},
        "speaker_caps":  {"render": _r_ts_speaker_caps},
        "speaker_lower": {"render": _r_ts_speaker_lower},
        "json":          {"render": _r_ts_json},
    },
    "timestamps_asr": {
        "bracket_colon": {"render": _r_tsa_bracket_colon},
        "bare_letter":   {"render": _r_tsa_bare_letter},
        "s_num":         {"render": _r_tsa_s_num},
        "speaker_upper": {"render": _r_tsa_speaker_upper},
        "inline":        {"render": _r_tsa_inline},
        "arrow":         {"render": _r_tsa_arrow},
        "srt":           {"render": _r_tsa_srt},
        "vtt":           {"render": _r_tsa_vtt},
        "speaker_caps":  {"render": _r_tsa_speaker_caps},
        "speaker_lower": {"render": _r_tsa_speaker_lower},
        "json":          {"render": _r_tsa_json},
    },
}

# First key of each variant is its default format (used by generic prompts).
DIAR_DEFAULT_FORMAT = {v: next(iter(fmts)) for v, fmts in _DIAR_FORMATS.items()}

# JSON is a structured, schema-heavy format; keep it a small slice of all rows.
JSON_FORMAT_RATIO = 0.05


def choose_format(variant, formats, rng, format_variety=True,
                  generic_ratio=0.4, json_ratio=JSON_FORMAT_RATIO):
    """Pick (format_key, prompt_style) for one diarization row.

    - Without `format_variety`: always the variant's default format + a generic
      (no-format) prompt.
    - 'json' is emitted for a fixed small fraction (`json_ratio`) of rows, always
      paired with an explicit, format-specifying prompt. It is never the default
      format and is never drawn by the generic/uniform path, so its overall share
      is exactly `json_ratio`.
    - Otherwise: with probability `generic_ratio` a generic prompt on the default
      format, else an explicit prompt on a uniformly-random non-json format.

    `rng` is a `random.Random`; `formats` is the variant's format dict (typically
    `_DIAR_FORMATS[variant]` or `formats_for(language, variant)`)."""
    if not format_variety:
        return DIAR_DEFAULT_FORMAT[variant], "generic"
    if "json" in formats and rng.random() < json_ratio:
        return "json", "explicit"
    if rng.random() < generic_ratio:
        return DIAR_DEFAULT_FORMAT[variant], "generic"
    non_json = [k for k in formats if k != "json"]
    return rng.choice(non_json), "explicit"


# --------------------------------------------------------------------------- #
# Language-specific EXTRA formats, merged on top of `_DIAR_FORMATS` only for    #
# rows in that language (see `formats_for`). These are NOT in the base table,   #
# so they never appear in other languages' output and are not subject to the    #
# English-completeness check; instead `_validate_prompts` checks each extra     #
# format has prompts in its own language. Format keys must not collide with the #
# base ones. The default format (generic prompts) stays the base default.       #
# --------------------------------------------------------------------------- #
_LANG_EXTRA_FORMATS = {
    "fr": {
        "asr": {
            "locuteur_colon": {"render": _r_asr_locuteur_colon},
            "locuteur_dash":  {"render": _r_asr_locuteur_dash},
            "locuteur_caps":  {"render": _r_asr_locuteur_caps},
            "locuteur_lower": {"render": _r_asr_locuteur_lower},
        },
        "timestamps": {
            "locuteur_bracket": {"render": _r_ts_locuteur_bracket},
            "locuteur_caps":    {"render": _r_ts_locuteur_caps},
            "locuteur_lower":   {"render": _r_ts_locuteur_lower},
        },
        "timestamps_asr": {
            "locuteur_bracket_colon": {"render": _r_tsa_locuteur_bracket_colon},
            "locuteur_inline":        {"render": _r_tsa_locuteur_inline},
            "locuteur_caps":          {"render": _r_tsa_locuteur_caps},
            "locuteur_lower":         {"render": _r_tsa_locuteur_lower},
        },
    },
}


def formats_for(language, variant):
    """Return the format dict for (language, variant): the base formats plus any
    language-specific extras. Use this instead of `_DIAR_FORMATS[variant]` when
    picking/rendering a format so French rows can also draw 'Locuteur' formats."""
    base = dict(_DIAR_FORMATS[variant])
    extra = _LANG_EXTRA_FORMATS.get(language, {}).get(variant, {})
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# Prompts: keyed by language. Add a new language by adding a top-level entry    #
# mirroring this structure (see the module docstring). Missing prompt lists     #
# fall back to DEFAULT_LANGUAGE.                                                #
# --------------------------------------------------------------------------- #
_DIAR_PROMPTS = {
    "en": {
        # Generic prompts: no format specified / no example. They map to each
        # variant's DEFAULT format, teaching the model its default behaviour.
        "generic": {
            "asr": [
                "Transcribe this conversation, labeling each speaker turn.",
                "Who said what in this recording? Provide a speaker-labeled transcript.",
                "Write the dialogue with one line per speaker turn.",
                "Produce a speaker-attributed transcript of this audio.",
                "Transcribe the audio and label each speaker.",
                "Write out what is said, indicating who speaks each part.",
                "Provide a transcript and identify the different speakers.",
                "Transcribe this recording, separating it by speaker.",
                "Give me the full transcription with speaker labels.",
                "Transcribe the audio and attribute every utterance to a speaker.",
                "Transcribe and diarize this conversation.",
                "Diarise this audio and transcribe each speaker's words.",
                "Transcribe this recording with speaker diarization.",
                "Diarize the speakers and write down what each one says.",
            ],
            "timestamps": [
                "Identify who spoke when in this recording.",
                "Perform speaker diarization on this audio.",
                "List the speaker segments with their start and end times.",
                "Give each speaker turn with its start and end time.",
                "When did each speaker talk? Give the time spans.",
                "Segment this audio by speaker, with timing for each turn.",
                "Mark out the speaker turns and their start and end times.",
                "Diarize this recording and report the time intervals per speaker.",
                "For each speaker, give the intervals during which they were talking.",
                "Produce a speaker timeline with start and end times for every turn.",
                "Diarise this audio.",
                "Perform speaker diarisation and give the timing of each turn.",
                "Diarize this clip: give each speaker turn with its time span.",
                "Run speaker diarization and output the speaker segments.",
            ],
            "timestamps_asr": [
                "Transcribe this conversation with speaker labels and timestamps.",
                "Provide a time-stamped, speaker-attributed transcript of this audio.",
                "For each speaker turn, give the time span, the speaker, and what was said.",
                "Transcribe the audio with timestamps and speaker labels.",
                "Identify who spoke when and who said what.",
                "Transcribe and diarize this audio.",
                "Give a full transcription and diarisation of the clip.",
                "Give a transcript with, for each turn, the times, the speaker and the words.",
                "Write out the dialogue with timing and speaker labels on every turn.",
                "Produce a timed, speaker-attributed transcript of this recording.",
                "Transcribe everything, noting when each speaker talks and what they say.",
                "Transcribe and diarise this recording, with timestamps for each turn.",
                "Perform speaker diarization and transcribe each turn with its time span.",
                "Diarize this audio and produce a timed transcript of every speaker turn.",
            ],
        },
        # Per-format prompts: each explicitly specifies that format (with an
        # example), so the default format is reachable both generically and here.
        "formats": {
            "asr": {
                "speaker_colon": [
                    "Transcribe the dialogue. Use the format 'Speaker N: <words>' for each turn.",
                    "Write each turn as 'Speaker N: <words>'.",
                    "Produce a speaker-labeled transcript, one line per turn as 'Speaker N: <words>'.",
                    "Transcribe this audio, prefixing each turn with 'Speaker N:'.",
                    "Give the transcript with every turn formatted as 'Speaker N: <words>'.",
                ],
                "bare_letter": [
                    "Transcribe the dialogue. Use the format 'A: <words>' "
                    "(label speakers A, B, C, ...).",
                    "Write each turn as 'X: <words>', using bare letters A, B, C for speakers.",
                    "Transcribe this audio, prefixing each turn with a speaker letter: 'A: <words>'.",
                    "Give the transcript with turns as 'A: <words>', 'B: <words>', and so on.",
                    "Label speakers with single letters and write each turn as 'A: <words>'.",
                ],
                "s_num": [
                    "Transcribe the dialogue. Use the format 'S1: <words>' "
                    "(label speakers S1, S2, S3, ...).",
                    "Write each turn as 'SN: <words>' where N is the speaker number.",
                    "Transcribe this audio, prefixing each turn with 'S1:', 'S2:', etc.",
                    "Give the transcript with turns as 'S1: <words>', 'S2: <words>', and so on.",
                    "Label speakers S1, S2, S3, ... and write each turn as 'SN: <words>'.",
                ],
                "speaker_upper": [
                    "Transcribe the dialogue. Use the format 'SPEAKER_00: <words>' "
                    "(label speakers SPEAKER_00, SPEAKER_01, ...).",
                    "Write each turn as 'SPEAKER_NN: <words>' with zero-padded indices from 00.",
                    "Transcribe this audio, prefixing each turn with 'SPEAKER_00:', 'SPEAKER_01:', etc.",
                    "Give the transcript with turns as 'SPEAKER_00: <words>', 'SPEAKER_01: <words>', ...",
                    "Label speakers SPEAKER_00, SPEAKER_01, ... and write each turn as 'SPEAKER_NN: <words>'.",
                ],
                "dash": [
                    "Transcribe with speaker labels using the format 'Speaker N - <words>'.",
                    "Separate each speaker label and their words with a dash: 'Speaker N - <words>'.",
                    "Write each turn as 'Speaker N - <words>'.",
                    "Give the transcript with every turn as 'Speaker N - <words>'.",
                    "Transcribe this audio, formatting each turn as 'Speaker N - <words>'.",
                ],
                "letter": [
                    "Transcribe the dialogue. Use the format 'Speaker A: <words>' "
                    "(label speakers A, B, C, ...).",
                    "Write each turn as 'Speaker X: <words>', using letters A, B, C for speakers.",
                    "Transcribe this audio, prefixing each turn with 'Speaker A:', 'Speaker B:', etc.",
                    "Give the transcript with turns as 'Speaker A: <words>', 'Speaker B: <words>', ...",
                    "Label speakers with letters and write each turn as 'Speaker A: <words>'.",
                ],
                "speaker_caps": [
                    "Transcribe the dialogue. Use the format 'SPEAKER N: <words>' "
                    "(the word SPEAKER in all caps).",
                    "Write each turn as 'SPEAKER N: <words>' with SPEAKER in uppercase.",
                    "Transcribe this audio, prefixing each turn with 'SPEAKER 1:', 'SPEAKER 2:', etc.",
                    "Give the transcript with turns as 'SPEAKER N: <words>', SPEAKER fully capitalised.",
                    "Label each turn 'SPEAKER N: <words>' using the uppercase word SPEAKER.",
                ],
                "speaker_lower": [
                    "Transcribe the dialogue. Use the format 'speaker N: <words>' "
                    "(the word speaker in lowercase).",
                    "Write each turn as 'speaker N: <words>' with speaker in lowercase.",
                    "Transcribe this audio, prefixing each turn with 'speaker 1:', 'speaker 2:', etc.",
                    "Give the transcript with turns as 'speaker N: <words>', speaker in lowercase.",
                    "Label each turn 'speaker N: <words>' using the lowercase word speaker.",
                ],
                "json": [
                    # Loose prompts: ask for JSON without spelling out the schema.
                    "Transcribe this conversation with speaker labels and return the result as JSON.",
                    "Give a speaker-labeled transcript of this audio in JSON format.",
                    "Diarize and transcribe this recording; output the result as JSON.",
                    # Schema-specifying prompts.
                    "Transcribe the dialogue as JSON: a \"transcription_result\" string "
                    "('spk1: ...\\nspk2: ...') and a \"segments\" array of "
                    "{\"segment\": \"<words>\", \"spk_id\": \"spkN\"}.",
                    "Output JSON with \"transcription_result\" (speaker-labeled lines) and "
                    "\"segments\": a list of {\"segment\": ..., \"spk_id\": \"spkN\"}.",
                    "Produce a speaker-attributed transcript as a JSON object with keys "
                    "\"transcription_result\" and \"segments\" (each segment has \"segment\" and \"spk_id\").",
                    "Return JSON: \"transcription_result\" joining the turns as 'spkN: <words>', and "
                    "\"segments\" with one {\"segment\", \"spk_id\"} object per turn.",
                    "Give the transcript as JSON with a \"transcription_result\" string and a "
                    "\"segments\" array of {\"segment\": \"<words>\", \"spk_id\": \"spkN\"}.",
                ],
            },
            "timestamps": {
                "plain": [
                    "Output one line per turn as 'Speaker N <start> <end>' in seconds.",
                    "For each turn, give 'Speaker N <start> <end>' (times in seconds).",
                    "Diarize this audio, one line per turn as 'Speaker N <start> <end>'.",
                    "List the speaker turns as 'Speaker N <start> <end>', times in seconds.",
                    "Give each turn as 'Speaker N <start> <end>' with start and end in seconds.",
                    "Diarise this recording, one line per turn as 'Speaker N <start> <end>'.",
                ],
                "rttm": [
                    "Perform speaker diarization and output the result in RTTM format.",
                    "Give the diarization as standard RTTM (SPEAKER ...) lines.",
                    "Diarize this audio and return RTTM-format lines.",
                    "Diarize this audio, following the RTTM format.",
                    "Perform speaker diarization following the RTTM format.",
                    "Output the speaker segmentation as NIST RTTM lines.",
                    "Diarise this audio and return the result as RTTM lines.",
                    "Perform speaker diarisation, following the RTTM format.",
                ],
                "bracket": [
                    "List each segment as '[start - end] Speaker N' in seconds.",
                    "For each speaker turn, give '[<start> - <end>] Speaker N'.",
                    "Diarize this audio, one line per turn as '[start - end] Speaker N'.",
                    "Output the speaker turns as '[<start> - <end>] Speaker N', times in seconds.",
                    "Give each turn as '[start - end] Speaker N'.",
                    "Diarise this recording, one line per turn as '[start - end] Speaker N'.",
                ],
                "bare_letter": [
                    "List each segment as '[start - end] A' in seconds "
                    "(label speakers A, B, C, ...).",
                    "For each speaker turn, give '[<start> - <end>] X' using letters A, B, C.",
                    "Diarize this audio, one line per turn as '[start - end] A'.",
                    "Output the speaker turns as '[<start> - <end>] A', '[<start> - <end>] B', ...",
                    "Give each turn as '[start - end] A', labeling speakers A, B, C, ...",
                    "Diarise this recording, one line per turn as '[start - end] A'.",
                ],
                "s_num": [
                    "List each segment as '[start - end] S1' in seconds "
                    "(label speakers S1, S2, S3, ...).",
                    "For each speaker turn, give '[<start> - <end>] SN'.",
                    "Diarize this audio, one line per turn as '[start - end] S1'.",
                    "Output the speaker turns as '[<start> - <end>] S1', '[<start> - <end>] S2', ...",
                    "Give each turn as '[start - end] SN', labeling speakers S1, S2, S3, ...",
                    "Diarise this recording, one line per turn as '[start - end] S1'.",
                ],
                "speaker_upper": [
                    "List each segment as '[start - end] SPEAKER_00' in seconds "
                    "(label speakers SPEAKER_00, SPEAKER_01, ...).",
                    "For each speaker turn, give '[<start> - <end>] SPEAKER_NN' "
                    "with zero-padded indices from 00.",
                    "Diarize this audio, one line per turn as '[start - end] SPEAKER_00'.",
                    "Output the speaker turns as '[<start> - <end>] SPEAKER_00', "
                    "'[<start> - <end>] SPEAKER_01', ...",
                    "Give each turn as '[start - end] SPEAKER_NN', labeling SPEAKER_00, SPEAKER_01, ...",
                    "Diarise this recording, one line per turn as '[start - end] SPEAKER_00'.",
                ],
                "speaker_caps": [
                    "List each segment as '[start - end] SPEAKER N' in seconds "
                    "(the word SPEAKER in all caps).",
                    "For each speaker turn, give '[<start> - <end>] SPEAKER N', SPEAKER uppercase.",
                    "Diarize this audio, one line per turn as '[start - end] SPEAKER 1'.",
                    "Output the speaker turns as '[<start> - <end>] SPEAKER 1', '[<start> - <end>] SPEAKER 2', ...",
                    "Give each turn as '[start - end] SPEAKER N' using the uppercase word SPEAKER.",
                ],
                "speaker_lower": [
                    "List each segment as '[start - end] speaker N' in seconds "
                    "(the word speaker in lowercase).",
                    "For each speaker turn, give '[<start> - <end>] speaker N', speaker lowercase.",
                    "Diarize this audio, one line per turn as '[start - end] speaker 1'.",
                    "Output the speaker turns as '[<start> - <end>] speaker 1', '[<start> - <end>] speaker 2', ...",
                    "Give each turn as '[start - end] speaker N' using the lowercase word speaker.",
                ],
                "json": [
                    # Loose prompts: ask for JSON without spelling out the schema.
                    "Perform speaker diarization on this audio and return the result as JSON.",
                    "Identify who spoke when and give the result in JSON format.",
                    "Diarize this recording; output the speaker segments as JSON.",
                    # Schema-specifying prompts.
                    "Diarize this audio as JSON with a \"speakers\" array of "
                    "{\"spk_id\": \"spkN\", \"duration\": <s>, \"nbr_seg\": <n>} and a \"segments\" "
                    "array of {\"seg_id\": <i>, \"spk_id\": \"spkN\", \"seg_begin\": <s>, \"seg_end\": <s>}.",
                    "Output JSON with \"speakers\" (per-speaker total \"duration\" and segment count "
                    "\"nbr_seg\") and \"segments\" ({\"seg_id\", \"spk_id\", \"seg_begin\", \"seg_end\"}).",
                    "Perform speaker diarization and return a JSON object: \"speakers\" summarising each "
                    "spk_id's duration and nbr_seg, and \"segments\" listing every turn with seg_begin/seg_end.",
                    "Give the diarization as JSON, times in seconds: a \"speakers\" summary "
                    "({spk_id, duration, nbr_seg}) and a \"segments\" list ({seg_id, spk_id, seg_begin, seg_end}).",
                    "Diarise this recording into JSON with two arrays: \"speakers\" "
                    "({\"spk_id\", \"duration\", \"nbr_seg\"}) and \"segments\" "
                    "({\"seg_id\", \"spk_id\", \"seg_begin\", \"seg_end\"}).",
                ],
            },
            "timestamps_asr": {
                "bracket_colon": [
                    "Format each turn as '[start-end] Speaker N: <words>' in seconds.",
                    "Write each turn as '[<start>-<end>] Speaker N: <words>'.",
                    "Transcribe with timestamps, each turn as '[start-end] Speaker N: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] Speaker N: <words>'.",
                    "Output each turn as '[start-end] Speaker N: <words>', times in seconds.",
                    "Transcribe and diarise this audio, each turn as '[start-end] Speaker N: <words>'.",
                ],
                "bare_letter": [
                    "Format each turn as '[start-end] A: <words>' in seconds "
                    "(label speakers A, B, C, ...).",
                    "Write each turn as '[<start>-<end>] X: <words>' using letters A, B, C.",
                    "Transcribe with timestamps, each turn as '[start-end] A: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] A: <words>', "
                    "'[<start>-<end>] B: <words>', ...",
                    "Output each turn as '[start-end] A: <words>', labeling speakers A, B, C, ...",
                    "Transcribe and diarise this audio, each turn as '[start-end] A: <words>'.",
                ],
                "s_num": [
                    "Format each turn as '[start-end] S1: <words>' in seconds "
                    "(label speakers S1, S2, S3, ...).",
                    "Write each turn as '[<start>-<end>] SN: <words>'.",
                    "Transcribe with timestamps, each turn as '[start-end] S1: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] S1: <words>', "
                    "'[<start>-<end>] S2: <words>', ...",
                    "Output each turn as '[start-end] SN: <words>', labeling speakers S1, S2, S3, ...",
                    "Transcribe and diarise this audio, each turn as '[start-end] S1: <words>'.",
                ],
                "speaker_upper": [
                    "Format each turn as '[start-end] SPEAKER_00: <words>' in seconds "
                    "(label speakers SPEAKER_00, SPEAKER_01, ...).",
                    "Write each turn as '[<start>-<end>] SPEAKER_NN: <words>' "
                    "with zero-padded indices from 00.",
                    "Transcribe with timestamps, each turn as '[start-end] SPEAKER_00: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] SPEAKER_00: <words>', "
                    "'[<start>-<end>] SPEAKER_01: <words>', ...",
                    "Output each turn as '[start-end] SPEAKER_NN: <words>', "
                    "labeling SPEAKER_00, SPEAKER_01, ...",
                    "Transcribe and diarise this audio, each turn as '[start-end] SPEAKER_00: <words>'.",
                ],
                "inline": [
                    "Transcribe with timestamps using the format 'Speaker N (start-end): <words>'.",
                    "Write each turn as 'Speaker N (<start>-<end>): <words>'.",
                    "Give a timed transcript with turns as 'Speaker N (start-end): <words>'.",
                    "Output each turn as 'Speaker N (<start>-<end>): <words>', times in seconds.",
                    "Transcribe this audio, formatting each turn as 'Speaker N (start-end): <words>'.",
                    "Transcribe and diarise this audio, each turn as 'Speaker N (start-end): <words>'.",
                ],
                "arrow": [
                    "Transcribe with timestamps. Use the format '<start> --> <end>  Speaker N: <words>'.",
                    "Use simple cues: '<start> --> <end>  Speaker N: <words>'.",
                    "Give a timed transcript with turns as '<start> --> <end>  Speaker N: <words>'.",
                    "Output each turn as '<start> --> <end>  Speaker N: <words>', times in seconds.",
                    "Transcribe this audio, formatting each turn as '<start> --> <end>  Speaker N: <words>'.",
                    "Transcribe and diarise this audio, each turn as '<start> --> <end>  Speaker N: <words>'.",
                ],
                "srt": [
                    "Transcribe with timestamps as SRT subtitles.",
                    "Produce SubRip (.srt) subtitles with speaker labels.",
                    "Output the transcript as SRT: numbered cues, 'HH:MM:SS,mmm' times, "
                    "and 'Speaker N: <words>'.",
                    "Transcribe this audio, following the SRT format.",
                    "Transcribe the speakers following the SRT subtitle format.",
                    "Generate SRT subtitles with each cue labeled by speaker.",
                    "Diarise this audio and write the transcript as SRT subtitles.",
                ],
                "vtt": [
                    "Transcribe with timestamps as WebVTT (.vtt) subtitles.",
                    "Output the transcript in WebVTT format with speaker labels.",
                    "Produce WEBVTT cues ('HH:MM:SS.mmm' times) with 'Speaker N: <words>'.",
                    "Transcribe this audio, following the WebVTT format.",
                    "Transcribe the speakers following the WebVTT (.vtt) format.",
                    "Generate WebVTT subtitles with each cue labeled by speaker.",
                    "Diarise this audio and write the transcript as WebVTT subtitles.",
                ],
                "speaker_caps": [
                    "Format each turn as '[start-end] SPEAKER N: <words>' in seconds "
                    "(the word SPEAKER in all caps).",
                    "Write each turn as '[<start>-<end>] SPEAKER N: <words>', SPEAKER uppercase.",
                    "Transcribe with timestamps, each turn as '[start-end] SPEAKER 1: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] SPEAKER N: <words>', SPEAKER capitalised.",
                    "Output each turn as '[start-end] SPEAKER N: <words>' using the uppercase word SPEAKER.",
                ],
                "speaker_lower": [
                    "Format each turn as '[start-end] speaker N: <words>' in seconds "
                    "(the word speaker in lowercase).",
                    "Write each turn as '[<start>-<end>] speaker N: <words>', speaker lowercase.",
                    "Transcribe with timestamps, each turn as '[start-end] speaker 1: <words>'.",
                    "Give a timed transcript with turns as '[<start>-<end>] speaker N: <words>', speaker in lowercase.",
                    "Output each turn as '[start-end] speaker N: <words>' using the lowercase word speaker.",
                ],
                "json": [
                    # Loose prompts: ask for JSON without spelling out the schema.
                    "Transcribe this audio with speaker labels and timestamps, and return the result as JSON.",
                    "Give a timed, speaker-attributed transcript in JSON format.",
                    "Transcribe and diarize this recording; output the result as JSON.",
                    # Schema-specifying prompts.
                    "Transcribe with timestamps as JSON: a \"transcription_result\" string "
                    "('spkN: ...') and a \"segments\" array of {\"segment\": \"<words>\", "
                    "\"start\": <s>, \"end\": <s>, \"duration\": <s>, \"spk_id\": \"spkN\"}.",
                    "Output JSON with \"transcription_result\" (speaker-labeled lines) and \"segments\", "
                    "each {\"segment\", \"start\", \"end\", \"duration\", \"spk_id\"}, times in seconds.",
                    "Produce a timed, speaker-attributed transcript as a JSON object with "
                    "\"transcription_result\" and a \"segments\" list carrying segment text, start, end, "
                    "duration and spk_id.",
                    "Transcribe and diarize this audio into JSON: a \"transcription_result\" string plus "
                    "\"segments\" of {\"segment\", \"start\", \"end\", \"duration\", \"spk_id\"}.",
                    "Give a timed transcript as JSON with \"transcription_result\" and a \"segments\" array "
                    "where each turn has \"segment\", \"start\", \"end\", \"duration\" and \"spk_id\".",
                ],
            },
        },
        # Appended to the prompt of the backchannel-included version of a row.
        "backchannel_suffixes": {
            # Variants that emit words (asr, timestamps_asr): talk about transcribing them.
            "transcribed": [
                "Include backchannels, acknowledgements and disfluencies (e.g. 'mm-hmm', 'yeah', 'uh').",
                "Keep all backchannels and acknowledgements such as 'mm', 'hmm', 'uh-huh'.",
                "Include every utterance, even short backchannels and fillers.",
                "Do not omit backchannels or filler words like 'mm', 'uh' and 'yeah'.",
                "Transcribe everything, including acknowledgements and hesitations.",
                "Keep the backchannels, fillers and disfluencies in the output.",
            ],
            # Timestamps-only (who-spoke-when, no words): talk about keeping the segments.
            "timestamps": [
                "Include short backchannel turns (e.g. 'mm-hmm', 'yeah') as separate segments.",
                "Keep every turn, even brief backchannels and acknowledgements.",
                "Do not drop short backchannel or filler turns; segment them too.",
                "Include all turns, including momentary backchannels and fillers.",
            ],
        },
    },
    "fr": {
        # Generic prompts: no format specified / no example. They map to each
        # variant's DEFAULT format, teaching the model its default behaviour.
        "generic": {
            "asr": [
                "Transcris cette conversation en indiquant chaque tour de parole.",
                "Qui dit quoi dans cet enregistrement ? Fournis une transcription avec les locuteurs.",
                "Écris le dialogue avec une ligne par tour de parole.",
                "Produis une transcription attribuée aux locuteurs de cet audio.",
                "Transcris l'audio et identifie chaque locuteur.",
                "Écris ce qui est dit en indiquant qui parle à chaque fois.",
                "Fournis une transcription et distingue les différents locuteurs.",
                "Transcris cet enregistrement en le séparant par locuteur.",
                "Donne-moi la transcription complète avec les étiquettes de locuteur.",
                "Transcris l'audio et attribue chaque énoncé à un locuteur.",
                "Transcris et segmente cette conversation par locuteur.",
                "Effectue la diarisation de cet audio et transcris les paroles de chaque locuteur.",
                "Transcris cet enregistrement avec diarisation des locuteurs.",
                "Identifie les locuteurs et écris ce que dit chacun d'eux.",
            ],
            "timestamps": [
                "Identifie qui a parlé et quand dans cet enregistrement.",
                "Effectue la diarisation des locuteurs sur cet audio.",
                "Liste les segments de parole avec leurs temps de début et de fin.",
                "Donne chaque tour de parole avec son temps de début et de fin.",
                "Quand chaque locuteur a-t-il parlé ? Donne les intervalles de temps.",
                "Segmente cet audio par locuteur, avec le minutage de chaque tour.",
                "Délimite les tours de parole et leurs temps de début et de fin.",
                "Effectue la diarisation et indique les intervalles de temps par locuteur.",
                "Pour chaque locuteur, donne les intervalles pendant lesquels il parle.",
                "Produis une chronologie des locuteurs avec les temps de début et de fin de chaque tour.",
                "Effectue la diarisation de cet audio.",
                "Effectue la diarisation et donne le minutage de chaque tour.",
                "Diarise ce clip : donne chaque tour de parole avec son intervalle de temps.",
                "Lance la diarisation des locuteurs et restitue les segments de parole.",
            ],
            "timestamps_asr": [
                "Transcris cette conversation avec les étiquettes de locuteur et les horodatages.",
                "Fournis une transcription horodatée et attribuée aux locuteurs de cet audio.",
                "Pour chaque tour de parole, donne l'intervalle de temps, le locuteur et ce qui est dit.",
                "Transcris l'audio avec les horodatages et les étiquettes de locuteur.",
                "Identifie qui a parlé et quand, et qui a dit quoi.",
                "Transcris et diarise cet audio.",
                "Donne une transcription et une diarisation complètes de ce clip.",
                "Donne une transcription avec, pour chaque tour, les temps, le locuteur et les paroles.",
                "Écris le dialogue avec le minutage et les étiquettes de locuteur sur chaque tour.",
                "Produis une transcription horodatée et attribuée aux locuteurs de cet enregistrement.",
                "Transcris tout, en notant quand chaque locuteur parle et ce qu'il dit.",
                "Transcris et diarise cet enregistrement, avec les horodatages de chaque tour.",
                "Effectue la diarisation et transcris chaque tour avec son intervalle de temps.",
                "Diarise cet audio et produis une transcription horodatée de chaque tour de parole.",
            ],
        },
        # Per-format prompts: each explicitly specifies that format (with an
        # example). The example tokens (Speaker N, RTTM, SRT, ...) are
        # language-independent because the renderers output them verbatim.
        "formats": {
            "asr": {
                "speaker_colon": [
                    "Transcris le dialogue. Utilise le format 'Speaker N: <paroles>' pour chaque tour.",
                    "Écris chaque tour sous la forme 'Speaker N: <paroles>'.",
                    "Produis une transcription par locuteur, une ligne par tour, au format 'Speaker N: <paroles>'.",
                    "Transcris cet audio en préfixant chaque tour par 'Speaker N:'.",
                    "Donne la transcription avec chaque tour au format 'Speaker N: <paroles>'.",
                ],
                "bare_letter": [
                    "Transcris le dialogue. Utilise le format 'A: <paroles>' "
                    "(étiquette les locuteurs A, B, C, ...).",
                    "Écris chaque tour sous la forme 'X: <paroles>', avec des lettres A, B, C pour les locuteurs.",
                    "Transcris cet audio en préfixant chaque tour par une lettre de locuteur : 'A: <paroles>'.",
                    "Donne la transcription avec les tours sous la forme 'A: <paroles>', 'B: <paroles>', etc.",
                    "Étiquette les locuteurs par une seule lettre et écris chaque tour 'A: <paroles>'.",
                ],
                "s_num": [
                    "Transcris le dialogue. Utilise le format 'S1: <paroles>' "
                    "(étiquette les locuteurs S1, S2, S3, ...).",
                    "Écris chaque tour sous la forme 'SN: <paroles>' où N est le numéro du locuteur.",
                    "Transcris cet audio en préfixant chaque tour par 'S1:', 'S2:', etc.",
                    "Donne la transcription avec les tours sous la forme 'S1: <paroles>', 'S2: <paroles>', etc.",
                    "Étiquette les locuteurs S1, S2, S3, ... et écris chaque tour 'SN: <paroles>'.",
                ],
                "speaker_upper": [
                    "Transcris le dialogue. Utilise le format 'SPEAKER_00: <paroles>' "
                    "(étiquette les locuteurs SPEAKER_00, SPEAKER_01, ...).",
                    "Écris chaque tour sous la forme 'SPEAKER_NN: <paroles>' avec des indices à partir de 00.",
                    "Transcris cet audio en préfixant chaque tour par 'SPEAKER_00:', 'SPEAKER_01:', etc.",
                    "Donne la transcription avec les tours 'SPEAKER_00: <paroles>', 'SPEAKER_01: <paroles>', ...",
                    "Étiquette les locuteurs SPEAKER_00, SPEAKER_01, ... et écris chaque tour 'SPEAKER_NN: <paroles>'.",
                ],
                "dash": [
                    "Transcris avec les étiquettes de locuteur au format 'Speaker N - <paroles>'.",
                    "Sépare l'étiquette du locuteur et ses paroles par un tiret : 'Speaker N - <paroles>'.",
                    "Écris chaque tour sous la forme 'Speaker N - <paroles>'.",
                    "Donne la transcription avec chaque tour 'Speaker N - <paroles>'.",
                    "Transcris cet audio en formatant chaque tour 'Speaker N - <paroles>'.",
                ],
                "letter": [
                    "Transcris le dialogue. Utilise le format 'Speaker A: <paroles>' "
                    "(étiquette les locuteurs A, B, C, ...).",
                    "Écris chaque tour sous la forme 'Speaker X: <paroles>', avec des lettres A, B, C.",
                    "Transcris cet audio en préfixant chaque tour par 'Speaker A:', 'Speaker B:', etc.",
                    "Donne la transcription avec les tours 'Speaker A: <paroles>', 'Speaker B: <paroles>', ...",
                    "Étiquette les locuteurs par des lettres et écris chaque tour 'Speaker A: <paroles>'.",
                ],
                "locuteur_colon": [
                    "Transcris le dialogue. Utilise le format 'Locuteur N: <paroles>' pour chaque tour.",
                    "Écris chaque tour sous la forme 'Locuteur N: <paroles>'.",
                    "Produis une transcription par locuteur, une ligne par tour, au format 'Locuteur N: <paroles>'.",
                    "Transcris cet audio en préfixant chaque tour par 'Locuteur N:'.",
                    "Donne la transcription avec chaque tour au format 'Locuteur N: <paroles>'.",
                ],
                "locuteur_dash": [
                    "Transcris avec les étiquettes de locuteur au format 'Locuteur N - <paroles>'.",
                    "Sépare l'étiquette du locuteur et ses paroles par un tiret : 'Locuteur N - <paroles>'.",
                    "Écris chaque tour sous la forme 'Locuteur N - <paroles>'.",
                    "Donne la transcription avec chaque tour 'Locuteur N - <paroles>'.",
                    "Transcris cet audio en formatant chaque tour 'Locuteur N - <paroles>'.",
                ],
                "locuteur_caps": [
                    "Transcris le dialogue. Utilise le format 'LOCUTEUR N: <paroles>' "
                    "(le mot LOCUTEUR en majuscules).",
                    "Écris chaque tour sous la forme 'LOCUTEUR N: <paroles>', LOCUTEUR en majuscules.",
                    "Transcris cet audio en préfixant chaque tour par 'LOCUTEUR 1:', 'LOCUTEUR 2:', etc.",
                    "Donne la transcription avec chaque tour 'LOCUTEUR N: <paroles>', LOCUTEUR tout en majuscules.",
                    "Étiquette chaque tour 'LOCUTEUR N: <paroles>' avec le mot LOCUTEUR en majuscules.",
                ],
                "locuteur_lower": [
                    "Transcris le dialogue. Utilise le format 'locuteur N: <paroles>' "
                    "(le mot locuteur en minuscules).",
                    "Écris chaque tour sous la forme 'locuteur N: <paroles>', locuteur en minuscules.",
                    "Transcris cet audio en préfixant chaque tour par 'locuteur 1:', 'locuteur 2:', etc.",
                    "Donne la transcription avec chaque tour 'locuteur N: <paroles>', locuteur en minuscules.",
                    "Étiquette chaque tour 'locuteur N: <paroles>' avec le mot locuteur en minuscules.",
                ],
                "speaker_caps": [
                    "Transcris le dialogue. Utilise le format 'SPEAKER N: <paroles>' "
                    "(le mot SPEAKER en majuscules).",
                    "Écris chaque tour sous la forme 'SPEAKER N: <paroles>', SPEAKER en majuscules.",
                    "Transcris cet audio en préfixant chaque tour par 'SPEAKER 1:', 'SPEAKER 2:', etc.",
                    "Donne la transcription avec chaque tour 'SPEAKER N: <paroles>', SPEAKER tout en majuscules.",
                    "Étiquette chaque tour 'SPEAKER N: <paroles>' avec le mot SPEAKER en majuscules.",
                ],
                "speaker_lower": [
                    "Transcris le dialogue. Utilise le format 'speaker N: <paroles>' "
                    "(le mot speaker en minuscules).",
                    "Écris chaque tour sous la forme 'speaker N: <paroles>', speaker en minuscules.",
                    "Transcris cet audio en préfixant chaque tour par 'speaker 1:', 'speaker 2:', etc.",
                    "Donne la transcription avec chaque tour 'speaker N: <paroles>', speaker en minuscules.",
                    "Étiquette chaque tour 'speaker N: <paroles>' avec le mot speaker en minuscules.",
                ],
                "json": [
                    # Prompts génériques : demandent du JSON sans détailler le schéma.
                    "Transcris cette conversation avec les étiquettes de locuteur et renvoie le résultat en JSON.",
                    "Donne une transcription par locuteur de cet audio au format JSON.",
                    "Transcris et diarise cet enregistrement ; restitue le résultat en JSON.",
                    # Prompts spécifiant le schéma.
                    "Transcris le dialogue en JSON : une chaîne \"transcription_result\" "
                    "('spk1: ...\\nspk2: ...') et un tableau \"segments\" d'objets "
                    "{\"segment\": \"<paroles>\", \"spk_id\": \"spkN\"}.",
                    "Restitue du JSON avec \"transcription_result\" (lignes étiquetées par locuteur) et "
                    "\"segments\" : une liste de {\"segment\": ..., \"spk_id\": \"spkN\"}.",
                    "Produis une transcription par locuteur sous forme d'objet JSON avec les clés "
                    "\"transcription_result\" et \"segments\" (chaque segment a \"segment\" et \"spk_id\").",
                    "Renvoie du JSON : \"transcription_result\" joignant les tours sous la forme 'spkN: <paroles>', "
                    "et \"segments\" avec un objet {\"segment\", \"spk_id\"} par tour.",
                    "Donne la transcription en JSON avec une chaîne \"transcription_result\" et un tableau "
                    "\"segments\" de {\"segment\": \"<paroles>\", \"spk_id\": \"spkN\"}.",
                ],
            },
            "timestamps": {
                "plain": [
                    "Produis une ligne par tour sous la forme 'Speaker N <début> <fin>' en secondes.",
                    "Pour chaque tour, donne 'Speaker N <début> <fin>' (temps en secondes).",
                    "Diarise cet audio, une ligne par tour 'Speaker N <début> <fin>'.",
                    "Liste les tours de parole 'Speaker N <début> <fin>', temps en secondes.",
                    "Donne chaque tour 'Speaker N <début> <fin>' avec début et fin en secondes.",
                    "Diarise cet enregistrement, une ligne par tour 'Speaker N <début> <fin>'.",
                ],
                "rttm": [
                    "Effectue la diarisation et restitue le résultat au format RTTM.",
                    "Donne la diarisation sous forme de lignes RTTM standard (SPEAKER ...).",
                    "Diarise cet audio et renvoie des lignes au format RTTM.",
                    "Diarise cet audio en suivant le format RTTM.",
                    "Effectue la diarisation des locuteurs en suivant le format RTTM.",
                    "Restitue la segmentation des locuteurs sous forme de lignes RTTM (NIST).",
                    "Diarise cet audio et renvoie le résultat sous forme de lignes RTTM.",
                ],
                "bracket": [
                    "Liste chaque segment sous la forme '[début - fin] Speaker N' en secondes.",
                    "Pour chaque tour, donne '[<début> - <fin>] Speaker N'.",
                    "Diarise cet audio, une ligne par tour '[début - fin] Speaker N'.",
                    "Restitue les tours sous la forme '[<début> - <fin>] Speaker N', temps en secondes.",
                    "Donne chaque tour '[début - fin] Speaker N'.",
                    "Diarise cet enregistrement, une ligne par tour '[début - fin] Speaker N'.",
                ],
                "bare_letter": [
                    "Liste chaque segment sous la forme '[début - fin] A' en secondes "
                    "(étiquette les locuteurs A, B, C, ...).",
                    "Pour chaque tour, donne '[<début> - <fin>] X' avec des lettres A, B, C.",
                    "Diarise cet audio, une ligne par tour '[début - fin] A'.",
                    "Restitue les tours '[<début> - <fin>] A', '[<début> - <fin>] B', ...",
                    "Donne chaque tour '[début - fin] A', en étiquetant les locuteurs A, B, C, ...",
                    "Diarise cet enregistrement, une ligne par tour '[début - fin] A'.",
                ],
                "s_num": [
                    "Liste chaque segment sous la forme '[début - fin] S1' en secondes "
                    "(étiquette les locuteurs S1, S2, S3, ...).",
                    "Pour chaque tour, donne '[<début> - <fin>] SN'.",
                    "Diarise cet audio, une ligne par tour '[début - fin] S1'.",
                    "Restitue les tours '[<début> - <fin>] S1', '[<début> - <fin>] S2', ...",
                    "Donne chaque tour '[début - fin] SN', en étiquetant les locuteurs S1, S2, S3, ...",
                    "Diarise cet enregistrement, une ligne par tour '[début - fin] S1'.",
                ],
                "speaker_upper": [
                    "Liste chaque segment sous la forme '[début - fin] SPEAKER_00' en secondes "
                    "(étiquette les locuteurs SPEAKER_00, SPEAKER_01, ...).",
                    "Pour chaque tour, donne '[<début> - <fin>] SPEAKER_NN' avec des indices à partir de 00.",
                    "Diarise cet audio, une ligne par tour '[début - fin] SPEAKER_00'.",
                    "Restitue les tours '[<début> - <fin>] SPEAKER_00', '[<début> - <fin>] SPEAKER_01', ...",
                    "Donne chaque tour '[début - fin] SPEAKER_NN', en étiquetant SPEAKER_00, SPEAKER_01, ...",
                    "Diarise cet enregistrement, une ligne par tour '[début - fin] SPEAKER_00'.",
                ],
                "locuteur_bracket": [
                    "Liste chaque segment sous la forme '[début - fin] Locuteur N' en secondes.",
                    "Pour chaque tour, donne '[<début> - <fin>] Locuteur N'.",
                    "Diarise cet audio, une ligne par tour '[début - fin] Locuteur N'.",
                    "Restitue les tours sous la forme '[<début> - <fin>] Locuteur N', temps en secondes.",
                    "Donne chaque tour '[début - fin] Locuteur N'.",
                    "Diarise cet enregistrement, une ligne par tour '[début - fin] Locuteur N'.",
                ],
                "locuteur_caps": [
                    "Liste chaque segment sous la forme '[début - fin] LOCUTEUR N' en secondes "
                    "(le mot LOCUTEUR en majuscules).",
                    "Pour chaque tour, donne '[<début> - <fin>] LOCUTEUR N', LOCUTEUR en majuscules.",
                    "Diarise cet audio, une ligne par tour '[début - fin] LOCUTEUR 1'.",
                    "Restitue les tours '[<début> - <fin>] LOCUTEUR 1', '[<début> - <fin>] LOCUTEUR 2', ...",
                    "Donne chaque tour '[début - fin] LOCUTEUR N', LOCUTEUR tout en majuscules.",
                ],
                "locuteur_lower": [
                    "Liste chaque segment sous la forme '[début - fin] locuteur N' en secondes "
                    "(le mot locuteur en minuscules).",
                    "Pour chaque tour, donne '[<début> - <fin>] locuteur N', locuteur en minuscules.",
                    "Diarise cet audio, une ligne par tour '[début - fin] locuteur 1'.",
                    "Restitue les tours '[<début> - <fin>] locuteur 1', '[<début> - <fin>] locuteur 2', ...",
                    "Donne chaque tour '[début - fin] locuteur N', locuteur en minuscules.",
                ],
                "speaker_caps": [
                    "Liste chaque segment sous la forme '[début - fin] SPEAKER N' en secondes "
                    "(le mot SPEAKER en majuscules).",
                    "Pour chaque tour, donne '[<début> - <fin>] SPEAKER N', SPEAKER en majuscules.",
                    "Diarise cet audio, une ligne par tour '[début - fin] SPEAKER 1'.",
                    "Restitue les tours '[<début> - <fin>] SPEAKER 1', '[<début> - <fin>] SPEAKER 2', ...",
                    "Donne chaque tour '[début - fin] SPEAKER N', SPEAKER tout en majuscules.",
                ],
                "speaker_lower": [
                    "Liste chaque segment sous la forme '[début - fin] speaker N' en secondes "
                    "(le mot speaker en minuscules).",
                    "Pour chaque tour, donne '[<début> - <fin>] speaker N', speaker en minuscules.",
                    "Diarise cet audio, une ligne par tour '[début - fin] speaker 1'.",
                    "Restitue les tours '[<début> - <fin>] speaker 1', '[<début> - <fin>] speaker 2', ...",
                    "Donne chaque tour '[début - fin] speaker N', speaker en minuscules.",
                ],
                "json": [
                    # Prompts génériques : demandent du JSON sans détailler le schéma.
                    "Effectue la diarisation des locuteurs sur cet audio et renvoie le résultat en JSON.",
                    "Identifie qui a parlé et quand, et donne le résultat au format JSON.",
                    "Diarise cet enregistrement ; restitue les segments de parole en JSON.",
                    # Prompts spécifiant le schéma.
                    "Diarise cet audio en JSON avec un tableau \"speakers\" de "
                    "{\"spk_id\": \"spkN\", \"duration\": <s>, \"nbr_seg\": <n>} et un tableau \"segments\" de "
                    "{\"seg_id\": <i>, \"spk_id\": \"spkN\", \"seg_begin\": <s>, \"seg_end\": <s>}.",
                    "Restitue du JSON avec \"speakers\" (durée totale \"duration\" et nombre de segments "
                    "\"nbr_seg\" par locuteur) et \"segments\" ({\"seg_id\", \"spk_id\", \"seg_begin\", \"seg_end\"}).",
                    "Effectue la diarisation et renvoie un objet JSON : \"speakers\" résumant la duration et "
                    "le nbr_seg de chaque spk_id, et \"segments\" listant chaque tour avec seg_begin/seg_end.",
                    "Donne la diarisation en JSON, temps en secondes : un résumé \"speakers\" "
                    "({spk_id, duration, nbr_seg}) et une liste \"segments\" ({seg_id, spk_id, seg_begin, seg_end}).",
                    "Diarise cet enregistrement en JSON avec deux tableaux : \"speakers\" "
                    "({\"spk_id\", \"duration\", \"nbr_seg\"}) et \"segments\" "
                    "({\"seg_id\", \"spk_id\", \"seg_begin\", \"seg_end\"}).",
                ],
            },
            "timestamps_asr": {
                "bracket_colon": [
                    "Formate chaque tour sous la forme '[début-fin] Speaker N: <paroles>' en secondes.",
                    "Écris chaque tour sous la forme '[<début>-<fin>] Speaker N: <paroles>'.",
                    "Transcris avec horodatage, chaque tour '[début-fin] Speaker N: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '[<début>-<fin>] Speaker N: <paroles>'.",
                    "Restitue chaque tour '[début-fin] Speaker N: <paroles>', temps en secondes.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] Speaker N: <paroles>'.",
                ],
                "bare_letter": [
                    "Formate chaque tour sous la forme '[début-fin] A: <paroles>' en secondes "
                    "(étiquette les locuteurs A, B, C, ...).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] X: <paroles>' avec des lettres A, B, C.",
                    "Transcris avec horodatage, chaque tour '[début-fin] A: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '[<début>-<fin>] A: <paroles>', "
                    "'[<début>-<fin>] B: <paroles>', ...",
                    "Restitue chaque tour '[début-fin] A: <paroles>', en étiquetant les locuteurs A, B, C, ...",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] A: <paroles>'.",
                ],
                "s_num": [
                    "Formate chaque tour sous la forme '[début-fin] S1: <paroles>' en secondes "
                    "(étiquette les locuteurs S1, S2, S3, ...).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] SN: <paroles>'.",
                    "Transcris avec horodatage, chaque tour '[début-fin] S1: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '[<début>-<fin>] S1: <paroles>', "
                    "'[<début>-<fin>] S2: <paroles>', ...",
                    "Restitue chaque tour '[début-fin] SN: <paroles>', en étiquetant les locuteurs S1, S2, S3, ...",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] S1: <paroles>'.",
                ],
                "speaker_upper": [
                    "Formate chaque tour sous la forme '[début-fin] SPEAKER_00: <paroles>' en secondes "
                    "(étiquette les locuteurs SPEAKER_00, SPEAKER_01, ...).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] SPEAKER_NN: <paroles>' "
                    "avec des indices à partir de 00.",
                    "Transcris avec horodatage, chaque tour '[début-fin] SPEAKER_00: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '[<début>-<fin>] SPEAKER_00: <paroles>', "
                    "'[<début>-<fin>] SPEAKER_01: <paroles>', ...",
                    "Restitue chaque tour '[début-fin] SPEAKER_NN: <paroles>', "
                    "en étiquetant SPEAKER_00, SPEAKER_01, ...",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] SPEAKER_00: <paroles>'.",
                ],
                "inline": [
                    "Transcris avec horodatage au format 'Speaker N (début-fin): <paroles>'.",
                    "Écris chaque tour sous la forme 'Speaker N (<début>-<fin>): <paroles>'.",
                    "Donne une transcription horodatée avec les tours 'Speaker N (début-fin): <paroles>'.",
                    "Restitue chaque tour 'Speaker N (<début>-<fin>): <paroles>', temps en secondes.",
                    "Transcris cet audio en formatant chaque tour 'Speaker N (début-fin): <paroles>'.",
                    "Transcris et diarise cet audio, chaque tour 'Speaker N (début-fin): <paroles>'.",
                ],
                "arrow": [
                    "Transcris avec horodatage. Utilise le format '<début> --> <fin>  Speaker N: <paroles>'.",
                    "Utilise des repères simples : '<début> --> <fin>  Speaker N: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '<début> --> <fin>  Speaker N: <paroles>'.",
                    "Restitue chaque tour '<début> --> <fin>  Speaker N: <paroles>', temps en secondes.",
                    "Transcris cet audio en formatant chaque tour '<début> --> <fin>  Speaker N: <paroles>'.",
                    "Transcris et diarise cet audio, chaque tour '<début> --> <fin>  Speaker N: <paroles>'.",
                ],
                "srt": [
                    "Transcris avec horodatage sous forme de sous-titres SRT.",
                    "Produis des sous-titres SubRip (.srt) avec les étiquettes de locuteur.",
                    "Restitue la transcription au format SRT : repères numérotés, temps 'HH:MM:SS,mmm', "
                    "et 'Speaker N: <paroles>'.",
                    "Transcris cet audio en suivant le format SRT.",
                    "Transcris les locuteurs en suivant le format de sous-titres SRT.",
                    "Génère des sous-titres SRT avec chaque repère étiqueté par locuteur.",
                    "Diarise cet audio et écris la transcription sous forme de sous-titres SRT.",
                ],
                "vtt": [
                    "Transcris avec horodatage sous forme de sous-titres WebVTT (.vtt).",
                    "Restitue la transcription au format WebVTT avec les étiquettes de locuteur.",
                    "Produis des repères WEBVTT (temps 'HH:MM:SS.mmm') avec 'Speaker N: <paroles>'.",
                    "Transcris cet audio en suivant le format WebVTT.",
                    "Transcris les locuteurs en suivant le format WebVTT (.vtt).",
                    "Génère des sous-titres WebVTT avec chaque repère étiqueté par locuteur.",
                    "Diarise cet audio et écris la transcription sous forme de sous-titres WebVTT.",
                ],
                "locuteur_bracket_colon": [
                    "Formate chaque tour sous la forme '[début-fin] Locuteur N: <paroles>' en secondes.",
                    "Écris chaque tour sous la forme '[<début>-<fin>] Locuteur N: <paroles>'.",
                    "Transcris avec horodatage, chaque tour '[début-fin] Locuteur N: <paroles>'.",
                    "Donne une transcription horodatée avec les tours '[<début>-<fin>] Locuteur N: <paroles>'.",
                    "Restitue chaque tour '[début-fin] Locuteur N: <paroles>', temps en secondes.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] Locuteur N: <paroles>'.",
                ],
                "locuteur_inline": [
                    "Transcris avec horodatage au format 'Locuteur N (début-fin): <paroles>'.",
                    "Écris chaque tour sous la forme 'Locuteur N (<début>-<fin>): <paroles>'.",
                    "Donne une transcription horodatée avec les tours 'Locuteur N (début-fin): <paroles>'.",
                    "Restitue chaque tour 'Locuteur N (<début>-<fin>): <paroles>', temps en secondes.",
                    "Transcris cet audio en formatant chaque tour 'Locuteur N (début-fin): <paroles>'.",
                    "Transcris et diarise cet audio, chaque tour 'Locuteur N (début-fin): <paroles>'.",
                ],
                "locuteur_caps": [
                    "Formate chaque tour sous la forme '[début-fin] LOCUTEUR N: <paroles>' en secondes "
                    "(le mot LOCUTEUR en majuscules).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] LOCUTEUR N: <paroles>', LOCUTEUR en majuscules.",
                    "Transcris avec horodatage, chaque tour '[début-fin] LOCUTEUR 1: <paroles>'.",
                    "Restitue chaque tour '[début-fin] LOCUTEUR N: <paroles>', LOCUTEUR tout en majuscules.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] LOCUTEUR N: <paroles>'.",
                ],
                "locuteur_lower": [
                    "Formate chaque tour sous la forme '[début-fin] locuteur N: <paroles>' en secondes "
                    "(le mot locuteur en minuscules).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] locuteur N: <paroles>', locuteur en minuscules.",
                    "Transcris avec horodatage, chaque tour '[début-fin] locuteur 1: <paroles>'.",
                    "Restitue chaque tour '[début-fin] locuteur N: <paroles>', locuteur en minuscules.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] locuteur N: <paroles>'.",
                ],
                "speaker_caps": [
                    "Formate chaque tour sous la forme '[début-fin] SPEAKER N: <paroles>' en secondes "
                    "(le mot SPEAKER en majuscules).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] SPEAKER N: <paroles>', SPEAKER en majuscules.",
                    "Transcris avec horodatage, chaque tour '[début-fin] SPEAKER 1: <paroles>'.",
                    "Restitue chaque tour '[début-fin] SPEAKER N: <paroles>', SPEAKER tout en majuscules.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] SPEAKER N: <paroles>'.",
                ],
                "speaker_lower": [
                    "Formate chaque tour sous la forme '[début-fin] speaker N: <paroles>' en secondes "
                    "(le mot speaker en minuscules).",
                    "Écris chaque tour sous la forme '[<début>-<fin>] speaker N: <paroles>', speaker en minuscules.",
                    "Transcris avec horodatage, chaque tour '[début-fin] speaker 1: <paroles>'.",
                    "Restitue chaque tour '[début-fin] speaker N: <paroles>', speaker en minuscules.",
                    "Transcris et diarise cet audio, chaque tour '[début-fin] speaker N: <paroles>'.",
                ],
                "json": [
                    # Prompts génériques : demandent du JSON sans détailler le schéma.
                    "Transcris cet audio avec les étiquettes de locuteur et les horodatages, et renvoie le résultat en JSON.",
                    "Donne une transcription horodatée et attribuée aux locuteurs au format JSON.",
                    "Transcris et diarise cet enregistrement ; restitue le résultat en JSON.",
                    # Prompts spécifiant le schéma.
                    "Transcris avec horodatage en JSON : une chaîne \"transcription_result\" "
                    "('spkN: ...') et un tableau \"segments\" de {\"segment\": \"<paroles>\", "
                    "\"start\": <s>, \"end\": <s>, \"duration\": <s>, \"spk_id\": \"spkN\"}.",
                    "Restitue du JSON avec \"transcription_result\" (lignes étiquetées par locuteur) et "
                    "\"segments\", chacun {\"segment\", \"start\", \"end\", \"duration\", \"spk_id\"}, temps en secondes.",
                    "Produis une transcription horodatée et attribuée aux locuteurs sous forme d'objet JSON avec "
                    "\"transcription_result\" et une liste \"segments\" portant le texte du segment, start, end, "
                    "duration et spk_id.",
                    "Transcris et diarise cet audio en JSON : une chaîne \"transcription_result\" et "
                    "\"segments\" de {\"segment\", \"start\", \"end\", \"duration\", \"spk_id\"}.",
                    "Donne une transcription horodatée en JSON avec \"transcription_result\" et un tableau "
                    "\"segments\" où chaque tour a \"segment\", \"start\", \"end\", \"duration\" et \"spk_id\".",
                ],
            },
        },
        # Appended to the prompt of the backchannel-included version of a row.
        "backchannel_suffixes": {
            "transcribed": [
                "Inclus les régulateurs, acquiescements et disfluences (par ex. 'mhm', 'ouais', 'euh').",
                "Conserve tous les acquiescements et régulateurs comme 'mm', 'hmm', 'ouais'.",
                "Inclus chaque énoncé, même les brefs acquiescements et mots de remplissage.",
                "N'omets pas les régulateurs ni les mots de remplissage comme 'mm', 'euh' et 'ouais'.",
                "Transcris tout, y compris les acquiescements et les hésitations.",
                "Conserve les régulateurs, mots de remplissage et disfluences dans la sortie.",
            ],
            "timestamps": [
                "Inclus les brefs tours de régulation (par ex. 'mhm', 'ouais') comme segments distincts.",
                "Conserve chaque tour, même les brefs acquiescements et régulateurs.",
                "N'omets pas les courts tours de régulation ou de remplissage ; segmente-les aussi.",
                "Inclus tous les tours, y compris les brefs régulateurs et hésitations.",
            ],
        },
    },
}


def _validate_prompts():
    """At import time, ensure the fallback language is complete (every variant and
    every format has a non-empty prompt list). Catches typos when editing prompts."""
    base = _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    for variant, fmts in _DIAR_FORMATS.items():
        assert base["generic"].get(variant), f"missing generic prompts for {variant!r}"
        for fmt in fmts:
            assert base["formats"].get(variant, {}).get(fmt), \
                f"missing {DEFAULT_LANGUAGE!r} prompts for {variant!r}/{fmt!r}"
    for lang, block in _DIAR_PROMPTS.items():
        suffixes = block.get("backchannel_suffixes")
        assert isinstance(suffixes, dict), f"{lang!r} backchannel_suffixes must be a dict"
        for key in ("transcribed", "timestamps"):
            assert suffixes.get(key), f"{lang!r} missing {key!r} backchannel suffixes"

    # Each language's EXTRA formats must have prompts in that same language (they
    # have no English fallback, being language-native), and must not shadow a base
    # format key.
    for lang, by_variant in _LANG_EXTRA_FORMATS.items():
        lang_block = _DIAR_PROMPTS.get(lang, {})
        for variant, fmts in by_variant.items():
            for fmt in fmts:
                assert fmt not in _DIAR_FORMATS[variant], \
                    f"extra format {lang!r}/{variant!r}/{fmt!r} collides with a base format"
                assert lang_block.get("formats", {}).get(variant, {}).get(fmt), \
                    f"missing {lang!r} prompts for extra format {variant!r}/{fmt!r}"


_validate_prompts()


def _prompts_for(language, variant, fmt, style):
    """Prompt list for (language, variant, format, style), falling back to
    DEFAULT_LANGUAGE for any list the requested language does not define."""
    block = _DIAR_PROMPTS.get(language) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    if style == "generic":
        prompts = block.get("generic", {}).get(variant)
    else:
        prompts = block.get("formats", {}).get(variant, {}).get(fmt)
    if not prompts and language != DEFAULT_LANGUAGE:
        return _prompts_for(DEFAULT_LANGUAGE, variant, fmt, style)
    return prompts


def _backchannel_suffixes_for(language, variant):
    """Backchannel-version suffix list for a language/variant. The 'timestamps'
    variant emits no words, so it uses segment-oriented wording; 'asr' and
    'timestamps_asr' use transcription-oriented wording."""
    block = _DIAR_PROMPTS.get(language) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    suffixes = block.get("backchannel_suffixes") or _DIAR_PROMPTS[DEFAULT_LANGUAGE]["backchannel_suffixes"]
    key = "timestamps" if variant == "timestamps" else "transcribed"
    return suffixes.get(key) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]["backchannel_suffixes"][key]


# --------------------------- backchannel detection --------------------------- #
# Backchannels / acknowledgements / fillers. A turn whose every lexical token is
# one of these is treated as a backchannel and dropped from the "clean" version.
# Keyed by language; the language-specific set is unioned with the universal
# vocalic core (mm/hmm/ah/oh/...). English is the default.
_BACKCHANNEL_CORE = {
    "mm", "mmm", "mhm", "mm-hmm", "mmhmm", "mm-hm", "hmm", "hm", "hmmm",
    "ah", "aha", "oh", "ooh", "huh", "mm-mm", "hm-mm", "mm-mmm", "mh",
}
_BACKCHANNEL_WORDS_BY_LANG = {
    "en": {
        "uh-huh", "uhhuh", "uh", "uhh", "uhm", "um", "umm", "er", "erm",
        "yeah", "yep", "yup", "nah", "kay", "okay", "ok", "right", "sure",
    },
    "fr": {
        # acquiescements / régulateurs / hésitations courants à l'oral
        "ouais", "oui", "ouaip", "non", "nan", "voilà", "voila",
        "d'accord", "daccord", "ok", "okay", "hein", "bah", "ben", "beh",
        "euh", "heu", "hum", "humhum", "mhmh", "mmh", "mouais", "ouf",
        "ah-ouais", "ah-oui", "ah-bon", "bon-ben",
    },
}
_BC_EDGE = re.compile(r"^[^\w']+|[^\w']+$")


def _backchannel_words_for(language):
    lang = language if language in _BACKCHANNEL_WORDS_BY_LANG else DEFAULT_LANGUAGE
    return _BACKCHANNEL_CORE | _BACKCHANNEL_WORDS_BY_LANG[lang]


def _is_backchannel(text: str, language: str = DEFAULT_LANGUAGE) -> bool:
    """True if every lexical token in `text` is a backchannel/acknowledgement/filler
    in the given language (falling back to the default language's set)."""
    words = _backchannel_words_for(language)
    toks = text.split()
    if not toks or len(toks) > 4:
        return False
    has_word = False
    for t in toks:
        w = _BC_EDGE.sub("", t).lower().strip("'")
        if not w:  # pure punctuation token
            continue
        has_word = True
        if w not in words:
            return False
    return has_word


# Non-lexical hesitation/filler sounds. Unlike the acknowledgement words above
# (oui/non/voilà/...), these carry no meaning and are safe to remove *anywhere* in
# a turn — so the "clean" (no-backchannel) version strips them in-line, while the
# "full" version keeps them. Lexical acknowledgements are only dropped when they
# form a whole turn (see _is_backchannel), never mid-sentence.
_FILLER_WORDS = {
    # English
    "uh", "uhh", "uhm", "um", "umm", "er", "erm", "uh-huh", "uhhuh",
    # French
    "euh", "heu", "heum", "euhm", "heuh", "ben-euh",
    # cross-lingual vocalic hesitations
    "hum", "humhum", "hmm", "hmmm", "hm", "mh", "mm", "mmm", "mmh", "mhm", "mhmh",
}


def strip_fillers(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Remove in-line filler/hesitation tokens (euh, uh, hum, mh, ...) from `text`,
    leaving real words — including lexical acknowledgements like oui/non/voilà —
    untouched. Collapses the whitespace left behind."""
    out = []
    for t in text.split():
        w = _BC_EDGE.sub("", t).lower().strip("'")
        if w in _FILLER_WORDS:
            continue
        out.append(t)
    return " ".join(out)


def clean_window_pieces(window, language: str = DEFAULT_LANGUAGE):
    """Build the 'clean' (no-backchannel) view of a window: drop turns that are
    entirely backchannels/fillers, and strip in-line fillers from the rest. Returns
    new piece dicts (the originals, used by the verbatim 'full' view, are
    untouched). Turns that become empty after filler removal are dropped."""
    out = []
    for p in window:
        text = p.get("text", "")
        if _is_backchannel(text, language):
            continue
        stripped = strip_fillers(text, language)
        if not stripped.strip():
            continue
        out.append({**p, "text": stripped})
    return out


def _defines_prompt(language, variant, fmt, style) -> bool:
    """True if `language` defines a prompt for (variant, fmt, style) itself, without
    falling back to DEFAULT_LANGUAGE."""
    block = _DIAR_PROMPTS.get(language)
    if not block:
        return False
    if style == "generic":
        return bool(block.get("generic", {}).get(variant))
    return bool(block.get("formats", {}).get(variant, {}).get(fmt))


def make_diar_lean_row(r: NemoDatasetRow, cross_lingual_ratio: float = 0.0) -> NemoDatasetRow:
    """Lean training row: prepend a prompt matching the row's prompt_style/format,
    in the row's language (falling back to DEFAULT_LANGUAGE), and — for the
    backchannel-included version — asking to keep them; drop metadata.

    With probability `cross_lingual_ratio` the prompt is instead drawn from a
    different language (prompt-language augmentation: a French prompt on English
    audio and vice versa). Only languages that define a prompt for the row's exact
    (variant, format, style) are eligible — so French-only formats (e.g. the
    'Locuteur' styles) never get an English prompt. The row's `language` field is
    left unchanged; only the instruction wording differs."""
    md = r.custom_metadata or {}
    variant, fmt, style = md.get("variant"), md.get("format"), md.get("prompt_style")
    language = r.language or DEFAULT_LANGUAGE

    prompt_language = language
    if cross_lingual_ratio and random.random() < cross_lingual_ratio:
        others = [lang for lang in _DIAR_PROMPTS
                  if lang != language and _defines_prompt(lang, variant, fmt, style)]
        if others:
            prompt_language = random.choice(others)

    turns = list(r.turns)
    prompts = _prompts_for(prompt_language, variant, fmt, style)
    if prompts:
        prompt = random.choice(prompts)
        if md.get("backchannels"):
            prompt = prompt + " " + random.choice(_backchannel_suffixes_for(prompt_language, variant))
        turns = [NemoTurn(role="User", value=prompt, turn_type="text")] + turns
    return NemoDatasetRow(
        id=r.id, dataset_name=r.dataset_name,
        split=r.split, language=r.language, turns=turns,
    )
