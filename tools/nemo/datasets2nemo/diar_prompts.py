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
    },
    "timestamps": {
        "plain":         {"render": _r_ts_plain},
        "rttm":          {"render": _r_ts_rttm},
        "bracket":       {"render": _r_ts_bracket},
        "bare_letter":   {"render": _r_ts_bare_letter},
        "s_num":         {"render": _r_ts_s_num},
        "speaker_upper": {"render": _r_ts_speaker_upper},
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
    },
}

# First key of each variant is its default format (used by generic prompts).
DIAR_DEFAULT_FORMAT = {v: next(iter(fmts)) for v, fmts in _DIAR_FORMATS.items()}


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
            },
        },
        # Appended to the prompt of the backchannel-included version of a row.
        "backchannel_suffixes": [
            "Include backchannels, acknowledgements and disfluencies (e.g. 'mm-hmm', 'yeah', 'uh').",
            "Keep all backchannels and acknowledgements such as 'mm', 'hmm', 'uh-huh'.",
            "Include every utterance, even short backchannels and fillers.",
            "Do not omit backchannels or filler words like 'mm', 'uh' and 'yeah'.",
            "Transcribe everything, including acknowledgements and hesitations.",
            "Keep the backchannels, fillers and disfluencies in the output.",
        ],
    },
    # "fr": { ... }  # add French here, mirroring the "en" structure above.
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
    assert base.get("backchannel_suffixes"), "missing backchannel suffixes"


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


def _backchannel_suffixes_for(language):
    block = _DIAR_PROMPTS.get(language) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    return block.get("backchannel_suffixes") or _DIAR_PROMPTS[DEFAULT_LANGUAGE]["backchannel_suffixes"]


# --------------------------- backchannel detection --------------------------- #
# Backchannels / acknowledgements / fillers. A turn whose every lexical token is
# one of these is treated as a backchannel and dropped from the "clean" version.
_BACKCHANNEL_WORDS = {
    "mm", "mmm", "mhm", "mm-hmm", "mmhmm", "mm-hm", "hmm", "hm", "hmmm",
    "uh-huh", "uhhuh", "uh", "uhh", "uhm", "um", "umm", "er", "erm",
    "ah", "aha", "oh", "ooh", "huh", "mm-mm", "hm-mm", "mm-mmm",
    "yeah", "yep", "yup", "nah", "kay", "okay", "ok", "right", "sure",
}
_BC_EDGE = re.compile(r"^[^\w']+|[^\w']+$")


def _is_backchannel(text: str) -> bool:
    """True if every lexical token in `text` is a backchannel/acknowledgement/filler."""
    toks = text.split()
    if not toks or len(toks) > 4:
        return False
    has_word = False
    for t in toks:
        w = _BC_EDGE.sub("", t).lower().strip("'")
        if not w:  # pure punctuation token
            continue
        has_word = True
        if w not in _BACKCHANNEL_WORDS:
            return False
    return has_word


def make_diar_lean_row(r: NemoDatasetRow) -> NemoDatasetRow:
    """Lean training row: prepend a prompt matching the row's prompt_style/format,
    in the row's language (falling back to DEFAULT_LANGUAGE), and — for the
    backchannel-included version — asking to keep them; drop metadata."""
    md = r.custom_metadata or {}
    variant, fmt, style = md.get("variant"), md.get("format"), md.get("prompt_style")
    language = r.language or DEFAULT_LANGUAGE
    turns = list(r.turns)
    prompts = _prompts_for(language, variant, fmt, style)
    if prompts:
        prompt = random.choice(prompts)
        if md.get("backchannels"):
            prompt = prompt + " " + random.choice(_backchannel_suffixes_for(language))
        turns = [NemoTurn(role="User", value=prompt, turn_type="text")] + turns
    return NemoDatasetRow(
        id=r.id, dataset_name=r.dataset_name,
        split=r.split, language=r.language, turns=turns,
    )
