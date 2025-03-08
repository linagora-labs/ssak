#!/usr/bin/env python3

import re
import sys

from num2words import num2words

from ssak.utils.curl import curl_post
from ssak.utils.format_transcription import to_linstt_transcription
from ssak.utils.text import remove_punctuations

INF = float("inf")


def strip_punctuation(w):
    return w.rstrip(".!?,;:").rstrip(" ")


def quick_format(w):
    return remove_punctuations(w).lower()


REPLACEMENT = {}
sys.setrecursionlimit(5000)


def get_all_replacements(_from, _to, minimum_len_char=3, minimum_len_word=1):
    global REPLACEMENT
    if _from == _to or not _from.strip() or _from in REPLACEMENT:
        return []

    if len(_from) < minimum_len_char:
        return []

    wf = _from.split(" ")

    if len(wf) < minimum_len_word:
        return []
    wt = _to.split(" ")
    wt2 = strip_punctuation(_to).split(" ")

    REPLACEMENT[_from] = _to
    assert "  " not in _from
    assert "  " not in _to
    first_wf = wf[0]
    last_wf = wf[-1]
    i_left = 1
    i_right = 1
    first_wt = quick_format(wt[0])
    last_wt = quick_format(wt2[-1])

    try:
        int(first_wt)
        first_wt = num2words(first_wt, lang="fr")
    except:
        pass
    try:
        int(last_wt)
        last_wt = num2words(last_wt, lang="fr")
    except:
        pass

    if first_wf.endswith("'"):
        first_wf = "".join(wf[:2])
        i_left = 2
    if len(wf) > 2 and wf[-2].endswith("'"):
        last_wf = "".join(wf[-2:])
        i_right = 2

    if first_wf == first_wt:
        _from2 = " ".join(wf[i_left:])
        _to2 = " ".join(wt[1:])
        if _from2 not in REPLACEMENT:
            REPLACEMENT.update(get_all_replacements(_from2, _to2, minimum_len_word=min(3, minimum_len_word + 1)))
    # else:
    #     print("WARNING: first word mismatch", first_wf, first_wt)
    if len(wf) > 1:
        if last_wf == last_wt:
            _from2 = " ".join(wf[:-i_right])
            _to2 = " ".join(wt2[:-1])
            if _from2 not in REPLACEMENT:
                REPLACEMENT.update(get_all_replacements(_from2, _to2, minimum_len_word=min(3, minimum_len_word + 1)))
        # else:
        #     print("WARNING: last word mismatch", last_wf, last_wt)

    return REPLACEMENT


def simple_text_normalize(text):
    # text = re.sub(r"(\w)' +(\w)", r"\1'\2", text)
    # text = re.sub(r"(\w) +'(\w)", r"\1'\2", text)
    # text = re.sub(r"(\w)- +(\w)", r"\1-\2", text)
    # text = re.sub(r"(\w) +-(\w)", r"\1-\2", text)
    return text.strip()  # .capitalize()


def merge_transcriptions(
    transcriptions,
    remove_isolated_words=[
        # "hum",
        # "euh",
    ],
    minimum_duration_isolated=0,  # 0.2,
    keep_isolated_words=[
        "allo",
        "oui",
        "non",
        "accord",
        "'accord",
    ],
    disable_punctuation=False,
    punctuation_server=None,
    keep_segments=False,
    join_segments_from_same_speaker=True,
):
    disable_punctuation = disable_punctuation or punctuation_server

    num = len(transcriptions)
    current_segments = [0] * len(transcriptions)
    current_words = [0] * len(transcriptions)

    def _current_segment(i):
        segments = transcriptions[i]["segments"]
        k = current_segments[i]
        if k >= len(segments):
            return {"words": [], "start": INF, "end": INF}
        return segments[k]

    def _current_word(i):
        words = _current_segment(i)["words"]
        k = current_words[i]
        if k >= len(words):
            return {"word": "", "start": INF, "end": INF}
        return words[current_words[i]]

    def _current_start_time(i):
        return max(
            _current_segment(i)["start"],
            _current_word(i)["start"],
        )

    def _current_end_time(i):
        return min(
            _current_segment(i)["end"],
            _current_word(i)["end"],
        )

    if isinstance(keep_segments, float | int):
        transcriptions = [cut_segments(transcription, keep_segments) for transcription in transcriptions]

    if keep_segments:

        def _next(i):
            current_segments[i] += 1
    else:

        def _next(i):
            word = _current_word(i)
            current_words[i] += 1
            seg = _current_segment(i)
            if current_words[i] > len(seg["words"]):
                current_words[i] = 0
                current_segments[i] += 1
            return word

    def _current_t0():
        times = [_current_start_time(i) for i in range(num)]
        minimum = min(times)
        return minimum, times.index(minimum)

    segments = []

    previous_spk = None
    t0, spk = _current_t0()
    last_spk = None

    if keep_segments:
        while t0 < INF:
            segment = _current_segment(spk)

            if not join_segments_from_same_speaker or spk != last_spk:
                segments.append(
                    {
                        "spk_id": f"spk{spk+1}",
                        "start": segment["start"],
                        "end": segment["end"],
                        "raw_segment": segment["raw_segment"],
                        "segment": segment["segment"],
                        "words": segment["words"],
                    }
                )
            else:
                segments[-1]["end"] = segment["end"]
                segments[-1]["raw_segment"] += " " + segment["raw_segment"]
                segments[-1]["segment"] += " " + segment["segment"]
                segments[-1]["words"] += segment["words"]
            last_spk = spk

            _next(spk)

            t0, spk = _current_t0()

    else:
        while t0 < INF:
            t1 = _current_end_time(spk)

            # Consume the next word
            word = _next(spk)
            next_t0, next_spk = _current_t0()

            t1 = min(t1, next_t0)
            word["end"] = t1

            do_add = True

            if previous_spk != spk:
                if next_spk == previous_spk:
                    if minimum_duration_isolated and t1 - t0 < minimum_duration_isolated and remove_punctuations(word["word"]).lower() not in keep_isolated_words:
                        do_add = False
                    elif word["word"] in remove_isolated_words:
                        do_add = False

                if do_add:
                    segments.append(
                        {
                            "spk_id": f"spk{spk+1}",
                            "start": t0,
                            "end": t1,
                            "raw_segment": word["word"],
                            "segment": None,
                            "words": [word],
                        }
                    )
                    previous_spk = spk
            else:
                segments[-1]["end"] = min(t1, next_t0)
                segments[-1]["raw_segment"] += " " + word["word"]
                segments[-1]["words"].append(word)

            t0 = next_t0
            spk = next_spk

        if disable_punctuation:
            for seg in segments:
                text = seg["raw_segment"]
                seg["segment"] = simple_text_normalize(text)

        else:
            # Recover punctuation
            replacements = {}
            for trans in transcriptions[::-1]:
                for seg in trans["segments"]:
                    _from = seg["raw_segment"]
                    _to = seg["segment"]
                    replacements = get_all_replacements(_from, _to)

            # Cheap recovery of punctuation
            for seg in segments:
                text = seg["raw_segment"]
                founded = False
                for _from, _to in replacements.items():
                    if text == _from:
                        text = _to
                        founded = True
                        break
                if not founded:
                    for _from, _to in replacements.items():
                        if len(_from) > 10 and _from in text:
                            text = re.sub(r"\b" + _from + r"\b", _to, text)
                    text = re.sub(r",+", ",", text)
                if text:
                    text = text.capitalize()
                if len(text) > 100 and text[-1] not in [".", "?", "!"]:
                    text += "."
                seg["segment"] = text

    if punctuation_server:
        texts = [seg["segment"] for seg in segments]
        new_texts = curl_post(
            punctuation_server + "/punctuation",
            {"sentences": texts},
            headers=["Content-Type: application/json"],
            verbose=False,
            post_as_fields=True,
        )
        assert "punctuated_sentences" in new_texts, "Invalid response from punctuation server"
        new_texts = new_texts["punctuated_sentences"]
        assert len(new_texts) == len(texts)
        assert len(new_texts) == len(segments)
        # for t1, t2 in zip(texts, new_texts):
        #     print(t1, "->", t2)
        for i, text in enumerate(new_texts):
            segments[i]["segment"] = text

    return {
        "transcription_result": "\n".join([f"{seg['spk_id']}: {seg['segment']}" for seg in segments]),
        "raw_transcription": " ".join([seg["raw_segment"] for seg in segments]),
        "segments": segments,
    }


def cut_segments(transcription, max_silence_duration):
    segments = []
    last_end = -2 * max_silence_duration
    for segment in transcription["segments"]:
        for word in segment["words"]:
            start = word["start"]
            end = word["end"]
            if start - last_end > max_silence_duration:
                segments.append(
                    {
                        "spk_id": segment["spk_id"],
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "raw_segment": word["word"],
                        "segment": word["word"],
                        "words": [word],
                    }
                )
            else:
                segments[-1]["end"] = end
                segments[-1]["duration"] = end - segments[-1]["start"]
                segments[-1]["raw_segment"] += " " + word["word"]
                segments[-1]["segment"] += " " + word["word"]
                segments[-1]["words"].append(word)
            last_end = end
            if word["word"][-1] in [".", "?", "!"]:
                last_end = -2 * max_silence_duration

    return {
        "transcription_result": transcription["transcription_result"],
        "raw_transcription": transcription["raw_transcription"],
        "confidence": transcription["confidence"],
        "segments": segments,
    }


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Combine several mono-channel transcriptions to a single transcription, assuming one speaker per channel")
    parser.add_argument("transcription_files", type=str, help="Transcription files", nargs="+")
    parser.add_argument("output_file", type=str, help="Output file")
    parser.add_argument("--disable_punctuation", action="store_true", default=False, help="Disable punctuation recovery")
    parser.add_argument("--punctuation_server", type=str, default=None, help="Transcription service URL")
    parser.add_argument("--segments", action="store_true", default=False, help="Keep segments as they are")
    args = parser.parse_args()

    transcriptions = [to_linstt_transcription(f) for f in args.transcription_files]

    transcription = merge_transcriptions(
        transcriptions,
        disable_punctuation=args.disable_punctuation,
        punctuation_server=args.punctuation_server,
        keep_segments=0.25 if args.segments else 0,
    )

    outdir = os.path.dirname(args.output_file)
    os.makedirs(outdir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)
