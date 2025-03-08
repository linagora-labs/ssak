#!/usr/bin/env python3

import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import Levenshtein
import numpy as np
import soxbindings as sox

from ssak.infer.speechbrain_infer import speechbrain_infer, speechbrain_load_model
from ssak.utils.align_transcriptions import *
from ssak.utils.audio import load_audio
from ssak.utils.env import *
from ssak.utils.monitoring import tic, toc
from ssak.utils.text import collapse_whitespace, format_text_latin


def additional_normalization(text):
    return text.replace("ö", "o").replace("ü", "u").replace("ä", "a").replace("ß", "ss").replace("ñ", "n").replace("á", "a")


def get_transcription_file_for_audio(audiofilename):
    bname = os.path.basename(audiofilename)
    fname = os.path.splitext(bname)[0]
    while os.path.splitext(fname)[-1] in [".audio"]:
        fname = os.path.splitext(fname)[0]
    for root, dirs, files in os.walk(os.path.dirname(audiofilename)):
        for f in files:
            if f.startswith(fname + ".") and f != bname and (f.endswith("annotations.json")):
                return os.path.join(root, f)


def get_transcription_for_audio(audiofilename):
    f = get_transcription_file_for_audio(audiofilename)
    assert f is not None, f"Cannot find transcription for {audiofilename}"
    if f.endswith(".json"):
        d = json.load(open(f))
        if "transcripts" in d:
            transcripts = d["transcripts"]
        else:
            transcripts = [d]
        texts = []
        for t in transcripts:
            t = get_texts_and_speakers(t["transcript"])
            texts.extend(t)
        return texts
    raise NotImplementedError(f"Cannot handle {f}")


# def get_texts(s):
#     if s.startswith("<?xml version"):
#         root = ET.fromstring(s)
#         texts = [""]
#         num_newlines = 0
#         for t in root.itertext():
#             t =  t.strip()
#             if t:
#                 texts[-1] += t + " "
#                 num_newlines = 0
#             else:
#                 num_newlines += 1
#                 if num_newlines > 1:
#                     texts.append("")
#         return [t.strip() for t in texts if t.strip()]
#     return [s]


def get_texts_and_speakers(s):
    if s.startswith("<?xml version"):
        root = ET.fromstring(s)
        texts = [[]]
        num_newlines = 0
        for t in root:
            if t.tag == "content":
                for s in t:
                    speaker = s.attrib.get("speaker")
                    t = s.text
                    if t is None:
                        t = ""
                    t = t.strip()
                    if t:
                        texts[-1].append((t, speaker))
                        num_newlines = 0
                    else:
                        num_newlines += 1
                        if num_newlines > 1 and texts[-1] != []:
                            texts.append([])
        return texts
    return [(s, None)]


def generalized_levenshtein_distance(s, t):
    return Levenshtein.distance(" ".join(s), " ".join(t), weights=(1, 1, 1))  #  (insertion, deletion, substitution)


def find_best_position(subsequence, sequence, flexibility=10, mini=0):
    wsubsequence = subsequence.split()
    wsequence = sequence.split()
    assert sum(len(w) for w in wsubsequence) == len(subsequence) - len(wsubsequence) + 1, f"{sum(len(w) for w in wsubsequence)} != {len(subsequence) - len(wsubsequence) + 1,}"
    assert sum(len(w) for w in wsequence) == len(sequence) - len(wsequence) + 1, f"{sum(len(w) for w in wsequence)} != {len(sequence) - len(wsequence) + 1,}"

    m = len(wsubsequence)
    i = 0
    min_distance = np.inf
    match = None
    for i in range(mini, len(wsequence) - m + 1):
        for j in range(i + m - flexibility, i + m + flexibility):
            if j > len(wsequence) or j <= i:
                continue
            distance = generalized_levenshtein_distance(wsubsequence, wsequence[i:j])
            if distance < min_distance:
                match = (i, j)
                min_distance = distance
    if match is None:
        print("sequence", sequence)
        print("subsequence", subsequence)
        print(len(wsequence), m, mini)
        raise RuntimeError("No match")
    (wi, wj) = match
    i = sum(len(w) for w in wsequence[:wi]) + wi
    j = sum(len(w) for w in wsequence[:wj]) + wj - 1
    return (i, j, wj)


def time2str(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def save_annotation(annot, filename):
    now = datetime.now()

    dic = {
        "transcripts": [
            {
                "date_created": time2str(now),
                "transcript": a[0],
                "timestamp_start_milliseconds": a[1] * 1000,
                "timestamp_end_milliseconds": a[2] * 1000,
                "extra": {
                    "speaker": a[3],
                },
            }
            if a[3]
            else {
                "date_created": time2str(now),
                "transcript": a[0],
                "timestamp_start_milliseconds": a[1] * 1000,
                "timestamp_end_milliseconds": a[2] * 1000,
            }
            for a in annot
        ]
    }
    with open(filename, "w") as f:
        json.dump(dic, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # sys.argv = ["cut_audio_around_transcription.py", "data/000000.audio.mp3"]

    if len(sys.argv) < 2:
        dname = "dev-data-2"
        # dname = "aday-private-corpus"
        sys.argv = sys.argv + [os.path.join(dname, f) for f in sorted(os.listdir(dname)) if f.endswith(".audio.mp3")]

    debug_path = None  # "new_cut"
    out_path = dname + "/annotation_align1"
    os.makedirs(out_path, exist_ok=True)
    if debug_path:
        os.makedirs(debug_path, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(description="Cut audiofiles around transcriptions", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audiofiles", type=str, help="Audio file(s) to cut", nargs="+")
    parser.add_argument(
        "--get_transcript",
        help="A method to get the transcription for a given audio file",
        default=get_transcription_for_audio,
    )
    args = parser.parse_args()

    WORD2VEC_PATH = "best_model"  # _ESTER"
    BASE_MODEL = WORD2VEC_PATH  # "Ilyes/wav2vec2-large-xlsr-53-french"
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    processor = transformers.Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model_transformers = transformers.Wav2Vec2ForCTC.from_pretrained(WORD2VEC_PATH).to(DEVICE)
    sample_rate = processor.feature_extractor.sampling_rate

    USE_SPEECHBRAIN = False
    if USE_SPEECHBRAIN:
        model_speechbrain = speechbrain_load_model(
            "/home/jlouradour/projects/SpeechBrain/models_trained/speechbrain_Split1-Eval480_LeBenchmark-wav2vec2-FR-7K-large_len-0.5-15_frFalse_lr1.0-0.0001_bs4_s1234_ascending/save/CKPT+2022-10-11+07-01-16+00"
        )

    tic_start = time.time()
    num_processed = 0

    for filename in args.audiofiles:
        bname = os.path.basename(filename)
        out_file = bname.replace("audio.mp3", "annotations.json")
        assert bname != out_file
        out_file = os.path.join(out_path, out_file)
        if os.path.exists(out_file):
            print("Skipping", out_file)
            continue

        print("Processing", filename)

        transcripts = get_transcription_for_audio(filename)

        tic_start0 = time.time()

        audio = load_audio(filename, sample_rate=sample_rate)

        tic()
        # print("Run Automatic Transcription")
        if USE_SPEECHBRAIN:
            full_reco = list(speechbrain_infer(model_speechbrain, [audio]))[0]
        else:
            # full_reco = list(kaldi_infer("vosk-model-fr-0.22", [audio]))[0]
            cmd = [
                sys.executable,
                "/home/jlouradour/src/stt-end2end-expes/ssak/infer/kaldi_infer.py",
                "--model",
                "vosk-model-fr-0.22",
                "--gpus",
                "1",
                filename,
            ]
            p = subprocess.Popen(
                cmd,
                env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),  # Otherwise ".local" path might be missing
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            (full_reco, stderr) = p.communicate()
            full_reco = full_reco.decode("utf-8")

        print("Reco:", full_reco)

        toc("ASR")
        full_reco = collapse_whitespace(additional_normalization(format_text_latin(full_reco.strip().replace("\n", " "))))
        # print(full_reco)
        # print(filename)

        new_transcripts = []
        idebug = 0
        wj = -1
        for t in transcripts:
            transcript0 = " ".join([ti[0] for ti in t])
            transcript = additional_normalization(format_text_latin(transcript0))
            transcript = collapse_whitespace(transcript)

            tic()
            (i, j, wj) = find_best_position(transcript, full_reco, mini=wj + 1)
            toc("Find transcription position")

            before = full_reco[:i].strip()
            after = full_reco[j + 1 :].strip()
            full_transcript = (before + " " + transcript + " " + after).strip()

            # print("Align transcription")
            try:
                tic()
                labels, emission, trellis, segments, word_segments = compute_alignment(audio, full_transcript, model_transformers, processor)
                toc("Alignment")
            except:
                print(">>> FULL")
                print(full_transcript)
                print(">>> SELECTED BEFORE")
                print(transcript0)
                print(">>> SELECTED AFTER")
                print(transcript)
                labels, emission, trellis, segments, word_segments = compute_alignment(audio, full_transcript, model_transformers, processor)

            ratio = len(audio) / (trellis.size(0) * sample_rate)

            all_words = full_transcript.split()
            import pickle

            with open("all_words.pkl", "wb") as f:
                pickle.dump(full_transcript, f)
            assert len(word_segments) == len(all_words), f"{len(word_segments)} != {len(all_words)}"

            istart = len(before.split())
            for ti in t:
                ftext = format_text_latin(ti[0])
                num_words = len(ftext.split())
                start, end = word_segments[istart].start, word_segments[istart + num_words - 1].end
                start *= ratio
                end *= ratio
                new_transcripts.append((ti[0], start, end, ti[1]))

                if debug_path:
                    subaudio = audio[int(start * sample_rate) : int(end * sample_rate)]
                    fname = bname.split(".")[0] + "_" + f"{idebug:02}" + "_" + ftext.replace(" ", "_").replace("'", "") + ".mp3"
                    idebug += 1
                    if len(fname) > 200 - 4:
                        fname = fname[: 200 - 4 - 23] + "..." + fname[-20:]
                    fname = os.path.join(debug_path, fname)
                    sox.write(fname, subaudio, sample_rate)

                istart += num_words

        save_annotation(new_transcripts, out_file)
        num_processed += 1
        time_per_item0 = time.time() - tic_start0
        time_per_item = (time.time() - tic_start) / num_processed
        print(f"{time_per_item0} sec / file - average {time_per_item} sec / file")

    time_per_item = (time.time() - tic_start) / num_processed
    print(f"{time_per_item} sec / file")
