import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from math import ceil, floor

from ssak.utils.align import find_best_position_dtw
from ssak.utils.text import format_text_latin, split_around_space_and_apostrophe

# Convert absolute timestamp to date time


def concert_timestamp(t):
    """
    t = 1554180140
    concert_timestamp(t) = "2019-04-02 06:42:20"
    """

    dt = datetime.fromtimestamp(t)
    return dt


def realign_annotations(annot_file, word_strategy=True, plot=False, verbose=False):
    if plot:
        outdir = "plots"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        plot = os.path.join(outdir, os.path.basename(annot_file))

    data = json.load(open(annot_file))
    transcripts = data["transcripts"]
    assert len(transcripts) == 1
    transcripts = transcripts[0]
    assert "transcript" in transcripts
    transcripts = transcripts["transcript"].strip()

    if transcripts.startswith("<?xml"):
        transcripts_xml = transcripts
        transcripts_xml = re.sub(r"<\/speech>\n\s+\n", "</speech>\n <speech />\n", transcripts_xml)
        transcripts = []
        for elt in ET.fromstring(transcripts_xml):
            if elt.tag == "content":
                for elt in elt:
                    if elt.tag == "speech":
                        if elt.text and elt.text.strip():
                            transcript = elt.text.strip().replace("’", "'")
                            if transcript[-1] not in [".", "!", "?"]:
                                transcript += "."
                            if not len(transcripts) or transcripts[-1] is None:
                                transcripts.append(transcript)
                            else:
                                transcripts[-1] += " " + transcript
                        elif len(transcripts) and transcripts[-1] is not None:
                            transcripts.append(None)
                    else:
                        print("WARNING: Unknown tag", elt.tag)
        transcripts = [t for t in transcripts if t is not None]
    else:
        print("WARNING: NO XML !!!!")
        transcripts = [transcripts]

    assert "extra" in data
    extra = data["extra"]
    assert "automatic_transcription" in extra
    assert "transcription_start" in extra
    auto_transcript = ET.fromstring(extra["automatic_transcription"])
    auto_transcript_start = extra["transcription_start"]
    # convert '2019-03-14T07:11:00+01:00' to absolute timing
    if isinstance(auto_transcript_start, int):
        auto_transcript_start = auto_transcript_start
    else:
        auto_transcript_start = datetime.strptime(auto_transcript_start[:-6], "%Y-%m-%dT%H:%M:%S")

    new_transcripts = []

    audio_start = None
    audio_end = None

    # Get words and timestamps from automatic transcript
    auto_words = []
    auto_starts = []
    for elt in auto_transcript:
        words = [elt.text]
        if elt.tag == "s2t":
            start = elt.get("datetime")
            # convert '2019-03-14T07:12:00.410000' to absolute timing
            try:
                start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
            starts = [start]
        elif elt.tag == "term":
            starts = [elt.get("start")]
        elif elt.tag == "speech":
            words = []
            starts = []
            for word in elt:
                assert word.tag == "term"
                words.append(word.text)
                start = word.get("start")
                assert start is not None
                if isinstance(start, str):
                    start = float(start)
                starts.append(start)
        else:
            raise ValueError(f"Unknown tag {elt.tag}")
        for start, word in zip(starts, words):
            if audio_start is None:
                audio_start = start
            audio_end = start
            assert start is not None
            start = start - auto_transcript_start
            # concert datetime.timedelta to seconds
            if isinstance(start, datetime):
                start = start.total_seconds()
            auto_words.append(word)
            auto_starts.append(start)

    audio_duration = audio_end - audio_start
    annotation_duration = 0

    assert len(auto_words) > 0, "No words in automatic transcript"

    for i_transcript, transcript in enumerate(transcripts):
        annotation_start = None
        annotation_end = None

        # Get words from manual transcript
        # (split around ' and spaces)
        words = split_around_space_and_apostrophe(transcript)

        if word_strategy:
            indices = find_best_position_dtw(
                [format_text_latin(w) for w in words],
                [format_text_latin(w) for w in auto_words],
                finetune_start_end=True,
                pad=True,
                plot=f"{plot}-{i_transcript}",
            )
            indices = indices["indices"]
        else:
            raise NotImplementedError("Not implemented yet")
            # Make character
            text, indices = words_to_character(words)
            auto_text, auto_indices = words_to_character(auto_words)

            indices = find_best_position_dtw(text, auto_text, plot=plot)

        assert len(indices) == len(words)

        current_segment = None
        current_start = None
        for i, (word, index) in enumerate(zip(words, indices)):
            if verbose:
                mindex = max(index + 1, indices[i + 1]) if i < (len(indices) - 1) else (index + 1)
                print(word, "->", " ".join(auto_words[index:mindex]))

            start = auto_starts[index]
            if index < len(auto_starts) - 1:
                end = auto_starts[index + 1]
            else:
                end = start + (auto_starts[index] - auto_starts[index - 1])

            if current_start is None:
                current_segment = ""
                current_start = start

            if current_segment and current_segment[-1] not in ["'", "-"]:
                current_segment += " "
            current_segment += word

            if word[-1] in [".", "!", "?"]:
                if annotation_start is None:
                    annotation_start = current_start
                annotation_end = end
                new_transcripts.append(
                    [
                        current_segment,
                        current_start,
                        end,
                        None,
                    ]
                )
                current_segment = None
                current_start = None

        last_i = indices[-1]
        auto_starts = auto_starts[last_i:]
        auto_words = auto_words[last_i:]

        # import pdb; pdb.set_trace()
        # print(indices, auto_starts, auto_words)

        if current_segment is not None:
            # Malformed transcript?
            print("WARNING: TRANSCRIPTION LOOKS WRONG")
            return
        assert current_segment is None

        annotation_duration += annotation_end - annotation_start

    return new_transcripts, audio_duration, annotation_duration


def time2str(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def save_annotation(annot, filename):
    dname = os.path.dirname(filename)
    if not os.path.exists(dname):
        os.makedirs(dname)

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
    import sys

    DIRIN = "/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-2/annotation_batch"
    DIROUT = "/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-2/annotation_new"

    audio_durations = []
    annotation_durations = []

    for file_in in sorted(os.listdir(DIRIN)):
        if not file_in.endswith(".annotations.json"):
            continue

        file_in = os.path.join(DIRIN, file_in)
        file_out = os.path.join(DIROUT, os.path.basename(file_in))

        # if os.path.isfile(file_out):
        #     print(f"WARNING: file already exists: {file_out}")
        #     continue

        print("================================")
        print(file_in)
        annot, audio_duration, annotation_duration = realign_annotations(file_in, word_strategy=True, plot="--plot" in sys.argv, verbose=False)
        print(f"Annotation duration: {annotation_duration}/{audio_duration} ({annotation_duration/audio_duration*100:.2f}%)")
        audio_durations.append(audio_duration)
        annotation_durations.append(annotation_duration)
        if annot is None:
            print("WARNING: SOMETHING WENT WRONG")
            continue
        save_annotation(annot, file_out)

    total_annotation_duration = sum(annotation_durations)
    total_audio_duration = sum(audio_durations)

    print(f"Total annotated duration: {total_annotation_duration}/{total_audio_duration} ({total_annotation_duration/total_audio_duration*100:.2f}%)")
    print(f"Audio duration: min = {min(audio_durations)/60}, max = {max(audio_durations)/60}, mean = {sum(audio_durations)/len(audio_durations)/60}, total = {sum(audio_durations)/60}")

    import matplotlib.pyplot as plt

    ratios = [100 * a / audio_duration for a, audio_duration in zip(annotation_durations, audio_durations)]
    plt.figure()
    plt.hist(ratios, bins=100, range=(floor(min(ratios) - 0.1), ceil(max(ratios) + 0.1)))
    plt.xlabel("Portion d'audio annotée manuellement (%)")

    # Plot cumulative distribution
    ratios.sort()
    plt.figure()
    plt.plot(ratios, [100 * (i + 1) / len(ratios) for i in range(len(ratios))])
    plt.xlabel("Portion d'audio annotée manuellement (%)")
    plt.ylabel("Distribution cumulée (%)")

    plt.show()
