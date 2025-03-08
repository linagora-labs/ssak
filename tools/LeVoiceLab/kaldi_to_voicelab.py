#!/usr/bin/env python3

import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import groupby

from tqdm import tqdm

from ssak.utils.audio import get_audio_total_duration
from ssak.utils.misc import commonprefix

CONVERT_ALL_WAV_TO_THE_SAME = False


# def timestamp2sec(str):
#     date = datetime.strptime(str, "%H:%M:%S.%f")
#     return (date -  datetime(1900, 1, 1)).total_seconds()
def timestamp2sec(str):
    hr = int(str.split(":")[0])
    str = str[str.find(":") + 1 :]
    date = datetime.strptime(str, "%M:%S.%f")
    return hr * 3600 + (date - datetime(1900, 1, 1)).total_seconds()


def time2str(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def json_dump(dic, f):
    json.dump(dic, f, indent=2, ensure_ascii=False)


def extract_info_from_soxi_output(infos, what, use_percent):
    all = [int(infos[m.start() : m.end()].split()[-1]) for m in re.finditer(what + r"\s*:\s+\d+", infos)]
    res = list(set(all))
    if len(res) == 1:
        chosen = res[0]
        notes = None
    else:
        chosen = max(res, key=all.count)
        print("Choosing", chosen, "among", res, "for", what)
        print(what, ":")
        for e in sorted(res):
            print("*", e, ":", "%.1f" % (all.count(e) * 100.0 / len(all)), "%")
        # notes = ", ".join([
        #     "%d (%d, %.2f%%)" % (e, all.count(e), all.count(e) * 100. / len(all))
        #     for e in sorted(res)
        # ])
        if use_percent:
            notes = dict([(e, "~%.1f%%" % (all.count(e) * 100.0 / len(all))) for e in sorted(res)])
        else:
            notes = dict([(e, all.count(e)) for e in sorted(res)])
    return chosen, notes


def extract_from_xml(xml_file, field="date"):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_file)
    res = list(_extract_all(tree.getroot(), field))
    if len(res) != 1:
        raise RuntimeError("Found %d %s in %s" % (len(res), field, xml_file))
    return res[0]


def _extract_all(root, field):
    for child in root:
        if child.tag.lower() == field:
            yield child.text
        for elt in _extract_all(child, field):
            yield elt


# def unquote(s):
#     if (s.startswith("'") or s.startswith('"')) and (s.endswith("'") or s.endswith('"')):
#         return s[1:-1]
#     return s


# TODO: use "from ssak.utils.audio import get_audio_duration" instead?
def get_audio_duration(audio_file):
    if audio_file in _durations:
        return _durations[audio_file]
    return float(subprocess.check_output(["soxi", "-D", audio_file]))


# TODO: generalize and use "get_audio_total_duration" instead of "get_audio_durations"?
_durations = {}


def get_audio_durations(audio_files):
    currdir = os.path.realpath(os.curdir)

    commondir = commonprefix(audio_files)
    while len(commondir) > 0 and commondir[-1] != "/":
        commondir = commondir[:-1]
    if commondir:
        print(commondir)
        os.chdir(commondir)
        audio_files = [f.replace(commondir, "") for f in audio_files]

    import time

    global _durations
    audio_files = list(audio_files)
    if len(" ".join(audio_files)) >= 2097152 - 5:
        i = int(len(audio_files) / 2)
        return get_audio_durations(audio_files[:i]) + get_audio_durations(audio_files[i:])
    tic = time.time()
    print("Running soxi on %d files" % len(audio_files))
    d = [float(d) for d in subprocess.check_output(["soxi", "-D"] + audio_files).split()]
    toc = time.time()
    print("Took %.2f sec to compute duration on %d files" % (toc - tic, len(audio_files)))
    _durations.update(zip(audio_files, d))
    os.chdir(currdir)
    return d

    # if audio_file.endswith(".wav"):
    #     import wave
    #     import contextlib
    #     with contextlib.closing(wave.open(audio_file,'r')) as f:
    #         frames = f.getnframes()
    #         rate = f.getframerate()
    #         duration = frames / float(rate)
    #         return duration
    # elif audio_file.endswith(".mp3"):
    #     from mutagen.mp3 import MP3
    #     audio = MP3(audio_file)
    #     return audio.info.length
    # else:
    #     raise RuntimeError("File extension of {} not supported".format(audio_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert kaldi-formatted dataset into the format for LeVoiceLab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dir_in", help="input folder", type=str)
    parser.add_argument("dir_out", help="output folder", type=str)
    parser.add_argument(
        "--raw",
        help="input folder for raw transcription (same as input folder if not specified)",
        type=str,
    )
    parser.add_argument("--max", help="Maximum number of audio files", default=1e10, type=int)
    parser.add_argument("--license", help="License", default="", type=str)
    parser.add_argument("--description", help="Description", default="", type=str)
    parser.add_argument(
        "--private",
        help="Whether the db is private",
        default=False,
        action="store_true",
    )
    parser.add_argument("--name", help="Corpus name", default=None, type=str)
    parser.add_argument("--ext", help="Audio extension (.wav, .mp3 ...)", default=".wav", type=str)
    parser.add_argument("--languages", help="Languages spoken in the dataset", default=["fr"], nargs="+")
    parser.add_argument(
        "--version",
        help="Format version",
        choices=["0.0.1", "0.0.2"],
        default="0.0.2",
        type=str,
    )
    parser.add_argument(
        "--disable_kaldi_checks",
        help="To disable checking that all ids have all info in the input kaldi files",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable_file_checks",
        help="To disable checking that input file exists",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore_existing",
        help="To ignore existing output file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--subsample_checks",
        help="To take only a subsample of audio to check formats",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no_overlapping",
        default=False,
        action="store_true",
        help="To ensure there is no overlapping between segments",
    )
    parser.add_argument(
        "--folder_depth",
        help="Number of folder name to include in the final id",
        default=0,
        type=int,
    )
    parser.add_argument("--folder_metadata", help="Folder with metadata about each file", default=None)
    parser.add_argument(
        "--ignore_speakers",
        default=False,
        action="store_true",
        help="To ignore speaker information (when it's not reliable)",
    )
    parser.add_argument(
        "--unknown_speaker_ids",
        default=None,
        nargs="+",
        help="The speaker id associed to unknown speaker",
    )
    parser.add_argument(
        "--preview_mode",
        default=False,
        action="store_true",
        help="If True, audios are not copied, making it faster and lighter",
    )
    args = parser.parse_args()

    if not args.license and args.private:
        args.license = "Private"
    assert args.license, "Please specify a license with --license"

    # params used from data processing
    folder = args.dir_in  # "/home/jlouradour/projects/VoiceLabData/data_kaldi/TCOF"
    folder_raw = args.raw if args.raw else folder
    output_folder = args.dir_out  # "to_upload/tcof"
    MAX = args.max

    audio_ext = args.ext
    # params used for data saving
    if args.name:
        corpus_name = args.name
    else:
        corpus_name = os.path.basename(folder)  # .lower()
    corpus_name_lower = corpus_name.split()[0].lower()

    # start processing

    with open(os.path.join(folder, "text")) as f:
        lines = f.readlines()
    print("Parsing text")
    text = {}
    for line in lines:
        line = line.strip().split()
        text[line[0]] = " ".join(line[1:])

    file_raw = os.path.join(folder_raw, "text") if folder_raw != folder else os.path.join(folder, "text_raw")
    HAS_RAW = os.path.exists(file_raw)
    if HAS_RAW:
        print("Parsing", file_raw)
        with open(file_raw) as f:
            lines = f.readlines()
        rawtext = {}
        for line in lines:
            line = line.strip().split()
            rawtext[line[0]] = " ".join(line[1:])
        assert len(rawtext) == len(text), f"rawtext and text have different number of lines: {len(rawtext)} != {len(text)}"

    with open(os.path.join(folder, "wav.scp")) as f:
        print("Parsing wav.scp")
        sys.stdout.flush()

        lines = f.readlines()

        wavs = {}
        for line in tqdm(lines):
            fields = line.strip().split()
            audio_path = None
            for elem in fields[1:]:
                if elem.endswith(audio_ext):
                    audio_path = elem
                    break
                elif elem.endswith(audio_ext + "'") or elem.endswith(audio_ext + '"'):
                    match = re.search(r"['\"].*" + audio_ext + r"['\"]", line)
                    assert match is not None, "Could not parse: " + line
                    start, end = match.span()
                    audio_path = line[start + 1 : end - 1]
                    break

            assert audio_path is not None, f"Could not find audio file in: {line} are you sure that --ext={args.ext} corresponds to the extension of the audio file?"
            wavs[fields[0]] = audio_path
            # This verification takes too long. Whatever will fail later...
            # if os.path.exists(audio_path):
            #     wavs[fields[0]] = audio_path
            # else:
            #     print("WARNING: could not find "+audio_path)

    total_duration = None
    has_segments = False
    if os.path.exists(os.path.join(folder, "segments")):
        print("Parsing segments")
        has_segments = True
        sys.stdout.flush()
        with open(os.path.join(folder, "segments")) as f:
            lines = f.readlines()

            segments = {}
            for line in tqdm(lines):
                line = line.strip().split()
                segments[line[0]] = line[1:]
    else:
        utt2dur_file = os.path.join(folder, "utt2dur")
        if os.path.exists(utt2dur_file):

            def convert(line):
                f = line.split()
                try:
                    assert len(f) == 2
                    return (f[0], (f[0], "0", f[1]))
                except:
                    raise RuntimeError("Problem to extract duration in line %s" % line)

            segments = dict([convert(line) for line in open(utt2dur_file).readlines()])
        else:
            print("Computing audio length")
            ids = text.keys()
            from operator import itemgetter

            audio_files = itemgetter(*ids)(wavs)
            segments = dict(
                zip(
                    text.keys(),
                    map(
                        lambda i, dur: (i, "0", str(dur)),
                        ids,
                        get_audio_durations(audio_files),
                    ),
                )
            )
        total_duration = sum([float(seg[-1]) for seg in segments.values()])

    with open(os.path.join(folder, "utt2spk")) as f:
        print("Parsing utt2spk")
        sys.stdout.flush()

        lines = f.readlines()

        spks = {}
        for line in tqdm(lines):
            line = line.strip().split()
            spks[line[0]] = line[1]

    gender = {}
    if os.path.exists(os.path.join(folder, "spk2gender")):
        with open(os.path.join(folder, "spk2gender")) as f:
            lines = f.readlines()

        for line in tqdm(lines):
            line = line.strip().split()
            gender[line[0]] = line[1]

    extra_utt = {}
    if os.path.exists(os.path.join(folder, "extra_utt.csv")):
        with open(os.path.join(folder, "extra_utt.csv")) as f:
            for row in csv.DictReader(f):
                extra_utt[row["id"]] = row

    output_folder_annots = output_folder + ("/annotation_processed" if HAS_RAW else "/annotation")
    os.makedirs(output_folder_annots, exist_ok=True)
    if HAS_RAW:
        output_folder_annots_raw = output_folder + "/annotation_raw"
        os.makedirs(output_folder_annots_raw, exist_ok=True)

    keys = text.keys()
    if args.disable_kaldi_checks:
        if keys != segments.keys() or keys != spks.keys():
            n = len(keys)
            print("WARNING: not all ids have all information")
            keys = set(text.keys()).intersection(segments.keys()).intersection(spks.keys())
            print("Found %d / %d ids with all informations" % (len(keys), n))
    else:
        assert keys == segments.keys() == spks.keys(), "kaldi data error in %s" % folder

    data = []
    genders = {}
    speakers = set()
    durations = []
    several_utt_per_wav = False
    all_wavs = {}
    is_a_speaker_missing = False
    for key in tqdm(keys):
        _text = text[key]
        _spk = spks[key]
        _gender = gender[_spk] if _spk in gender.keys() else ""
        if (isinstance(args.unknown_speaker_ids, list) and _spk in args.unknown_speaker_ids) or (_spk == args.unknown_speaker_ids):
            is_a_speaker_missing = True
            _spk = None
        else:
            speakers.add(_spk)
            genders[_spk] = _gender
        _segment = segments[key]
        _wav_id = _segment[0]
        _start = float(_segment[1])
        _end = float(_segment[2])
        _duration = _end - _start
        _wav = wavs[_wav_id] if _wav_id in wavs.keys() else ""

        if has_segments and not several_utt_per_wav:
            if _wav in all_wavs:
                several_utt_per_wav = True
        all_wavs[_wav] = _wav_id

        data.append(
            {
                "id": key,
                "text": _text,
                "rawtext": rawtext[key] if HAS_RAW else _text,
                "wav": _wav,
                "speaker": _spk,
                "gender": _gender,
                "start": _start,
                "end": _end,
                "duration": _duration,
            }
        )
        durations.append(_end - _start)

    num_wavs = None
    if several_utt_per_wav:
        data = sorted(data, key=lambda x: x["wav"])  # Data must be sorted before groupby

        def get_iterator():
            return groupby(data, lambda x: x["wav"])

    else:

        def get_iterator():
            for d in data:
                yield d["wav"], [d]

        num_wavs = len(data)

    now = datetime.now()

    annotation_extra_fields = {}
    if os.path.isfile(os.path.join(folder, "extra_spk.csv")):
        speaker_metadata = {}
        with open(os.path.join(folder, "extra_spk.csv")) as f:
            for row in csv.DictReader(f):
                spk = row["id"]
                assert spk in speakers, f"Speaker {spk} not found in data (ex: {list(speakers)[:10]})"
                speaker_metadata[spk] = row
                speaker_metadata[spk].pop("id")
        annotation_extra_fields["speaker_metadata"] = speaker_metadata

    if not os.path.isfile(os.path.join(output_folder, "meta.json")):
        audio_files = []
        num_audio_files = 0
        for _wav, utt in tqdm(get_iterator(), total=num_wavs):
            if args.disable_file_checks or os.path.exists(_wav):
                num_audio_files += 1
                if num_audio_files > MAX:
                    break
                audio_files.append(_wav)
            else:
                print("WARNING: could not find " + _wav)
        if num_wavs is None:
            num_wavs = num_audio_files
        sample = audio_files
        if args.subsample_checks:
            if len(sample) > 1000:
                # assert total_duration is not None, "The total duration cannot be computed with option --subsample_checks and without precomputed duration info"
                if total_duration is None:
                    wav_files = list(set([d["wav"] for d in data]))
                    if num_wavs is None:
                        num_wavs = len(wav_files)
                    _, total_duration = get_audio_total_duration(wav_files, verbose=True)
                    del wav_files
                assert isinstance(total_duration, float), f"total_duration should be a float, not {type(total_duration)}"
                sample = random.sample(audio_files, 1000)
        if len(" ".join(sample)) >= 2097152 - 5:
            while len(" ".join(sample)) >= 2097152 - 5:
                sample = sample[: int(len(sample) / 2)]

        print("Running soxi on %d/%d audio files" % (len(sample), len(audio_files)))
        infos = subprocess.check_output(["soxi"] + sample).decode("utf8")
        print("ok")

        use_percent = len(sample) < len(audio_files)
        sample_rate, note_sample_rate = extract_info_from_soxi_output(infos, "Sample Rate", use_percent)
        bit_depth, note_bit_depth = extract_info_from_soxi_output(infos, "Precision", use_percent)  # "Sample Encoding"
        channels, note_channels = extract_info_from_soxi_output(infos, "Channels", use_percent)
        if total_duration is None and len(sample) == len(audio_files):
            total_duration = infos.strip().split("\n")[-1].split()[-1]
            total_duration = timestamp2sec(total_duration)

    wav_dates = {}
    if os.path.isfile(folder + "/extra_wav.csv"):
        with open(folder + "/extra_wav.csv") as f:
            for row in csv.DictReader(f):
                wav_dates[row["id"]] = row["date"]

    # if args.subsample_checks:
    #     assert note_sample_rate is None
    #     assert note_bit_depth is None
    #     assert note_channels is None

    min_date = None
    max_date = None
    min_date_annot = None
    max_date_annot = None

    num_audio_files = 0
    total_duration_speech = 0
    for _wav, utt in tqdm(get_iterator(), total=num_wavs):
        num_audio_files += 1
        if num_audio_files > MAX:
            break

        filename = os.path.splitext(os.path.basename(_wav))[0]
        if args.folder_depth > 0:
            d = os.path.dirname(_wav)
            for i in range(args.folder_depth):
                de = os.path.basename(d)
                if de not in ["wav"]:
                    filename = de + "_" + filename
                d = os.path.dirname(d)
        out_name = f"{corpus_name_lower}_{filename}"
        _out_wav = os.path.join(output_folder, out_name + ".audio" + audio_ext)
        if not args.preview_mode:
            if os.path.exists(_out_wav):
                if not args.ignore_existing:
                    raise RuntimeError("Conflict on %s" % _out_wav)
            else:
                # convert audio to wav
                if not CONVERT_ALL_WAV_TO_THE_SAME and _wav.endswith(audio_ext):
                    shutil.copyfile(_wav, _out_wav)
                else:
                    cmd = [
                        "sox",
                        _wav,
                        #'-t', 'wav',
                    ]
                    if CONVERT_ALL_WAV_TO_THE_SAME:
                        cmd += [
                            "-r",
                            str(sample_rate),  # 16k
                            "-b",
                            str(bit_depth),  # 16
                            "-c",
                            str(channels),  # 1/2
                        ]
                    cmd += [_out_wav]
                    subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

        # Look for a date
        date = None
        xml_file = re.sub(os.path.splitext(_wav)[-1] + "$", ".xml", _wav)
        date_file = None
        if os.path.isfile(xml_file):
            date = extract_from_xml(xml_file, "date")
            date_file = xml_file
        if date is None:
            date = wav_dates.get(all_wavs[_wav])
            date_file = folder + "/extra_wav.csv"
        if date is None and args.folder_metadata:
            metadata = os.path.join(
                args.folder_metadata,
                os.path.splitext(os.path.basename(_wav))[0] + ".json",
            )
            if os.path.isfile(metadata):
                with open(metadata) as f:
                    metadata = json.load(f)
                for key in (
                    "publish_date",
                    "date_recorded",
                    "date_created",
                    "date_modified",
                ):
                    if key in metadata:
                        date = metadata[key]
                        break
                if date is None:
                    print(f"WARNING: could not find date in metadata {metadata}")
                else:
                    date_file = metadata
            else:
                print(f"WARNING: could not find metadata {metadata}")
        # Conversion to timestamp
        if isinstance(date, str):
            try:
                if "/" in date:
                    if date.endswith("0000"):
                        date = None
                    else:
                        date = date.replace("/15/", "/12/").replace("00/", "01/")  # fixing annot
                        date = datetime.strptime(date, "%d/%m/%Y")
                elif "-" in date:
                    if "T" in date:
                        if date.endswith("+00:00"):
                            date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+00:00")
                        else:
                            date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
                    else:
                        date = datetime.strptime(date, "%Y-%m-%d")
                else:
                    raise RuntimeError('Could not parse "%s" in %s' % (date, date_file))
            except Exception as err:
                raise RuntimeError('Could not parse "%s" in %s' % (date, date_file)) from err
        if date != None:
            min_date = date if min_date is None else min(date, min_date)
            max_date = date if max_date is None else max(date, max_date)

        # Accumulate transcriptions
        transcriptions = []
        transcriptions_raw = []
        previous_start = 0
        previous_end = 0
        for utterance in sorted(utt, key=lambda x: float(x["start"])):
            if args.ignore_speakers or utterance["speaker"] is None or args.version >= "0.0.2":
                extra = {}
            else:
                extra = {"speaker": utterance["speaker"]}
            if utterance["gender"] and not args.ignore_speakers:
                extra.update({"gender": utterance["gender"]})
            start = int(utterance["start"] * 1000)
            end = int(utterance["end"] * 1000)
            if args.no_overlapping:
                if start < previous_end:
                    middle = (start + previous_end) / 2
                    if end <= middle:
                        print(f"WARNING: overlapping segment ignored (case 1): \"{utterance['text']}\" ({start/1000:.2f}->{end/1000:.2f} included in {previous_start/1000:.2f}->{previous_end/1000:.2f})")
                        continue
                    if middle <= previous_start:
                        print(f"WARNING: overlapping segment ignored (case 2): \"{utterance['text']}\" ({start/1000:.2f}->{end/1000:.2f} included in {previous_start/1000:.2f}->{previous_end/1000:.2f})")
                        continue
                    start = int(middle)
                    transcriptions[-1]["timestamp_end_milliseconds"] = start
                    if HAS_RAW:
                        transcriptions_raw[-1]["timestamp_end_milliseconds"] = start
            previous_end = end
            previous_start = start
            other = {}
            if utterance["id"] in extra_utt:
                extra_this = extra_utt[utterance["id"]]
                for f in ("channel",):
                    assert f in extra_this, f"Missing field {f} in extra_utt.csv"
                    if f in extra_this:
                        other[f] = int(extra_this[f])
            extra = {"extra": extra} if extra else {}
            transcriptions.append(
                (
                    {"date_created": time2str(now)}  # Overkill, in first version
                    if args.version == "0.0.1"
                    else {}
                )
                | {
                    "transcript": utterance["text"],
                    "timestamp_start_milliseconds": start,
                    "timestamp_end_milliseconds": end,
                }
                | (
                    {
                        "speaker": utterance["speaker"],
                    }
                    if (args.version >= "0.0.2" and not args.ignore_speakers and utterance["speaker"] is not None)
                    else {}
                )
                | other
                | extra
            )
            if HAS_RAW:
                t_raw = transcriptions[-1].copy()
                t_raw["transcript"] = utterance["rawtext"]
                transcriptions_raw.append(t_raw)
            total_duration_speech += utterance["duration"]

        transcript_base = (
            {
                "version": args.version,
            }
            if args.version >= "0.0.2"
            else {}
        ) | {
            "format_specification_uri": f"http://levoicelab.org/schemas/{args.version}/annotation-single.schema.transcription.json",
        }
        if not os.path.isfile(os.path.join(output_folder_annots, out_name + ".annotations.json")):
            with open(os.path.join(output_folder_annots, out_name + ".annotations.json"), "w") as f:
                json_dump(transcript_base | {"transcripts": transcriptions}, f)

        if HAS_RAW:
            with open(
                os.path.join(output_folder_annots_raw, out_name + ".annotations.json"),
                "w",
            ) as f:
                json_dump(transcript_base | {"transcripts": transcriptions_raw}, f)

        if not os.path.isfile(os.path.join(output_folder, out_name + ".meta.json")):
            if not args.preview_mode:
                duration = get_audio_duration(_out_wav)
            else:
                duration = 0
            with open(os.path.join(output_folder, out_name + ".meta.json"), "w") as f:
                json_dump(
                    (
                        {
                            "version": args.version,
                        }
                        if args.version >= "0.0.2"
                        else {}
                    )
                    | {
                        "format_specification_uri": f"http://levoicelab.org/schemas/{args.version}/audio-format.schema.json",
                        "duration_milliseconds": int(duration * 1000),
                        "is_natural": True,
                        "is_augmented": False,
                        "is_synthetic": False,
                        "date_created": time2str(now),
                        "collection_date": time2str(date),
                    },
                    f,
                )

    if total_duration is None:
        total_duration = total_duration_speech

    if not os.path.isfile(os.path.join(output_folder, "meta.json")):
        with open(os.path.join(output_folder, "meta.json"), "w") as f:
            extra = (
                {}
                if args.ignore_speakers
                else {
                    "num_speakers": len(speakers),
                }
            )
            fcount = list(genders.values()).count("f")
            mcount = list(genders.values()).count("m")
            if fcount > 0 and mcount > 0:
                extra.update(
                    {
                        "gender": {
                            "female": fcount,  # "f" in genders.values(),
                            "male": mcount,  # "m" in genders.values(),
                        },
                    }
                )
            if note_sample_rate or note_channels or note_bit_depth:
                concerned = (["sample_rate"] if note_sample_rate else []) + (["num_channels"] if note_channels else []) + (["bit_depth"] if note_bit_depth else [])
                extra.update({"notes": "This corpus contains audio files with several values for %s." % (" / ".join(concerned))})
            if note_sample_rate:
                extra.update({"sample_rate": note_sample_rate})
            if note_channels:
                extra.update({"num_channels": note_channels})
            if note_bit_depth:
                extra.update({"bit_depth": note_bit_depth})

            extra = {"extra": extra} if extra else {}

            metadata = (
                {
                    "name": corpus_name,
                    "description": (args.description if args.description else corpus_name),
                    "date_created": time2str(now),
                    "collection_date_from": time2str(min_date),
                    "collection_date_to": time2str(max_date),
                }
                | (
                    {
                        "version": args.version,
                    }
                    if args.version >= "0.0.2"
                    else {}
                )
                | {
                    # "format_specification_uri": "http://www.levoicelab.org/annotation_conventions/batvoice_transcription_conventions-v1.1",
                    "format_specification_uri": f"http://levoicelab.org/schemas/{args.version}/main-db.schema.json",
                    "num_channels": channels,
                    "sample_rate": sample_rate,
                    "license": args.license,
                    "is_private": args.private,
                    "contact": {
                        "organization": "LINAGORA",
                        "name": "Jean-Pierre LORRE",
                        "email": "jplorre@linagora.com",
                        "uri": "https://labs.linagora.com/",
                    },
                    "contains_augmented_speech": False,
                    "contains_synthetic_speech": False,
                    "contains_natural_speech": True,
                    "total_duration_seconds": round(total_duration),
                    "natural_speech_duration_seconds": round(total_duration_speech),
                    "audio_format": args.ext.split(".")[-1],
                    "bit_depth": bit_depth,
                    "num_audio_files": num_audio_files,
                    "augmented_speech_duration_seconds": 0,
                    "synthetic_speech_duration_seconds": 0,
                }
                | (
                    {
                        "languages": args.languages,
                    }
                    if args.version >= "0.0.2"
                    else {}
                )
                | extra
            )
            json_dump(metadata, f)

    if not os.path.isfile(os.path.join(output_folder_annots, "meta.json")):
        speaker_information = "none" if args.ignore_speakers else "uuid"
        if speaker_information == "uuid" and is_a_speaker_missing:
            speaker_information += "-with-missing"
        metadata = (
            (
                {
                    "version": args.version,
                }
                if args.version >= "0.0.2"
                else {}
            )
            | {
                "format_specification_uri": f"http://levoicelab.org/schemas/{args.version}/annotation-batch.schema.json",
                # "format_specification_uri": "http://www.levoicelab.org/annotation_conventions/batvoice_transcription_conventions-v1.1",
                "date_created": time2str(now),
                "annotation_date_from": time2str(min_date_annot),
                "annotation_date_to": time2str(max_date_annot),
            }
            | (
                {
                    "annotation_type": "transcription",
                    "speaker_information": speaker_information,
                }
                if args.version >= "0.0.2"
                else {}
            )
            | {
                "contact": {
                    "organization": "LINAGORA",
                    "name": "Jean-Pierre LORRE",
                    "email": "jplorre@linagora.com",
                    "uri": "https://labs.linagora.com/",
                },
                "extra": {
                    "word_alignement": False,
                    "utt_alignement": True,
                    "avg_utt_alignement_duration_second": sum(durations) / len(durations),
                }
                | annotation_extra_fields,
            }
        )
        with open(os.path.join(output_folder_annots, "meta.json"), "w") as f:
            json_dump(metadata, f)

    if HAS_RAW:
        shutil.copy(
            os.path.join(output_folder_annots, "meta.json"),
            os.path.join(output_folder_annots_raw, "meta.json"),
        )
