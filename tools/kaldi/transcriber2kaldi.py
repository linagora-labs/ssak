#!/usr/bin/env python3

import os
import re
import warnings
from glob import glob

from tqdm import tqdm

from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.transcriber import read_transcriber


def do_ignore_text(text):
    return len(text.strip().strip("._").lower()) == 0


# Note: this is because xmltodict.parse removes trailing \n
#       and \n are specially important to check number of speakers in a turn, with an xml like:
# """
# <?xml version="1.0" encoding="ISO-8859-1"?>
# <!DOCTYPE Trans SYSTEM "trans-14.dtd">
# <Trans scribe="Authot" audio_filename="Linagora_A1_0:05:27--end" version="3" version_date="200114">
# <Episode>
# <Section type="report" startTime="0" endTime="797.718">
# <Turn speaker="spk1 spk4" startTime="105.576" endTime="112.150">
# <Sync time="105.576">
# </Sync>
# <Sync time="105.576">
# <Who nb="1"/>
# Speaker 1
# <Who nb="2"/>
#
# </Sync>
# </Turn>
# </Section>
# </Episode>
# </Trans>
# """


def transcriber2kaldi_add_file(
    trs_file,
    audio_file,
    output_folder,
    max_speakers=1,
    anonymization_level=0,
    max_text_length=None,
    remove_extra_speech=True,
):
    alldata = read_transcriber(
        trs_file,
        anonymization_level=anonymization_level,
        remove_extra_speech=remove_extra_speech,
    )

    # Save the data
    os.makedirs(output_folder, exist_ok=True)

    basename = os.path.basename(trs_file.split(".")[0])
    basename = basename.lower()

    spk2gender = output_folder + "/spk2gender"
    with open(output_folder + "/segments", "a") as segments_fid, open(output_folder + "/utt2spk", "a") as utt2spk_fid, open(output_folder + "/text", "a") as text_fid, open(output_folder + "/wav.scp", "a") as wav_fid, open(
        spk2gender, "a"
    ) as spk2gender_fid:
        final_list_of_speakers = {}
        for data in alldata:
            if do_ignore_text(data["text"]):
                continue
            elif not data["text"].isprintable():
                warnings.warn(f"Skipping non-printable characters for {data['id']} in text: {data['text']}")
                continue
            elif float(data["sTime"]) >= float(data["eTime"]):
                warnings.warn(f"Skipping empty segment for {data['id']}: {data['sTime']}>={data['eTime']} (with text: {data['text']})")
                continue
            if data["nbrSpk"] > max_speakers:
                if max_speakers > 1:
                    warnings.warn("Skipping segment with %d speakers > %d" % (data["nbrSpk"], max_speakers))
                continue
            if max_text_length and len(data["text"]) > max_text_length:
                warnings.warn("Skipping segment with text of length %d > %d" % (len(data["text"]), max_text_length))
                continue
            final_list_of_speakers[data["spkId"]] = data["gender"]
            segments_fid.write(data["id"] + " " + basename + " " + data["sTime"] + " " + data["eTime"] + "\n")
            utt2spk_fid.write(data["id"] + " " + data["spkId"] + "\n")
            text_fid.write(data["id"] + " " + data["text"] + "\n")

        for spk_id, gender in final_list_of_speakers.items():
            spk2gender_fid.write(spk_id + " " + re.sub("^[^mf].*", "m", gender) + "\n")

        wav_fid.write(basename + " sox " + audio_file + " -t wav -r 16k -b 16 -c 1 - |\n")

    assert os.system("sort %s | uniq > %s.tmp" % (spk2gender, spk2gender)) == 0
    assert os.system("mv %s.tmp %s" % (spk2gender, spk2gender)) == 0


def list_files(folder, subfolders=False, extension=None):
    if subfolders:
        list_files = []
        for root, dirs, files in os.walk(folder):
            root = root[len(folder) :].lstrip("/")
            to_add = [os.path.join(root, i) for i in files if (i.lower().endswith(extension) or extension is None)]
            list_files.extend(to_add)
        return list_files
    else:
        return [os.path.join(folder, i) for i in os.listdir(folder) if (i.lower().endswith(extension) or extension is None)]


def to_eslo_id(basename):
    """For ESL, we remove the last part of the basename"""
    return "_".join(basename.split("_")[:-1])


def to_clapi_id(basename):
    return re.sub(r"transcription_transcriber$", "audio_", basename)
    # return re.sub(r"transcription_orthographe_standard_transcriber$", "audio_wav", basename)


def transcriber2kaldi(
    trs_folder,
    audio_folder,
    output_folder,
    language=None,
    extension=".trs",
    audio_extensions=[".wav", ".mp3"],
    subfolders=False,
    function_to_id=None,
    ignore_missing_audio=False,
    force=False,
    **kwargs,
):
    if os.path.isdir(output_folder) and not force:
        raise RuntimeError(f"Output folder {output_folder} already exists. Please remove it to overwrite.")

    for filename in tqdm(list_files(trs_folder, subfolders=subfolders, extension=extension)):
        basename = os.path.splitext(filename)[0]
        if function_to_id is not None:
            basename = function_to_id(basename)
        trs_file = os.path.join(trs_folder, filename)
        audio_files = []
        for audio_extension in audio_extensions:
            audio_filepath = os.path.join(audio_folder, basename + audio_extension)
            audio_files += glob(audio_filepath)
        if len(audio_files) == 0:
            if not ignore_missing_audio:
                raise RuntimeError(f"Audio file not found for {filename} ({audio_filepath}) (in {audio_folder})")
            else:
                warnings.warn(f"Skipping {filename}, audio file not found in {audio_folder}")
                continue
        audio_file = audio_files[0]
        transcriber2kaldi_add_file(trs_file, audio_file, output_folder, **kwargs)

    check_kaldi_dir(output_folder, language=language)


def case_insensitive(s):
    return "".join(["[" + c.upper() + c.lower() + "]" if c.isalpha() else c for c in s])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a dataset in Transcriber format (.xml with extension .trs) into kaldi format")
    parser.add_argument("trs_folder", help="Folder with trs files")
    parser.add_argument("audio_folder", help="Folder with audio files (if different from trs folder)", nargs="?", default=None)
    parser.add_argument("output_folder", help="output directory")
    parser.add_argument(
        "--anonymization_level",
        default=0,
        type=int,
        choices=[0, 1, 2],
        help="0: No anonymization, 1: Change speaker names, 2: Total anonymization (default: 1)",
    )
    parser.add_argument("--max_speakers", default=1, type=int, help="Default number of speakers at the same time")
    parser.add_argument(
        "--remove_extra_speech",
        default=False,
        action="store_true",
        help="Remove extra speech (events, comments, background)",
    )
    parser.add_argument("--max_text_length", default=None, type=int, help="Maximum text length in number of charachers (default: None)")
    parser.add_argument("--language", default="fr", type=str, help="Main language (only for checking the charset and giving warnings)")
    parser.add_argument("--subfolders", default=False, action="store_true", help="Search for trs files in subfolders")
    parser.add_argument("--ignore_missing_audio", default=False, action="store_true", help="Ignore missing audio files")
    parser.add_argument(
        "--audio_extensions",
        default=[".wav", ".mp3"],
        nargs="+",
        type=str,
        help="Audio extensions to look for (default: .wav .mp3)",
    )
    parser.add_argument("--force", default=False, action="store_true", help="Force overwrite of output folder")
    parser.add_argument(
        "--function_to_id",
        default=None,
        type=str,
        help="Function to convert basename to id (e.g. to_eslo_id, to_clapi_id)",
    )
    parser.add_argument("--trs_extension", default=".trs", type=str, help="Extension of the transcriber files")
    args = parser.parse_args()

    if not args.audio_folder:
        args.audio_folder = args.trs_folder

    assert os.path.isdir(args.trs_folder), f"Folder {args.trs_folder} does not exist"
    assert os.path.isdir(args.audio_folder), f"Folder {args.audio_folder} does not exist"
    if not args.force:
        assert not os.path.isdir(args.output_folder), f"Folder {args.output_folder} already exists"

    assert args.anonymization_level >= 0
    assert args.max_speakers > 0, f"Invalid number of speakers: {args.max_speakers} (must be a strictly positive integer)"
    assert args.max_text_length is None or args.max_text_length > 0, f"Invalid maximum text length: {args.max_text_length} (must be a strictly positive integer or None)"

    functions_to_id = {
        "to_eslo_id": to_eslo_id,
        "to_clapi_id": to_clapi_id,
    }

    transcriber2kaldi(
        args.trs_folder,
        args.audio_folder,
        args.output_folder,
        max_speakers=args.max_speakers,
        anonymization_level=args.anonymization_level,
        max_text_length=args.max_text_length,
        remove_extra_speech=args.remove_extra_speech,
        language=args.language,
        subfolders=args.subfolders,
        ignore_missing_audio=args.ignore_missing_audio,
        audio_extensions=args.audio_extensions,
        force=args.force,
        function_to_id=functions_to_id[args.function_to_id] if args.function_to_id is not None else None,
        extension=args.trs_extension,
    )
