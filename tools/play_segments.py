#!/usr/bin/env python3

import csv
import os
import tempfile

from ssak.utils.dataset import kaldi_folder_to_dataset
from ssak.utils.format_transcription import to_linstt_transcription
from ssak.utils.player import play_audiofile


def play_segments(
    audio_file,
    transcript,
    min_sec=0,
    wordlevel=False,
    play_silences=False,
    other_commands={},
    quit_command={"q": "quit"},
):
    assert audio_file and os.path.isfile(audio_file), f"Cannot find audio file {audio_file}"

    name = "word" if wordlevel else "segment"

    assert len(quit_command) == 1, f"Quit command must be a single key. Got {quit_command}"
    QUIT = list(quit_command.keys())[0]

    if other_commands:
        for o in other_commands:
            assert o not in list(quit_command.keys()) + ["s", "dbg"] or isinstance(o, (float, int)), f"Command {o} is already defined"
    additional_commands = dict(quit_command)
    additional_commands.update(other_commands)
    additional_commands.update(
        {
            "dbg": "debug",
            20.05: "seek (forward or rewind) to 20.05 sec",
            "p": "go to previous audio file",
        }
    )
    if wordlevel:
        additional_commands.update(
            {
                "s": "skip remaining words in segment",
            }
        )

    previous_start = 0
    global previous_wav
    global previous_transcript

    for i, segment in enumerate(transcript["segments"]):
        # print(f'{segment["text"]} : {segment["start"]}-{segment["end"]}')
        # play_audiofile(audio_file, segment["start"], segment["end"], ask_for_replay = True)

        current_wav = audio_file
        current_transcript = transcript

        if not wordlevel:
            segment["words"] = [segment]

        text_key = None

        for iw, word in enumerate(segment["words"]):
            if text_key is None:
                if "text" in word:
                    text_key = "text"
                elif "segment" in word:
                    text_key = "segment"
                elif "word" in word:
                    text_key = "word"
                else:
                    raise ValueError(f"Cannot find text key in {word}")

            txt = word[text_key]
            start = word["start"]
            end = word["end"]
            if end < min_sec:
                previous_start = end
                continue

            if play_silences and previous_start < start:
                print(f"Silence : {previous_start}-{start}")
                x = play_audiofile(audio_file, previous_start, start, additional_commands=additional_commands)
            else:
                x = None
            previous_start = end

            if x not in additional_commands.keys() and not isinstance(x, (float, int)):
                # Regular play
                if wordlevel:
                    print(f"== segment {i+1}/{len(transcript['segments'])}, {name} {iw+1}/{len(segment['words'])} : {start}-{end}")
                else:
                    print(f"== segment {i+1}/{len(transcript['segments'])} : {start}-{end}")
                print(txt)

                x = play_audiofile(audio_file, start, end, additional_commands=additional_commands)

            if x == "p":
                if previous_wav and previous_transcript:
                    return play_segments(
                        previous_wav,
                        previous_transcript,
                        min_sec=min_sec,
                        wordlevel=wordlevel,
                        play_silences=play_silences,
                        other_commands=other_commands,
                        quit_command=quit_command,
                    )
            previous_wav = current_wav
            previous_transcript = current_transcript

            if x == QUIT:
                return QUIT
            elif x == "s":
                break
            elif x == "dbg":
                import pdb

                pdb.set_trace()
            elif isinstance(x, (float, int)):
                min_sec = x
                if min_sec < start:
                    # Rewind
                    return play_segments(
                        audio_file,
                        transcript,
                        min_sec=min_sec,
                        wordlevel=wordlevel,
                        play_silences=play_silences,
                        other_commands=other_commands,
                        quit_command=quit_command,
                    )
            elif x in other_commands.keys():
                return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play audio file using transcript from a segmentation into words / segments")
    parser.add_argument("transcripts", type=str, help="Transcription file or Kaldi folder")
    parser.add_argument(
        "audio_file",
        type=str,
        help="Audio file (not necessary when transcript if a Kaldi folder)",
        default=None,
        nargs="?",
    )
    parser.add_argument("--words", default=False, action="store_true", help="Play words instead of segments")
    parser.add_argument("--min_sec", default=0, type=float, help="Minimum second to start playing from (default: 0)")
    parser.add_argument("--play_silences", default=False, action="store_true", help="Play silence between words")
    args = parser.parse_args()

    audio_file = args.audio_file
    transcripts = args.transcripts

    if os.path.isdir(transcripts):
        # Kaldi folder
        # We will filter corresponding to the wav file (or not  )

        _, tmp_csv_in = kaldi_folder_to_dataset(transcripts, return_format="csv")
        current_audio_file = audio_file if audio_file else None
        tmp_csv_out = tempfile.mktemp(suffix=".csv")
        fid_csv_out = None
        csvwriter = None
        header = None

        try:
            do_quit = False
            with open(tmp_csv_in, encoding="utf8") as fin:
                csvreader = csv.reader(fin)
                for i, row in enumerate(csvreader):
                    if i == 0:
                        # Read header
                        ipath = row.index("path")
                        header = row
                        fid_csv_out = open(tmp_csv_out, "w", encoding="utf8")
                        csvwriter = csv.writer(fid_csv_out)
                        csvwriter.writerow(header)
                    else:
                        path = row[ipath]
                        if audio_file and os.path.basename(path) != os.path.basename(audio_file):
                            continue
                        if (csvwriter is None) if audio_file else (path != current_audio_file):
                            x = None
                            if csvwriter is not None and current_audio_file:
                                if fid_csv_out:
                                    fid_csv_out.close()
                                fid_csv_out = None
                                transcript = to_linstt_transcription(tmp_csv_out, warn_if_missing_words=args.words)
                                x = play_segments(
                                    current_audio_file,
                                    transcript,
                                    wordlevel=args.words,
                                    min_sec=args.min_sec,
                                    play_silences=args.play_silences,
                                    other_commands={"n": "skip audio file"} if not audio_file else {},
                                )
                            current_audio_file = path
                            print("Playing audio file:", current_audio_file)
                            fid_csv_out = open(tmp_csv_out, "w", encoding="utf8")
                            csvwriter = csv.writer(fid_csv_out)
                            csvwriter.writerow(header)
                            if x == "q":
                                do_quit = True
                                break
                        csvwriter.writerow(row)

            if not do_quit and csvwriter is not None:
                if fid_csv_out:
                    fid_csv_out.close()
                fid_csv_out = None
                transcript = to_linstt_transcription(tmp_csv_out, warn_if_missing_words=args.words)
                play_segments(
                    current_audio_file,
                    transcript,
                    wordlevel=args.words,
                    min_sec=args.min_sec,
                    play_silences=args.play_silences,
                )

        finally:
            if fid_csv_out:
                fid_csv_out.close()
            if os.path.isfile(tmp_csv_in):
                os.remove(tmp_csv_in)
            if os.path.isfile(tmp_csv_out):
                os.remove(tmp_csv_out)
    else:
        if not audio_file:
            raise ValueError("Please provide an audio file when a transcript file is provided")
        assert os.path.isfile(audio_file), f"Cannot find audio file {audio_file}"
        assert os.path.isfile(transcripts), f"Cannot find transcription file {transcripts}"
        transcript = to_linstt_transcription(transcripts, warn_if_missing_words=args.words)

        play_segments(audio_file, transcript, wordlevel=args.words, min_sec=args.min_sec, play_silences=args.play_silences)
