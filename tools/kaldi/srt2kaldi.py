#!/usr/bin/env python3

import csv
import os
import re
from datetime import datetime

from slugify import slugify
from tqdm import tqdm

from ssak.utils.audio import AUDIO_EXTENSIONS, mix_audios
from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text_utils import format_special_characters


def get_channel_funcgen(prefix="Speaker "):
    def get_channel(speaker):
        assert speaker.startswith(prefix)
        return int(speaker[len(prefix) :])

    return get_channel


def split_channel_funcgen(separator=":", get_channel=get_channel_funcgen()):
    # Generate a function to split an annotation line into channel, text
    def split_speaker(line):
        if separator not in line:
            # print(f"Separator {separator} not found in line: {line}")
            return 1, line
        speaker, text = line.split(separator, 1)
        return get_channel(speaker), text

    return split_speaker


def parse_timestamp(str):
    # Parse timestamp at format "00:00:01,710" in second
    return sum(float(x) * 60**i for i, x in enumerate(reversed(str.replace(",", ".").split(":"))))


def read_srt(filename, split_channel=split_channel_funcgen()):
    try:
        with open(filename) as fn:
            lines = fn.readlines()
        # Remove last empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        num_turns = 0
        turn_bias = None
        turn_number_error = False
        channel = None
        extra_empty_lines = 0
        speaker_overlap = False
        last_speaker_line = None
        for iline, line in enumerate(lines):
            line_orig = line.rstrip("\n")
            line = line.strip().strip("\ufeff")
            n = (iline - extra_empty_lines) % 4

            # Workaround for Atril livraison 2
            # if n == 0 and line == "UNKNOWN":
            #     assert last_speaker_line
            #     if last_speaker_line == "Speaker 1":
            #         line = "Speaker 2"
            #     elif last_speaker_line == "Speaker 2":
            #         line = "Speaker 1"
            #     else:
            #         raise ValueError(f"Unexpected last speaker line {last_speaker_line}")

            try:
                if n != 3 and not line:
                    extra_empty_lines += 1
                    continue
                if n < 2 and line.lower().startswith("speaker") and ":" not in line:
                    channel = line.lower().replace("speaker", "").replace("turn", "").strip()
                    channel = int(channel)
                    if n == 0:
                        if turn_bias:
                            iturn = num_turns + turn_bias
                        else:
                            iturn = num_turns
                    last_speaker_line = line
                    speaker_overlap = False
                elif n == 0:
                    iturn = int(line)
                    if turn_bias is None:
                        turn_bias = iturn - num_turns
                        if turn_bias not in [0, 1, 2]:
                            print(f"Warning: Unexpected turn bias {turn_bias} (turn {iturn} != {num_turns}) in {filename}")
                    if not iturn == num_turns + turn_bias and not turn_number_error:
                        print(f"Warning: Unexpected turn number {iturn} != {num_turns + turn_bias} in {filename}")
                        turn_number_error = True
                    speaker_overlap = False
                elif n == 1:
                    if "->" not in line and channel is not None:
                        int(line)
                        extra_empty_lines += 1
                        continue
                    start, end = line.split(" --> ")
                    start = parse_timestamp(start)
                    end = parse_timestamp(end.split()[0])
                    speaker_overlap = False
                elif n == 2 or (n == 3 and speaker_overlap and line):
                    is_overlap = n == 3 and speaker_overlap and line
                    if is_overlap:
                        assert channel is not None
                        yield {
                            "start": start,
                            "end": end,
                            "channel": channel,
                            "text": text,
                        }
                        if channel == 1:
                            channel = 2
                        else:
                            assert channel == 2
                            channel = 1
                        extra_empty_lines += 1
                        speaker_overlap = False
                    if channel is None:
                        channel, text = split_channel(line)
                    else:
                        text = line
                    if text.startswith("-- "):
                        text = text[3:].lstrip()
                        speaker_overlap = True
                    else:
                        speaker_overlap = False
                    text = format_special_characters(text)
                elif n == 3 and line.lower().startswith("speaker ") and ":" not in line:
                    # Dump previous segment
                    assert channel is not None
                    assert text
                    assert end
                    yield {
                        "start": start,
                        "end": end,
                        "channel": channel,
                        "text": text,
                    }
                    # register new speaker and overlap
                    channel = line.lower().replace("speaker", "").strip()
                    channel = int(channel)
                    speaker_overlap = False
                    extra_empty_lines += 2
                    last_speaker_line = line
                else:
                    assert n == 3
                    assert line == ""
                    assert channel is not None
                    yield {
                        "start": start,
                        "end": end,
                        "channel": channel,
                        "text": text,
                    }
                    num_turns += 1
                    channel = None
                    speaker_overlap = False
            except Exception as err:
                # Print error stack
                import traceback

                traceback.print_exc()
                raise ValueError(f"{err}\nUnexpected line {iline+1} ({n=}): '{line_orig}'") from err
    except Exception as err:
        raise RuntimeError(f"Error while reading {filename}") from err


def find_all_audio_files(audio_folder):
    audio_files = {}
    for root, dirs, files in os.walk(audio_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in AUDIO_EXTENSIONS:
                continue
            assert filename not in audio_files, f"Duplicate audio file {filename}"
            audio_files[filename] = os.path.join(root, filename)
    return audio_files


def find_audio_files(audio_files, name, same_exact_name=True):
    if same_exact_name:
        all_candidates = []
        for i in audio_files:
            if os.path.splitext(i)[0] == os.path.splitext(name)[0]:
                all_candidates.append(audio_files[i])
        if len(all_candidates):
            return name, exclude_stereo_audio(all_candidates)
    else:
        possible_prefixes = [name] + [name[: s.start()] for s in re.finditer(r"[\-_\.]", name)][-1::-1]
        for prefix in possible_prefixes:
            all_candidates = []
            for relpath, fullpath in audio_files.items():
                if relpath.startswith(prefix):
                    all_candidates.append(fullpath)
            if len(all_candidates):
                return prefix, exclude_stereo_audio(all_candidates)
    if "-" in name:
        name, ext = os.path.splitext(name)
        name = name.split("-")[0] + ext
        return find_audio_files(audio_files, name, same_exact_name=same_exact_name)
    raise RuntimeError(f"Could not find audio file for {name}")


def exclude_stereo_audio(audio_files):
    lens = [len(os.path.splitext(f)[0]) for f in audio_files]
    min_len = min(lens)
    max_len = max(lens)
    if min_len != max_len and min_len:
        if all([f[:min_len] == audio_files[0][:min_len] for f in audio_files]):
            return [f for f in audio_files if len(os.path.splitext(f)[0]) != min_len]
    return audio_files


def srt2kaldi(
    srt_folder,
    audio_folder,
    output_folder,
    new_audio_folder=None,
    language=None,
    metadata=None,
    missing_gender=False,
    ignore_existing_mixed_audio=True,
    encoding="utf8",
    stop_on_error=False,
    same_exact_name=True,
):
    utt_spk = {}
    spk_metadata = {}
    if isinstance(metadata, str):
        # Load csv
        def normalize_key(k):
            k = k.lower()
            if k in ["participant", "participant id"]:
                k = "speaker"
            if k in ["role"]:
                k = "channel"
            return k

        reader = csv.DictReader(open(metadata, encoding=encoding))
        utt_spk = {}
        spk_metadata = {}
        for row in reader:
            channel = None
            spk = None
            call_id = None
            for k, v in row.items():
                k = normalize_key(k)
                if k == "gender":
                    v = v.lower()[:1]
                    assert v in ["m", "f"]
                if k == "channel":
                    v = int(v)
                    assert v > 0

                if k == "id":
                    assert not call_id
                    call_id = v
                elif k == "channel":
                    assert not channel
                    channel = v
                elif k == "speaker":
                    assert not spk
                    spk = v
                else:
                    assert spk
                    spk_metadata[spk] = spk_metadata.get(spk, {})
                    if k in spk_metadata[spk]:
                        assert spk_metadata[spk][k] == v, f"Conflicting metadata for speaker {spk} and key {k}: {spk_metadata[spk][k]} != {v}"
                    else:
                        spk_metadata[spk][k] = v

            utt_spk[(call_id, channel)] = spk

    database_name = slugify(os.path.basename(output_folder))

    if new_audio_folder is None:
        new_audio_folder = os.path.join(output_folder, "audio")

    audio_files = find_all_audio_files(audio_folder)

    transcriptions_and_audio = {}
    for root, dirs, files in os.walk(srt_folder):
        for filename in files:
            if not filename.endswith(".srt"):
                continue
            trs_file = os.path.join(root, filename)
            try:
                basename, audios = find_audio_files(audio_files, filename, same_exact_name=same_exact_name)
            except Exception as err:
                if stop_on_error:
                    basename, audios = find_audio_files(audio_files, filename, same_exact_name=same_exact_name)
                print(f"Error: {err}")
                continue
            audios = sorted(audios)
            transcriptions_and_audio[basename] = (trs_file, audios)

    speakers_to_genders = {}
    speakers_to_metadata = {}
    os.makedirs(output_folder, exist_ok=True)

    with open(output_folder + "/text", "w", encoding=encoding) as text_file, open(output_folder + "/segments", "w", encoding=encoding) as segments_file, open(output_folder + "/wav.scp", "w", encoding=encoding) as wavscp_file, open(
        output_folder + "/utt2spk", "w", encoding=encoding
    ) as utt2spk_file, open(output_folder + "/utt2dur", "w", encoding=encoding) as utt2dur_file, open(output_folder + "/extra_utt.csv", "w", encoding=encoding) as extra_utt_file, open(
        output_folder + "/extra_wav.csv", "w", encoding=encoding
    ) as extra_wav_file:
        extra_utt = csv.writer(extra_utt_file)
        extra_utt.writerow(["id", "channel"])

        extra_wav = csv.writer(extra_wav_file)
        extra_wav.writerow(["id", "date"])

        for basename in tqdm(sorted(transcriptions_and_audio.keys())):
            trs_file, audio_files = transcriptions_and_audio[basename]

            # Check audio. Mix audios if several channels
            assert len(audio_files) > 0
            if len(audio_files) == 1:
                audio_file = audio_files[0]
            else:
                audio_file = os.path.join(new_audio_folder, basename + os.path.splitext(audio_files[0])[1])
                os.makedirs(new_audio_folder, exist_ok=True)
                assert len(audio_files) == 2, f"Unexpected number of audio files: {audio_files}"
                mix_audios(audio_files, audio_file, ignore_existing=ignore_existing_mixed_audio)
            assert os.path.isfile(audio_file)

            try:
                turns = list(read_srt(trs_file))
            except Exception as err:
                # if stop_on_error:
                if True:  # NOCOMMIT
                    # if any([c in trs_file for c in ["184813", "184814"]]):
                    #     continue
                    turns = list(read_srt(trs_file))
                print(f"Error: {err}")
                continue
            trs_channels = sorted(list(set([turn["channel"] for turn in turns])))
            expected_channels = list(range(1, 1 + len(audio_files)))
            if trs_channels < expected_channels and not stop_on_error:
                print(f"Unexpected channels: {trs_channels} != {expected_channels} (audio files: {audio_files})")
                continue
            assert trs_channels == expected_channels, f"Unexpected channels: {trs_channels} != {expected_channels} (audio files: {audio_files})"

            wav_id = f"{database_name}_{basename}"
            wavscp_file.write(f"{wav_id} sox {os.path.realpath(audio_file)} -t wav -r 16k -b 16 -c 1 - |\n")

            extra_wav.writerow(
                [
                    wav_id,
                    datetime.fromtimestamp(max([os.path.getmtime(f) for f in audio_files])).isoformat(timespec="seconds") + "+00:00",
                ]
            )

            use_dummy_meta_if_missing = True

            missing_spk_info = []
            if utt_spk:
                for channel in range(1, 1 + len(expected_channels)):
                    if (basename, channel) not in utt_spk:
                        missing_spk_info.append((basename, channel))
                if missing_spk_info:
                    msg = f"Speaker info missing on {missing_spk_info} (expected {len(expected_channels)} channels)"
                    if stop_on_error:
                        raise ValueError("ERROR: " + msg)
                    print(f"WARNING: {msg}")

                    if not use_dummy_meta_if_missing:
                        print("Skipping sample!!!")
                        continue

                    # Put dummy info
                    for basename, channel in missing_spk_info:
                        spk = f"{basename}_{channel}"
                        utt_spk[(basename, channel)] = spk
                        assert len(spk_metadata)
                        first_spk = list(spk_metadata.keys())[0]
                        spk_metadata[spk] = {}
                        for k in spk_metadata[first_spk].keys():
                            spk_metadata[spk][k] = "" if k != "gender" else "f"  # More female speakers by default

            for iturn, turn in enumerate(turns):
                channel = turn["channel"]
                text = turn["text"]
                start = turn["start"]
                end = turn["end"]
                duration = end - start
                if utt_spk:
                    assert (basename, channel) in utt_spk, f"Speaker info missing for {basename} channel {channel}"
                    spk = utt_spk[(basename, channel)]
                    spk_id = f"{database_name}_{spk}"
                    utt_id = f"{spk_id}_{basename}_{iturn+1:03d}"
                    gender = spk_metadata[spk]["gender"]
                    speakers_to_metadata[spk_id] = spk_metadata[spk]
                else:
                    spk_id = f"{database_name}_{basename}_{channel}"
                    utt_id = f"{database_name}_{basename}_{channel}_{iturn+1:03d}"
                    gender = "m"

                if spk_id not in speakers_to_genders:
                    speakers_to_genders[spk_id] = gender

                text_file.write(f"{utt_id} {text}\n")
                segments_file.write(f"{utt_id} {wav_id} {start:.3f} {end:.3f}\n")
                utt2spk_file.write(f"{utt_id} {spk_id}\n")
                utt2dur_file.write(f"{utt_id} {duration}\n")
                extra_utt.writerow([utt_id, channel])

    if not missing_gender:
        with open(output_folder + "/spk2gender", "w") as spk2gender_file:
            for speaker, gender in speakers_to_genders.items():
                spk2gender_file.write(f"{speaker} {gender}\n")

    if speakers_to_metadata:
        with open(output_folder + "/extra_spk.csv", "w", encoding=encoding) as extra_speaker_file:
            extra_spk = csv.writer(extra_speaker_file)
            keys = list(list(speakers_to_metadata.values())[0].keys())
            extra_spk.writerow(["id"] + keys)
            for speaker, metadata in speakers_to_metadata.items():
                extra_spk.writerow([speaker] + [metadata[k] for k in keys])

    return check_kaldi_dir(output_folder, language=language)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a dataset in CSV or TSV format (like CommonVoice) into kaldi format")
    parser.add_argument("srt_folder", type=str, help="Input folder with TRS files")
    parser.add_argument("audio_folder", type=str, help="Input folder with audio files (same as srt_folder if not specified)", nargs="?")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--metadata", default=None, type=str, help="Metadata file")
    parser.add_argument("--language", default=None, type=str, help="Main language (only for checking the charset and giving warnings)")
    parser.add_argument("--missing_gender", action="store_true", help="Specify if generate spk2gender file")
    parser.add_argument("--ignore_errors", action="store_true", help="Ignore errors")
    parser.add_argument("--approx_name", action="store_true", help="Use approximate name matching")

    args = parser.parse_args()

    srt_folder = args.srt_folder
    output_folder = args.output_folder
    audio_folder = args.audio_folder
    if not audio_folder:
        audio_folder = srt_folder

    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isdir(srt_folder), f"Input folder not found: {srt_folder}"
    # assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"

    new_audio_folder = None
    # TODO: make it possible to save everything on the side ?

    srt2kaldi(
        srt_folder,
        audio_folder,
        output_folder,
        new_audio_folder=new_audio_folder,
        language=args.language,
        metadata=args.metadata,
        missing_gender=args.missing_gender,
        stop_on_error=not args.ignore_errors,
        same_exact_name=not args.approx_name,
    )
