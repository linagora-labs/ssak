#!/usr/bin/env python3

import csv
import json
import shutil

from tqdm import tqdm

from ssak.utils.language import (
    check_language,
    is_hate_speech,
)
from ssak.utils.text import (
    collapse_whitespace,
    format_special_characters,
    format_text,
    remove_parenthesis,
    remove_special_characters,
)


def load_csv_text(filename):
    with open(filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=";")
        assert next(reader)[0] == "text"
        text = "\n".join([row[0] for row in reader if row])  # Add a check for non-empty rows
        return text


def load_csv_times(filename):
    with open(filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=";")
        assert next(reader) == ["text", "start", "duration"]
        return [(float(row[1]), float(row[1]) + float(row[2])) for row in reader if row]


def custom_clean_text(
    text,
    do_remove_special_characters=True,
    do_format_special_characters=True,
    do_remove_parenthesis=True,
    do_unupper_case=False,
):
    if do_remove_parenthesis:
        text = remove_parenthesis(text)
    if do_format_special_characters:
        text = format_special_characters(text)
    if do_remove_special_characters:
        text = remove_special_characters(text)
    if do_unupper_case:
        text = text.lower().capitalize()
    text = collapse_whitespace(text)
    return text


def rewrite_csv(
    filename_in,
    filename_out,
    do_unupper_case=False,
):
    with open(filename_in, encoding="utf8") as f_in, open(filename_out, "w", encoding="utf8") as f_out:
        reader = csv.reader(f_in, delimiter=";")
        writer = csv.writer(f_out, delimiter=";")
        num_fields = 0
        for i, row in enumerate(reader):
            if i == 0:
                num_fields = len(row)
                assert num_fields, "Got empty header"
            else:
                if len(row) == 0:  # Ignore empty rows
                    continue
                assert len(row) == num_fields, f"Got {len(row)} fields instead of {num_fields} on line {i+1}"
                text = custom_clean_text(row[0], do_unupper_case=do_unupper_case)
                if not text:  # Ignore empty text
                    continue
                row[0] = text
            writer.writerow(row)


def transcription_dont_match(
    csv_file,
    mp3_file,
    model,
    language,
    max_duration=30,
    threshold=0.6,
):
    from ssak.infer.general import infer
    from ssak.utils.audio import load_audio
    from ssak.utils.wer import compute_wer

    def text_normalize(text):
        text = custom_clean_text(text)
        return format_text(text, language)

    def remove_spaces(text):
        return text.replace(" ", "").replace("-", "").replace("'", "")

    start, end = None, None
    transcription = ""
    with open(csv_file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=";")
        assert next(reader) == ["text", "start", "duration"]
        for row in reader:
            if start is None:
                start = float(row[1])
            end = float(row[1]) + float(row[2])
            text = text_normalize(row[0])
            if text:
                if transcription:
                    transcription += " "
                transcription += text
            if end - start > max_duration:
                break

    try:
        audio = load_audio(mp3_file, start=start, end=end)
    except Exception as err:
        return f"Failed to load audio {mp3_file}: {err}"

    reco = infer(model, [audio], language=language)
    reco = next(reco)
    reco = text_normalize(reco)

    cer = compute_wer([remove_spaces(transcription)], [remove_spaces(reco)], character_level=True)
    cer = cer["wer"]

    if cer > threshold:
        return json.dumps(
            {
                "CER": cer,
                "transcription": transcription,
                "reco": reco,
            },
            indent=4,
            ensure_ascii=False,
        )
    return False


def looks_like_generated_from_ASR(text, language):
    """
    Return True if the text looks like it was obtained from an ASR.
    """
    if language == "ar":
        # TODO?
        return False
    text = format_special_characters(text, format_whitespace=False)
    if text.islower():
        return "Lower case"
    if "," not in text:
        return "No coma"
    return False


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Check if transcriptions are in the right language (in a folder obtained by scrape_youtube.py). Sort them in different folders accordingly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).",
        type=str,
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.",
        type=str,
    )
    parser.add_argument(
        "--model",
        help="An ASR to check that the audio content seems to be right",
        default=None,
    )
    parser.add_argument("--check_no_hate", help="Classify into hate_speech or not", default=False, action="store_true")
    parser.add_argument("--min_num_words", default=7, type=int, help="Minimum number of words to be retained")
    parser.add_argument(
        "--max_char",
        default=1000,
        type=int,
        help="Maximum number of characters in a transcription to consider for language identification",
    )
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("-v", "--verbose", help="Print more information", action="store_true")
    args = parser.parse_args()

    lang = args.language
    path = args.path
    if not path:
        # YouTubeEn, YouTubeFr, etc.
        path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"

    model = None
    if args.model:
        from ssak.infer.general import load_model

        model = load_model(args.model)  # device?

    do_stt = bool(model)
    do_hate = args.check_no_hate

    csv_folder = os.path.join(path, lang)
    mp3_folder = os.path.join(path, "mp3")
    for folder in [csv_folder, mp3_folder]:
        assert os.path.exists(folder), f"Folder {folder} does not exist."

    mp3_folder_ok_lang = mp3_folder + "_checked_" + lang
    mp3_folder_ok_noasr_stream = mp3_folder_ok_lang + "_noasr_stream"
    mp3_folder_ok_noasr_nostream = mp3_folder_ok_lang + "_noasr_nostream"
    mp3_folder_ok_stt_stream = mp3_folder_ok_noasr_stream + "_stt"
    mp3_folder_ok_stt_nostream = mp3_folder_ok_noasr_nostream + "_stt"
    mp3_folder_ok_stt_deprecated = mp3_folder_ok_lang + "_stt"

    csv_folder_ok_lang = csv_folder + "_checked_" + lang
    csv_folder_ok_noasr_stream = csv_folder + "_checked_" + lang + "_noasr_stream"
    csv_folder_ok_noasr_nostream = csv_folder + "_checked_" + lang + "_noasr_nostream"
    csv_folder_ok_stt_stream = csv_folder_ok_noasr_stream + "_stt"
    csv_folder_ok_stt_nostream = csv_folder_ok_noasr_nostream + "_stt"
    csv_folder_ko_lang = csv_folder + "_discarded_" + lang
    csv_folder_ko_noasr = csv_folder + "_discarded_" + lang + "_noasr"
    csv_folder_ko_stt = csv_folder_ko_noasr + "_stt"

    # From a time where STT test was done after checking language but no ASR looking selection
    csv_folder_ok_stt_deprecated = csv_folder_ok_lang + "_stt"
    csv_folder_ko_stt_deprecated = csv_folder_ko_lang + "_stt"

    for folder in [
        mp3_folder_ok_lang,
        mp3_folder_ok_noasr_stream,
        mp3_folder_ok_noasr_nostream,
        mp3_folder_ok_stt_stream,
        mp3_folder_ok_stt_nostream,
        csv_folder_ok_lang,
        csv_folder_ok_noasr_stream,
        csv_folder_ok_noasr_nostream,
        csv_folder_ok_stt_stream,
        csv_folder_ok_stt_nostream,
        csv_folder_ko_lang,
        csv_folder_ko_noasr,
        csv_folder_ko_stt,
    ]:
        os.makedirs(folder, exist_ok=True)

    def rewrite_formatted(csv_file, do_unupper_case):
        dirname = os.path.dirname(csv_file) + "_formatted"
        os.makedirs(dirname, exist_ok=True)
        rewrite_csv(csv_file, os.path.join(dirname, os.path.basename(csv_file)), do_unupper_case=do_unupper_case)

    def add_folder_suffix(filename, suffix):
        return os.path.join(os.path.dirname(filename) + suffix, os.path.basename(filename))

    for filename in tqdm(os.listdir(csv_folder)):
        csv_file = os.path.join(csv_folder, filename)
        mp3_file = os.path.join(mp3_folder, filename.replace(".csv", ".mp3"))
        output_file_ok_lang = os.path.join(csv_folder_ok_lang, filename)
        output_file_ok_noasr_stream = os.path.join(csv_folder_ok_noasr_stream, filename)
        output_file_ok_noasr_nostream = os.path.join(csv_folder_ok_noasr_nostream, filename)
        output_file_ok_stt_stream = os.path.join(csv_folder_ok_stt_stream, filename)
        output_file_ok_stt_nostream = os.path.join(csv_folder_ok_stt_nostream, filename)
        output_file_ko_lang = os.path.join(csv_folder_ko_lang, filename)
        output_file_ko_noasr = os.path.join(csv_folder_ko_noasr, filename)
        output_file_ko_stt = os.path.join(csv_folder_ko_stt, filename)

        output_file_ok_stt_deprecated = os.path.join(csv_folder_ok_stt_deprecated, filename)
        output_file_ko_stt_deprecated = os.path.join(csv_folder_ko_stt_deprecated, filename)

        # Skip if done
        if do_stt:
            can_be_files = [
                output_file_ok_stt_stream,
                output_file_ok_stt_nostream,
                output_file_ko_stt,
                output_file_ko_noasr,
                output_file_ko_lang,
            ]
        else:
            can_be_files = [
                output_file_ok_noasr_stream,
                output_file_ok_noasr_nostream,
                output_file_ko_noasr,
                output_file_ko_lang,
            ]
        if do_hate:
            for i in (
                1,
                0,
            ):
                can_be_files.append(add_folder_suffix(can_be_files[i], "_hate"))
                can_be_files[i] = add_folder_suffix(can_be_files[i], "_nohate")

        if max([os.path.isfile(f) for f in can_be_files]):
            if args.verbose > 1:
                dirname = os.path.dirname([f for f in can_be_files if os.path.isfile(f)][0])
                print(f"Already processed: {filename} -> {dirname}")
            continue

        # Skip if audio is missing (may arrive later...)
        if not os.path.exists(mp3_file):
            continue

        text = load_csv_text(csv_file)
        text_one_line = collapse_whitespace(text)
        do_unupper_case = text_one_line.isupper()

        discarded = False

        if os.path.isfile(output_file_ko_lang) or os.path.isfile(output_file_ok_lang):
            discarded = not os.path.isfile(output_file_ok_lang)

        else:
            num_chars = len(text_one_line)
            num_words = len(text_one_line.split())

            if os.path.isfile(output_file_ko_lang) or os.path.isfile(output_file_ok_lang):
                dicarded = not os.path.isfile(output_file_ok_lang)

            # Discard too short text
            elif args.min_num_words and num_words < args.min_num_words:
                discarded = f"Text too short: {text_one_line}"

            # Discard text in paranthesis
            elif text_one_line.startswith("(") and text_one_line.endswith(")") and len(text_one_line) < 200:
                discarded = f"Text between parenthesis: {text_one_line}"

            else:
                # Take the start
                text_start = text_one_line
                if args.max_char and len(text_start) > args.max_char:
                    text_start = text_start[: args.max_char + 1]
                    while text_start[-1] not in " .,;:!?":
                        text_start = text_start[:-1]
                    assert len(text_start)

                # Check language on the start
                meta = check_language(custom_clean_text(text_start, do_remove_parenthesis=False), lang, return_meta=True)
                is_lang = meta["result"]
                detected_language = meta["best"]

                # Discard audio in other language
                if is_lang:
                    if args.verbose and detected_language != lang:
                        print(f">> {filename} -- Borderline detected -- {lang} or {detected_language}? ({text_one_line[:100]})")
                else:
                    discarded = f"Other language detected: {detected_language} -- ({text_one_line[:100]})"

            if discarded:
                if args.verbose:
                    print(f">> {filename} -- {discarded}")
                with open(output_file_ko_lang, "w") as f:
                    f.write(f"{discarded}\n")
            else:
                if args.verbose and do_unupper_case:
                    print(f">> {filename} -- Upper text detected ({text_one_line[:100]})")

                mp3_file_ok = os.path.join(mp3_folder_ok_lang, os.path.basename(mp3_file))
                if not os.path.exists(mp3_file_ok):
                    os.symlink(os.path.relpath(mp3_file, mp3_folder_ok_lang), mp3_file_ok)
                shutil.copy2(csv_file, output_file_ok_lang)
                # rewrite_formatted(output_file_ok_lang, do_unupper_case)

        has_stream = False

        if not discarded:
            mp3_folder_ok_noasr = mp3_folder_ok_noasr_stream if has_stream else mp3_folder_ok_noasr_nostream
            output_file_ok_noasr = output_file_ok_noasr_stream if has_stream else output_file_ok_noasr_nostream
            output_file_ok = output_file_ok_noasr
            mp3_file_ok = os.path.join(mp3_folder_ok_noasr, os.path.basename(mp3_file))

            if not os.path.exists(output_file_ok_noasr) and not os.path.exists(output_file_ko_noasr):
                discarded = looks_like_generated_from_ASR(text, lang)

                if discarded:
                    if args.verbose:
                        print(f">> {filename} -- {discarded}")
                    shutil.copy2(csv_file, output_file_ko_noasr)
                else:
                    # Check for "streaming-like" subtitles, usually generated from ASR
                    times = load_csv_times(csv_file)
                    num_overlaps = 0
                    previous_end = 0
                    for start, end in times:
                        if start < previous_end - 0.5:
                            num_overlaps += 1
                        previous_end = end
                    has_stream = num_overlaps and num_overlaps > (len(times) // 2)

                    if not os.path.exists(mp3_file_ok):
                        os.symlink(os.path.relpath(mp3_file, mp3_folder_ok_noasr), mp3_file_ok)
                    shutil.copy2(csv_file, output_file_ok_noasr)
                    if not do_stt:
                        rewrite_formatted(output_file_ok_noasr, do_unupper_case)

        if do_stt and not discarded:
            mp3_folder_ok_stt = mp3_folder_ok_stt_stream if has_stream else mp3_folder_ok_stt_nostream
            output_file_ok_stt = output_file_ok_stt_stream if has_stream else output_file_ok_stt_nostream
            output_file_ok = output_file_ok_stt
            mp3_file_ok = os.path.join(mp3_folder_ok_stt, os.path.basename(mp3_file))

            if not os.path.exists(output_file_ok_stt) and not os.path.exists(output_file_ko_stt):
                verbose = args.verbose
                if os.path.exists(output_file_ok_stt_deprecated) or os.path.exists(output_file_ko_stt_deprecated):
                    discarded = not os.path.exists(output_file_ok_stt_deprecated)
                    if discarded:
                        assert os.path.exists(output_file_ko_stt_deprecated)
                        with open(output_file_ko_stt_deprecated) as f:
                            discarded = f.read().rstrip("\n")
                    verbose = False
                else:
                    res = transcription_dont_match(csv_file, mp3_file, model, language=lang)
                    if res:
                        discarded = f"Audio does not match transcription: {res}"

                if discarded:
                    if verbose:
                        print(f">> {filename} -- {discarded}")
                    with open(output_file_ko_stt, "w") as f:
                        f.write(f"{discarded}\n")
                else:
                    if not os.path.exists(mp3_file_ok):
                        os.symlink(os.path.relpath(mp3_file, mp3_folder_ok_stt), mp3_file_ok)
                    shutil.copy2(csv_file, output_file_ok_stt)
                    rewrite_formatted(output_file_ok_stt, do_unupper_case)

        if do_hate and not discarded:
            hate_score = is_hate_speech(text, lang=lang, return_type="score")
            is_hate = hate_score > 0.5
            if is_hate:
                print("Hate score", os.path.basename(csv_file), ":", hate_score)

            # Link mp3
            dirname = os.path.dirname(mp3_file_ok) + ("_hate" if is_hate else "_nohate")
            os.makedirs(dirname, exist_ok=True)
            output_file = os.path.join(dirname, os.path.basename(mp3_file_ok))
            if not os.path.exists(output_file):
                os.symlink(os.path.relpath(mp3_file, dirname), output_file)

            # Copy transcription
            dirname = os.path.dirname(output_file_ok) + ("_hate" if is_hate else "_nohate")
            os.makedirs(dirname, exist_ok=True)
            output_file = os.path.join(dirname, os.path.basename(output_file_ok))
            shutil.copy2(csv_file, output_file)
            rewrite_formatted(output_file, do_unupper_case)

    # for folder in [
    #     mp3_folder_ok_stt_deprecated,
    #     csv_folder_ok_stt_deprecated,
    #     csv_folder_ko_stt_deprecated,
    # ]:
    #     if os.path.exists(folder):
    #         shutil.rmtree(folder)
