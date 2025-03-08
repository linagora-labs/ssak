#!/usr/bin/env python3

import os
import shutil
import sys

from tqdm import tqdm

from ssak.utils.kaldi import check_kaldi_dir
from ssak.utils.text_latin import format_text_latin


def clean_text_fr(
    input,
    output,
    file_clean_mode="file",
    keep_punc=False,
    keep_num=False,
    keep_case=False,
    empty_string_policy="fail",
    linebreak_policy="fail",
    remove_suspicious_entry=False,
    extract_parenthesis=False,
    ignore_first=0,
    file_acronyms=None,
    file_special_char=None,
):
    if file_clean_mode == "kaldi":
        ignore_first = 1
        if not os.path.isdir(input):
            if os.path.isfile(input):
                raise FileNotFoundError(f"Input folder {input} is a file, not a folder")
            else:
                raise FileNotFoundError(f"Input folder {input} not found")
        if not os.path.exists(os.path.join(input, "text")):
            raise FileNotFoundError(f"Input folder {input} does not contain a 'text' file")
        if os.path.exists(output):
            raise RuntimeError(f"Output folder {output} already exists")
        os.makedirs(output)
        fout = open(os.path.join(output, "text"), "w", encoding="utf-8")
        num_lines = sum(1 for _ in open(os.path.join(input, "text")))
        gen = open(os.path.join(input, "text"), encoding="utf-8")
        raw_file = open(os.path.join(output, "text_raw"), "w", encoding="utf-8")
        for fn in "utt2spk", "utt2dur", "segments", "wav.scp", "spk2utt":
            if os.path.exists(os.path.join(input, fn)):
                # shutil.copyfile(os.path.join(input, fn), os.path.join(output, fn))    # use it when you don't have write permissions (CORPUS_FINAL)
                shutil.copy2(os.path.join(input, fn), os.path.join(output, fn))
    else:
        if output:
            output_file = output
            if os.path.exists(output_file):
                raise RuntimeError(f"Output file {output_file} already exists")
                # os.remove(output_file)

            dname = os.path.dirname(output_file)
            if dname and not os.path.isdir(dname):
                os.makedirs(dname)
            fout = open(output_file, "a", encoding="utf-8")
        else:
            fout = sys.stdout

        # Get the number of lines
        # Note: This is ~10 times slower than wc -l
        #       but it's reasonnable (20 sec for ~70 000 000)
        # see https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
        if os.path.isfile(input):
            num_lines = sum(1 for _ in open(input))
            gen = open(input, encoding="utf-8")
        else:
            print(f"WARNING: File {input} not found. Interpreting that as an input")
            num_lines = 1
            gen = [input]

    fid_acronyms = open(file_acronyms, "a", encoding="utf-8") if file_acronyms else None
    fid_special_char = open(file_special_char, "a", encoding="utf-8") if file_special_char else None

    try:
        for line in tqdm(gen, total=num_lines, desc=f"Cleaning text from {input}"):
            full_line = line
            if ignore_first:
                words = line.split()
                assert len(words) >= ignore_first, f"Line {line} has less than {ignore_first} words"
                line = " ".join(words[ignore_first:])
            line = format_text_latin(
                line,
                lower_case=not keep_case,
                keep_punc=keep_punc,
                convert_numbers=not keep_num,
                extract_parenthesis=extract_parenthesis,
                fid_acronyms=fid_acronyms,
                fid_special_chars=fid_special_char,
                remove_suspicious_entry=remove_suspicious_entry,
            )
            num_dumps = 0
            for subline in line.splitlines():
                subline = subline.strip()
                if subline or empty_string_policy == "allow":
                    if ignore_first:
                        subline = " ".join(words[:ignore_first]) + " " + subline
                    fout.write(subline + "\n")
                    if file_clean_mode == "kaldi":
                        raw_file.write(full_line)
                    fout.flush()
                    num_dumps += 1
            if not num_dumps and empty_string_policy != "ignore":
                raise RuntimeError(f"Empty string found (on '{full_line}').\nUse option --empty_string_policy=allow or --empty_string_policy=ignore to explicitly allow or ignore empty strings")
            if num_dumps > 1 and linebreak_policy == "fail":
                line_ = line.replace("\n", "\\n")
                raise RuntimeError(f"Line break found when normalizing '{full_line}' (into '{line_}').\nUse option --linebreak_policy=allow to explicitly allow line breaks")
    finally:
        if fout is not sys.stdout:
            fout.close()
        if hasattr(gen, "close"):
            gen.close()
        if file_clean_mode == "kaldi":
            raw_file.close()
    if file_clean_mode == "kaldi":
        check_kaldi_dir(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean input text (in order to train a language model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument(
        "output",
        help="Output file (if not specified, the text will be outputed on stdout",
        type=str,
        nargs="?",
        default=None,
    )
    parser.add_argument("--keep_punc", help="Keep punctuations", default=False, action="store_true")
    parser.add_argument("--keep_num", help="Keep numbers and symbols", default=False, action="store_true")
    parser.add_argument("--keep_case", help="Keep case (otherwise, everything will be lowercased)", default=False, action="store_true")
    parser.add_argument(
        "--empty_string_policy",
        choices=["fail", "allow", "ignore"],
        default="fail",
        help="What to do with empty strings",
    )
    parser.add_argument(
        "--linebreak_policy",
        choices=["fail", "allow"],
        default="fail",
        help="What to do when a line break is introduced",
    )
    parser.add_argument(
        "--remove_suspicious_entry",
        help="To ignore entries that are probably written in bad French",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--extract_parenthesis",
        help="To pull out parenthesis and process them separately (as new lines)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore_first",
        default=0,
        type=int,
        help="Ignore the first N words (can be set to 1 to ignore the first word that can be an ID)",
    )
    parser.add_argument("--file_acronyms", help="A file to list acronyms found", default=None, type=str)
    parser.add_argument("--file_special_char", help="A file to list special characters that were removed", default=None, type=str)
    parser.add_argument(
        "--file_clean_mode",
        choices=["file", "kaldi"],
        default="file",
        help="Type of input and output (file or kaldi folder). If kaldi, it sets ignore_first to 1",
    )
    args = parser.parse_args()

    clean_text_fr(
        input=args.input,
        output=args.output,
        file_clean_mode=args.file_clean_mode,
        keep_punc=args.keep_punc,
        keep_num=args.keep_num,
        keep_case=args.keep_case,
        empty_string_policy=args.empty_string_policy,
        linebreak_policy=args.linebreak_policy,
        remove_suspicious_entry=args.remove_suspicious_entry,
        extract_parenthesis=args.extract_parenthesis,
        ignore_first=args.ignore_first,
        file_acronyms=args.file_acronyms,
        file_special_char=args.file_special_char,
    )
