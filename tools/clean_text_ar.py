#!/usr/bin/env python3

from ssak.utils.text import format_text_ar

if __name__ == "__main__":
    import argparse
    import os
    import sys

    from tqdm import tqdm

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
    parser.add_argument("--keep_punc", help="Whether to keep punctuations", default=False, action="store_true")
    parser.add_argument(
        "--keep_latin_chars",
        help="Whether to keep latin characters (otherwise, only arabic characters)",
        default=False,
        action="store_true",
    )
    parser.add_argument("--bw", help="Whether to transliterate text into buckwalter encoding.", default=False, action="store_true")
    parser.add_argument(
        "--ignore_first",
        default=0,
        type=int,
        help="Ignore the first N words (can be set to 1 to ignore the first word that can be an ID)",
    )
    parser.add_argument("--language", default="ar", type=str, help="Whether to use 'ar or ar_tn'")
    parser.add_argument("--normalize_dialect_words", help="Whether to Normalize language words", default=False, action="store_true")
    args = parser.parse_args()

    input_file = args.input

    if args.language == "tn":
        args.language = "ar_tn"

    if args.output:
        output_file = args.output
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
    if os.path.isfile(input_file):
        num_lines = sum(1 for _ in open(input_file))
        gen = open(input_file, encoding="utf-8")
    else:
        num_lines = 1
        gen = [input_file]

    try:
        formatted_lines = []
        for line in tqdm(gen, total=num_lines):
            if args.ignore_first:
                words = line.split()
                assert len(words) >= args.ignore_first, f"Line {line} has less than {args.ignore_first} words"
                line = " ".join(words[args.ignore_first :])

            line = format_text_ar(
                line,
                keep_punc=args.keep_punc,
                keep_latin_chars=args.keep_latin_chars,
                bw=args.bw,
                lang=args.language,
                normalize_dialect_words=args.normalize_dialect_words,
            )
            for subline in line.splitlines():
                subline = subline.strip()
                if subline:
                    if args.ignore_first:
                        subline = " ".join(words[: args.ignore_first]) + " " + subline
                    fout.write(subline + "\n")
                    fout.flush()
    finally:
        if fout is not sys.stdout and fout is not None:
            fout.close()
