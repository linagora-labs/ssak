#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm
from ssak.utils.nemo_dataset import NemoDataset
from ssak.utils.text_latin import format_text_latin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text_fr(
    nemo_dataset,
    keep_punc=True,
    keep_num=False,
    keep_case=True,
    empty_string_policy="fail",
    linebreak_policy="fail",
    remove_suspicious_entry=False,
    extract_parenthesis=False,
    file_acronyms=None,
    file_special_char=None,
    wer_format=True,
    replacements=None,
):
    fid_acronyms = open(file_acronyms, "a", encoding="utf-8") if file_acronyms else None
    fid_special_char = open(file_special_char, "a", encoding="utf-8") if file_special_char else None

    new_data = NemoDataset()
    for row in tqdm(nemo_dataset, desc=f"Cleaning text {nemo_dataset}"):
        full_line = row
        row.text = format_text_latin(
            row.text,
            lower_case=not keep_case,
            keep_punc=keep_punc,
            convert_numbers=not keep_num,
            extract_parenthesis=extract_parenthesis,
            fid_acronyms=fid_acronyms,
            fid_special_chars=fid_special_char,
            remove_suspicious_entry=remove_suspicious_entry,
            wer_format=wer_format,
            replacements=replacements,
        )

        if len(row.text) > 0 and row.text[-1] == '"' and row.text[0] == '"':
            row.text = row.text[1:-1]
        num_dumps = 0
        if row.text or empty_string_policy == "allow":
            new_data.append(row)
            num_dumps += 1
        if not num_dumps and empty_string_policy != "ignore":
            raise RuntimeError(f"Empty string found (on '{full_line}').\nUse option --empty_string_policy=allow or --empty_string_policy=ignore to explicitly allow or ignore empty strings")
        if num_dumps > 1 and linebreak_policy == "fail":
            line_ = row.text.replace("\n", "\\n")
            raise RuntimeError(f"Line break found when normalizing '{full_line}' (into '{line_}').\nUse option --linebreak_policy=allow to explicitly allow line breaks")
    return new_data

def clean_text_fr_file(
    input,
    output_file,
    keep_punc=True,
    keep_num=False,
    keep_case=True,
    empty_string_policy="fail",
    linebreak_policy="fail",
    remove_suspicious_entry=False,
    extract_parenthesis=False,
    file_acronyms=None,
    file_special_char=None,
    wer_format=True,
    replacements=None,
):
    """
    Clean the text of a manifest file for French language (remove special characters, numbers, etc.)
    Args:
        input (str): input manifest file
        output_file (str): output manifest file
        keep_punc (bool): keep punctuations
        keep_num (bool): keep numbers and symbols
        keep_case (bool): keep case (otherwise, everything will be lowercased)
        empty_string_policy (str): what to do with empty strings
        linebreak_policy (str): what to do when a line break is introduced
        remove_suspicious_entry (bool): to ignore entries that are probably written in bad French
        extract_parenthesis (bool): to pull out parenthesis and process them separately (as new lines)
        file_acronyms (str): a file to list acronyms found
        file_special_char (str): a file to list special characters that were removed
    """
    if output_file:
        if os.path.exists(output_file):
            raise FileExistsError(f"Output file {output_file} already exists")
        dname = os.path.dirname(output_file)
        os.makedirs(dname, exist_ok=True)
    
    if os.path.isfile(input):
        gen = NemoDataset()
        gen.load(input)
    else:
        print(f"WARNING: File {input} not found. Interpreting that as an input")
        gen = NemoDataset()
        gen.append({"text": input})

    fid_acronyms = open(file_acronyms, "a", encoding="utf-8") if file_acronyms else None
    fid_special_char = open(file_special_char, "a", encoding="utf-8") if file_special_char else None

    new_data = clean_text_fr(
        gen, 
        keep_punc, 
        keep_num, 
        keep_case, 
        fid_acronyms, 
        fid_special_char)

    if output_file:
        new_data.save(output_file)
    else:
        for i in new_data:
            print(i)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean input text (in order to train a language model). Can remove punctuation, out of vocabulary words, numbers, and uppercases the text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input manifest file", type=str)
    parser.add_argument(
        "output",
        help="Output file (if not specified, the text will be outputed on stdout)",
        type=str,
        nargs="?",
        default=None,
    )
    parser.add_argument("--keep_punc", help="Keep punctuations", default=True, action="store_false")
    parser.add_argument("--keep_num", help="Keep numbers and symbols", default=False, action="store_true")
    parser.add_argument("--keep_case", help="Keep case (otherwise, everything will be lowercased)", default=True, action="store_false")
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
    parser.add_argument("--file_acronyms", help="A file to list acronyms found", default=None, type=str)
    parser.add_argument("--file_special_char", help="A file to list special characters that were removed", default=None, type=str)
    args = parser.parse_args()

    clean_text_fr(
        input=args.input,
        output=args.output,
        keep_punc=args.keep_punc,
        keep_num=args.keep_num,
        keep_case=args.keep_case,
        empty_string_policy=args.empty_string_policy,
        linebreak_policy=args.linebreak_policy,
        remove_suspicious_entry=args.remove_suspicious_entry,
        extract_parenthesis=args.extract_parenthesis,
        file_acronyms=args.file_acronyms,
        file_special_char=args.file_special_char,
    )
