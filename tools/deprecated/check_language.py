import argparse
from pathlib import Path

import langid


def is_language(text, lang_code):
    lang, score = langid.classify(text)
    return lang == lang_code


def test_language(path, lang_code):
    path = Path(path)
    if not path.exists():
        print("Invalid path:", path)
        return

    non_lang_files = []
    if path.is_file():
        with path.open("r") as f:
            text = f.read()
            if is_language(text, lang_code):
                print(f"The file {path} contains {lang_code} text")
            else:
                print(f"The file {path} does not contain {lang_code} text")
    elif path.is_dir():
        for file_path in path.iterdir():
            if file_path.is_file():
                with file_path.open("r") as f:
                    text = f.read()
                    if is_language(text, lang_code):
                        print(f"The file {file_path} contains {lang_code} text")
                        print()
                    else:
                        print(f"The file {file_path} does not contain {lang_code} text")
                        non_lang_files.append(file_path)

    # to delete the non-lang files, uncomment the code below
    # for file_path in non_lang_files:
    #     file_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test if a file or folder contains a given language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", help="Path to the input file or folder", type=str, required=True)
    parser.add_argument("-l", help="Language code to detect (e.g. ar for Arabic, en for English)", type=str, required=True)

    args = parser.parse_args()
    test_language(args.i, args.l)
