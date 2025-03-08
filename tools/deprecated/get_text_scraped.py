import csv
import os

from tqdm import tqdm


def extract_text_from_transcriptions(transcription_files, output_folder):
    """
    Extract text from a directory of transcription files in CSV format and save as separate text files.
    """
    extension = ".csv"
    os.makedirs(output_folder, exist_ok=True)
    for entry in tqdm(sorted(os.scandir(transcription_files), key=lambda e: e.name)):
        if not entry.is_file() or not entry.name.endswith(extension):
            continue

        clean_file = os.path.join(output_folder, os.path.splitext(entry.name)[0] + ".txt")
        texts = []

        with open(entry.path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                text = row["text"]
                texts.append(text)

        with open(clean_file, "w", encoding="utf-8") as clean_text:
            clean_text.write("\n".join(texts))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from transcription files and save as separate text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", help="Input directory containing transcription files.", type=str)
    parser.add_argument("-o", help="Output directory for text files.", type=str)

    args = parser.parse_args()
    extract_text_from_transcriptions(args.i, args.o)
