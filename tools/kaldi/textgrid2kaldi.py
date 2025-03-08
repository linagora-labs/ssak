import argparse
import csv
import os

from textgrids import TextGrid

from tools.scraping.scrape_youtube_to_kaldi import generate_kaldi_data


def textgrid_to_csv(textgrid_file: str, output_dir: str) -> None:
    """Convert a Praat TextGrid file to a CSV file.

    Args:
        textgrid_file (str): Path to the TextGrid file.
        output_dir (str): Path to the output directory.

    Returns:
        None
    """
    try:
        # Read the TextGrid file using the local TextGrid class
        tg_file_name = os.path.basename(textgrid_file).split(".")[0]
        csv_file = f"{tg_file_name}.csv"
        textgrid = TextGrid()
        textgrid.read(textgrid_file)

        # Extract text, start time, and end time from each interval in the TextGrid file
        data = []
        for _, intervals in textgrid.items():
            for interval in intervals:
                if interval.text.strip():  # Ignore empty text
                    text = interval.text
                    start = interval.xmin
                    end = interval.xmax
                    duration = end - start
                    if len(text) > 1:
                        data.append([text, start, duration])
            # break # Needed for OFROM

        # Write the extracted data to a CSV file
        csv_path = os.path.join(output_dir, csv_file)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")  # Use semicolon as delimiter
            writer.writerow(["text", "start", "duration"])
            writer.writerows(data)
    except Exception as e:
        print(f"Error processing {textgrid_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("textgrid_path", help="Path to folder containing the TextGrid files", type=str)
    parser.add_argument("audios", help="Path to folder containing the audio files", type=str)
    parser.add_argument("output", help="Path to Kaldi data folder", type=str)
    parser.add_argument("--extension", help="File extension of audio files (e.g., mp3, wav, ogg)", type=str, default="mp3")
    parser.add_argument(
        "--audio_suffix",
        help="Suffix for audio files if different from their corresponding transcription files",
        type=str,
    )
    parser.add_argument("--utt_prefix", default="", type=str)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)  # Ensure output directory exists

    csv_path = os.path.join(os.path.dirname(args.output), "CSV_folder")
    os.makedirs(csv_path, exist_ok=True)  # Create CSV folder inside the output directory

    # Convert TextGrid files to CSV
    for root, _, files in os.walk(args.textgrid_path):
        for file in files:
            if file.endswith(".TextGrid"):
                textgrid_to_csv(os.path.join(root, file), csv_path)

    # Generate Kaldi data
    generate_kaldi_data(args.audios, csv_path, args.output, extension=args.extension, audio_suffix=args.audio_suffix, utt_prefix="")
