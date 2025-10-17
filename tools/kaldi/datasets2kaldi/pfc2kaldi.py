import argparse
import logging
import os
import shutil

from ssak.utils.kaldi_converter import *
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PFC dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/PFC/2")
    parser.add_argument("--output", type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/PFC")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw = os.path.join(output_path, "raw")

    nocasepunc = os.path.join(output_path, "nocasepunc")

    if os.path.exists(nocasepunc) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc):
        shutil.rmtree(nocasepunc)

    transcripts = TextGrid2Kaldi("", ["text", "start", "duration"], execute_order=0, subfolders=True, extract_items=[0])
    audios = AudioFolder2Kaldi("", execute_order=1, extracted_id="audio_id", audio_extensions=[".mp3"])
    dev_reader = Reader2Kaldi(input_dataset, processors=[transcripts, audios])
    dataset = dev_reader.load(check_if_segments_in_audio=True)
    dataset.save(raw, True)

    clean_text_fr(raw, nocasepunc, ignore_first=1, file_clean_mode="kaldi")
