import argparse
import logging
import os
import shutil

from ssak.utils.kaldi_converter import AudioFolder2Kaldi, CsvFile2Kaldi, ListFile2Kaldi, Reader2Kaldi, Row2Info
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TEDX dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/TEDX/fr-fr/data")
    parser.add_argument("--output", type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/TEDX_fr")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw_dev = os.path.join(output_path, "casepunc/dev")
    raw_test = os.path.join(output_path, "casepunc/test")
    raw_train = os.path.join(output_path, "casepunc/train")

    nocasepunc_dev = os.path.join(output_path, "nocasepunc/dev")
    nocasepunc_test = os.path.join(output_path, "nocasepunc/test")
    nocasepunc_train = os.path.join(output_path, "nocasepunc/train")

    if os.path.exists(nocasepunc_dev) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc_dev):
        shutil.rmtree(nocasepunc_dev)
        shutil.rmtree(nocasepunc_test, ignore_errors=True)
        shutil.rmtree(nocasepunc_train, ignore_errors=True)

    segments = CsvFile2Kaldi(
        "txt/segments",
        ["id", "audio_id", "start", "end"],
        separator=" ",
        execute_order=0,
        merge_on="id",
        sort_merging=False,
    )
    texts = ListFile2Kaldi("txt/valid.fr", ["text"], separator=None, execute_order=1)
    audios = AudioFolder2Kaldi("wav", execute_order=2, extracted_id="audio_id", audio_extensions=".flac")
    speakers = Row2Info("audio_id", ["speaker"], execute_order=3, separator=None, info_position=None)

    dev_reader = Reader2Kaldi(os.path.join(input_dataset, "valid"), processors=[segments, audios, texts, speakers])
    dataset = dev_reader.load()
    dataset.save(raw_dev, True)

    clean_text_fr(raw_dev, nocasepunc_dev, ignore_first=1, file_clean_mode="kaldi")

    segments = CsvFile2Kaldi(
        "txt/segments",
        ["id", "audio_id", "start", "end"],
        separator=" ",
        execute_order=0,
        merge_on="id",
        sort_merging=False,
    )
    texts = ListFile2Kaldi("txt/test.fr", ["text"], separator=None, execute_order=1)
    audios = AudioFolder2Kaldi("wav", execute_order=2, extracted_id="audio_id", audio_extensions=".flac")
    speakers = Row2Info("audio_id", ["speaker"], execute_order=3, separator=None, info_position=None)

    dev_reader = Reader2Kaldi(os.path.join(input_dataset, "test"), processors=[segments, audios, texts, speakers])
    dataset = dev_reader.load()
    dataset.save(raw_test, True)

    clean_text_fr(raw_test, nocasepunc_test, ignore_first=1, file_clean_mode="kaldi")

    segments = CsvFile2Kaldi(
        "txt/segments",
        ["id", "audio_id", "start", "end"],
        separator=" ",
        execute_order=0,
        merge_on="id",
        sort_merging=False,
    )
    texts = ListFile2Kaldi("txt/train.fr", ["text"], separator=None, execute_order=1)
    audios = AudioFolder2Kaldi("wav", execute_order=2, extracted_id="audio_id", audio_extensions=".flac")
    speakers = Row2Info("audio_id", ["speaker"], execute_order=3, separator=None, info_position=None)

    dev_reader = Reader2Kaldi(os.path.join(input_dataset, "train"), processors=[segments, audios, texts, speakers])
    dataset = dev_reader.load()
    dataset.save(raw_train, True)

    clean_text_fr(raw_train, nocasepunc_train, ignore_first=1, file_clean_mode="kaldi")
