import argparse
import logging
import os
import shutil

from ssak.utils.kaldi_converter import *
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LesVocaux dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/LesVocaux/lesvocaux/2")
    parser.add_argument("--output", type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/LesVocaux")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw = os.path.join(output_path, "raw/")

    nocasepunc = os.path.join(output_path, "nocasepunc/")

    raw_cleaned = os.path.join(output_path, "raw_cleaned/")

    function_to_copy_metadata = shutil.copyfile
    # function_to_copy_metadata = shutil.copy2

    if os.path.exists(nocasepunc) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc):
        shutil.rmtree(nocasepunc)

    transcripts = TextFolder2Kaldi("txt", execute_order=0, extracted_id="id", supported_extensions=[".txt"])
    audios = AudioFolder2Kaldi("audio", execute_order=1, extracted_id="id", audio_extensions=[".wav"])
    durations = Row2Duration(execute_order=2)
    speakers = Row2Info(input="id", return_columns=["speaker"], execute_order=3, separator="_", info_position=0)
    genders = ColumnFile2Kaldi(
        input="Speaker_metadata_fixed.csv",
        return_columns=["speaker", "gender"],
        merge_on="speaker",
        separator=",",
        header=False,
        execute_order=4,
    )
    dev_reader = Reader2Kaldi(input_dataset, processors=[transcripts, audios, durations, speakers, genders])
    dataset = dev_reader.load(check_if_segments_in_audio=True)
    dataset.save(raw, True)
    function_to_copy_metadata(os.path.join(input_dataset, "Speaker_metadata_fixed.csv"), os.path.join(raw, "metadata.csv"))

    for i in dataset:
        i.text = i.text.replace("<nib>", "[beep]")
        i.text = i.text.replace("<usb>", "[beep]")
        i.text = i.text.replace("…", "...")

    dataset.save(raw_cleaned, True)
    function_to_copy_metadata(os.path.join(raw, "metadata.csv"), os.path.join(raw_cleaned, "metadata.csv"))

    clean_text_fr(raw_cleaned, nocasepunc, ignore_first=1, file_clean_mode="kaldi")
    function_to_copy_metadata(os.path.join(raw_cleaned, "metadata.csv"), os.path.join(nocasepunc, "metadata.csv"))
