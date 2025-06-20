import argparse
import logging
import os
import shutil

from ssak.utils.kaldi_converter import *
from tools.clean_text_fr import clean_text_fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert yodas dataset to Kaldi format")
    parser.add_argument("--force", action="store_true", default=True)
    parser.add_argument("--input", type=str, default="/data-server/datasets/audio/transcript/fr/YODAS/fr000")
    parser.add_argument("--output", type=str, default="/data-server/datasets/audio/kaldi/fr/YODAS/fr000")
    args = parser.parse_args()

    input_dataset = args.input

    output_path = args.output

    raw = os.path.join(output_path, "raw")

    nocasepunc = os.path.join(output_path, "nocasepunc")

    if os.path.exists(nocasepunc) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc):
        shutil.rmtree(nocasepunc)

    audios = AudioFolder2Kaldi("audio", execute_order=3, extracted_id="id", audio_extensions=[".wav"], sort_merging="only_new")
    file_reader = TextFile2Kaldi("", return_columns=["id", "text"], execute_order=0, separator=" ")
    texts = ColumnFileFolder2Kaldi(
        "text",
        execute_order=1,
        columnfile2kaldi=file_reader,
        extracted_id="id",
        extracted_info="text",
        files_extensions=[".txt"],
    )
    file_reader = TextFile2Kaldi("", return_columns=["id", "duration"], execute_order=0, separator=" ")
    durations = ColumnFileFolder2Kaldi(
        "duration",
        execute_order=2,
        columnfile2kaldi=file_reader,
        extracted_id="id",
        extracted_info="text",
        files_extensions=[".txt"],
    )
    audio_ids = Row2Info("id", ["audio_id"], 3, None, None)
    spk_ids = Row2Info("id", ["speaker"], 4, None, None)
    dev_reader = Reader2Kaldi(input_dataset, processors=[texts, durations, audios, audio_ids, spk_ids])
    dataset = dev_reader.load(debug=False, accept_missing_speaker=False)

    def filter(row):
        if row.id.startswith("E--pPwqi_50-"):
            return True
        elif "#" in row.text:
            return True
        return False

    removed_lines = dataset.apply_filter(filter)

    dataset.normalize_audios(os.path.join(input_dataset, "converted"), target_extension="wav", num_workers=16)

    logger.info(f"Dataset duration: {dataset.get_duration('sum')/3600:.2f}h")
    dataset.save(raw, False)

    clean_text_fr(raw, nocasepunc, ignore_first=1, empty_string_policy="ignore", file_clean_mode="kaldi")
