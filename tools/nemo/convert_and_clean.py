import os
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from clean_manifest_text_fr import clean_text_fr
from convert_kaldi_dataset_to_nemo import convert_dataset, get_dataset_name
from convert_kaldi_datasets_to_nemo import convert_datasets
from concat_segments import concat_segments as f_concat_segments
from find_long_transcriptions import filter_incoherent_segments

from ssak.utils.kaldi_dataset import KaldiDataset
from ssak.utils.nemo_dataset import NemoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def kaldi_to_nemo(input_folder, name, output_folder):
    kaldi_dataset = KaldiDataset(name=name, log_folder=output_folder, row_checking_kwargs=dict(show_warnings=False))
    kaldi_dataset.load(input_folder)
    nemo_dataset = NemoDataset(name=name, log_folder=output_folder)
    nemo_dataset.kaldi_to_nemo(kaldi_dataset)
    return nemo_dataset

def process_dataset(input_kaldi_folder=None, input_nemo_file=None, config=dict(), dataset_name=None, output_folder=None, output_converted_wav_folder=None, casepunc=True, nemo_format="multiturn"):
    if not config:
        config = dict()
    
    output_intermediate = output_folder / "intermediate" / dataset_name
            
    if input_kaldi_folder:
        if not os.path.exists(input_kaldi_folder) or not os.path.isdir(input_kaldi_folder):
            raise FileNotFoundError(f"Non-existing path (or not a directory) {input_kaldi_folder}")
        nemo_dataset = kaldi_to_nemo(input_kaldi_folder, dataset_name, output_intermediate)
    elif input_nemo_file:
        nemo_dataset = NemoDataset(name=dataset_name, log_folder=output_intermediate)
        nemo_dataset.load(input_nemo_file)
    else:
        raise ValueError(f"input_kaldi_folder or input_nemo_file must be provided")
    # TO DO: add filter
    
    if config.get("check_if_in_audio", False):
        logger.info("Check if segments are in audios")
        nemo_dataset.check_if_segments_in_audios()
    if config.get("remove_incoherent_texts", False):
        if isinstance(config["remove_incoherent_texts"], dict):
            logger.info(f"Check for incoherent texts using {config['remove_incoherent_texts']}")
            if "charset" in config["remove_incoherent_texts"] and config["remove_incoherent_texts"]["charset"]:
                logger.info(f"Filtering out segments with wrong charset")
                nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_charset.jsonl")
            if "too_long" in config["remove_incoherent_texts"] and config["remove_incoherent_texts"]["too_long"]:
                logger.info(f"Filtering out segments with too long transcriptions")
                nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_time_long.jsonl", mode="too_long")
            if "too_short" in config["remove_incoherent_texts"] and config["remove_incoherent_texts"]["too_short"]:
                logger.info(f"Filtering out segments with too short transcriptions")
                nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_time_short.jsonl", mode="too_short")
        else:
            logger.info("Check for incoherent texts: wrong charset, transcriptions too long or too short")
            nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_charset.jsonl")
            nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_time_long.jsonl", mode="too_long")
            nemo_dataset = filter_incoherent_segments(nemo_dataset, output_intermediate / "filtered_out_incoherent_segments_time_short.jsonl", mode="too_short")
    if config.get("concat_segments", False):
        logger.info("Concatenating segments")
        nemo_dataset = f_concat_segments(
            nemo_dataset,
            max_duration=30,
            acceptance=1.0,
            acceptance_punc=0.2,
            merge_audios=config.get("concat_audios", False),
            merged_audio_folder = Path(str(output_converted_wav_folder) + "_merged"),
            keep_audio_structure=True,
            num_threads=config.get("num_threads", 8),
        )
    if config.get("check_audio", False) and output_converted_wav_folder:
        logger.info("Checking (and transforming if needed) audio files")
        nemo_dataset.normalize_audios(
            output_converted_wav_folder,
            target_sample_rate=16000,
            target_extension="wav",
            num_workers=config.get("num_threads", 8),
        )
    if config.get("clean_text", True):
        replacements = [("!", "."), (":", ","), (";", ",")] if nemo_format=="asr" else []
        logger.info(f"Cleaning texts with replacements {replacements}")
        nemo_dataset = clean_text_fr(
            nemo_dataset=nemo_dataset,
            keep_punc=casepunc,
            keep_case=casepunc,
            empty_string_policy="ignore",
            wer_format=False,
            replacements=replacements,
        )
    if nemo_format=="multiturn":
        if config.get("set_context", False):
            nemo_dataset.save(output_intermediate / f"manifest_{dataset_name}_no_context.jsonl", type=nemo_format)
            nemo_dataset.set_context_if_none(config["set_context"])
    nemo_dataset.save(output_folder/ f"manifest_{dataset_name}.jsonl", type=nemo_format)
    logger.info(f"Saved {output_folder/ f'manifest_{dataset_name}.jsonl'}")

def process_datasets(input_datasets, output_folder, output_wav_folder=None, nemo_format="multiturn", casepunc=True):
    logger.info(f"Converting datasets from {input_datasets}")
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    if output_wav_folder is not None:
        output_wav_folder = Path(output_wav_folder)
        output_wav_folder.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(input_datasets, desc=f"Converting datasets")
    for input_folder in pbar:
        dataset_name = get_dataset_name(input_folder, remove_casing=False, remove_max_duration=False, remove_split=False)
        pbar.set_description(f"Converting {dataset_name}")
        dataset_output_folder = output_folder / get_dataset_name(input_folder)
        if output_wav_folder:
            dataset_audio_output_folder = output_wav_folder / get_dataset_name(input_folder, remove_max_duration=False)
        else:
            dataset_audio_output_folder = dataset_output_folder / "audios"
        
        output_manifest = dataset_output_folder / f"manifest_{dataset_name}.jsonl"
        if not output_manifest.exists():
            process_dataset(
                input_kaldi_folder=input_folder, 
                config=input_datasets[input_folder], 
                dataset_name=dataset_name, 
                output_folder=dataset_output_folder, 
                output_converted_wav_folder=dataset_audio_output_folder,
                casepunc=casepunc,
                nemo_format=nemo_format
            )
        else:
            logger.info(f"Skipping {input_folder} as it already exists ({output_manifest})")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a list of Kaldi datasets to Nemo format")
    parser.add_argument("datasets", help="Input datasets", type=str, nargs="+")
    parser.add_argument("--output", help="Output file", type=str, default="/data-server/datasets/audio/nemo/multi-turn/asr/fr/nocontext")
    # parser.add_argument("--output_wav_dir", type=str, default=None)
    # parser.add_argument("--check_audio", action="store_true", default=False)
    parser.add_argument("--input_data_path", default="/data-server/datasets/audio/kaldi/fr")
    parser.add_argument("--patterns", type=str, nargs="+", default=["casepunc", "recasepunc"])
    parser.add_argument("--nemo_format", type=str, default="multiturn")
    parser.add_argument("--nocasepunc", action="store_true", default=False)
    args = parser.parse_args()
    input_files = args.datasets
    if len(input_files) == 1 and os.path.isfile(input_files[0]):
        logger.warning("One input file, considering it as containing a list of files")
        with open(input_files[0]) as f:
            input_files = json.load(f)
    else:
        input_files = {input_file: None for input_file in input_files}
    new_input_files = dict()
    for input_folder in input_files:
        if input_files[input_folder] is not None and "kaldi_subpath" in input_files[input_folder]:
            new_path = os.path.join(args.input_data_path, input_files[input_folder]["kaldi_subpath"])
        else:
            new_path = os.path.join(args.input_data_path, input_folder)
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Input folder {new_path} does not exist")
        elif not os.path.exists(os.path.join(new_path, "wav.scp")):
            for pattern in args.patterns:
                if os.path.exists(os.path.join(new_path, pattern, "wav.scp")):
                    new_input_files[os.path.join(new_path, pattern)] = input_files[input_folder]
                    break
                elif os.path.exists(os.path.join(new_path, pattern)):
                    dirs = os.listdir(os.path.join(new_path, pattern))
                    for dir in dirs:
                        new_input_files[os.path.join(new_path, pattern, dir)] = input_files[input_folder]
                    break
            else:
                logger.warning(f"Input folder {new_path} does not contain a wav.scp file")
    process_datasets(
        new_input_files, 
        output_folder=args.output, 
        nemo_format=args.nemo_format,
        casepunc=not args.nocasepunc    
    )