import argparse
import logging
import os
import shutil
from functools import partial

from clean_manifest_text_fr import clean_text_fr
from convert_kaldi_datasets_to_nemo import convert_datasets
from generate_dataset_list_files import generate_dataset_list_files
from merge_manifest import merge_manifests

CHECK_AUDIO = True
CHECK_IF_SEGMENT_IN_AUDIO = False
REMOVE_INCOHERENT_TEXTS = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Nemo: generate dataset list files, convert datasets from kaldi format, merge them, clean them and if needed create a tarred version and a tokenizer.")
    parser.add_argument("--train_input_datasets", help="Input datasets", type=str, default=None)
    parser.add_argument("--test_input_datasets", help="Input datasets", type=str, default=None)
    parser.add_argument("--dev_input_datasets", help="Input datasets", type=str, default=None)
    parser.add_argument("--datasets_folder", help="Dataset folder", type=str, default=None)
    parser.add_argument("--output_wav_dir", help="Place to store converted audios (if audios are not in wav for example)", type=str, default="processed_dataset")
    parser.add_argument("--manifest_dir", default="input_manifests", help="Place to store/load manifests")
    parser.add_argument("--subset_pattern", nargs="+", default="nocasepunc_max30", help="Subset folders to search for in datasets", type=str)
    parser.add_argument("--nocasepunc", default=False, action="store_true", help="Remove casing and punctuations when cleaning")
    # Options for creating a tokenizer using all splits
    parser.add_argument("--create_tokenizer", default=None, help="Folder to save tokenizer (if not set, no tokenizer is created)")
    parser.add_argument("--vocab_size", help="Vocab size of the tokenizer", type=int, default=1024)
    parser.add_argument("--spe_type", help="Type of tokenizer", choices=["unigram", "bpe", "char", "word"], type=str, default="bpe")
    # Options for making buckets for the training data
    parser.add_argument("--create_tarred", default=None, help="Folder to save tarred dataset (if not set, no tarred dataset is created)")
    parser.add_argument("--num_shards", default=24, type=int)
    parser.add_argument("--num_buckets", default=12, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--max_duration", default=30.1, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    vocab_size = args.vocab_size
    input_datasets = args.train_input_datasets
    output_wav_dir = args.output_wav_dir
    tmp_manifest_dir = args.manifest_dir
    casepunc = not args.nocasepunc

    logger.info(f"Train input is set to {input_datasets}")
    logger.info(f"Test input is set to {args.test_input_datasets}")
    logger.info(f"Dev input is set to {args.dev_input_datasets}")

    logger.info(f"Output_wav_dir is set to {output_wav_dir}")
    logger.info(f"Manifest_dir is set to {tmp_manifest_dir}")

    if args.create_tokenizer:
        path_to_tokenizer = "tokenizer" if args.create_tokenizer is True else args.create_tokenizer
        logger.info(f"Create_tokenizer is set to {args.create_tokenizer}")
        logger.info(f"\tVocab_size is set to {vocab_size}")
    else:
        path_to_tokenizer = None
        logger.info("No tokenizer will be created")

    if args.create_tarred:
        output_tarred_dir = "tarred_dataset" if args.create_tarred is True else args.create_tarred
        logger.info(f"Output_tarred_dir is set to {output_tarred_dir}")
        logger.info(f"\tNum_shards is set to {args.num_shards}")
        logger.info(f"\tNum_buckets is set to {args.num_buckets}")
        logger.info(f"\tNum_workers is set to {args.num_workers}")
        logger.info(f"\tMax_duration is set to {args.max_duration}")
    else:
        output_tarred_dir = None
        logger.info("No tarred dataset will be created")

    datasets_folder = args.datasets_folder
    if datasets_folder is None:
        datasets_folder = ""

    splits_to_process = []
    os.makedirs(os.path.join(tmp_manifest_dir, "datasets_list"), exist_ok=True)
    if args.train_input_datasets:
        splits_to_process.append("train")
        generate_dataset_list_files(
            input_datasets,
            datasets_folder,
            dest=os.path.join(tmp_manifest_dir, "datasets_list", "train_datasets"),
            mode="train",
            subset_patterns=args.subset_pattern,
        )
    if args.test_input_datasets:
        splits_to_process.append("test")
        generate_dataset_list_files(
            args.test_input_datasets,
            datasets_folder,
            dest=os.path.join(tmp_manifest_dir, "datasets_list", "test_datasets"),
            mode="test",
            subset_patterns=args.subset_pattern,
        )
    if args.dev_input_datasets:
        splits_to_process.append("dev")
        generate_dataset_list_files(
            args.dev_input_datasets,
            datasets_folder,
            dest=os.path.join(tmp_manifest_dir, "datasets_list", "dev_datasets"),
            mode="dev",
            subset_patterns=args.subset_pattern,
        )
    if len(splits_to_process) == 0:
        raise ValueError("No splits to process")

    for i in splits_to_process:
        try:
            convert_datasets(
                [os.path.join(tmp_manifest_dir, "datasets_list", f"{i}_datasets")],
                os.path.join(tmp_manifest_dir, f"{i}_manifests"),
                output_wav_dir,
                check_audio=CHECK_AUDIO,
                check_if_in_audio=CHECK_IF_SEGMENT_IN_AUDIO,
                remove_incoherent_texts=REMOVE_INCOHERENT_TEXTS,
            )
        except FileExistsError:
            pass
        try:
            merge_manifests(
                [os.path.join(tmp_manifest_dir, f"{i}_manifests")],
                os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest.jsonl"),
            )
        except FileExistsError:
            logger.info(f"{i} merged manifest already exists")
        clean_path = os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest_clean.jsonl")
        if not os.path.exists(clean_path):
            if os.path.exists(clean_path + ".tmp"):
                os.remove(clean_path + ".tmp")
            clean_text_fr(
                input=os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest.jsonl"), output=clean_path + ".tmp", keep_punc=casepunc, keep_case=casepunc, empty_string_policy="ignore", wer_format=False, replacements=[("!", "."), (":", ","), (";", ",")]
            )
            shutil.move(clean_path + ".tmp", clean_path)
        else:
            logger.info(f"{i} cleaned manifest already exists")
    if len(splits_to_process) > 1:
        try:
            merge_manifests(
                [os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest_clean.jsonl") for i in splits_to_process],
                os.path.join(f"{tmp_manifest_dir}", "all_manifest_clean.jsonl"),
            )
        except FileExistsError:
            logger.info("Merged manifest for tokenizer already exists")
    else:
        import shutil

        if not os.path.exists(os.path.join(f"{tmp_manifest_dir}", "all_manifest_clean.jsonl")):
            logger.info(f"Copying {splits_to_process[0]} manifest to all manifest")
            shutil.copy2(
                os.path.join(f"{tmp_manifest_dir}", f"{splits_to_process[0]}_manifest_clean.jsonl"),
                os.path.join(f"{tmp_manifest_dir}", "all_manifest_clean.jsonl"),
            )
    if args.create_tokenizer:
        from process_asr_text_tokenizer import process_asr_text_tokenizer

        if not os.path.exists(path_to_tokenizer):
            logging.info(f"Creating tokenizer in {path_to_tokenizer}")
            process_asr_text_tokenizer(
                manifests=os.path.join(tmp_manifest_dir, "all_manifest_clean.jsonl"),
                data_root=path_to_tokenizer,
                vocab_size=vocab_size,
                tokenizer="spe",
                spe_type=args.spe_type,
                spe_split_digits=True,
            )
            logging.info("Tokenizer created")
        else:
            logging.info(f"Tokenizer already exists in {path_to_tokenizer}")
    if args.create_tarred:
        from convert_to_tarred_audio_dataset import convert_to_tarred_audio_dataset, hybrid_bucketing_times

        logging.info(f"Creating tarred dataset in {output_tarred_dir}")
        if args.num_buckets == 6:
            custom_hybrid_method = partial(hybrid_bucketing_times, threshold=10, num_buckets_linear=4, num_buckets_log=2)
        else:
            custom_hybrid_method = "linear"
        convert_to_tarred_audio_dataset(
            manifest_path=os.path.join(tmp_manifest_dir, "train_manifest_clean.jsonl"),
            target_dir=output_tarred_dir,
            num_shards=args.num_shards,
            max_duration=args.max_duration,
            min_duration=0.1,
            method=custom_hybrid_method,
            workers=args.num_workers,
            buckets_num=args.num_buckets,
            shuffle_seed=42,
            shuffle=True,
            sort_in_shards=False,
        )
