#!/usr/bin/env python3

import datetime
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Optional, Union

import datasets
import numpy as np
import torch
import transformers

from ssak.utils.augment import SpeechAugment
from ssak.utils.dataset import kaldi_folder_to_dataset, process_dataset
from ssak.utils.env import *  # handle option --gpus (and set environment variables at the beginning)
from ssak.utils.misc import hashmd5, remove_commonprefix, save_source_dir
from ssak.utils.monitoring import get_num_gpus, tic, toc, vram_free, vram_usage
from ssak.utils.text import remove_special_words


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: transformers.Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    log_stream: Optional[str] = None
    verbose: bool = False

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        if self.log_stream is not None:
            vram_usage("collater", stream=self.log_stream)

        # split inputs and labels since they have to be of different lenghts and need different padding methods
        if isinstance(features, datasets.Dataset):
            # The second version below works but is quite slow
            features = features.to_pandas()
            input_features = list(map(lambda x: dict([("input_values", x)]), features["input_values"]))
            label_features = list(map(lambda x: dict([("input_ids", x)]), features["labels"]))
            if self.verbose:
                len = list(features["input_values"].apply(lambda x: x.shape))
        else:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            if self.verbose:
                len = [feature["input_values"].shape for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.verbose:
            print(len, "->", batch["input_values"].shape)

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


wer_metric = datasets.load_metric("wer")
format_words_for_wer = lambda x: remove_special_words(x, glue_apostrophe=False)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    pred_str = list(map(format_words_for_wer, pred_str))
    label_str = list(map(format_words_for_wer, label_str))

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    import argparse

    # # Output folders generated in subfolders relative to this script
    # workdir = os.path.realpath(os.path.dirname(__file__))
    # currdir = os.path.realpath(os.curdir)
    # print("Working directory:", currdir, "->", workdir)
    # os.chdir(workdir)

    # # For huggingface caching (datasets.config.HF_DATASETS_CACHE)
    # os.environ["HOME"] = os.path.dirname(os.path.dirname(workdir))

    parser = argparse.ArgumentParser(description="Train wav2vec2 on a given dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("train", help="A kaldi folder, or a file containing a list of kaldi folders, with training data")
    parser.add_argument("valid", help="A kaldi folder, or a file containing a list of kaldi folders, with validation data")
    parser.add_argument("--debug", help="to perform small experiment, check if things are running", default=False, action="store_true")
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("--online", help="load and process audio files on the fly", default=False, action="store_true")
    parser.add_argument("--max_duration", help="maximum signal length", default=15, type=int)
    parser.add_argument("--min_duration", help="minimum signal length", default=1, type=int)
    parser.add_argument(
        "--base_model",
        default="Ilyes/wav2vec2-large-xlsr-53-french",
        type=str,
        help="Model to adapt. \
        ex: samirt8/wav2vec2-xls-r-1b-fr / \
            Ilyes/wav2vec2-large-xlsr-53-french",
    )
    # Hyperparameters
    parser.add_argument("--no_freeze", help="To avoid freezing the feature extractor", default=False, action="store_true")
    parser.add_argument("--data_augment", help="To augment data", default=False, action="store_true")
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--num_epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    parser.add_argument("--attention_dropout", help="attention dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout", help="hidden dropout", default=0.05, type=float)
    parser.add_argument("--feat_proj_dropout", help="feat proj dropout", default=0.0, type=float)
    parser.add_argument("--layer_dropout", help="layer dropout", default=0.1, type=float)
    parser.add_argument("--mask_time_prob", help="mask time prob", default=0.05, type=float)
    parser.add_argument("--disable_first_eval", help="to disable the evaluation of the init model", default=False, action="store_true")
    parser.add_argument("--seed", help="Random seed", default=69, type=int)
    parser.add_argument("--eval_steps", help="Validation and checkpoint model every n steps", default=400, type=int)
    parser.add_argument(
        "--data_augment_noise",
        help="Folder with audio files to simulate noises (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises",
        type=str,
    )
    parser.add_argument(
        "--data_augment_rir",
        help="Folder with audio files to simulate reverberation (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/[simulated_rirs_16k/smallroom/rir_list,simulated_rirs_16k/mediumroom/rir_list,simulated_rirs_16k/largeroom/rir_list]",
        type=str,
    )
    parser.add_argument("--output_dir", help="output parent folder", default=".", type=str)
    args = parser.parse_args()

    if not args.gpus:
        args.gpus = ",".join([str(i) for i in range(get_num_gpus())])

    # GPU with the most of memory first
    gpus = list(reversed(sorted([int(i) for i in args.gpus.split(",") if i and int(i) >= 0], key=vram_free)))
    print("Using gpus:", gpus)

    args.online = args.online or args.data_augment
    online_dev = False
    use_gpu = len(gpus) > 0
    USE_MIXED_PRECISION = False  # use_gpu
    USE_MIXED_PRECISION_CPU = False  # Too many problems
    if args.debug and args.num_epochs > 5:
        args.num_epochs = 5
        args.eval_steps = 2

    # Shortcuts
    args.base_model = {
        "ilyes": "Ilyes/wav2vec2-large-xlsr-53-french",
        "samir": "samirt8/wav2vec2-xls-r-1b-fr",
        "facebook": "facebook/wav2vec2-base-10k-voxpopuli-ft-fr",
    }.get(args.base_model.lower(), args.base_model)
    args.base_model = args.base_model

    base_processor_name = args.base_model
    if args.base_model in ["facebook/wav2vec2-base"] or "LeBenchmark" in args.base_model:
        args.disable_first_eval = True
        base_processor_name = "Ilyes/wav2vec2-large-xlsr-53-french"

    def args_to_str(args, no_training_hyperparams=False):
        d = args.__dict__
        if no_training_hyperparams:
            # Make a dictionary with the first 8 elements (/!\ very specific to this script: the first 8 options are not learning hyperparameters)
            d = dict(list(d.items())[:8])

        s = "_".join(
            (
                "{}-{}".format(
                    "".join([a[0] for a in k.replace("-", "_").split("_")]),
                    {True: 1, False: 0}.get(v, str(v).replace("/", "_")),
                )
            )  # if v != 0 else ""
            for k, v in d.items()
            if k
            not in [
                "verbose",
                "disable_first_eval",
                "output_dir",
                "gpus",
                "eval_steps",  # No influence on the results
                "num_epochs",  # We ignore this, to be able to continue training in the same folder
                "data_augment_noise",
                "data_augment_rir",  # We ignore this arbitrarily
                "train",
                "valid",
                "debug",  # Will be handled differently
                "online",
                "no_freeze",
                "data_augment",  # Will be handled differently
            ]
        )
        if not no_training_hyperparams:
            s += ("_adamwt") + ("_nofreeze" if args.no_freeze else "") + ("_augment" if args.data_augment else "") + ("_online" if args.online else "")
        if args.debug:
            s = "DEBUG_" + s
        else:
            (_, train_path, dev_path) = remove_commonprefix([os.path.realpath(f) for f in [sys.argv[0], args.train, args.valid]], "/")
            s = hashmd5((train_path, dev_path)) + "_" + s
        while "__" in s:
            s = s.replace("__", "_")
        return "hf_" + s

    output_folder = f"{args.output_dir}/{args_to_str(args, False)}"
    untrained_dir = f"{args.output_dir}/{args_to_str(args, True)}"
    print("Output Folder:", output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    save_source_dir(output_folder)

    resume_from_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_folder)

    if True:  # not os.path.isfile(output_folder+"/README.txt"):
        readme = open(output_folder + "/README.txt", "a")

        # Print the date and time
        print(datetime.datetime.now(), file=readme)
        print(" ".join(sys.argv), file=readme)
        print(
            sys.argv[0] + " --" + " --".join([k if v is True else k + "=" + str(v) for k, v in args.__dict__.items() if v is not False]),
            file=readme,
        )
        print("", file=readme)
    else:
        readme = None

    data_augmenter = None
    if args.data_augment:
        if "[" not in args.data_augment_rir:
            raise RuntimeError("--data_augment_rir syntax must be /root/folder/[rir/file1,rir/file2,...]")
        rir_dir = args.data_augment_rir.split("[")[0].rstrip("/")
        rir_lists = args.data_augment_rir.split("[")[1].split("]")[0].split(",")
        for f in rir_lists:
            if not os.path.isfile(os.path.join(rir_dir, f)):
                raise RuntimeError(f"RIR list file {os.path.join(rir_dir, f)} does not exist")
        data_augmenter = SpeechAugment(
            noise_dir=args.data_augment_noise,
            rir_dir=rir_dir,
            rir_lists=rir_lists,
            apply_prob=1,
            sample_rate=16000,
        )

    trainsetmeta, trainset = kaldi_folder_to_dataset(
        args.train,
        shuffle=True,
        online=args.online,
        max_data=(2 * args.batch_size) if args.debug else None,
        choose_data_with_max_duration=args.debug,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        logstream=readme,
    )
    testsetmeta, testset = kaldi_folder_to_dataset(
        args.valid,
        shuffle=False,
        online=online_dev,
        max_data=(2 * args.batch_size) if args.debug else 480,
        choose_data_with_max_duration=args.debug,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        logstream=readme,
    )

    trainset = trainset.shuffle(seed=69)

    trainsetlen = trainsetmeta["samples"]
    trainsetmeta = ", ".join(f"{v} {k}" for k, v in trainsetmeta.items())
    testsetmeta = ", ".join(f"{v} {k}" for k, v in testsetmeta.items())
    print("Training set:", trainsetmeta)
    print("Test set:", testsetmeta)
    if readme:
        print("", file=readme)
        print("Training set:", trainsetmeta, file=readme)
        print("Test set:", testsetmeta, file=readme)
        print("", file=readme)
        readme.flush()

    # Load model to finetune
    processor = transformers.Wav2Vec2Processor.from_pretrained(base_processor_name)

    model = transformers.Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        layerdrop=args.layer_dropout,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        # vocab_size = len(processor.tokenizer), # Could reinitialize the last weights
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float if USE_MIXED_PRECISION_CPU else "auto",
    )
    model.config.ctc_zero_infinity = True  # Important to avoid NaN during training when the input sequence is too short
    if not args.no_freeze:
        model.freeze_feature_encoder()
    model.train(True)
    model.gradient_checkpointing_enable()

    gpu_log = open(os.path.join(output_folder, "gpu_log_{}.txt".format("-".join([str(g) for g in gpus]))), "a") if args.gpus else None

    vram_usage("START", stream=gpu_log)
    if use_gpu:
        model = model.to("cuda:" + str(gpus[0]))  # torch.device(type='cuda', index=gpus[0]).
        mem = vram_usage("Model loaded", stream=gpu_log)
        min_mem = +mem + (0.5 * mem if USE_MIXED_PRECISION else 0) + 2 * mem + mem
        print("Estimation of minimal GPU memory:", min_mem)

    batch_size_preprocessing = 8  # could be args.batch_size ?
    trainset = process_dataset(processor, trainset, data_augmenter=data_augmenter, batch_size=batch_size_preprocessing, logstream=readme)
    testset = process_dataset(processor, testset, batch_size=batch_size_preprocessing, logstream=readme)
    if readme is not None:
        readme.flush()

    # WTF?
    use_max_step = True  # args.online

    num_devices = len(gpus) or 1

    random.seed(args.seed)
    transformers.set_seed(args.seed)
    training_args = transformers.TrainingArguments(
        output_dir=output_folder,
        group_by_length=not args.online,
        per_device_train_batch_size=args.batch_size // num_devices,
        per_device_eval_batch_size=args.batch_size // num_devices,
        gradient_accumulation_steps=1,  # 2?,
        gradient_checkpointing=False,
        dataloader_num_workers=6,  # if "ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)", use 0 or increase /dev/shm
        num_train_epochs=args.num_epochs,
        max_steps=-1 if not use_max_step else round(args.num_epochs * trainsetlen / args.batch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=500,
        evaluation_strategy=transformers.IntervalStrategy.STEPS,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=2,  # Only last 2 models are saved. Older ones are deleted.
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
        resume_from_checkpoint=resume_from_checkpoint,
        fp16=USE_MIXED_PRECISION,
        fp16_full_eval=USE_MIXED_PRECISION,  # Mixed precision
        use_ipex=USE_MIXED_PRECISION_CPU,
        bf16=USE_MIXED_PRECISION_CPU,
        bf16_full_eval=USE_MIXED_PRECISION_CPU,
        # /usr/local/lib/python3.9/site-packages/intel_extension_for_pytorch/optim/_optimizer_utils.py:179:
        # Avoid UserWarning: Does not suport fused step for <class 'torch.optim.adamw.AdamW'>, will use non-fused step
        no_cuda=not use_gpu,
        optim="adamw_torch",  # Avoid warning .../transformers/optimization.py:306: FutureWarning:
        # This implementation of AdamW is deprecated and will be removed in a future version.
        # Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
        logging_dir=output_folder,
        do_eval=True,
        # logging_first_step = False, # eval perfs is not included anyway
        # eval_delay = -1,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, log_stream=gpu_log, verbose=False)
    trainer = transformers.Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=15)],
        train_dataset=trainset.with_format("torch"),
        eval_dataset=testset.with_format("torch"),
        tokenizer=processor.feature_extractor,
    )

    # Evaluate initial model
    if not args.disable_first_eval:
        init_results = output_folder + "/init_eval.json"
        if not os.path.isfile(init_results):
            init_results0 = untrained_dir + "/init_eval.json"
            if not os.path.exists(init_results0):
                print("Evaluating initial model", init_results0)
                if not os.path.exists(untrained_dir):
                    os.makedirs(untrained_dir)

                res = trainer.evaluate()
                json.dump(res, open(init_results0, "w"), indent=2)

            shutil.copy(init_results0, init_results)

    # Train
    tic()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    toc("Training", stream=readme)

    # Save model
    processor.save_pretrained(output_folder + "/final")
    model.save_pretrained(output_folder + "/final")
