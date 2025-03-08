#!/usr/bin/env python3

import datetime
import os
import shutil
import sys
import time

import hyperpyyaml
import speechbrain as sb
import torch
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from torch.utils.data import DataLoader

from ssak.infer.speechbrain_infer import speechbrain_cachedir, speechbrain_load_model
from ssak.train.speechbrain.wav2vec_finalize import finalize_folder
from ssak.utils.audio import load_audio
from ssak.utils.dataset import kaldi_folder_to_dataset, to_annotation_text
from ssak.utils.env import auto_device  # handle option --gpus (and set environment variables at the beginning)
from ssak.utils.misc import get_cache_dir, hashmd5, save_source_dir
from ssak.utils.monitoring import get_num_gpus
from ssak.utils.text import remove_special_words
from ssak.utils.text_utils import collapse_whitespace
from ssak.utils.yaml_utils import easy_yaml_load

USE_HF_METRIC = True
if USE_HF_METRIC:
    import datasets  # For metrics

######################################################################


# Define training procedure
class Trainer(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        # batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs, wav_lens)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            sequence = sb.decoders.ctc_greedy_decode(p_ctc, wav_lens, blank_id=self.hparams.blank_index)

            # predicted_words = self.tokenizer(sequence, task="decode_from_list")
            predicted_words = self.tokenizer.decode(sequence)

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            # target_words = self.tokenizer(target_words, task="decode_from_list")
            target_words = self.tokenizer.decode(target_words)

            target_words = [t.replace("⁇", "") for t in target_words]  # WTF
            # print("Predicted:", predicted_words)
            # print("Target:", target_words)

            if USE_HF_METRIC:
                self.wer_metric.add_batch(predictions=predicted_words, references=target_words)
                self.cer_metric.add_batch(predictions=predicted_words, references=target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        if self.auto_mix_prec:
            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            if not self.hparams.wav2vec2.freeze:
                self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.model_optimizer)

            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.model_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        self.optimizer_step += 1
        self.on_fit_batch_end(batch, outputs, loss, True)
        loss = loss.detach()
        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TRAIN:
            self.train_stats = {}
            self.batch_count = 0
            self.total_loss = 0
            print("Training epoch", epoch)
        else:
            self.eval_start_time = time.time()
            print("Validation epoch", epoch, "batch", self.total_batches)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if USE_HF_METRIC:
                stage_stats["CER"] = 100 * self.cer_metric.compute()
                stage_stats["WER"] = 100 * self.wer_metric.compute()
            else:
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                self.cer_metric.clear()
                self.wer_metric.clear()
            self.time_eval += time.time() - self.eval_start_time
            train_stats = {}
            train_stats.update(self.train_stats)
            within_epoch = "loss" not in train_stats
            if within_epoch:
                train_stats["loss"] = self.total_loss / self.batch_count

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(stage_stats["loss"])
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)
            current_time = time.time()
            for train_logger in self.hparams.train_loggers:
                train_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "epoch_finished": not within_epoch,
                        "total_samples": self.total_batches * self.batch_size,
                        "total_audio_h": self.total_frames / (self.hparams.sample_rate * 3600.0),
                        "total_tokens": self.total_tokens,
                        "train_time_h": (current_time - self.time_global - self.time_eval) / 3600.0 + self.train_time_extra,
                        "valid_time_h": self.time_eval / 3600.0 + self.valid_time_extra,
                        "lr_model": old_lr_model,
                        "lr_wav2vec": old_lr_wav2vec,
                    },
                    train_stats=train_stats,
                    valid_stats=stage_stats,
                    verbose=True,
                )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        if stage == sb.Stage.TEST:
            raise NotImplementedError(f"Not implemented for {stage}")

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if not self.batch_size:
            self.batch_size = batch.batchsize
        self.total_batches += 1
        self.total_frames += torch.count_nonzero(batch.sig.data).item()
        self.total_tokens += torch.count_nonzero(batch.tokens.data).item()
        self.total_loss += loss.detach().item()
        self.batch_count += 1
        if self.total_batches % self.hparams.eval_steps == 0:
            # intra-epoch validation
            step = self.step
            self.step = 0
            self._fit_valid(valid_set=self.valid_set, epoch=self.epoch, enable=False)
            # Recover same state after valid...
            self.modules.train()
            self.step = step

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"

        if not hasattr(self, "total_batches"):
            self.total_batches = 0
            self.total_frames = 0
            self.total_tokens = 0
            self.train_time_extra = 0
            self.valid_time_extra = 0
        else:
            assert hasattr(self, "total_frames")
            assert hasattr(self, "total_tokens")
            assert hasattr(self, "train_time_extra")
            assert hasattr(self, "valid_time_extra")

        self.batch_size = 0
        self.total_loss = 0
        self.batch_count = 0
        self.time_eval = 0
        self.time_global = time.time()

        if USE_HF_METRIC:
            self.cer_metric = datasets.load_metric("cer")
            self.wer_metric = datasets.load_metric("wer")
        else:
            self.cer_metric = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)  # self.hparams.cer_computer()
            self.wer_metric = sb.utils.metric_stats.ErrorRateStats()  # self.hparams.error_rate_computer()

        # If the wav2vec encoder is unfrozen, we create the optimizer
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(self.modules.wav2vec2.parameters())
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)

        self.model_optimizer = self.hparams.model_opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Overriding fit function to add the possibility to access (self.)valid_set and (self.)epoch (for intra-epoch validation)"""

        # self.sb.core.Brain.fit(...)
        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            train_set = self.make_dataloader(train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs)
        if valid_set is not None and not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )
        self.valid_set = valid_set

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for self.epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=self.epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=self.epoch, enable=enable)

            # Debug mode only runs a few epochs
            if self.debug and self.epoch == self.debug_epochs or self._optimizer_step_limit_exceeded:
                break


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the trainer class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    DEBUG = hparams["debug"]
    if DEBUG:
        hparams["sorting"] = "descending"

    readme = open(hparams["output_folder"] + "/README.txt", "w")

    # Fill the readme
    print(datetime.datetime.now(), file=readme)
    print(" ".join(sys.argv), file=readme)
    print("", file=readme)

    trainsetmeta, traincsv = kaldi_folder_to_dataset(
        hparams["train"],
        return_format="csv",
        include_duration=True,
        shuffle=True,
        max_data=max(1, int(hparams["debug_num_batches"] * hparams["batch_size"])) if DEBUG else None,
        choose_data_with_max_duration=DEBUG,
        min_duration=hparams["min_duration"],
        max_duration=hparams["max_duration"],
        logstream=readme,
    )
    testsetmeta, testcsv = kaldi_folder_to_dataset(
        hparams["valid"],
        return_format="csv",
        include_duration=True,
        shuffle=False,
        max_data=max(1, int(2 * hparams["test_batch_size"])) if DEBUG else 480,
        choose_data_with_max_duration=DEBUG,
        min_duration=max(1, hparams["min_duration"]),
        max_duration=hparams["max_duration"],
        logstream=readme,
    )

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

    print(traincsv)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=traincsv,  # replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            # key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            # key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["dataloader_options"]["shuffle"] = True

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=testcsv,
    )

    # We sort the validation data by duration so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("path", "start", "end")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(path, start, end):
        audio = load_audio(path, start, end, sample_rate=hparams["sample_rate"], return_format="torch")
        return audio

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    pretrained_tokenizer = "base_model" in hparams and hparams["base_model"] in ["speechbrain/asr-wav2vec2-commonvoice-fr"]

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("tokens")
    def text_pipeline(wrd):
        wrd = remove_special_words(wrd)
        if pretrained_tokenizer:
            wrd = wrd.upper()
            wrd = wrd.replace("'", " ")
            wrd = collapse_whitespace(wrd)
        tokens_list = tokenizer.encode_as_ids(wrd)
        return torch.LongTensor(tokens_list)

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens"],
    )

    print("Training set:", len(train_data), "samples")
    print("Validation set:", len(valid_data), "samples")

    return train_data, valid_data


def to_dict(obj):
    """Converts a class instance to a dictionary"""
    if hasattr(obj, "keys"):
        return obj
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


def find_sub_obj_of_type(obj, t):
    """Finds a sub-object of a given type in a class instance"""
    if isinstance(obj, t):
        yield obj
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if isinstance(v, t):
                yield v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, t):
                        yield v2
            elif isinstance(v, list):
                for v2 in v:
                    if isinstance(v2, t):
                        yield v2
            elif isinstance(v, tuple):
                for v2 in v:
                    if isinstance(v2, t):
                        yield v2
            elif isinstance(v, set):
                for v2 in v:
                    if isinstance(v2, t):
                        yield v2
            elif isinstance(v, type):
                if issubclass(v, t):
                    yield v
            elif isinstance(v, object):
                for i in find_sub_obj_of_type(v, t):
                    yield i


if __name__ == "__main__":
    # Remove --gpus option from sys.argv (because it is not handled by speeechbrain)
    args = []
    skip = False
    for arg in sys.argv[1:]:
        if arg.startswith("--gpus") or skip:
            skip = False
            if not arg.startswith("--gpus="):
                skip = True
            continue
        args.append(arg)

    # Parse options
    try:
        hparams_file, run_opts, overrides = sb.parse_arguments(args)
    except SystemExit as err:
        # Print options found in the yaml file
        if "--help" in args or "-h" in args:
            fargs = [a for a in args if not a.startswith("-")]
            if len(fargs) > 0:
                hparams_file = fargs[0]
                import argparse

                parser = argparse.ArgumentParser(
                    usage=f"\n{os.path.basename(sys.argv[0])} {hparams_file} [options]",
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                )
                for key, value in easy_yaml_load(hparams_file, default="Mandatory").items():
                    if isinstance(value, (int, float, str, bool)):
                        optional = value != "Mandatory"
                        parser.add_argument(
                            "--" + key,
                            default=value if optional else None,
                            help="Optional" if optional else "Mandatory",
                            type=type(value),
                            metavar=str(type(value)).split("'")[1].upper(),
                        )
                parser.parse_args(args)
        sys.exit(err.code)

    hparams = hyperpyyaml.load_hyperpyyaml(open(hparams_file), overrides + "\ndebug: " + str(run_opts["debug"]))

    DEBUG = hparams["debug"]

    device = auto_device()

    run_opts = run_opts | {
        "device": str(device),
        "data_parallel_backend": get_num_gpus() > 1,
        "ckpt_interval_minutes": 0.00001 if DEBUG else hparams["ckpt_interval_minutes"],
        "nonfinite_patience": 0,
        "debug": False,
        "debug_batches": 3,
        "debug_epochs": 3,
    }
    print("- " + "\n- ".join([f"{k}: {v}" for k, v in run_opts.items()]))

    # Make output folder
    output_folder = hparams["output_folder"]
    print("Training in", output_folder)
    save_source_dir(output_folder)  # , [sb, hyperpyyaml])

    freeze = hparams["freeze_wav2vec"]

    if "base_model" not in hparams:
        assert "modules" in hparams
        modules = hparams["modules"]

    else:
        model_name = hparams["base_model"]

        model = speechbrain_load_model(model_name, device=device)
        model = model.train(True)
        model = model.requires_grad_(True)

        # "/home/jlouradour/.cache/speechbrain",
        modules = model.mods.encoder

        modules.wav2vec2.freeze = freeze

        hparams["model"] = torch.nn.ModuleList([modules.enc, modules.ctc_lin])
        hparams["wav2vec2"] = modules.wav2vec2

    print("- " + "\n- ".join([f"{k}: {v}" for k, v in hparams.items() if isinstance(v, (int, float, str, bool))]))

    # Set seed for reproducibility
    seed = hparams["seed"]
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer_dir = hparams["save_folder"]
    if "token_type" in hparams:
        # "Train" the tokenizer
        cdir = get_cache_dir("linacache/" + hashmd5(hparams["train"]))
        txt_corpus = cdir + "/text.txt"
        if not os.path.isfile(txt_corpus):
            os.makedirs(cdir, exist_ok=True)
            with open(txt_corpus, "a") as fout:
                for text in to_annotation_text(hparams["train"]):
                    text = remove_special_words(text)
                    fout.write(text + "\n")
        if not os.path.exists(tokenizer_dir + "/text.txt"):
            os.symlink(os.path.realpath(txt_corpus), tokenizer_dir + "/text.txt")
        tokenizer = SentencePiece(
            model_dir=tokenizer_dir,
            vocab_size=hparams["output_neurons"],
            annotation_train=txt_corpus,
            annotation_read="text",
            model_type=hparams["token_type"],
            character_coverage=hparams["character_coverage"],
        )
        tokenizer = tokenizer.sp
    else:
        tokenizer = model.tokenizer

        # This is an ugly copy, but I could not find out how to properly serialize the tokenizer (SentencePieceProcessor object, it has a load method, but no save method)
        cachedir = speechbrain_cachedir(model_name)
        shutil.copyfile(cachedir + "/tokenizer.ckpt", tokenizer_dir + "/tokenizer.ckpt")

    train_data, valid_data = dataio_prepare(hparams, tokenizer)

    trainer = Trainer(
        modules=modules,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        # opt_class=lambda x: torch.optim.SGD(x, 1e-5),
    )
    trainer.tokenizer = tokenizer
    trainer.sample_rate = hparams["sample_rate"]

    if not freeze:
        hparams["checkpointer"].add_recoverable("wav2vec2", trainer.modules.wav2vec2)
    hparams["checkpointer"].add_recoverable("enc", trainer.modules.enc)
    hparams["checkpointer"].add_recoverable("ctc_lin", trainer.modules.ctc_lin)

    # If resuming from checkpoint, load previous values
    if os.path.isfile(output_folder + "/train_log.txt"):
        with open(output_folder + "/train_log.txt") as fin:
            last_line = fin.readlines()[-1].strip()

        def extract(what, t=int):
            return t(last_line.split(what + ":")[-1].split(",")[0])

        trainer.total_batches = extract("total_samples") // hparams["batch_size"]
        trainer.total_frames = extract("total_audio_h", float) * (hparams["sample_rate"] * 3600)
        trainer.total_tokens = extract("total_tokens")
        trainer.train_time_extra = extract("train_time_h", float)
        trainer.valid_time_extra = extract("valid_time_h", float)

    trainer.fit(
        trainer.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    finalize_folder(output_folder)
