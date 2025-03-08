#!/usr/bin/env python3

import glob
import os
import shutil

import hyperpyyaml
import speechbrain as sb

from ssak.infer.speechbrain_infer import speechbrain_load_model
from ssak.utils.yaml_utils import copy_yaml_fields, make_yaml_overrides


def finalize_folder(
    folder,
    hparams_file=None,
    delete_extra=False,
):
    # Create a symbolic link to the last checkpoint with validation
    ckpt_can_be_deleted = []
    ckpt_all = []
    best_cpkt_folder = None
    best_cpkt_WER = None
    for root, dirs, files in os.walk(folder):
        for d in sorted(dirs, reverse=True):
            if d.startswith("CKPT"):
                ckpt_folder = os.path.join(root, d)
                ckpt_all.append(ckpt_folder)
                ckpt_file = os.path.join(ckpt_folder, "CKPT.yaml")
                wer = None
                if os.path.exists(ckpt_file):
                    with open(ckpt_file) as f:
                        ckpt = hyperpyyaml.load_hyperpyyaml(f)
                        if "WER" in ckpt:
                            wer = ckpt["WER"]
                if best_cpkt_WER is None or (wer is not None and wer < best_cpkt_WER):
                    if best_cpkt_folder and best_cpkt_WER:
                        ckpt_can_be_deleted.append(best_cpkt_folder)
                    best_cpkt_WER = wer
                    best_cpkt_folder = ckpt_folder
                else:
                    ckpt_can_be_deleted.append(ckpt_folder)

    assert best_cpkt_folder is not None, f"Could not find checkpoint folder in {folder}"
    print("Choose checkpoint folder", best_cpkt_folder)
    final_folder = os.path.join(folder, "final")
    if os.path.exists(final_folder) or os.path.islink(final_folder):
        os.remove(final_folder)
    os.symlink(os.path.relpath(best_cpkt_folder, folder), final_folder)

    # Get parameters passed to the command line
    readme_file = os.path.join(folder, "README.txt")
    assert os.path.exists(readme_file), f"Could not find {readme_file}"
    args = ""
    with open(readme_file) as f:
        for line in f:
            if ".py" in line:
                args = line.split()[1:]
                break
    _, _, overrides = sb.parse_arguments(args)
    overrides = dict([[e.strip() for e in f.split(":")] for f in overrides.split("\n")])

    # Get hyperparameters yaml file
    if hparams_file is None:
        for d in sorted(os.listdir(folder), reverse=True):
            if d.startswith("src"):
                files = glob.glob(os.path.join(folder, d, "*.yaml"))
                assert len(files) == 1, f"Found {len(files)} yaml files in {root}"
                hparams_file = files[0]
                break
    assert hparams_file is not None, f"Could not find hyperparams yaml file in {folder}"
    overrides = overrides | make_yaml_overrides(hparams_file, {"augmentation": None})
    hparams_in = hyperpyyaml.load_hyperpyyaml(open(hparams_file), overrides=overrides)
    save_folder = hparams_in["save_folder"]

    # Create the hyperparams.yaml file
    keep_fields = [
        "sample_rate",
        "blank_index",
        "wav2vec2",
        "ctc_lin",
        "enc",
        "log_softmax",
    ]

    hparams_file_out = os.path.join(final_folder, "hyperparams.yaml")
    copied_fields = copy_yaml_fields(hparams_file, hparams_file_out, keep_fields, overrides=overrides)
    with open(hparams_file_out, "a") as f:
        f.write(
            """
tokenizer: !new:sentencepiece.SentencePieceProcessor

modules:
    encoder: !ref <encoder>

encoder: !new:speechbrain.nnet.containers.LengthsCapableSequential
    wav2vec2: !ref <wav2vec2>
    enc: !ref <enc>
    ctc_lin: !ref <ctc_lin>

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        wav2vec2: !ref <wav2vec2>
        enc: !ref <enc>
        ctc_lin: !ref <ctc_lin>
        tokenizer: !ref <tokenizer>

decoding_function: !name:speechbrain.decoders.ctc_greedy_decode
    blank_id: !ref <blank_index>
"""
        )
        if "wav2vec2" not in copied_fields:
            f.write(
                """
wav2vec2: !new:speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
    source: LeBenchmark/wav2vec2-FR-7K-large
    save_path: !apply:ssak.utils.misc.get_cache_dir [ speechbrain/wav2vec2_checkpoint ] # ~/.cache/speechbrain/wav2vec2_checkpoint
"""
            )

    # Copy the tokenizer
    tokenizer_in = None
    for filename in os.listdir(save_folder):
        if filename.endswith(".model") or filename == "tokenizer.ckpt":
            assert tokenizer_in is None, f"Found multiple tokenizers in {save_folder}"
            tokenizer_in = os.path.join(save_folder, filename)
    assert tokenizer_in is not None, f"Could not find tokenizer in {save_folder}"
    tokenizer_out = os.path.join(final_folder, "tokenizer.ckpt")
    shutil.copyfile(tokenizer_in, tokenizer_out)

    # Save the wav2vec2 model
    wav2vec_file = os.path.join(final_folder, "wav2vec2.ckpt")
    if not os.path.exists(wav2vec_file):
        if "wav2vec2" in hparams_in:
            wav2vec2 = hparams_in["wav2vec2"]
        else:
            model = speechbrain_load_model(hparams_in["base_model"])
            wav2vec2 = model.hparams.wav2vec2
        sb.utils.checkpoints.torch_save(wav2vec2, wav2vec_file)

    # print("-----------------------------")
    # print("\n".join(sorted(ckpt_can_be_deleted)))
    # print("+++++++++++++++++++++++++++++")
    # print("\n".join(sorted(list(set(ckpt_all) - set(ckpt_can_be_deleted)))))
    if delete_extra:
        kept = sorted(list(set(ckpt_all) - set(ckpt_can_be_deleted)))
        assert len(kept) > 0
        for ckpt_folder in sorted(ckpt_can_be_deleted):
            print("Delete", ckpt_folder)
            shutil.rmtree(ckpt_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finalize a training model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("folder", help="Folder with the trained model")
    parser.add_argument("hparams", help="hparam.yaml file", default=None, nargs="?")
    parser.add_argument("--delete-extra", help="Remove extra checkpoint folders", default=False, action="store_true")
    args = parser.parse_args()

    finalize_folder(args.folder, args.hparams, delete_extra=args.delete_extra)
