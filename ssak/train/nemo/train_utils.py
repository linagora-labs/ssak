import time

import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils import logging, model_utils
from nemo.utils.get_rank import is_global_rank_zero
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")


def get_base_model(trainer, cfg, wait=False) -> ASRModel:
    """
    Returns the base model to be fine-tuned.
    Currently supports two types of initializations:
    1) `init_from_nemo_model`, and
    2) `init_from_pretrained_model`.
    Args:
        trainer: PyTorch Lightning Trainer
        cfg: config
    Returns:
        asr_model: ASRModel instance
    """
    asr_model = None
    nemo_model_path = cfg.get("init_from_nemo_model", None)
    pretrained_name = cfg.get("init_from_pretrained_model", None)
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError("Both `init_from_nemo_model` and `init_from_pretrained_model cannot be None, should pass atleast one of them")
    elif nemo_model_path is not None:
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path, trainer=trainer)
    elif pretrained_name is not None:
        # Due to potential first time download of the model on the cluster, we need to make sure that only one
        # rank downloads the model and the others wait for the download to finish.
        num_ranks = trainer.num_devices * trainer.num_devices

        if num_ranks > 1 and is_global_rank_zero():
            asr_model = ASRModel.from_pretrained(model_name=pretrained_name, trainer=trainer)
        else:
            # Sleep on all ranks for at least 60 seconds
            if wait:
                wait_time = int(cfg.get("exp_manager", {}).get("seconds_to_sleep", 60))
                if wait_time < 60:
                    wait_time = 60

                logging.info(f"Sleeping for at least {wait_time} seconds to wait for model download to finish.")

                time.sleep(wait_time)

            # restore model from cached model dir
            override_config_path = None
            if cfg.get("model", {}).get("override_config_path", None) is not None:
                override_config_path = cfg.model.override_config_path

            asr_model = ASRModel.from_pretrained(model_name=pretrained_name, trainer=trainer, override_config_path=override_config_path)

    return asr_model


def check_vocabulary(asr_model: ASRModel, cfg):
    """
    Checks if the decoder and vocabulary of the model needs to be updated.
    If either of them needs to be updated, it updates them and returns the updated model.
    else vocabulary will be reused from the pre-trained model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated decoder and vocabulary
    """
    if hasattr(cfg.model.tokenizer, "update_tokenizer") and cfg.model.tokenizer.update_tokenizer:
        if hasattr(cfg.model.char_labels, "update_labels") and cfg.model.char_labels.update_labels:
            raise ValueError("Both `model.tokenizer.update_tokenizer` and `model.char_labels.update_labels` cannot be passed together")
        else:
            asr_model = update_tokenizer(asr_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type)
    elif hasattr(cfg.model, "char_labels") and cfg.model.char_labels.update_labels:
        asr_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning("The vocabulary of the model has been updated with provided char labels.")
    else:
        logging.info("Reusing the vocabulary from the pre-trained model.")

    return asr_model


def update_tokenizer(asr_model: ASRModel, tokenizer_dir, tokenizer_type):
    """
    Updates the tokenizer of the model and also reinitializes the decoder if the vocabulary size
    of the new tokenizer differs from that of the loaded model.
    Args:
        asr_model: ASRModel instance
        tokenizer_dir: tokenizer directory
        tokenizer_type: tokenizer type
    Returns:
        asr_model: ASRModel instance with updated tokenizer and decoder
    """
    vocab_size = asr_model.tokenizer.vocab_size
    decoder = asr_model.decoder.state_dict()
    if hasattr(asr_model, "joint"):
        joint_state = asr_model.joint.state_dict()
    else:
        joint_state = None

    if tokenizer_dir is None:
        raise ValueError("dir must be specified if update_tokenizer is True")
    logging.info("Using the tokenizer provided through config")
    asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type)
    if asr_model.tokenizer.vocab_size != vocab_size:
        logging.warning("The vocabulary size of the new tokenizer differs from that of the loaded model. As a result, finetuning will proceed with the new vocabulary, and the decoder will be reinitialized.")
    else:
        asr_model.decoder.load_state_dict(decoder)
        if joint_state is not None:
            asr_model.joint.load_state_dict(joint_state)

    return asr_model


def setup_dataloaders(asr_model: ASRModel, cfg: OmegaConf):
    """
    Sets up the training, validation and test dataloaders for the model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated dataloaders
    """
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_multiple_test_data(cfg.model.test_ds)

    return asr_model
