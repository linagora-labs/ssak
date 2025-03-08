import os

import pytorch_lightning as pl
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from train_utils import check_vocabulary, get_base_model, setup_dataloaders


@hydra_runner(config_path="yamls", config_name="finetuning_small_model.yaml")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    # set seed
    pl.seed_everything(cfg.get("seed", 42))
    logging.info(f"Trainer seed set to: {cfg.get('seed', 42)}")

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    if hasattr(cfg, "init_from_ptl_ckpt") and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError("Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`")

    asr_model = get_base_model(trainer, cfg)
    logging.info(type(asr_model))
    if cfg.model.freeze_encoder:
        asr_model.encoder.freeze()
        # asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
    # del cfg.model.freeze_encoder

    # Check vocabulary type and update if needed
    asr_model = check_vocabulary(asr_model, cfg)

    # Setup Data
    asr_model = setup_dataloaders(asr_model, cfg)

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    if hasattr(cfg.model, "spec_augment") and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    asr_model.decoding.cfg.strategy = "greedy_batch"
    logging.info(f"Decoding strategy: {asr_model.decoding.cfg.strategy}")
    asr_model.wer.log_prediction = False

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
