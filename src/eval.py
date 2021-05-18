import logging

from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything
import torch
import copy

from omegaconf import DictConfig

from src.experiments.static_quantization import quantize_static
from src.utils.quantization_util import get_model_size
from src.utils.evaluation import test
from src.utils import config_utils

log = logging.getLogger(__name__)


def eval(config: DictConfig, model: LightningModule, trainer: Trainer, datamodule: LightningDataModule):
    """Contains the evaluation pipeline.

    Uses the configuration to execute the evaluation pipeline on a given model.

    args:
        config (DictConfig): Configuration composed by Hydra.
        model (LightningModule): The model that is evaluated
        trainer (Trainer)
        datamodule (LightningDataModule)
    """

    if 'seed' in config:
        seed_everything(config.seed)

    # Send some parameters from config to all lightning loggers
    log.info('Logging hyperparameters!')
    config_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        logger=trainer.logger,
    )
    cpu = torch.device("cpu")

    logger = trainer.logger
    trainer.logger = None
    model.eval()

    # log test result before applying quantization
    # test(model, datamodule, logger, config=config)

    # quantization
    if config.get('quantization'):
        log.info(f'Starting quantization: {config.quantization.type}')

        # pre_q_size = get_model_size(model)
        model.to(cpu)

        assert config.quantization.type in ['static']
        if config.quantization.type == 'static':
            q_model = quantize_static(model, datamodule.train_dataloader(), **config.quantization)

        log.info('Quantization finished')
        test(q_model, datamodule, logger, 'q', config)

    logger.finalize(status="FINISHED")
