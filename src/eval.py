import logging

from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything
import torch
import copy
import uuid
from pathlib import Path
from omegaconf import DictConfig

from src.experiments.static_quantization import quantize_static
from src.experiments.pruning import prune
from src.utils.quantization_util import get_model_size
from src.utils.evaluation import test
from src.utils import config_utils

log = logging.getLogger(__name__)


def eval(config: DictConfig, model: LightningModule,
         trainer: Trainer, datamodule: LightningDataModule):
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

    # log path of csv files
    path = f"{config.get('log_dir')}/{uuid.uuid4()}"
    logger.log_hyperparams({'csv_path': path})
    Path(path).mkdir(parents=True, exist_ok=True)

    # if config.get('ensemble'):
    #     # save ensemble checkpoint
    #     model_path = f"{path}/model.ckpt"
    #     logger.log_hyperparams({f'model_path': model_path})
    #     trainer.accelerator.model = model
    #     trainer.save_checkpoint(model_path)

    # log test result before applying quantization
    test(model, datamodule, logger, config=config, path=path)

    if config.get('pruning'):
        log.info(f'Starting pruning: {config.pruning.method}')

        prune(model, datamodule, config.pruning, logger, path=path)

        log.info('Pruning finished')
        test(model, datamodule, logger, f'p', config, path=path)

    # quantization
    if config.get('quantization'):
        log.info(f'Starting quantization: {config.quantization.type}')

        pre_q_size = get_model_size(model)
        model.to(cpu)

        assert config.quantization.type in ['static']
        if config.quantization.type == 'static':
            q_model = quantize_static(model, datamodule.train_dataloader(), **config.quantization)

        log.info('Quantization finished')
        q_size = get_model_size(q_model)
        log.info(f"model size: {q_size}, before: {pre_q_size}")
        test(q_model, datamodule, logger, 'q', config, path=path)

    logger.finalize(status="FINISHED")
