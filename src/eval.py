import logging

from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything

from omegaconf import DictConfig

from src.utils.static_quantization import quantize_static
from src.utils.quantization_util import get_model_size
from src.utils import config_utils

log = logging.getLogger(__name__)


def eval(config: DictConfig, model: LightningModule, trainer: Trainer, datamodule: LightningDataModule) -> Optional[float]:
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

    # quantization
    if config.get('quantization'):
        log.info(f'Starting quantization: {config.quantization.type}')

        pre_q_size = get_model_size(model)

        assert config.quantization.type in ['static']
        if config.quantization.type == 'static':
            q_model = quantize_static(
                model, datamodule.train_dataloader(), **config.quantization)

        log.info('Quantization finished')
        result = trainer.test(q_model, test_dataloaders=[
                              datamodule.test_dataloader()])
        print(result)
        log.info('QTEST RESULT: ' +
                 ' '.join([f'{key}: {result[0][key]}' for key in result[0].keys()]))
        log.info(
            f'Model size: Before: {pre_q_size}MB. After: {get_model_size(q_model)}MB')
