import logging

from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything

from omegaconf import DictConfig
from pytorch_lightning.loggers.base import LightningLoggerBase

from src.experiments.static_quantization import quantize_static
from src.utils.quantization_util import get_model_size
from src.utils import config_utils, format_result

log = logging.getLogger(__name__)


def test_model(model: LightningModule, trainer: Trainer, datamodule: LightningDataModule, logger: LightningLoggerBase, name: str = None):
    test_result = trainer.test(model, test_dataloaders=[
                               datamodule.test_dataloader()])
    logger.log_metrics(format_result(test_result, name))

    c_result = trainer.test(model, test_dataloaders=[
                            datamodule.test_c_dataloader()])
    logger.log_metrics(format_result(c_result, f'{name}_c' if name else 'c'))

    return {'t': test_result[0], 'c': c_result}


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

    logger = trainer.logger
    trainer.logger = None

    # log test result before
    result_b = test_model(model, trainer, datamodule, logger)

    # quantization
    if config.get('quantization'):
        log.info(f'Starting quantization: {config.quantization.type}')

        pre_q_size = get_model_size(model)

        assert config.quantization.type in ['static']
        if config.quantization.type == 'static':
            q_model = quantize_static(
                model, datamodule.train_dataloader(), **config.quantization)

        log.info('Quantization finished')

        result_a = test_model(q_model, trainer, datamodule, logger, 'q')
