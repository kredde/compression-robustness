from pytorch_lightning import LightningModule, LightningDataModule, Trainer
import torch
import logging
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning.loggers.base import LightningLoggerBase

from src.utils.pruning_util import get_prunable_modules
from src.utils.evaluation import test, test_model
from src.utils.config_utils import format_result

log = logging.getLogger(__name__)

strategies = {
    'random_unstructured': torch.nn.utils.prune.random_unstructured,
    'l1_unstructured': torch.nn.utils.prune.l1_unstructured,
    'l1_structured': torch.nn.utils.prune.ln_structured,
    'random_structured': torch.nn.utils.prune.random_structured
}


def prune_model(model: LightningModule, config: DictConfig):
    """
        Applies a pruning strategy to a model
    """
    modules = get_prunable_modules(model.model)

    for module in modules:
        strategy = strategies[config.method]
        if config.strategy_conf:
            strategy(module, name="weight", amount=config.amount, **config.strategy_conf)
        else:
            strategy(module, name="weight", amount=config.amount)

    return model


def remove_pruning(model: LightningDataModule):
    """
        Removes the pruning weights and makes the pruning non-reversible
    """
    modules = get_prunable_modules(model.model)

    for module in modules:
        torch.nn.utils.prune.remove(module, "weight")

    return model


def prune(model: LightningModule, datamodule: LightningDataModule,
          config: DictConfig, logger: LightningLoggerBase, path: str = None, *args, **kwargs):
    """
        Iteratively prunes model with given strategy, amount, iterations and fine tune epochs.
        Tests model after each pruning step.
    """
    for epoch in range(config.iterations):
        model = prune_model(model, config)

        test_dataloader = datamodule.test_dataloader()
        test_result, _ = test_model(model, model.criterion, test_dataloader)

        log.info(f'After Pruning: Epoch {epoch}. Acc: {test_result[0]["test/acc"]}')

        log.info(f'Start finetuning')
        trainer = Trainer(max_epochs=config.fine_tuning_epochs, gpus=1)
        trainer.fit(model=model, datamodule=datamodule)

        test_result, _ = test_model(model, model.criterion, test_dataloader)
        prune_amount = str(config.amount ** (epoch + 1))
        logger.log_metrics(format_result(test_result, f'p_{prune_amount}'))

        model = remove_pruning(model)

        # save model
        model_path = f"{path}/model_{prune_amount}.ckpt"
        logger.log_hyperparams({f'model_{prune_amount}': model_path})
        trainer.save_checkpoint(model_path)

        log.info(f'After Finetuning: Epoch {epoch}. Acc: {test_result[0]["test/acc"]}')
