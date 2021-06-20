from pytorch_lightning import LightningModule, LightningDataModule, Trainer
import torch
import numpy as np
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
    test_dataloader = datamodule.test_dataloader()
    test_result, _ = test_model(model, model.criterion, test_dataloader)
    initial_acc = test_result[0]["test/acc"]

    for epoch in range(config.iterations):
        prune_model(model, config)

        test_result, _ = test_model(model, model.criterion, test_dataloader)

        log.info(f'After Pruning: Epoch {epoch}. Acc: {test_result[0]["test/acc"]}')

        log.info(f'Start finetuning')
        trainer = Trainer(max_epochs=config.fine_tuning_epochs, gpus=1)
        trainer.fit(model=model, datamodule=datamodule)

        test_result, _ = test_model(model, model.criterion, test_dataloader)
        prune_amount = str(config.amount ** (epoch + 1))
        logger.log_metrics(format_result(test_result, f'p_{prune_amount}'))

        current_acc = test_result[0]["test/acc"]
        log.info(f'After Finetuning: Epoch {epoch}. Acc: {current_acc}')

        if initial_acc - current_acc > 0.30 or epoch == config.iterations - 1:
            model = remove_pruning(model)

            # save model
            model_path = f"{path}/model_{prune_amount}.ckpt"
            logger.log_hyperparams({f'model_{prune_amount}': model_path})
            trainer.save_checkpoint(model_path)

            tot, nonz = model_size(model)
            pruned_percent = ((tot - nonz) / tot)
            logger.log_metrics({"pruned_percent": pruned_percent})
            break


def nonzero(tensor):
    """Returns absolute number of values different from 0
    Arguments:
        tensor {numpy.ndarray} -- Array to compute over
    Returns:
        int -- Number of nonzero elements
    """
    return np.sum(tensor != 0.0)


# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def model_size(model, as_bits=False):
    """Returns absolute and nonzero model size
    Arguments:
        model {torch.nn.Module} -- Network to compute model size over
    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype
    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """

    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    return int(total_params), int(nonzero_params)
