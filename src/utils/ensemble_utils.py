import torch
from pytorch_lightning.metrics.classification import Accuracy
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from tqdm import tqdm


def fit_meta(model: LightningModule, trainer: Trainer, datamodule: LightningDataModule):
    # set max expochs to 5
    me = trainer.max_epochs
    trainer.max_epochs = 5

    # freeze ensemble models
    for m in model.models:
        m.freeze()
    model.train()

    # train meta learner
    trainer.fit(model, datamodule)
    trainer.max_epochs = me
