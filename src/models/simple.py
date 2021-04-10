
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams['input_size'], hparams['lin1_size']),
            nn.BatchNorm1d(hparams['lin1_size']),
            nn.ReLU(),
            nn.Linear(hparams['lin1_size'], hparams['output_size'])
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        return self.model(x)


class SimpleModel(LightningModule):
    """
      Simple test model
    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        output_size: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

        self.model = SimpleDenseNet(hparams=self.hparams)

        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            'train/acc': [],
            'val/acc': [],
            'train/loss': [],
            'val/loss': [],
        }

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist['train/acc'].append(
            self.trainer.callback_metrics['train/acc'])
        self.metric_hist['train/loss'].append(
            self.trainer.callback_metrics['train/loss'])
        self.log('train/acc_best',
                 max(self.metric_hist['train/acc']), prog_bar=False)
        self.log('train/loss_best',
                 min(self.metric_hist['train/loss']), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log('val/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist['val/acc'].append(
            self.trainer.callback_metrics['val/acc'])
        self.metric_hist['val/loss'].append(
            self.trainer.callback_metrics['val/loss'])

        self.log('val/acc_best',
                 max(self.metric_hist['val/acc']), prog_bar=False)
        self.log('val/loss_best',
                 min(self.metric_hist['val/loss']), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
