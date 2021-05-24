import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.quantization.mobilenet import mobilenet_v3_large

from src.models.base import BaseModel


class MobileNetV2(BaseModel):
    def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = mobilenet_v3_large(num_classes=num_classes)

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200), 'name': 'cos_lr'}
        return [optimizer], [lr_scheduler]

    def fuse_model(self):
        self.model.fuse_model()
