import torch
from torch.quantization import fuse_modules, QuantStub, DeQuantStub

from src.models.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class LeNetModule(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(LeNetModule, self).__init__()

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.contiguous()
        x = x.view(x.size(0), -1)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        x = self.dequant(x)

        return x

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2'], ['fc1', 'relu3'], ['fc2', 'relu4']], inplace=True)


class LeNet(BaseModel):
    def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = LeNetModule(num_classes)

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200), 'name': 'cos_lr'}
        return [optimizer], [lr_scheduler]
