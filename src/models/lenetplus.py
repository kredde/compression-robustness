import torch
from torch.quantization import fuse_modules, QuantStub, DeQuantStub

from src.models.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import math


class LeNetPlusModule(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(LeNetPlusModule, self).__init__()

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.dequant = DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.quant(x)

        x = self.features(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        x = self.dequant(x)

        return x

    def fuse_model(self):
        fuse_modules(self.features, [['0', '1', '2'], ['4', '5', '6'], ['8', '9', '10'],
                     ['12', '13', '14'], ['16', '17', '18']], inplace=True)
        fuse_modules(self.classifier, [['0', '1'], ['2', '3']], inplace=True)


class LeNetPlus(BaseModel):
    def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = LeNetPlusModule(num_classes)

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
