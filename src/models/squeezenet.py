import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math
from torch.quantization import fuse_modules, QuantStub, DeQuantStub

from src.models.base import BaseModel


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)

        # expand 1x1
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # expand 3x3
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu3 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out1 = self.relu2(out1)

        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out = torch.cat([out1, out2], 1)
        return out

    def fuse_modules(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'],
                     ['conv3', 'bn3', 'relu3']], inplace=True)


class SqueezeNetModule(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNetModule, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)  # 32
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 16
        self.fire1 = Fire(64, 16, 64)
        self.fire2 = Fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 8
        self.fire3 = Fire(128, 32, 128)
        self.fire4 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 4
        self.fire5 = Fire(256, 48, 192)
        self.fire6 = Fire(384, 48, 192)
        self.fire7 = Fire(384, 64, 256)
        self.fire8 = Fire(512, 64, 256)

        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.maxpool2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool3(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)

        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, Fire):
                m.fuse_modules()

        fuse_modules(self, [['conv1', 'relu']], inplace=True)
        fuse_modules(self.classifier, [['1', '2']], inplace=True)


class SqueezeNet(BaseModel):
    def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = SqueezeNetModule(num_classes)

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
