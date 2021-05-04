import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.quantization import fuse_modules, QuantStub, DeQuantStub

from src.models.base import BaseModel
from src.models.blocks.base_block import BaseBlock


class MobileNetV2Module(nn.Module):
    def __init__(self, num_classes=10, alpha=1):
        super(MobileNetV2Module, self).__init__()
        self.output_size = num_classes

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # first conv layer
        self.conv0 = nn.Conv2d(3,
                               int(32 * alpha),
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))
        self.relu0 = nn.ReLU(inplace=False)

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t=1, downsample=False),
            BaseBlock(16, 24, downsample=False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample=False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample=True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample=False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample=True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample=False))

        self.conv1 = nn.Conv2d(int(320 * alpha), 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc = nn.Linear(1280, self.output_size)

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.quant(inputs)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        x = self.bottlenecks(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):
        fuse_modules(self, [['conv0', 'bn0', 'relu0'], [
                     'conv1', 'bn1', 'relu1']], inplace=True)
        for m in self.bottlenecks:
            m.fuse_modules()


class MobileNetV2(BaseModel):
    def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = MobileNetV2Module(num_classes=num_classes)

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 225], gamma=0.01), 'name': 'step_lr'}
        return [optimizer], [lr_scheduler]

    def fuse_model(self):
        self.model.fuse_model()
