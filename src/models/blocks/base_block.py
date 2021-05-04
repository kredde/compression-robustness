import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t=6, downsample=False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.skip_add = nn.quantized.FloatFunctional()

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)

        # for main path:
        c = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(
            c,
            c,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=c,
            bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        # main path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # shortcut path
        if self.shortcut:
            x = self.skip_add.add(x, inputs)

        return x

    def fuse_modules(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3']], inplace=True)
