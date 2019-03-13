import torch
import torch.nn as nn

import random
import numpy as np
from collections import OrderedDict

import hparams as hp
import audio


class PreLinear(nn.Module):
    """Pre Linear"""

    def __init__(self, input_size=hp.pre_input_size, output_size=hp.pre_output_size):
        super(PreLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        out = self.fc(x)
        return out


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = self.conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet"""

    def __init__(self, block=ResidualBlock, layers=hp.layers):
        super(ResNet, self).__init__()
        self.in_channels = 8
        self.conv = self.conv3x3(1, 8)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 8, layers[0])
        self.layer2 = self.make_layer(block, 16, layers[1], 2)
        self.layer3 = self.make_layer(block, 32, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)

    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1)or (self.in_channels != out_channels):
            downsample = nn.Sequential(self.conv3x3(
                self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(* layers)

    def forward(self, x):
        # torch.Size([2, 32, 45, 64])
        # torch.Size([2, 32, 5, 8])
        # torch.Size([2, 1280])

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return out


if __name__ == "__main__":
    # Test
    test_model = ResNet(ResidualBlock, [2, 2, 2])
    print(test_model)

    test_input = torch.randn(2, 1, 180, 256)
    output = test_model(test_input)
    print(output.size())
