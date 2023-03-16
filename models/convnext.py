import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNextBlock(nn.Module):
    def __init__(self, n_channels, layer_size):
        super(ConvNextBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=7, padding=3, groups=n_channels)
        self.ln1 = nn.LayerNorm([n_channels, layer_size, layer_size])
        self.conv2 = nn.Conv2d(n_channels, n_channels * 4, kernel_size=1)
        self.conv3 = nn.Conv2d(n_channels * 4, n_channels, kernel_size=1)

    def forward(self, x):
        out = self.ln1(self.conv1(x))
        out = F.gelu(self.conv2(out))
        out = self.conv3(out)
        out += x
        return out


class ConvNextModel(nn.Module):
    def __init__(self, num_classes=200, img_size=64, repeat_blocks=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super(ConvNextModel, self).__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.LayerNorm([96, img_size // 4, img_size // 4]),
            nn.GELU(),
        )

        body = []
        layer_size = img_size // 4
        for i in range(4):
            if i != 0:
                downsample = nn.Sequential(
                    nn.LayerNorm([dims[i - 1], layer_size, layer_size]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2, bias=False),
                )
                body.append(downsample)
                layer_size //= 2
            layer = self._make_layer(dims[i], layer_size, repeat_blocks[0])
            body.append(layer)
        body.append(nn.LayerNorm([dims[-1], img_size // 32, img_size // 32]))
        self.body = nn.Sequential(*body)

        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        fc = nn.Linear(dims[-1], num_classes)
        self.head = nn.Sequential(avgpool, flatten, fc)

    def _make_layer(self, n_planes, layer_size, n_blocks):
        layers = []

        for _ in range(n_blocks):
            layers.append(ConvNextBlock(n_planes, layer_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x


def ConvNext18(num_classes=200, img_size=64):
    return ConvNextModel(num_classes=num_classes, img_size=img_size, repeat_blocks=[2, 2, 2, 2])


def ConvNext26(num_classes=200, img_size=64):
    return ConvNextModel(num_classes=num_classes, img_size=img_size, repeat_blocks=[2, 2, 6, 2])


def ConvNext38(num_classes=200, img_size=64):
    return ConvNextModel(num_classes=num_classes, img_size=img_size, repeat_blocks=[3, 3, 9, 3])
