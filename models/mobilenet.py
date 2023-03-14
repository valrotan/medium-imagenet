import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=200):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x
