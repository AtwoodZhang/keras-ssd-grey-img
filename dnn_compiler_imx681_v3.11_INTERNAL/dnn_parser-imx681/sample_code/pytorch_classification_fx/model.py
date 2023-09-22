# ------------------------------------------------------------------------------
# Copyright 2022 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Conv2D -> BatchNorm -> ReLU
# Note that PyTorch does not support ReLU6 when fusing modules for QAT.
# Also note that fusing modules for QAT requires bias = False.
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU(inplace=False))
    ]))

# DepthwiseConv2D -> BatchNorm -> ReLU
# Note that Pytorch does not support ReLU6 when fusing modules for QAT.
# Also note that fusing modules for QAT requires bias = False.
def dw_bn_relu(in_channels, channel_multiplier=1, kernel_size=3, stride=1, padding=1):
    out_channels = in_channels * channel_multiplier
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU(inplace=False))
    ]))

def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight) 

# Sample Classification Model
class SampleModel(nn.Module):
    def __init__(self, num_classes=10, input_size=(120,160), mode='train'):
        super(SampleModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.mode = mode
        # initialize module layers
        self.layer1 = conv_bn_relu(1, 16, kernel_size=3, stride=2, padding=1)
        self.layer2 = dw_bn_relu(16, 1, kernel_size=3, stride=2, padding=1)
        self.layer3 = conv_bn_relu(16, 32, kernel_size=1, stride=1, padding=0)
        self.layer4 = dw_bn_relu(32, 1, kernel_size=3, stride=2, padding=1)
        self.layer5 = conv_bn_relu(32, 64, kernel_size=1, stride=1, padding=0)
        self.layer6 = dw_bn_relu(64, 1, kernel_size=3, stride=2, padding=1)
        self.layer7 = conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0)
        self.layer8 = nn.MaxPool2d(2, stride=2, padding=(0,0))
        self.layer9 = nn.Linear(2560, 32)
        self.layer10 = nn.Linear(32, 10)

    def init(self):
        self.apply(_xavier_init_)

    def forward(self, x):
        # Conv/BN/ReLU layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # MaxPooling layer
        x = self.layer8(x)
        # Linear layer
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer9(x))
        # Classification layer
        x = self.layer10(x)
        # Softmax
        if self.mode == 'train':
            return x
        else:
            return F.softmax(x, dim=-1)


