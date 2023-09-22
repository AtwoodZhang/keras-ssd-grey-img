import torch.nn as nn
import math
from collections import OrderedDict

#from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
#from brevitas.quant import IntBias
#from brevitas.inject.defaults import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
#from functools import partial

# Conv2D -> BN (optional) -> ReLU
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=False))
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('relu', nn.ReLU(inplace=False))
        ]))

# DepthwiseConv2D -> BN (optional) -> ReLU
def dw_bn_relu(in_channels, channel_multiplier=1, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
    out_channels = in_channels * channel_multiplier
    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=False))
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('relu', nn.ReLU(inplace=False))
        ]))

# 1x1 Conv2D -> BN (optional) -> ReLU
def conv_1x1_bn_relu(in_channels, out_channels, use_batch_norm=True):
    return conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_batch_norm=use_batch_norm)


# Inverted residual block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 stride=1, expand_ratio=1, channel_multiplier=1, 
                 use_batch_norm=True, keep_expansion=False):
        super(InvertedResidual, self).__init__()
        
        # only supports stride=1 or stride=2
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)
        dw_channels = hidden_dim * channel_multiplier
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.skip_add = nn.quantized.FloatFunctional()

        if expand_ratio == 1 and not keep_expansion:
            if use_batch_norm:
                self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(in_channels, dw_channels, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('bn1', nn.BatchNorm2d(dw_channels)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('conv2', nn.Conv2d(dw_channels, out_channels, 1, 1, 0, bias=False)),
                    ('bn2', nn.BatchNorm2d(out_channels)),
                ]))
            else:
                self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(in_channels, dw_channels, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('conv2', nn.Conv2d(dw_channels, out_channels, 1, 1, 0, bias=False)),
                ]))
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)),
                    ('bn1', nn.BatchNorm2d(hidden_dim)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('conv2', nn.Conv2d(hidden_dim, dw_channels, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('bn2', nn.BatchNorm2d(dw_channels)),
                    ('relu2', nn.ReLU(inplace=False)),
                    ('conv3', nn.Conv2d(dw_channels, out_channels, 1, 1, 0, bias=False)),
                    ('bn3', nn.BatchNorm2d(out_channels)),
                ]))
            else:
                self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('conv2', nn.Conv2d(hidden_dim, dw_channels, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('relu2', nn.ReLU(inplace=False)),
                    ('conv3', nn.Conv2d(dw_channels, out_channels, 1, 1, 0, bias=False)),
                ]))

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn_relu(3, input_channel, kernel_size=3, stride=2, padding=1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride=s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, stride=1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn_relu(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
