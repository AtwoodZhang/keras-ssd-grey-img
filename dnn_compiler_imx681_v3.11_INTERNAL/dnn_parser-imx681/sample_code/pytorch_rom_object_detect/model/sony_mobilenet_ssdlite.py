import torch
import math
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn, quantization
from collections import OrderedDict
from .mobilenet_v2 import InvertedResidual, dw_bn_relu, conv_bn_relu, conv_1x1_bn_relu
from .ssd import SSD, GraphPath
from .predictor import Predictor
from .config import sony_mobilenet_ssd_config as config

"""
Separable convolution building block.
  Depthwise Conv -> ReLU -> Conv (1x1)
"""
def SeperableConv2d(
    in_channels, 
    out_channels, 
    channel_multiplier=1, 
    kernel_size=1, 
    stride=1, 
    padding=0):
    
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(OrderedDict([
        ('conv1', Conv2d(in_channels=in_channels, out_channels=in_channels*channel_multiplier, 
                         kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=False)),
        ('bn1', BatchNorm2d(in_channels*channel_multiplier)),
        ('relu1', nn.ReLU(inplace=False)),
        ('conv2', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)),
        ('bn2', BatchNorm2d(out_channels))
    ]))
    
"""
Construct SSDLite + Mobilenet feature extractor model architecture.
"""
def create_sony_mobilenet_ssdlite(
    num_classes, 
    width_mult=1.0,
    use_batch_norm=True,
    quantize=True, 
    is_test=False):
    
    # Feature Extractor
    base_net = SonyMobileNet(width_mult=width_mult, use_batch_norm=use_batch_norm).features
    source_layer_indexes = [
        14, 18, 19, 20, 22
    ]
    
    extras = ModuleList([
        # no extra modules
    ])

    # Box regression headers
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(64 * width_mult), out_channels=2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(80 * width_mult), out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(64 * width_mult), out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(32 * width_mult), out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(16 * width_mult), out_channels=6 * 4, kernel_size=3, padding=1) 
    ])

    # Classification headers
    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(64 * width_mult), out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(80 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(64 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(32 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(16 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1) 
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, 
               is_test=is_test, config=config, quantize=quantize)


def create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
    
"""
Sony Mobilenet Feature Extractor Backbone
"""
class SonyMobileNet(nn.Module):
    def __init__(self, n_class=1000, width_mult=1, dropout_ratio=0.2, use_batch_norm=True):
        super(SonyMobileNet, self).__init__()
        block = InvertedResidual
        input_channel = 1
        first_channel = 8
        interverted_residual_setting = [
            # expansion_ratio, out_channels, number_repeats, stride, channel_multiplier, keep_expansion
            [1, 8, 1, 2, 1, False],
            [1, 8, 4, 2, 2, False],
            [2, 16, 1, 1, 1, True],
            [1, 16, 6, 2, 2, False],
            [2, 64, 1, 1, 1, True],
            [1, 40, 1, 2, 1, False],
            [1, 40, 2, 1, 2, False],
            [2, 80, 1, 1, 1, True],
            [1, 64, 1, 2, 1, True], 
            [1, 32, 1, 2, 1, True],
            [1, 32, 1, 2, 1, False],
            [1, 16, 1, 2, 1, False],
        ]
        
        # list of layers
        self.features = []
        
        # building first layer
        input_channel = int(input_channel * width_mult)
        first_channel = int(first_channel * width_mult)
        self.features.append(conv_bn_relu(input_channel, first_channel, kernel_size=3, stride=1, padding=1))
        input_channel = first_channel
        
        # building inverted residual blocks
        for er, oc, nr, st, cm, ke  in interverted_residual_setting:
            output_channel = int(oc * width_mult)
            for i in range(nr):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride=st,
                                               expand_ratio=er, channel_multiplier=cm,
                                               keep_expansion=ke, use_batch_norm=use_batch_norm))
                else:
                    self.features.append(block(input_channel, output_channel, stride=1,
                                               expand_ratio=er, channel_multiplier=cm,
                                               keep_expansion=ke, use_batch_norm=use_batch_norm))
                input_channel = output_channel
        
        # make it nn.Sequential
        submodule_names = ['in_conv', *[f'inv_conv_{i}' for i in range(1,22)]]
        self.features = torch.nn.Sequential(OrderedDict(list(zip(submodule_names, self.features))))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(input_channel, n_class),
        )
        self.classifier.qconfig = None

        # initialize weights
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
