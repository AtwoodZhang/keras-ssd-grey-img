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


def _xavier_init_(m: nn.Module):  # initial the weights of model.
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        
# Sample Classification Model
class SampleModel(nn.Module):
    def __init__(self, num_classes=10, input_size=(120, 160), quantize=True, mode='train'):
        super(SampleModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.quantize = quantize
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
        
        # quantization needs QuantStub and DeQuantStub; PTSQ API
        if self.quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            # List of modules specifying which layers to fuse
            # This is unique to a given model according to the names of the modules.
            self.layers_to_fuse = [ # list of layers to fuse together (e.g. conv -> batchnorm -> relu)
                [ 'layer1.conv', 'layer1.bn', 'layer1.relu' ],
                [ 'layer2.conv', 'layer2.bn', 'layer2.relu' ],
                [ 'layer3.conv', 'layer3.bn', 'layer3.relu' ],
                [ 'layer4.conv', 'layer4.bn', 'layer4.relu' ],
                [ 'layer5.conv', 'layer5.bn', 'layer5.relu' ],
                [ 'layer6.conv', 'layer6.bn', 'layer6.relu' ],
                [ 'layer7.conv', 'layer7.bn', 'layer7.relu' ],
            ]
    
    def init(self):
        self.apply(_xavier_init_)
    
    def forward(self, x):
        # QuantStub
        if self.quantize:
            x = self.quant(x)
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
        x = self.layer10(x)  # (32, 10)
        # DequantStub
        if self.quantize:
            x = self.dequant(x)
        
        # Softmax
        if self.mode == "train":
            return x
        else:
            return F.softmax(x, dim=-1)
            